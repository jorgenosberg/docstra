# File: ./docstra/core/services/chat_service.py
"""
Service responsible for handling interactive chat sessions with the codebase.
"""

import datetime
import json
import sqlite3  # For session storage
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console

from docstra.core.config.settings import UserConfig, ModelProvider
from docstra.core.llm.base import LLMClient  # Assuming a base class or common interface
from docstra.core.llm.anthropic import AnthropicClient
from docstra.core.llm.local import LocalModelClient
from docstra.core.llm.ollama import OllamaClient
from docstra.core.llm.openai import OpenAIClient
from docstra.core.services.query_service import QueryService
from docstra.core.utils.token_counter import (
    get_token_counter,
    ContextBudgetManager,
    count_tokens_in_messages,
)


# Database schema constants
SESSIONS_TABLE = "chat_sessions"
MESSAGES_TABLE = "chat_messages"


def _get_llm_client_for_chat_service(
    config: UserConfig, callbacks: Optional[List[Any]] = None
):
    """
    Helper to get LLM client based on config, applying callbacks.
    Ensures DocstraStatsCallbackHandler is included.
    """
    provider = config.model.provider

    # Currently only AnthropicClient and OpenAIClient support callbacks
    # So we need to handle each case separately
    if provider == ModelProvider.ANTHROPIC:
        return AnthropicClient(
            model_name=config.model.model_name_chat
            or config.model.model_name,  # Prefer chat-specific model
            api_key=config.model.api_key,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
        )
    elif provider == ModelProvider.OPENAI:
        return OpenAIClient(
            model_name=config.model.model_name_chat or config.model.model_name,
            api_key=config.model.api_key,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
        )
    elif provider == ModelProvider.OLLAMA:
        return OllamaClient(
            model_name=config.model.model_name_chat or config.model.model_name,
            api_base=config.model.api_base or "http://localhost:11434",
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            validate_connection=False,  # Don't validate during service creation
        )
    elif provider == ModelProvider.LOCAL:
        return LocalModelClient(
            model_name=config.model.model_name_chat or config.model.model_name,
            model_path=config.model.model_path,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            device=config.model.device,
            # callbacks removed
        )
    else:
        raise ValueError(f"Unsupported model provider in ChatService: {provider}")


class ChatService:
    """
    Manages interactive chat sessions, including history and context.
    """

    def __init__(
        self,
        user_config: UserConfig,
        console: Optional[Console] = None,
        callbacks: Optional[List[Any]] = None,
    ):
        self.user_config = user_config
        self.console = console or Console()
        self.callbacks = callbacks  # Will be passed to _get_llm_client_for_chat_service

        self.llm_client: LLMClient = _get_llm_client_for_chat_service(
            self.user_config, self.callbacks
        )

        # QueryService is used for RAG within the chat
        self.query_service = QueryService(user_config, self.console, self.callbacks)

        # Token management for conversation
        self.token_counter = get_token_counter(
            self.user_config.model.model_name_chat or self.user_config.model.model_name,
            self.user_config.model.provider,
        )
        self.budget_manager = ContextBudgetManager(
            self.token_counter, self.user_config.model.context_mode
        )

        self.db_path = self._get_db_path()
        self._ensure_db_tables()

        self.current_session_id: Optional[str] = None
        self.current_chat_history: List[Dict[str, str]] = []
        self.current_codebase_path_context: Optional[Path] = None

        # Track last interaction details for stats
        self.last_interaction: Optional[Dict[str, Any]] = None

    def _get_db_path(self) -> Path:
        persist_dir_name = self.user_config.storage.persist_directory
        base_persist_path = Path(persist_dir_name)

        # If persist_dir_name is relative, it should be relative to the project root where .docstra is.
        # The CLI usually handles making this absolute based on the current project.
        # For robustness, we resolve it here. If it's already absolute, resolve() does nothing.
        # If it's relative, it resolves against CWD, which should be project root when CLI runs.
        if not base_persist_path.is_absolute():
            # This assumes CWD is the project root, which is typical for CLI tools.
            # A more robust solution might involve finding a project root marker.
            base_persist_path = Path.cwd() / persist_dir_name

        resolved_path = base_persist_path.resolve()

        if not resolved_path.exists():
            try:
                resolved_path.mkdir(parents=True, exist_ok=True)
                self.console.print(
                    f"[dim]Created persistence directory for chat DB: {resolved_path}[/dim]"
                )
            except Exception as e:
                self.console.print(
                    f"[red]Error creating persistence directory {resolved_path}: {e}[/red]"
                )
                return Path(":memory:")  # Fallback

        return resolved_path / "chat_sessions.sqlite"

    def _get_db_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)  # Added timeout

    def _ensure_db_tables(self):
        try:
            with self._get_db_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                CREATE TABLE IF NOT EXISTS {SESSIONS_TABLE} (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_accessed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    codebase_path TEXT,
                    metadata TEXT
                )
                """
                )
                cursor.execute(
                    f"""
                CREATE TABLE IF NOT EXISTS {MESSAGES_TABLE} (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    role TEXT CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES {SESSIONS_TABLE}(id) ON DELETE CASCADE
                )
                """
                )
                conn.commit()
        except sqlite3.Error as e:
            self.console.print(
                f"[bold red]Error initializing chat database at {self.db_path}: {e}[/]"
            )
            self.db_path = Path(":memory:")
            self.console.print(
                "[yellow]Warning: Chat history will be in-memory for this session only.[/yellow]"
            )

    def start_new_session(
        self, codebase_path_str: str, name: Optional[str] = None
    ) -> str:
        self.current_session_id = str(uuid.uuid4())
        session_name = (
            name
            or f"Chat Session - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.current_codebase_path_context = Path(codebase_path_str).resolve()

        try:
            with self._get_db_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"INSERT INTO {SESSIONS_TABLE} (id, name, codebase_path, last_accessed_at) VALUES (?, ?, ?, ?)",
                    (
                        self.current_session_id,
                        session_name,
                        str(self.current_codebase_path_context),
                        datetime.datetime.now().isoformat(),
                    ),
                )
                conn.commit()
            self.current_chat_history = []
            self.console.print(
                f"New chat session started: [bold cyan]{session_name}[/] (ID: {self.current_session_id})"
            )
            self.console.print(
                f"Codebase context: [bold]{self.current_codebase_path_context}[/]"
            )
            return self.current_session_id
        except sqlite3.Error as e:
            self.console.print(
                f"[bold red]Error starting new chat session in DB: {e}[/]"
            )
            self.current_chat_history = []
            self.console.print(
                f"[yellow]Started in-memory session: {session_name}[/yellow]"
            )
            return self.current_session_id

    def load_session(self, session_id: str, codebase_path_str: str) -> bool:
        self.current_codebase_path_context = Path(codebase_path_str).resolve()
        try:
            with self._get_db_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT name, codebase_path FROM {SESSIONS_TABLE} WHERE id = ?",
                    (session_id,),
                )
                session_data = cursor.fetchone()
                if not session_data:
                    self.console.print(
                        f"[bold red]Error: Chat session with ID '{session_id}' not found.[/]"
                    )
                    return False

                session_name, stored_codebase_path_str = session_data
                stored_codebase_path = Path(stored_codebase_path_str).resolve()

                if stored_codebase_path != self.current_codebase_path_context:
                    self.console.print(
                        f"[bold yellow]Warning:[/yellow] Session '{session_name}' (ID: {session_id}) was for codebase '{stored_codebase_path}'."
                    )
                    self.console.print(
                        f"Current session context is for codebase '{self.current_codebase_path_context}'. Context may differ."
                    )

                cursor.execute(
                    f"SELECT role, content FROM {MESSAGES_TABLE} WHERE session_id = ? ORDER BY timestamp ASC",
                    (session_id,),
                )
                self.current_chat_history = [
                    {"role": row[0], "content": row[1]} for row in cursor.fetchall()
                ]
                self.current_session_id = session_id

                cursor.execute(
                    f"UPDATE {SESSIONS_TABLE} SET last_accessed_at = ?, codebase_path = ? WHERE id = ?",
                    (
                        datetime.datetime.now().isoformat(),
                        str(self.current_codebase_path_context),
                        session_id,
                    ),
                )
                conn.commit()

                self.console.print(
                    f"Resumed chat session: [bold cyan]{session_name}[/] (ID: {self.current_session_id})"
                )
                self.console.print(
                    f"Codebase context: [bold]{self.current_codebase_path_context}[/]"
                )
                # for msg in self.current_chat_history: # Optionally print history on load
                #     role_color = "cyan" if msg["role"] == "user" else "magenta"
                #     self.console.print(f"[{role_color}]{msg['role'].capitalize()}:[/] {msg['content']}")
                return True
        except sqlite3.Error as e:
            self.console.print(
                f"[bold red]Error loading chat session '{session_id}': {e}[/]"
            )
            return False

    def _add_message_to_history(
        self, role: str, content: str, metadata: Optional[Dict] = None
    ):
        if not self.current_session_id:
            self.current_chat_history.append({"role": role, "content": content})
            return

        message_id = str(uuid.uuid4())
        try:
            with self._get_db_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"INSERT INTO {MESSAGES_TABLE} (id, session_id, role, content, metadata) VALUES (?, ?, ?, ?, ?)",
                    (
                        message_id,
                        self.current_session_id,
                        role,
                        content,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                conn.commit()
            self.current_chat_history.append({"role": role, "content": content})
        except sqlite3.Error as e:
            self.console.print(
                f"[bold red]Error saving message to DB: {e}[/]. Message added to in-memory history only."
            )
            self.current_chat_history.append({"role": role, "content": content})

    def get_response(self, user_query: str) -> str:
        if not self.current_session_id or not self.current_codebase_path_context:
            self.console.print(
                "[bold red]Error: No active chat session or codebase context. Please start or load a session.[/]"
            )
            return "Error: No active session. Use `chat --session-id <id>` or start a new one."

        self._add_message_to_history("user", user_query)

        # Manage conversation context within budget
        budget_info = self.budget_manager.get_budget_info()

        # Reserve budget for RAG context and response
        rag_budget = int(budget_info["context_budget"] * 0.6)  # 60% for RAG
        conversation_budget = budget_info["context_budget"] - rag_budget

        # Trim conversation history if needed
        trimmed_history = self._trim_conversation_to_budget(conversation_budget)

        self.console.print(
            f"[dim]Chat context: {len(trimmed_history)} messages "
            f"(~{count_tokens_in_messages(trimmed_history, self.token_counter):,} tokens), "
            f"Mode: {budget_info['mode']}[/dim]"
        )

        # Use QueryService to get RAG context with reduced budget
        context_answer, sources = self.query_service.answer_question(
            question=user_query,
            codebase_path_str=str(self.current_codebase_path_context),
            n_results=3,
            suppress_status=True,  # Suppress status displays to avoid Rich Live display conflicts
        )

        # Create enhanced response using both conversation context and RAG
        enhanced_response = self._create_enhanced_response(
            user_query, context_answer, trimmed_history, sources
        )

        # Calculate token usage for this interaction
        conversation_tokens = count_tokens_in_messages(
            trimmed_history, self.token_counter
        )
        context_tokens = self.token_counter.count_tokens(context_answer)
        response_tokens = self.token_counter.count_tokens(enhanced_response)
        total_tokens = conversation_tokens + context_tokens + response_tokens

        # Store interaction details for stats
        self.last_interaction = {
            "user_query": user_query,
            "conversation_tokens": conversation_tokens,
            "context_tokens": context_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "context_mode": self.user_config.model.context_mode,
            "model_name": self.user_config.model.model_name,
            "provider": self.user_config.model.provider.value,
            "sources_count": len(sources),
        }

        response_metadata = {
            "sources": sources,
            "conversation_tokens": conversation_tokens,
            "total_messages": len(trimmed_history),
        }

        self._add_message_to_history(
            "assistant", enhanced_response, metadata=response_metadata
        )

        return enhanced_response

    def get_last_usage_summary(self) -> str:
        """Get a simple one-line summary of the last interaction's token usage."""
        if not self.last_interaction:
            return "no usage data available"

        interaction = self.last_interaction
        total_tokens = interaction["total_tokens"]
        context_tokens = interaction["context_tokens"]

        # Get context window for percentage calculation
        context_window = self.user_config.model.context_window
        if not context_window:
            context_window = self.token_counter.estimate_max_context()

        usage_percent = (total_tokens / context_window) * 100

        return f"{total_tokens:,} tokens ({usage_percent:.1f}% of {context_window:,}) | {context_tokens:,} context | {interaction['context_mode']} mode"

    def _trim_conversation_to_budget(self, budget: int) -> List[Dict[str, str]]:
        """Trim conversation history to fit within budget while preserving recent context."""

        if not self.current_chat_history:
            return []

        # Always keep the most recent user message
        recent_messages = []
        current_tokens = 0

        # Go backwards through history to keep most recent messages
        for message in reversed(self.current_chat_history):
            message_tokens = self.token_counter.count_tokens(
                f"{message.get('role', '')}: {message.get('content', '')}"
            )

            if current_tokens + message_tokens > budget and recent_messages:
                # Budget exceeded and we have at least one message
                break

            recent_messages.insert(0, message)  # Insert at beginning to maintain order
            current_tokens += message_tokens

        return recent_messages

    def _create_enhanced_response(
        self,
        user_query: str,
        rag_answer: str,
        conversation_history: List[Dict[str, str]],
        sources: List[Dict[str, Any]],
    ) -> str:
        """Create an enhanced response using conversation context and RAG results."""

        # For now, we'll use the RAG answer directly but could enhance it
        # with conversation-aware processing in the future

        # Check if this is a follow-up question
        if len(conversation_history) > 1:
            # Look for continuation words that suggest follow-up
            follow_up_indicators = [
                "also",
                "additionally",
                "furthermore",
                "what about",
                "how about",
            ]
            if any(
                indicator in user_query.lower() for indicator in follow_up_indicators
            ):
                # This might be a follow-up question
                prefix = "Based on our previous discussion and the codebase: "
                return f"{prefix}{rag_answer}"

        return rag_answer

    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            with self._get_db_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT id, name, codebase_path, created_at, last_accessed_at FROM {SESSIONS_TABLE} ORDER BY last_accessed_at DESC LIMIT ?",
                    (limit,),
                )
                sessions = []
                for row in cursor.fetchall():
                    sessions.append(
                        {
                            "id": row[0],
                            "name": row[1],
                            "codebase_path": row[2],
                            "created_at": row[3],
                            "last_accessed_at": row[4],
                        }
                    )
                return sessions
        except sqlite3.Error as e:
            self.console.print(f"[bold red]Error listing chat sessions: {e}[/]")
            return []

    def delete_session(self, session_id: str) -> bool:
        try:
            with self._get_db_conn() as conn:
                cursor = conn.cursor()
                # Check if session exists before deleting messages to avoid foreign key issues if any
                cursor.execute(
                    f"SELECT 1 FROM {SESSIONS_TABLE} WHERE id = ?", (session_id,)
                )
                if not cursor.fetchone():
                    self.console.print(
                        f"[yellow]Session '{session_id}' not found for deletion.[/yellow]"
                    )
                    return False

                cursor.execute(
                    f"DELETE FROM {MESSAGES_TABLE} WHERE session_id = ?", (session_id,)
                )
                cursor.execute(
                    f"DELETE FROM {SESSIONS_TABLE} WHERE id = ?", (session_id,)
                )
                conn.commit()

                if (
                    cursor.rowcount > 0 or True
                ):  # Deletion from messages might not return rowcount for session
                    self.console.print(
                        f"Session '{session_id}' and its messages deleted."
                    )
                    if self.current_session_id == session_id:
                        self.current_session_id = None
                        self.current_chat_history = []
                        self.current_codebase_path_context = None
                    return True
        except sqlite3.Error as e:
            self.console.print(
                f"[bold red]Error deleting session '{session_id}': {e}[/]"
            )
            return False
