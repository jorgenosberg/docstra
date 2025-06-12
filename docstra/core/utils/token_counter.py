"""
Token counting utilities for different LLM models and providers.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


class TokenCounter(ABC):
    """Abstract base class for token counters."""
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        pass
    
    @abstractmethod
    def estimate_max_context(self) -> int:
        """Get estimated maximum context window size."""
        pass


class TikTokenCounter(TokenCounter):
    """Token counter using tiktoken for OpenAI models."""
    
    def __init__(self, model_name: str):
        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken is required for OpenAI token counting")
        
        self.model_name = model_name
        
        # Model name mappings for tiktoken
        model_mappings = {
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo",
            "gpt-4o": "gpt-4o",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "text-embedding-ada-002": "text-embedding-ada-002",
            "text-embedding-3-small": "text-embedding-3-small",
            "text-embedding-3-large": "text-embedding-3-large",
        }
        
        tiktoken_model = model_mappings.get(model_name, "gpt-3.5-turbo")
        
        try:
            self.encoding = tiktoken.encoding_for_model(tiktoken_model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoding.encode(text))
    
    def estimate_max_context(self) -> int:
        """Get estimated maximum context window for OpenAI models."""
        context_limits = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-3.5-turbo": 4096,
        }
        return context_limits.get(self.model_name, 4096)


class AnthropicTokenCounter(TokenCounter):
    """Token counter for Anthropic Claude models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def count_tokens(self, text: str) -> int:
        """Estimate tokens for Anthropic models (roughly 4 chars per token)."""
        # Anthropic uses a different tokenizer, but this is a reasonable approximation
        # More accurate counting would require the actual Anthropic tokenizer
        return len(text) // 4
    
    def estimate_max_context(self) -> int:
        """Get estimated maximum context window for Anthropic models."""
        context_limits = {
            "claude-3-haiku": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-opus": 200000,
            "claude-3.5-sonnet": 200000,
            "claude-instant": 100000,
            "claude-2": 100000,
        }
        return context_limits.get(self.model_name, 100000)


class GenericTokenCounter(TokenCounter):
    """Generic token counter for local and other models."""
    
    def __init__(self, model_name: str, estimated_context: int = 4096):
        self.model_name = model_name
        self.estimated_context = estimated_context
    
    def count_tokens(self, text: str) -> int:
        """Estimate tokens based on character count (rough approximation)."""
        # Very rough approximation: ~4 characters per token
        return len(text) // 4
    
    def estimate_max_context(self) -> int:
        """Get estimated maximum context window."""
        return self.estimated_context


def get_token_counter(model_name: str, provider: str) -> TokenCounter:
    """Get appropriate token counter for the given model and provider."""
    
    if provider.lower() == "openai":
        if HAS_TIKTOKEN:
            return TikTokenCounter(model_name)
        else:
            # Fallback to generic counter
            context_limits = {
                "gpt-4": 8192,
                "gpt-4-turbo": 128000,
                "gpt-4o": 128000,
                "gpt-3.5-turbo": 4096,
            }
            estimated_context = context_limits.get(model_name, 4096)
            return GenericTokenCounter(model_name, estimated_context)
    
    elif provider.lower() == "anthropic":
        return AnthropicTokenCounter(model_name)
    
    elif provider.lower() == "ollama":
        # Common Ollama model context sizes
        ollama_contexts = {
            "llama3.2": 131072,
            "llama3.1": 131072, 
            "llama3": 8192,
            "llama2": 4096,
            "mistral": 8192,
            "codellama": 16384,
            "gemma": 8192,
            "qwen": 32768,
        }
        
        # Try to match model name
        estimated_context = 4096  # Default
        for model_key, context in ollama_contexts.items():
            if model_key in model_name.lower():
                estimated_context = context
                break
        
        return GenericTokenCounter(model_name, estimated_context)
    
    else:
        # Generic fallback for local models
        return GenericTokenCounter(model_name, 4096)


class ContextBudgetManager:
    """Manages token budgets and context allocation for different modes."""
    
    def __init__(self, token_counter: TokenCounter, context_mode: str = "balanced"):
        self.token_counter = token_counter
        self.context_mode = context_mode
        self.max_context = token_counter.estimate_max_context()
        
        # Reserve space for response (20% of context window)
        self.response_reserve = int(self.max_context * 0.2)
        self.available_context = self.max_context - self.response_reserve
        
        # Context mode budgets
        self.mode_multipliers = {
            "compact": 0.3,    # Use 30% of available context
            "balanced": 0.6,   # Use 60% of available context  
            "detailed": 0.9    # Use 90% of available context
        }
    
    def get_context_budget(self) -> int:
        """Get the token budget for the current context mode."""
        multiplier = self.mode_multipliers.get(self.context_mode, 0.6)
        return int(self.available_context * multiplier)
    
    def fits_in_budget(self, text: str) -> bool:
        """Check if text fits within the current context budget."""
        token_count = self.token_counter.count_tokens(text)
        return token_count <= self.get_context_budget()
    
    def get_budget_info(self) -> Dict[str, Any]:
        """Get information about the current budget allocation."""
        budget = self.get_context_budget()
        return {
            "mode": self.context_mode,
            "max_context": self.max_context,
            "response_reserve": self.response_reserve,
            "available_context": self.available_context,
            "context_budget": budget,
            "budget_percentage": (budget / self.max_context) * 100
        }
    
    def truncate_to_budget(self, text: str, preserve_end: bool = False) -> str:
        """Truncate text to fit within the context budget."""
        budget = self.get_context_budget()
        
        if self.fits_in_budget(text):
            return text
        
        # Binary search for the right length
        chars = list(text)
        left, right = 0, len(chars)
        best_length = 0
        
        while left <= right:
            mid = (left + right) // 2
            
            if preserve_end:
                test_text = "".join(chars[len(chars) - mid:])
            else:
                test_text = "".join(chars[:mid])
            
            if self.token_counter.count_tokens(test_text) <= budget:
                best_length = mid
                left = mid + 1
            else:
                right = mid - 1
        
        if preserve_end:
            truncated = "".join(chars[len(chars) - best_length:])
            return f"...[truncated]\n{truncated}"
        else:
            truncated = "".join(chars[:best_length])
            return f"{truncated}\n[truncated]..."


def count_tokens_in_messages(messages: List[Dict[str, str]], token_counter: TokenCounter) -> int:
    """Count total tokens in a list of chat messages."""
    total = 0
    for message in messages:
        # Count tokens in role and content
        total += token_counter.count_tokens(message.get("role", ""))
        total += token_counter.count_tokens(message.get("content", ""))
        # Add overhead for message formatting (rough estimate)
        total += 4
    return total


def format_context_with_budget(
    context_parts: Dict[str, str], 
    budget_manager: ContextBudgetManager,
    priorities: Optional[List[str]] = None
) -> str:
    """Format context parts within budget, prioritizing important sections."""
    
    if priorities is None:
        priorities = list(context_parts.keys())
    
    formatted_parts = []
    remaining_budget = budget_manager.get_context_budget()
    
    for priority_key in priorities:
        if priority_key not in context_parts:
            continue
            
        content = context_parts[priority_key]
        tokens_needed = budget_manager.token_counter.count_tokens(content)
        
        if tokens_needed <= remaining_budget:
            # Fits completely
            formatted_parts.append(content)
            remaining_budget -= tokens_needed
        elif remaining_budget > 100:  # Only truncate if we have meaningful space left
            # Truncate to fit
            truncated = budget_manager.truncate_to_budget(content)
            # Recalculate tokens for truncated content
            actual_tokens = budget_manager.token_counter.count_tokens(truncated)
            if actual_tokens <= remaining_budget:
                formatted_parts.append(truncated)
                remaining_budget -= actual_tokens
        
        if remaining_budget <= 50:  # Stop if very little budget remains
            break
    
    return "\n\n".join(formatted_parts)