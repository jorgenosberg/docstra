# File: ./docstra/core/llm/openai.py

"""
OpenAI API integration for LLM interactions.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Union

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from docstra.core.llm.prompt import PromptBuilder
from docstra.core.tracking.llm_tracker import (
    UniversalLLMTracker,
    get_global_tracker,
)


class OpenAIClient:
    """Client for interacting with OpenAI's models."""

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        enable_tracking: bool = True,
    ):
        """Initialize the OpenAI client.

        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0.0 to 1.0)
            enable_tracking: Whether to enable usage tracking
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_tracking = enable_tracking

        # Get API key from parameter or environment variable
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()

        # Initialize tracker
        if self.enable_tracking:
            self.tracker: Optional[UniversalLLMTracker] = get_global_tracker()
        else:
            self.tracker = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(self, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response from OpenAI.

        Args:
            prompt: Prompt for generation
            metadata: Optional metadata for tracking

        Returns:
            Generated response
        """
        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            output_text = response.choices[0].message.content or ""
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Track usage if enabled
            if self.tracker:
                # Extract token usage from response
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else None
                output_tokens = usage.completion_tokens if usage else None

                self.tracker.track_llm_call(
                    provider="openai",
                    model=self.model_name,
                    input_text=prompt,
                    output_text=output_text,
                    duration_ms=duration_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    metadata=metadata,
                )

            return output_text

        except Exception as e:
            # Track error if tracking enabled
            if self.tracker:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                error_metadata = (metadata or {}).copy()
                error_metadata.update({"error": str(e), "status": "error"})

                self.tracker.track_llm_call(
                    provider="openai",
                    model=self.model_name,
                    input_text=prompt,
                    output_text="",
                    duration_ms=duration_ms,
                    input_tokens=0,
                    output_tokens=0,
                    metadata=error_metadata,
                )

            # Log the error and re-raise for retry
            print(f"Error in OpenAI API call: {str(e)}")
            raise

    def document_code(
        self, code: str, language: str, additional_context: str = ""
    ) -> str:
        """Generate documentation for code.

        Args:
            code: Code to document
            language: Programming language
            additional_context: Additional context about the code

        Returns:
            Generated documentation
        """
        prompt = self.prompt_builder.build_document_code_prompt(
            code=code, language=language, additional_context=additional_context
        )

        return self.generate(
            prompt, metadata={"request_type": "document_code", "language": language}
        )

    def explain_code(
        self, code: str, language: str, additional_context: str = ""
    ) -> str:
        """Generate an explanation for code.

        Args:
            code: Code to explain
            language: Programming language
            additional_context: Additional context about the code

        Returns:
            Generated explanation
        """
        prompt = self.prompt_builder.build_explain_code_prompt(
            code=code, language=language, additional_context=additional_context
        )

        return self.generate(
            prompt, metadata={"request_type": "explain_code", "language": language}
        )

    def answer_question(
        self, question: str, context: Union[str, List[Dict[str, Any]]]
    ) -> str:
        """Answer a question based on context.

        Args:
            question: User question
            context: Context for answering the question

        Returns:
            Generated answer
        """
        prompt = self.prompt_builder.build_answer_question_prompt(
            question=question, context=context
        )

        # Calculate context size for metadata
        context_size = 0
        if isinstance(context, str):
            context_size = len(context)
        elif isinstance(context, list):
            context_size = sum(len(str(item)) for item in context)

        return self.generate(
            prompt,
            metadata={
                "request_type": "answer_question",
                "context_size": context_size,
                "question_length": len(question),
            },
        )

    def generate_examples(
        self, request: str, language: str, additional_context: str = ""
    ) -> str:
        """Generate code examples.

        Args:
            request: Request for examples
            language: Programming language
            additional_context: Additional context for the examples

        Returns:
            Generated examples
        """
        prompt = self.prompt_builder.build_generate_examples_prompt(
            request=request, language=language, additional_context=additional_context
        )

        return self.generate(
            prompt, metadata={"request_type": "generate_examples", "language": language}
        )

    def custom_request(self, template_name: str, **kwargs) -> str:
        """Make a custom request using a template.

        Args:
            template_name: Name of the template to use
            **kwargs: Values for template placeholders

        Returns:
            Generated response
        """
        prompt = self.prompt_builder.build_custom_prompt(
            template_name=template_name, **kwargs
        )

        return self.generate(
            prompt, metadata={"request_type": "custom", "template": template_name}
        )

    def add_template(self, name: str, template: str) -> None:
        """Add a new template or override an existing one.

        Args:
            name: Template name
            template: Template string
        """
        self.prompt_builder.add_template(name, template)

    def get_last_usage(self) -> Dict[str, Any]:
        """Get usage information from the last request.

        Returns:
            Dictionary containing usage information
        """
        if self.tracker and self.tracker.session_stats:
            return self.tracker.session_stats[-1]
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
        }
