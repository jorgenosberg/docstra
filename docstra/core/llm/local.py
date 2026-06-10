# File: ./docstra/core/llm/local.py

"""
Local model integration for LLM interactions using transformers.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )
except ImportError:
    raise ImportError(
        "Local model support requires transformers and torch. "
        "Install the project dependencies with: uv sync"
    )

from docstra.core.llm.prompt import PromptBuilder
from docstra.core.tracking.llm_tracker import (
    UniversalLLMTracker,
    get_global_tracker,
)


class KeywordsStoppingCriteria(StoppingCriteria):
    """Stopping criteria based on keywords."""

    def __init__(self, keywords, tokenizer):
        self.keywords = keywords
        self.tokenizer = tokenizer
        # Pre-encode keywords for efficiency
        self.keyword_ids = []
        for keyword in keywords:
            ids = self.tokenizer.encode(keyword, add_special_tokens=False)
            if ids:
                self.keyword_ids.append(ids)

    def __call__(self, input_ids, scores, **kwargs):
        # Check if any of the keyword sequences appear at the end of the generated sequence
        for keyword_ids in self.keyword_ids:
            if len(keyword_ids) <= input_ids.shape[1]:
                # Check if the last tokens match the keyword
                if input_ids[0, -len(keyword_ids) :].tolist() == keyword_ids:
                    return True
        return False


class LocalModelClient:
    """Client for interacting with local transformer models."""

    def __init__(
        self,
        model_name: str = "TheBloke/Llama-2-7b-Chat-GGUF",
        model_path: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        device: str = "auto",
        enable_tracking: bool = True,
    ):
        """Initialize the local model client.

        Args:
            model_name: Name or path of the model to use
            model_path: Optional local path to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0.0 to 1.0)
            device: Device to run the model on ('auto', 'cpu', 'cuda', etc.)
            enable_tracking: Whether to enable usage tracking
        """
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_tracking = enable_tracking

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize tracker
        if self.enable_tracking:
            self.tracker: Optional[UniversalLLMTracker] = get_global_tracker()
        else:
            self.tracker = None

        # Initialize model and tokenizer
        print(f"Loading model {self.model_path} on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        # Configure stopping criteria
        self.stopping_keywords = ["<|endoftext|>", "<|im_end|>", "</s>"]

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response from the local model.

        Args:
            prompt: Prompt for generation
            stream: Whether to stream the response
            metadata: Optional metadata for tracking

        Returns:
            Generated response
        """
        start_time = time.perf_counter()

        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Setup stopping criteria
            stopping_criteria = StoppingCriteriaList(
                [KeywordsStoppingCriteria(self.stopping_keywords, self.tokenizer)]
            )

            # Generate
            if stream:
                # Setup streamer
                streamer = TextIteratorStreamer(
                    self.tokenizer, skip_prompt=True, skip_special_tokens=True
                )

                # Start generation in a separate thread
                generation_kwargs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "do_sample": self.temperature > 0,
                    "stopping_criteria": stopping_criteria,
                    "streamer": streamer,
                }

                # Start generation thread
                import threading

                thread = threading.Thread(
                    target=self.model.generate, kwargs=generation_kwargs
                )
                thread.start()

                # Stream tokens
                generated_text = ""
                for text in streamer:
                    generated_text += text

                output_text = str(generated_text)
            else:
                # Generate in one go
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    stopping_criteria=stopping_criteria,
                )

                # Decode the generated tokens
                output_text = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                )

            # Track usage if enabled
            if self.tracker:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000

                # For local models, we can get exact token counts
                input_tokens = inputs.input_ids.shape[1]
                output_tokens = len(
                    self.tokenizer.encode(output_text, add_special_tokens=False)
                )

                self.tracker.track_llm_call(
                    provider="local",
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
                    provider="local",
                    model=self.model_name,
                    input_text=prompt,
                    output_text="",
                    duration_ms=duration_ms,
                    input_tokens=0,
                    output_tokens=0,
                    metadata=error_metadata,
                )

            print(f"Error in local model generation: {str(e)}")
            raise

    def document_code(
        self,
        code: str,
        language: str,
        additional_context: str = "",
        stream: bool = False,
    ) -> str:
        """Generate documentation for code.

        Args:
            code: Code to document
            language: Programming language
            additional_context: Additional context about the code
            stream: Whether to stream the response

        Returns:
            Generated documentation
        """
        prompt = self.prompt_builder.build_document_code_prompt(
            code=code, language=language, additional_context=additional_context
        )

        return self.generate(
            prompt,
            stream=stream,
            metadata={"request_type": "document_code", "language": language},
        )

    def explain_code(
        self,
        code: str,
        language: str,
        additional_context: str = "",
        stream: bool = False,
    ) -> str:
        """Generate an explanation for code.

        Args:
            code: Code to explain
            language: Programming language
            additional_context: Additional context about the code
            stream: Whether to stream the response

        Returns:
            Generated explanation
        """
        prompt = self.prompt_builder.build_explain_code_prompt(
            code=code, language=language, additional_context=additional_context
        )

        return self.generate(
            prompt,
            stream=stream,
            metadata={"request_type": "explain_code", "language": language},
        )

    def answer_question(
        self,
        question: str,
        context: Union[str, List[Dict[str, Any]]],
        stream: bool = False,
    ) -> str:
        """Answer a question based on context.

        Args:
            question: User question
            context: Context for answering the question
            stream: Whether to stream the response

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
            stream=stream,
            metadata={
                "request_type": "answer_question",
                "context_size": context_size,
                "question_length": len(question),
            },
        )

    def generate_examples(
        self,
        request: str,
        language: str,
        additional_context: str = "",
        stream: bool = False,
    ) -> str:
        """Generate code examples.

        Args:
            request: Request for examples
            language: Programming language
            additional_context: Additional context for the examples
            stream: Whether to stream the response

        Returns:
            Generated examples
        """
        prompt = self.prompt_builder.build_generate_examples_prompt(
            request=request, language=language, additional_context=additional_context
        )

        return self.generate(
            prompt,
            stream=stream,
            metadata={"request_type": "generate_examples", "language": language},
        )

    def custom_request(self, template_name: str, stream: bool = False, **kwargs) -> str:
        """Make a custom request using a template.

        Args:
            template_name: Name of the template to use
            stream: Whether to stream the response
            **kwargs: Values for template placeholders

        Returns:
            Generated response
        """
        prompt = self.prompt_builder.build_custom_prompt(
            template_name=template_name, **kwargs
        )

        return self.generate(
            prompt,
            stream=stream,
            metadata={"request_type": "custom", "template": template_name},
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
