# File: ./docstra/core/tracking/llm_tracker.py
"""
Utilities for tracking LLM operation statistics.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, ClassVar
import tiktoken
from pathlib import Path
import json
import datetime

# Global store for LLM stats - can be refactored for more sophisticated storage
_llm_stats_store: List[Dict[str, Any]] = []


# --- Helper functions for stats management ---
def get_llm_stats() -> List[Dict[str, Any]]:
    """Returns a copy of the collected LLM statistics."""
    return list(_llm_stats_store)


def clear_llm_stats() -> None:
    """Clears all collected LLM statistics."""
    _llm_stats_store.clear()


def _estimate_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Estimates token count for a given text using tiktoken.
    Defaults to gpt-3.5-turbo encoding if model-specific encoding is not found.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to a common encoding if the specific model is not found
        # This is a rough estimate.
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


class UniversalLLMTracker:
    """
    Universal LLM tracker that works with all providers.
    Tracks calls directly across supported providers.
    """

    # Enhanced pricing data with more models
    PRICING: ClassVar[Dict[str, Dict[str, Dict[str, float]]]] = {
        "anthropic": {
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3.5-haiku": {"input": 1.0, "output": 5.0},
            "default": {"input": 3.0, "output": 15.0},
        },
        "openai": {
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "default": {"input": 5.0, "output": 15.0},
        },
        "ollama": {
            "default": {"input": 0.0, "output": 0.0},  # Local models have no API cost
        },
        "local": {
            "default": {"input": 0.0, "output": 0.0},  # Local models have no API cost
        },
        "default": {"default": {"input": 1.0, "output": 2.0}},  # Conservative default
    }

    def __init__(self, stats_file: Optional[str] = None):
        """Initialize the universal tracker."""
        self.stats_file = stats_file
        if not self.stats_file:
            self.stats_file = str(Path.home() / ".docstra" / "llm_stats.json")

        self.session_stats: List[Dict[str, Any]] = []
        self.total_stats = {
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "total_duration_ms": 0,
        }

        # Load existing stats
        self._load_stats()

    def track_llm_call(
        self,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        duration_ms: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Track an LLM call with comprehensive metrics.

        Args:
            provider: LLM provider (openai, anthropic, ollama, local)
            model: Model name
            input_text: Input prompt text
            output_text: Generated output text
            duration_ms: Duration in milliseconds
            input_tokens: Actual input tokens (if available)
            output_tokens: Actual output tokens (if available)
            metadata: Additional metadata

        Returns:
            Dictionary with usage statistics
        """
        # Estimate tokens if not provided
        if input_tokens is None:
            input_tokens = _estimate_tokens(input_text, model)
        if output_tokens is None:
            output_tokens = _estimate_tokens(output_text, model)

        # Calculate cost
        cost = self._calculate_cost(provider, model, input_tokens, output_tokens)

        # Create usage record
        usage_record = {
            "call_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "duration_ms": duration_ms,
            "cost_usd": cost,
            "metadata": metadata or {},
        }

        # Add to session stats
        self.session_stats.append(usage_record)

        # Update totals
        self.total_stats["total_requests"] += 1
        self.total_stats["total_input_tokens"] += input_tokens
        self.total_stats["total_output_tokens"] += output_tokens
        self.total_stats["total_cost"] += cost
        self.total_stats["total_duration_ms"] += duration_ms

        # Save stats
        self._save_stats()

        return usage_record

    def _calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost based on provider and model."""
        provider_pricing = self.PRICING.get(provider.lower(), self.PRICING["default"])

        # Try to find exact model match, then fallback to default
        model_pricing = None
        for model_key in provider_pricing:
            if model_key in model.lower() or model_key == "default":
                model_pricing = provider_pricing[model_key]
                break

        if not model_pricing:
            model_pricing = provider_pricing.get(
                "default", self.PRICING["default"]["default"]
            )

        input_cost = (input_tokens / 1000) * model_pricing.get("input", 0.0)
        output_cost = (output_tokens / 1000) * model_pricing.get("output", 0.0)

        return input_cost + output_cost

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        if not self.session_stats:
            return {"message": "No LLM calls tracked in this session"}

        by_provider: Dict[str, Dict[str, Any]] = {}
        for stat in self.session_stats:
            provider = stat["provider"]
            if provider not in by_provider:
                by_provider[provider] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                    "avg_duration_ms": 0,
                }

            by_provider[provider]["requests"] += 1
            by_provider[provider]["input_tokens"] += stat["input_tokens"]
            by_provider[provider]["output_tokens"] += stat["output_tokens"]
            by_provider[provider]["cost"] += stat["cost_usd"]

        # Calculate averages
        for provider_stats in by_provider.values():
            if provider_stats["requests"] > 0:
                total_duration = sum(s["duration_ms"] for s in self.session_stats)
                provider_stats["avg_duration_ms"] = total_duration / len(
                    self.session_stats
                )

        return {
            "session_summary": self.total_stats,
            "by_provider": by_provider,
            "total_calls": len(self.session_stats),
        }

    def _load_stats(self) -> None:
        """Load existing stats from file."""
        try:
            if self.stats_file:
                stats_path = Path(self.stats_file)
                if stats_path.exists():
                    with open(stats_path, "r") as f:
                        data = json.load(f)
                        self.total_stats = data.get("totals", self.total_stats)
        except Exception as e:
            print(f"Warning: Could not load stats from {self.stats_file}: {e}")

    def _save_stats(self) -> None:
        """Save stats to file."""
        try:
            if self.stats_file:
                stats_path = Path(self.stats_file)
                stats_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data to save
            data = {
                "totals": self.total_stats,
                "last_updated": datetime.datetime.now().isoformat(),
                "recent_calls": self.session_stats[-50:],  # Keep last 50 calls
            }

            with open(stats_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save stats to {self.stats_file}: {e}")


# Global tracker instance
_global_tracker = UniversalLLMTracker()


def get_global_tracker() -> UniversalLLMTracker:
    """Get the global tracker instance."""
    return _global_tracker


# Example of how to potentially track embedding usage alongside direct model calls
# This would require modifying how embeddings are called.
# For now, this module focuses on direct LLM calls.

# def track_embedding_call(func):
#     """
#     Decorator or wrapper to track embedding calls.
#     This is a conceptual placeholder.
#     """
#     @functools.wraps(func)
#     def wrapper_track_embedding_call(*args, **kwargs):
#         start_time = time.perf_counter()
#         # Assuming the first arg or a kwarg 'texts' contains the list of texts
#         texts_to_embed = []
#         if args and isinstance(args[0], list): # Simple assumption
#             texts_to_embed = args[0]
#         elif kwargs.get("texts") and isinstance(kwargs.get("texts"), list):
#             texts_to_embed = kwargs.get("texts")

#         estimated_tokens = sum(_estimate_tokens(text) for text in texts_to_embed)

#         result = func(*args, **kwargs)
#         duration = time.perf_counter() - start_time

#         # How to get model_name for embeddings? Needs to be passed or inferred.
#         # Embedding model name might be part of the embedding object itself.
#         embedding_model_name = "unknown_embedding_model"
#         if hasattr(args[0], 'model'): # if 'self' is the embedding client
#             embedding_model_name = getattr(args[0], 'model', embedding_model_name)

#         stats_entry = {
#             "call_id": str(uuid.uuid4()),
#             "type": "embedding",
#             "model_name": embedding_model_name,
#             "duration_ms": round(duration * 1000, 2),
#             "num_texts": len(texts_to_embed),
#             "estimated_input_tokens": estimated_tokens,
#             "cost_usd": 0.0, # Placeholder
#             "timestamp": time.time(),
#         }
#         _llm_stats_store.append(stats_entry)
#         return result
#     return wrapper_track_embedding_call


class LLMTracker:
    """Tracks LLM usage statistics across sessions."""

    # Price constants per 1K tokens (example rates)
    PRICING: ClassVar[Dict[str, Dict[str, Dict[str, float]]]] = {
        "anthropic": {
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
            "default": {"input": 3.0, "output": 15.0},
        },
        "openai": {
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 1.0, "output": 3.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "default": {"input": 5.0, "output": 15.0},
        },
        "ollama": {
            "default": {"input": 0.0, "output": 0.0},  # Local models have no API cost
        },
        "default": {"default": {"input": 1.0, "output": 2.0}},  # Conservative default
    }

    def __init__(self, stats_file: Optional[str] = None):
        """Initialize the tracker.

        Args:
            stats_file: Optional path to save stats
        """
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_duration_ms = 0
        self.total_cost = 0.0
        self.last_usage: Dict[str, Any] = {}
        self.usage_history: List[Dict[str, Any]] = []

        self.stats_file = stats_file
        if not self.stats_file:
            # Default to a stats file in user's home directory
            self.stats_file = str(Path.home() / ".docstra" / "llm_stats.json")

        # Try to load existing stats
        self._load_stats()

    def _load_stats(self) -> None:
        """Load statistics from file if available."""
        try:
            if self.stats_file:
                stats_path = Path(self.stats_file)
                if stats_path.exists():
                    with open(stats_path, "r") as f:
                        data = json.load(f)
                        self.total_input_tokens = data.get("total_input_tokens", 0)
                        self.total_output_tokens = data.get("total_output_tokens", 0)
                        self.total_requests = data.get("total_requests", 0)
                        self.total_duration_ms = data.get("total_duration_ms", 0)
                        self.total_cost = data.get("total_cost", 0.0)
                        self.usage_history = data.get("usage_history", [])
        except Exception:
            # If loading fails, start with empty stats
            pass

    def _save_stats(self) -> None:
        """Save statistics to file."""
        try:
            if self.stats_file:
                stats_path = Path(self.stats_file)
                stats_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_requests": self.total_requests,
                "total_duration_ms": self.total_duration_ms,
                "total_cost": self.total_cost,
                "usage_history": self.usage_history[-100:],  # Keep last 100 entries
                "last_updated": datetime.datetime.now().isoformat(),
            }

            with open(stats_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            # If saving fails, continue without error
            pass

    def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int = 0,
        request_type: str = "unspecified",
    ) -> None:
        """Record LLM usage data.

        Args:
            provider: LLM provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            duration_ms: Request duration in milliseconds
            request_type: Type of request (e.g., "query", "chat", "document")
        """
        # Calculate cost based on provider and model
        provider_rates = self.PRICING.get(provider.lower(), self.PRICING["default"])
        model_rates = provider_rates.get(model.lower(), provider_rates.get("default"))

        # Ensure model_rates is not None
        if model_rates is None:
            model_rates = self.PRICING["default"]["default"]

        input_cost = (input_tokens / 1000) * model_rates.get("input", 0.0)
        output_cost = (output_tokens / 1000) * model_rates.get("output", 0.0)
        total_cost = input_cost + output_cost

        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        self.total_duration_ms += duration_ms
        self.total_cost += total_cost

        # Record this usage
        usage_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_ms": duration_ms,
            "cost": total_cost,
            "request_type": request_type,
        }

        self.last_usage = usage_data
        self.usage_history.append(usage_data)

        # Save stats
        self._save_stats()

    def get_session_stats(self) -> Dict[str, Any]:
        """Get the current session statistics.

        Returns:
            Dictionary containing session statistics
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_requests": self.total_requests,
            "total_duration_ms": self.total_duration_ms,
            "total_cost": self.total_cost,
        }

    def get_usage_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get usage summary for the specified period.

        Args:
            days: Number of days to include in the summary

        Returns:
            Dictionary containing usage summary
        """
        # Filter usage history by date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()

        recent_usage = [
            entry
            for entry in self.usage_history
            if entry.get("timestamp", "") >= cutoff_str
        ]

        # Calculate summary
        total_input = sum(entry.get("input_tokens", 0) for entry in recent_usage)
        total_output = sum(entry.get("output_tokens", 0) for entry in recent_usage)
        total_cost = sum(entry.get("cost", 0.0) for entry in recent_usage)

        # Group by model and provider
        by_provider: Dict[str, float] = {}
        by_model: Dict[str, float] = {}

        for entry in recent_usage:
            provider = entry.get("provider", "unknown")
            model = entry.get("model", "unknown")
            cost = entry.get("cost", 0.0)

            by_provider[provider] = by_provider.get(provider, 0.0) + cost
            by_model[model] = by_model.get(model, 0.0) + cost

        return {
            "period_days": days,
            "total_requests": len(recent_usage),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": total_cost,
            "cost_by_provider": by_provider,
            "cost_by_model": by_model,
        }
