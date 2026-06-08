"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize LLM provider.

        Args:
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.last_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _update_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0) -> None:
        computed_total = total_tokens or (prompt_tokens + completion_tokens)
        self.last_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": computed_total}
        self.total_usage["prompt_tokens"] += prompt_tokens
        self.total_usage["completion_tokens"] += completion_tokens
        self.total_usage["total_tokens"] += computed_total

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send chat messages and get response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for this call

        Returns:
            Generated text response
        """
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a single prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return {
            "provider": self.get_provider_name(),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.kwargs,
        }
