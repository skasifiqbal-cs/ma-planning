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
