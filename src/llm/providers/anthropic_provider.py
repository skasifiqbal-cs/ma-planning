"""Anthropic (Claude) provider implementation."""

from typing import List, Dict, Optional
from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Claude model name
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            **kwargs: Additional Anthropic parameters
        """
        super().__init__(model, **kwargs)
        self.api_key = api_key

        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Anthropic provider requires 'anthropic' package. "
                "Install with: pip install anthropic"
            )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages to Anthropic API."""
        # Convert messages format if needed
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        response = self.client.messages.create(
            model=self.model,
            messages=anthropic_messages,
            system=system_message,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens or 4096),
            **{
                k: v
                for k, v in {**self.kwargs, **kwargs}.items()
                if k not in ["temperature", "max_tokens"]
            }
        )
        return response.content[0].text

    def get_provider_name(self) -> str:
        return "anthropic"
