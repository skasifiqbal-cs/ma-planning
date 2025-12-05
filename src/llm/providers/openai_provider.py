"""OpenAI provider implementation."""

from typing import List, Dict, Optional
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider (GPT-3.5, GPT-4, etc.)"""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (gpt-4, gpt-3.5-turbo, etc.)
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            organization: OpenAI organization ID
            **kwargs: Additional OpenAI parameters
        """
        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.organization = organization

        # Lazy import to avoid requiring openai if not used
        try:
            import openai

            self.client = openai.OpenAI(api_key=api_key, organization=organization)
        except ImportError:
            raise ImportError(
                "OpenAI provider requires 'openai' package. "
                "Install with: pip install openai"
            )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages to OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **{
                k: v
                for k, v in {**self.kwargs, **kwargs}.items()
                if k not in ["temperature", "max_tokens"]
            }
        )
        return response.choices[0].message.content

    def get_provider_name(self) -> str:
        return "openai"
