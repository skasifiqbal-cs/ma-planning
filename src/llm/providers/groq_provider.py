"""Groq API provider implementation using OpenAI-compatible client."""

import os
from typing import List, Dict, Optional
from .base import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider (OpenAI-compatible API)"""

    def __init__(
        self,
        model: str = "llama-3.1-70b-versatile",
        api_key: Optional[str] = None,
        url: Optional[str] = None,  # Ignore url parameter (used by Ollama providers)
        **kwargs
    ):
        """
        Initialize Groq provider using OpenAI-compatible API.

        Args:
            model: Model name (llama-3.1-70b-versatile, mixtral-8x7b-32768, etc.)
            api_key: Groq API key (or use GROQ_API_KEY env var)
            url: Ignored (for compatibility with other providers)
            **kwargs: Additional Groq parameters
        """
        super().__init__(model, **kwargs)
        # Try: config api_key parameter -> GROQ_API_KEY env var -> MAP_PLANNING_LLM_API_KEY env var
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("MAP_PLANNING_LLM_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set one of:\n"
                "  1. GROQ_API_KEY environment variable\n"
                "  2. MAP_PLANNING_LLM_API_KEY environment variable\n"
                "  3. api_key in config.yaml under llm section\n"
                "  4. Pass api_key parameter programmatically\n\n"
                "Get your free Groq API key: https://console.groq.com/keys"
            )

        # Use OpenAI-compatible client
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        except ImportError:
            raise ImportError(
                "Groq provider requires 'openai' package. "
                "Install with: pip install openai"
            )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages to Groq API using OpenAI-compatible endpoint."""
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
        return "groq"
