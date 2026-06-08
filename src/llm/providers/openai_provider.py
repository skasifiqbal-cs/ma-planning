"""OpenAI provider implementation."""

import os
from typing import List, Dict, Optional
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider (GPT-3.5, GPT-4, etc.)"""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs,
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

        # Try: config api_key parameter -> OPENAI_API_KEY env var
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set one of:\n"
                "  1. OPENAI_API_KEY environment variable\n"
                "  2. api_key in config.yaml under llm section\n"
                "  3. Pass api_key parameter programmatically\n\n"
                "Get your OpenAI API key: https://platform.openai.com/account/api-keys"
            )

        # Lazy import to avoid requiring openai if not used
        try:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key, organization=organization)
        except ImportError:
            raise ImportError(
                "OpenAI provider requires 'openai' package. "
                "Install with: pip install openai"
            )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages to OpenAI API."""
        # Filter out parameters that shouldn't be passed to OpenAI API
        all_kwargs = {**self.kwargs, **kwargs}
        filtered_kwargs = {
            k: v
            for k, v in all_kwargs.items()
            if k
            not in [
                "temperature",
                "max_tokens",
                "max_completion_tokens",
                "url",
                "api_key",
                "base_url",
                "organization",
            ]
        }

        # GPT-4o and newer models use max_completion_tokens instead of max_tokens
        # This includes: gpt-4o*, o1*, gpt-5*, and future models
        max_tokens_value = all_kwargs.get("max_tokens", self.max_tokens)
        uses_completion_tokens = (
            self.model.startswith("gpt-4o")
            or self.model.startswith("o1")
            or self.model.startswith("gpt-5")
            or "2024" in self.model
            or "2025" in self.model
        )

        if uses_completion_tokens:
            token_param = {"max_completion_tokens": max_tokens_value}
        else:
            token_param = {"max_tokens": max_tokens_value}

        # o1 and gpt-5 models only support temperature=1 (default)
        # They don't allow custom temperature values
        temp_value = all_kwargs.get("temperature", self.temperature)
        if self.model.startswith("o1") or self.model.startswith("gpt-5"):
            temp_param = {}  # Don't pass temperature at all (uses default 1.0)
        else:
            temp_param = {"temperature": temp_value}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **temp_param,
            **token_param,
            **filtered_kwargs,
        )
        content = response.choices[0].message.content

        if hasattr(response, "usage") and response.usage:
            self._update_usage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                completion_tokens=getattr(response.usage, "completion_tokens", 0),
                total_tokens=getattr(response.usage, "total_tokens", 0),
            )

        # Debug logging for empty responses
        if not content or (isinstance(content, str) and not content.strip()):
            import sys

            print(f"[OPENAI] WARNING: Empty content from {self.model}", file=sys.stderr)
            print(f"[OPENAI] Response choices: {response.choices}", file=sys.stderr)
            print(f"[OPENAI] Message: {response.choices[0].message}", file=sys.stderr)
            print(
                f"[OPENAI] Finish reason: {response.choices[0].finish_reason}",
                file=sys.stderr,
            )

        return content

    def get_provider_name(self) -> str:
        return "openai"
