"""DeepSeek provider implementation."""

import os
from typing import List, Dict, Optional
from .base import BaseLLMProvider


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek LLM provider (deepseek-chat, deepseek-reasoner, etc.)"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        **kwargs
    ):
        """
        Initialize DeepSeek provider.

        Args:
            model: Model name (deepseek-chat, deepseek-reasoner, etc.)
            api_key: DeepSeek API key (or use DEEPSEEK_API_KEY env var)
            base_url: DeepSeek API base URL
            **kwargs: Additional API parameters
        """
        super().__init__(model, **kwargs)

        # Try: config api_key parameter -> DEEPSEEK_API_KEY env var
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. Set one of:\n"
                "  1. DEEPSEEK_API_KEY environment variable\n"
                "  2. api_key in config.yaml under llm section\n"
                "  3. Pass api_key parameter programmatically\n\n"
                "Get your DeepSeek API key: https://platform.deepseek.com/api_keys"
            )

        self.base_url = base_url

        # DeepSeek uses OpenAI-compatible API
        try:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)
        except ImportError:
            raise ImportError(
                "DeepSeek provider requires 'openai' package. "
                "Install with: pip install openai"
            )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages to DeepSeek API."""
        # Filter out parameters that shouldn't be passed to the API
        filtered_kwargs = {
            k: v
            for k, v in {**self.kwargs, **kwargs}.items()
            if k
            not in [
                "temperature",
                "max_tokens",
                "url",
                "api_key",
                "base_url",
                "use_batch",
            ]
        }

        # deepseek-reasoner does not support the temperature parameter
        is_reasoner = "reasoner" in self.model.lower()

        create_kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **filtered_kwargs
        )
        if not is_reasoner:
            create_kwargs["temperature"] = kwargs.get("temperature", self.temperature)

        response = self.client.chat.completions.create(**create_kwargs)

        if hasattr(response, "usage") and response.usage:
            self._update_usage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                completion_tokens=getattr(response.usage, "completion_tokens", 0),
                total_tokens=getattr(response.usage, "total_tokens", 0),
            )

        # deepseek-reasoner returns reasoning_content + content; return the final answer
        msg = response.choices[0].message
        return msg.content

    def get_provider_name(self) -> str:
        return "deepseek"
