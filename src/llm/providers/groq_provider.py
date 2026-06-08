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
        **kwargs,
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
        import time

        max_retries = kwargs.pop("max_retries", 2)
        retry_delay = kwargs.pop("retry_delay", 1.0)

        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    **{
                        k: v
                        for k, v in {**self.kwargs, **kwargs}.items()
                        if k not in ["temperature", "max_tokens"]
                    },
                )

                # Extract content
                content = (
                    response.choices[0].message.content if response.choices else None
                )

                if hasattr(response, "usage") and response.usage:
                    self._update_usage(
                        prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                        completion_tokens=getattr(response.usage, "completion_tokens", 0),
                        total_tokens=getattr(response.usage, "total_tokens", 0),
                    )

                # Check for empty response
                if not content or not content.strip():
                    print(
                        f"\n[GROQ WARNING] Empty response on attempt {attempt + 1}/{max_retries + 1}"
                        f" (completion_tokens={self.last_usage['completion_tokens']})"
                    )
                    if attempt < max_retries:
                        print(f"  Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"  ✗ Max retries exhausted, returning empty string")
                        return ""

                return content

            except Exception as e:
                print(
                    f"\n[GROQ ERROR] Request failed on attempt {attempt + 1}/{max_retries + 1}: {e}"
                )
                if attempt < max_retries:
                    print(f"  Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                raise

        return ""

    def get_provider_name(self) -> str:
        return "groq"
