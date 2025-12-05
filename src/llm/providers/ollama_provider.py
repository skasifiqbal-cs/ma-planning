"""Ollama provider implementation."""

from typing import List, Dict, Optional
import requests
from .base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    def __init__(
        self, model: str = "llama3:8b", url: str = "http://localhost:11434", **kwargs
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name (llama3:8b, mistral, codellama, etc.)
            url: Ollama server URL
            **kwargs: Additional Ollama parameters
        """
        super().__init__(model, **kwargs)
        self.url = url.rstrip("/")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages to Ollama API."""
        endpoint = f"{self.url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                **(kwargs.get("options", {})),
            },
        }

        # Add format=json for structured output if requested
        if kwargs.get("format") == "json" or self.kwargs.get("format") == "json":
            payload["format"] = "json"

        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()

        return response.json()["message"]["content"]

    def get_provider_name(self) -> str:
        return "ollama"
