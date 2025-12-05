"""LLM client for chat completion."""

import json
from typing import Optional, Union, List
import requests


class LLMClient:
    """
    Client for LLM inference via Ollama API.

    Supports Ollama's /api/chat endpoint with fallback handling.
    """

    def __init__(
        self,
        model: str,
        url: str,
        temperature: float = 0.7,
        max_tokens: int = 128,
        timeout: int = 120,
        debug: bool = False,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize LLM client.

        Args:
            model: Model name (e.g., "llama3:8b")
            url: Base URL or full chat endpoint URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            debug: Enable debug logging
            system_prompt: Default system prompt
        """
        self.model = model
        self.url = url.rstrip("/")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.debug = bool(debug)
        self.system_prompt = system_prompt or (
            "You are a PDDL planning assistant. "
            "Respond ONLY with plan actions. No explanations."
        )

    def _chat_endpoint(self) -> str:
        """Get the chat endpoint URL."""
        u = self.url
        if u.endswith("/api/chat"):
            return u
        if u.endswith("/api"):
            return f"{u}/chat"
        return f"{u}/api/chat"

    def chat(
        self,
        user_prompt: str,
        extra_system: Optional[str] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> Optional[str]:
        """
        Send a chat request to the LLM.

        Args:
            user_prompt: User message content
            extra_system: Optional system prompt override
            stop: Optional stop tokens (string or list)

        Returns:
            Assistant response text, or None on failure
        """
        sys_prompt = extra_system if extra_system is not None else self.system_prompt

        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        if stop:
            options["stop"] = [stop] if isinstance(stop, str) else stop

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt or ""},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": options,
        }

        endpoint = self._chat_endpoint()
        if self.debug:
            print(f"[LLM] POST {endpoint}")
            print(f"[LLM] Model: {self.model}, Temp: {self.temperature}")

        try:
            resp = requests.post(endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            if self.debug:
                print(f"[LLM ERROR] Request failed: {e}")
            return None

        try:
            data = resp.json()
        except Exception:
            if self.debug:
                print("[LLM ERROR] Failed to parse JSON response")
            return None

        # Extract content from Ollama response
        msg = data.get("message", {})
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content

        # Fallback to OpenAI-like response format
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            cmsg = choices[0].get("message", {})
            ccontent = cmsg.get("content")
            if isinstance(ccontent, str):
                return ccontent

        if self.debug:
            print("[LLM ERROR] Could not extract content from response")
        return None
