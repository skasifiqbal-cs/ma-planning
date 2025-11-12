import requests
import json
from typing import Optional
from config import Config


class LLMPrompt:
    """
    Wrapper for an Ollama-compatible /api/chat endpoint.
    - Forces non-streaming responses (stream: false).
    - If config.debug is True, prints the payload (including the user prompt) before sending.
    """

    def __init__(self, config: Config):
        self.model = config.llm_model
        self.url = config.llm_url
        self.temperature = getattr(config, "temperature", 0.1)
        self.max_tokens = getattr(config, "max_tokens", 128)
        self.request_timeout = 300
        self.debug = getattr(config, "debug", False)

    def chat(
        self, user_content: str, system_content: Optional[str] = None
    ) -> Optional[str]:
        if system_content is None:
            system_content = "You are a PDDL planning assistant. Respond ONLY with one valid grounded action. No explanations."

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if self.debug:
            print("\n[DEBUG] LLM REQUEST")
            print(f"POST {self.url}")
            print(
                f"model: {self.model}, temperature: {self.temperature}, max_tokens: {self.max_tokens}, stream: False"
            )
            print("---- SYSTEM ----")
            print(system_content)
            print("---- USER PROMPT ----")
            print(user_content)
            print("---- END PROMPT ----\n")

        try:
            resp = requests.post(self.url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[ERROR] LLM request failed: {e}")
            return None

        # Prefer structured; if fails, return raw text
        try:
            data = resp.json()
            content = None
            if isinstance(data, dict):
                msg = data.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                if content is None:
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices:
                        content = (choices[0].get("message") or {}).get("content")
            if not content:
                content = resp.text
            return (content or "").strip() or None
        except json.JSONDecodeError:
            return (resp.text or "").strip() or None

    @staticmethod
    def extract_first_action(text: str) -> str | None:
        # A simple heuristic: first balanced parenthesized expression
        # If your actions always look like (op arg1 arg2 ...), this is sufficient.
        s = (text or "").strip()
        start = s.find("(")
        end = s.find(")", start + 1)
        if start != -1 and end != -1:
            return s[start : end + 1].strip()
        # Fallback to first line
        line = s.splitlines()[0].strip() if s else ""
        return line or None
