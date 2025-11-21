import json
from typing import Optional, Union, List
import requests


class LLMPrompt:
    """
    Simple LLM client for Ollama's /api/chat (default), with fallback to OpenAI-like responses.

    Pass either:
      - Base URL:  "http://localhost:11434"
      - Full URL:  "http://localhost:11434/api/chat"

    The class normalizes to a single /api/chat endpoint without duplication.
    """

    def __init__(
        self,
        model: str,
        url: str,
        temperature: float = 0.7,
        max_tokens: int = 128,
        timeout: int = 120,
        debug: bool = False,
        system_prompt: Optional[
            str
        ] = "You are a PDDL planning assistant. Respond ONLY with plan actions. No explanations.",
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.debug = bool(debug)
        self.system_prompt = system_prompt

    def _chat_endpoint(self) -> str:
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
        stop: Optional[Union[str, List[str]]] = None,  # NEW: stop tokens
    ) -> Optional[str]:
        sys_prompt = extra_system if extra_system is not None else self.system_prompt

        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        # Ollama supports "stop" as string or list of strings
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
            print(f"POST {endpoint}")
            print(
                f"model: {self.model}, temperature: {self.temperature}, max_tokens: {self.max_tokens}, stream: False"
            )

        try:
            resp = requests.post(endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            if self.debug:
                print(f"[LLM ERROR] HTTP request failed: {e}")
            try:
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as e2:
                if self.debug:
                    print(f"[LLM ERROR] Fallback HTTP request failed: {e2}")
                return None

        try:
            data = resp.json()
        except Exception:
            if self.debug:
                print("[LLM ERROR] Failed to parse JSON response")
                print((resp.text or "")[:500])
            return None

        # Ollama response
        msg = data.get("message", {})
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content

        # OpenAI-like fallback
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            cmsg = choices[0].get("message", {})
            ccontent = cmsg.get("content")
            if isinstance(ccontent, str):
                return ccontent

        if self.debug:
            print("[LLM ERROR] Could not extract assistant content from response JSON")
            try:
                print(json.dumps(data, indent=2)[:1000])
            except Exception:
                print(str(data)[:1000])
        return None
