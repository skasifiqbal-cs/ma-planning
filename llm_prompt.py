import requests

class LLMPrompt:
    def __init__(self, model, url="http://localhost:11434/api/chat"):
        self.model = model
        self.url = url

    def prompt(self, messages, temp=0.1):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temp, "num_ctx": 32768},
        }
        resp = requests.post(self.url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["message"]["content"]