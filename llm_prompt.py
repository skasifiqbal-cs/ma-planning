import requests
from config import Config  # Import the Config class


class LLMPrompt:
    def __init__(self, config: Config):
        """Initialize the LLM prompt settings using the provided Config instance."""
        self.model = config.llm_model
        self.url = config.llm_url
        self.temperature = config.temperature
        self.num_completions = getattr(config, "num_completions", 1)

    def prompt(self, input_prompt: str) -> str:
        """Send a prompt to the LLM server and handle tokenized streaming responses."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 10000,
        }

        try:
            # Send the request to the LLM server
            with requests.post(self.url, json=payload, stream=True, timeout=60) as resp:
                resp.raise_for_status()  # Raise HTTPError for bad HTTP responses

                # Aggregate the "content" fields from streamed messages
                full_response = ""
                for line in resp.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            message = eval(line)  # Parse JSON-like string
                            content = message.get("message", {}).get("content", "")
                            full_response += content
                        except Exception:
                            continue  # Ignore invalid lines
                return full_response.strip()

        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP Error ({resp.status_code}): {resp.text}")
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected error: {str(e)}")
            raise
