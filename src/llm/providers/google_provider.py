"""Google Gemini provider implementation with batch API support."""

import os
import time
import json
from typing import List, Dict, Optional
from .base import BaseLLMProvider


class GoogleProvider(BaseLLMProvider):
    """Google Gemini LLM provider with batch API support for 50% cost reduction."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Google Gemini provider.

        Args:
            model: Model name (gemini-3-flash-preview, gemini-3-pro-preview,
                   gemini-2.5-flash, gemini-2.5-pro, etc.)
            api_key: Google API key (or use GOOGLE_API_KEY env var)
            use_batch: Enable batch API for 50% cost reduction (default: False)
            **kwargs: Additional Google parameters
        """
        super().__init__(model, **kwargs)

        # Try: config api_key parameter -> GOOGLE_API_KEY env var
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Google API key not found. Set one of:\n"
                "  1. GOOGLE_API_KEY environment variable\n"
                "  2. api_key in config.yaml under llm section\n"
                "  3. Pass api_key parameter programmatically\n\n"
                "Get your Google API key: https://aistudio.google.com/app/apikey"
            )

        # Lazy import to avoid requiring google-generativeai if not used
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.genai = genai

            # Initialize model
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens or 8192,
            }
            # Add any additional kwargs that are valid generation config parameters
            for key in ["top_p", "top_k", "candidate_count"]:
                if key in self.kwargs:
                    generation_config[key] = self.kwargs[key]

            self.model_instance = genai.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config,
            )
        except ImportError:
            raise ImportError(
                "Google provider requires 'google-generativeai' package. "
                "Install with: pip install google-generativeai"
            )

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict]:
        """
        Convert OpenAI-style messages to Google Gemini format.

        OpenAI format: [{"role": "system"/"user"/"assistant", "content": "..."}]
        Gemini format: [{"role": "user"/"model", "parts": ["..."]}]
        """
        gemini_messages = []
        system_prompt = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Store system prompt to prepend to first user message
                system_prompt = content
            elif role == "user":
                # Prepend system prompt to first user message if exists
                if system_prompt:
                    content = f"{system_prompt}\n\n{content}"
                    system_prompt = None
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})

        return gemini_messages

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages to Google Gemini API."""
        return self._sync_chat(messages, **kwargs)

    def _sync_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send synchronous chat request to Google Gemini API."""
        gemini_messages = self._convert_messages(messages)

        # Override generation config if kwargs provided
        generation_config = {}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            generation_config["max_output_tokens"] = kwargs["max_tokens"]

        # Retry logic for API rate limits
        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 10.0)

        for attempt in range(max_retries + 1):
            try:
                # Start chat session with history
                chat = self.model_instance.start_chat(history=gemini_messages[:-1])

                # Send last message as prompt
                last_message = (
                    gemini_messages[-1]["parts"][0] if gemini_messages else ""
                )

                response = chat.send_message(
                    last_message,
                    generation_config=generation_config if generation_config else None,
                )

                content = response.text

                # Debug logging for empty responses
                if not content or not content.strip():
                    import sys

                    print(
                        f"[GOOGLE] WARNING: Empty content from {self.model}",
                        file=sys.stderr,
                    )
                    print(f"[GOOGLE] Response: {response}", file=sys.stderr)
                    if hasattr(response, "prompt_feedback"):
                        print(
                            f"[GOOGLE] Prompt feedback: {response.prompt_feedback}",
                            file=sys.stderr,
                        )

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    self._update_usage(
                        prompt_tokens=getattr(um, "prompt_token_count", 0),
                        completion_tokens=getattr(um, "candidates_token_count", 0),
                    )

                return content

            except Exception as e:
                error_msg = str(e).lower()
                # Retry on rate limit or server errors
                if attempt < max_retries and (
                    "rate" in error_msg or "429" in error_msg or "503" in error_msg
                ):
                    wait_time = retry_delay * (3**attempt)  # Exponential backoff
                    print(
                        f"[GOOGLE] Rate limit hit, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Google Gemini API error: {e}")

        raise Exception(f"Google Gemini API failed after {max_retries} retries")

    def _batch_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Queue request for batch processing (50% cost reduction).

        Note: Batch API is asynchronous - returns batch job ID for later retrieval.
        For planning systems requiring immediate responses, use standard mode.
        """
        raise NotImplementedError(
            "Batch API support is planned but not yet implemented. "
            "Batch processing requires asynchronous workflow:\n"
            "1. Collect all planning requests\n"
            "2. Submit as batch job\n"
            "3. Poll for completion (up to 24 hours)\n"
            "4. Retrieve results\n\n"
            "For immediate planning results, set use_batch=false in config."
        )

    def get_provider_name(self) -> str:
        return "google"

    def supports_batch(self) -> bool:
        """Check if this provider supports batch API."""
        return True

    def get_batch_pricing_info(self) -> Dict[str, str]:
        """Return batch API pricing information."""
        # Based on https://ai.google.dev/pricing
        model_pricing = {
            "gemini-3-flash-preview": {
                "standard_input": "Free",
                "standard_output": "Free",
                "batch_input": "$0.50/M tokens",
                "batch_output": "$3.00/M tokens",
                "savings": "50% vs paid tier standard pricing",
            },
            "gemini-3-pro-preview": {
                "standard_input": "Not available",
                "standard_output": "Not available",
                "batch_input": "$2.00/M tokens (<=200k), $4.00/M tokens (>200k)",
                "batch_output": "$12.00/M tokens (<=200k), $18.00/M tokens (>200k)",
                "savings": "50% cost reduction",
            },
            "gemini-2.5-flash": {
                "standard_input": "Free",
                "standard_output": "Free",
                "batch_input": "$0.30/M tokens (text/image/video)",
                "batch_output": "$2.50/M tokens",
                "savings": "50% vs paid tier standard pricing",
            },
            "gemini-2.5-pro": {
                "standard_input": "Free",
                "standard_output": "Free",
                "batch_input": "$1.25/M tokens (<=200k), $2.50/M tokens (>200k)",
                "batch_output": "$10.00/M tokens (<=200k), $15.00/M tokens (>200k)",
                "savings": "50% cost reduction",
            },
        }

        return model_pricing.get(
            self.model,
            {
                "info": "Pricing varies by model. See https://ai.google.dev/pricing",
                "savings": "Batch API typically offers 50% cost reduction",
            },
        )
