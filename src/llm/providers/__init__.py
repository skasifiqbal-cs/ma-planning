"""LLM provider registry and factory."""

from typing import Dict, Type, Any
from .base import BaseLLMProvider


class LLMProviderRegistry:
    """Registry for LLM providers."""

    _providers: Dict[str, Type[BaseLLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLMProvider]):
        """Register a new LLM provider."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(cls, provider_name: str, **kwargs) -> BaseLLMProvider:
        """Create an LLM provider instance."""
        provider_name = provider_name.lower()

        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown LLM provider '{provider_name}'. "
                f"Available providers: {available}"
            )

        provider_class = cls._providers[provider_name]
        return provider_class(**kwargs)

    @classmethod
    def list_providers(cls) -> list:
        """List all registered providers."""
        return list(cls._providers.keys())


# Auto-register providers on import
def _register_providers():
    """Auto-register all available providers."""
    try:
        from .openai_provider import OpenAIProvider

        LLMProviderRegistry.register("openai", OpenAIProvider)
        LLMProviderRegistry.register("gpt", OpenAIProvider)
    except ImportError:
        pass

    try:
        from .anthropic_provider import AnthropicProvider

        LLMProviderRegistry.register("anthropic", AnthropicProvider)
        LLMProviderRegistry.register("claude", AnthropicProvider)
    except ImportError:
        pass

    try:
        from .ollama_provider import OllamaProvider

        LLMProviderRegistry.register("ollama", OllamaProvider)
    except ImportError:
        pass

    try:
        from .huggingface_provider import HuggingFaceProvider

        LLMProviderRegistry.register("huggingface", HuggingFaceProvider)
        LLMProviderRegistry.register("hf", HuggingFaceProvider)
    except ImportError:
        pass

    try:
        from .azure_provider import AzureOpenAIProvider

        LLMProviderRegistry.register("azure", AzureOpenAIProvider)
        LLMProviderRegistry.register("azure-openai", AzureOpenAIProvider)
    except ImportError:
        pass

    try:
        from .google_provider import GoogleProvider

        LLMProviderRegistry.register("google", GoogleProvider)
        LLMProviderRegistry.register("gemini", GoogleProvider)
    except ImportError:
        pass

    try:
        from .deepseek_provider import DeepSeekProvider

        LLMProviderRegistry.register("deepseek", DeepSeekProvider)
    except ImportError:
        pass

    try:
        from .groq_provider import GroqProvider

        LLMProviderRegistry.register("groq", GroqProvider)
    except ImportError:
        pass


_register_providers()


__all__ = ["LLMProviderRegistry", "BaseLLMProvider"]
