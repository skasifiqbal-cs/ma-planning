"""Factory for creating LLM providers with registry pattern."""

from typing import Any, Dict, Type, Optional
from .providers.base import BaseLLMProvider


class ProviderRegistry:
    """Registry for LLM providers."""

    _providers: Dict[str, Type[BaseLLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLMProvider]):
        """Register a new LLM provider."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """Create a provider instance."""
        provider = provider.lower()

        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider '{provider}'. Available providers: {available}"
            )

        provider_class = cls._providers[provider]
        return provider_class(
            model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    @classmethod
    def list_providers(cls) -> list:
        """List all registered providers."""
        return list(cls._providers.keys())


class ProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create(
        provider: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """
        Create an LLM provider.

        Args:
            provider: Provider name (ollama, openai, deepseek, groq, anthropic, azure, huggingface)
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific arguments

        Returns:
            LLM provider instance
        """
        return ProviderRegistry.create(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


# Auto-register all available providers
def _register_providers():
    """Auto-register all available provider implementations."""
    # Always available providers
    try:
        from .providers.ollama_provider import OllamaProvider

        ProviderRegistry.register("ollama", OllamaProvider)
    except ImportError:
        pass

    try:
        from .providers.groq_provider import GroqProvider

        ProviderRegistry.register("groq", GroqProvider)
    except ImportError:
        pass

    # Optional providers (require external libraries)
    try:
        from .providers.openai_provider import OpenAIProvider

        ProviderRegistry.register("openai", OpenAIProvider)
    except ImportError:
        pass

    try:
        from .providers.deepseek_provider import DeepSeekProvider

        ProviderRegistry.register("deepseek", DeepSeekProvider)
    except ImportError:
        pass

    try:
        from .providers.anthropic_provider import AnthropicProvider

        ProviderRegistry.register("anthropic", AnthropicProvider)
    except ImportError:
        pass

    try:
        from .providers.azure_provider import AzureOpenAIProvider

        ProviderRegistry.register("azure", AzureOpenAIProvider)
    except ImportError:
        pass

    try:
        from .providers.huggingface_provider import HuggingFaceProvider

        ProviderRegistry.register("huggingface", HuggingFaceProvider)
    except ImportError:
        pass

    try:
        from .providers.google_provider import GoogleProvider

        ProviderRegistry.register("google", GoogleProvider)
        ProviderRegistry.register("gemini", GoogleProvider)
    except ImportError:
        pass


_register_providers()


__all__ = ["ProviderFactory", "ProviderRegistry"]
