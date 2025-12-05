"""Factory for creating planning strategies with registry pattern."""

from typing import Any, Dict, Type
from .base import PlanGenerationStrategy
from .no_validation import OpenLoopNoValidationStrategy


class StrategyRegistry:
    """Registry for planning strategies."""

    _strategies: Dict[str, Type[PlanGenerationStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[PlanGenerationStrategy]):
        """Register a new planning strategy."""
        cls._strategies[name.lower()] = strategy_class

    @classmethod
    def create(
        cls, name: str, llm: Any, config: Any, **kwargs
    ) -> PlanGenerationStrategy:
        """Create a strategy instance."""
        name = name.lower()

        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy '{name}'. Available strategies: {available}"
            )

        strategy_class = cls._strategies[name]
        return strategy_class(llm=llm, config=config, **kwargs)

    @classmethod
    def list_strategies(cls) -> list:
        """List all registered strategies."""
        return list(cls._strategies.keys())


class StrategyFactory:
    """Factory for creating planning strategies (legacy compatibility)."""

    @staticmethod
    def create(mode: str, llm: Any, config: Any) -> PlanGenerationStrategy:
        """
        Create a planning strategy.

        Args:
            mode: Strategy mode (no-val, soft-val-ar, open-loop-repair, open-loop-rand)
            llm: LLM client instance
            config: Configuration object

        Returns:
            Strategy instance
        """
        return StrategyRegistry.create(
            mode,
            llm=llm,
            config=config,
            state_based_validation=getattr(config, "state_based_validation", False),
        )


# Auto-register available strategies
def _register_strategies():
    """Auto-register all available strategy implementations."""
    # Always available
    StrategyRegistry.register("no-val", OpenLoopNoValidationStrategy)
    StrategyRegistry.register("open-loop", OpenLoopNoValidationStrategy)

    # Try to register optional strategies
    try:
        from .autoregressive import LLM4PDDLZeroShotAutoregressiveStrategy

        StrategyRegistry.register("soft-val-ar", LLM4PDDLZeroShotAutoregressiveStrategy)
        StrategyRegistry.register(
            "autoregressive", LLM4PDDLZeroShotAutoregressiveStrategy
        )
    except ImportError:
        pass

    try:
        from .repair import OpenLoopSimilarityRepairStrategy

        StrategyRegistry.register("open-loop-repair", OpenLoopSimilarityRepairStrategy)
        StrategyRegistry.register("repair", OpenLoopSimilarityRepairStrategy)
    except ImportError:
        pass

    try:
        from .randomized import OpenLoopRandomizedStrategy

        StrategyRegistry.register("open-loop-rand", OpenLoopRandomizedStrategy)
        StrategyRegistry.register("randomized", OpenLoopRandomizedStrategy)
    except ImportError:
        pass


_register_strategies()


__all__ = ["StrategyFactory", "StrategyRegistry", "PlanGenerationStrategy"]
