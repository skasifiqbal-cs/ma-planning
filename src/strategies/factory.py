"""Strategy registry and factory."""

from typing import Any, Dict, Type
from .base import PlanGenerationStrategy
from .base_llm import BaseStrategy


class StrategyRegistry:
    _strategies: Dict[str, Type[PlanGenerationStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[PlanGenerationStrategy]):
        cls._strategies[name.lower()] = strategy_class

    @classmethod
    def create(cls, name: str, llm: Any, config: Any, **kwargs) -> PlanGenerationStrategy:
        name = name.lower()
        if name not in cls._strategies:
            available = ", ".join(sorted(cls._strategies.keys()))
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
        return cls._strategies[name](llm=llm, config=config, **kwargs)

    @classmethod
    def list_strategies(cls) -> list:
        return sorted(cls._strategies.keys())


class StrategyFactory:
    @staticmethod
    def create(mode: str, llm: Any, config: Any, **kwargs) -> PlanGenerationStrategy:
        return StrategyRegistry.create(mode, llm=llm, config=config, **kwargs)


def _register_strategies():
    # ── Core strategies (canonical names first, then aliases) ─────────────────

    # base: open-loop LLM generation, no validation
    StrategyRegistry.register("base",     BaseStrategy)
    StrategyRegistry.register("no-val",   BaseStrategy)   # alias
    StrategyRegistry.register("open-loop", BaseStrategy)  # alias

    # llm-modulo: generate → validate with VAL → backprompt with errors → retry
    try:
        from .llm_modulo import LLMModuloStrategy
        StrategyRegistry.register("llm-modulo",  LLMModuloStrategy)
        StrategyRegistry.register("val-feedback", LLMModuloStrategy)  # alias
        StrategyRegistry.register("backprompt",   LLMModuloStrategy)  # alias
    except ImportError:
        pass

    # llm-repair: similarity-based action repair using embeddings
    try:
        from .llm_repair import LLMRepairStrategy
        StrategyRegistry.register("llm-repair", LLMRepairStrategy)
        StrategyRegistry.register("repair",     LLMRepairStrategy)  # alias
    except ImportError:
        pass

    # llm-merge: per-agent decomposition via pyperplan + LLM merges subplans
    try:
        from .llm_merge import LLMMergeStrategy
        StrategyRegistry.register("llm-merge",    LLMMergeStrategy)
        StrategyRegistry.register("decomposition", LLMMergeStrategy)  # alias
        StrategyRegistry.register("multi-agent",   LLMMergeStrategy)  # alias
    except ImportError:
        pass

    # ── Secondary strategies (kept but not primary focus) ─────────────────────
    try:
        from .autoregressive import LLM4PDDLZeroShotAutoregressiveStrategy
        StrategyRegistry.register("autoregressive", LLM4PDDLZeroShotAutoregressiveStrategy)
        StrategyRegistry.register("soft-val-ar",    LLM4PDDLZeroShotAutoregressiveStrategy)
    except ImportError:
        pass

    try:
        from .randomized import OpenLoopRandomizedStrategy
        StrategyRegistry.register("randomized",    OpenLoopRandomizedStrategy)
        StrategyRegistry.register("open-loop-rand", OpenLoopRandomizedStrategy)
    except ImportError:
        pass

    try:
        from .task_decomposition import TaskDecompositionStrategy
        StrategyRegistry.register("task-decomp", TaskDecompositionStrategy)
    except ImportError:
        pass


_register_strategies()

__all__ = ["StrategyFactory", "StrategyRegistry", "PlanGenerationStrategy"]
