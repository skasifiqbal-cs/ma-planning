# Extensible Architecture Guide

## Overview

This codebase is designed for **research flexibility** - easily add new LLM providers, prompting strategies, and planning approaches without modifying core code.

## Key Design Patterns

### 1. **Registry Pattern**

All extensible components use registries for easy addition:

- **LLM Providers**: `LLMProviderRegistry`
- **Planning Strategies**: `StrategyRegistry`  
- **Prompt Templates**: `PromptRegistry`

### 2. **How to Add Components**

#### Adding a New LLM Provider

Create `/home/rr/ma-planning/src/llm/providers/my_provider.py`:

```python
from .base import BaseLLMProvider

class MyProvider(BaseLLMProvider):
    def __init__(self, model: str, api_key: str = None, **kwargs):
        super().__init__(model, **kwargs)
        # Initialize your client
        
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Implement API call
        pass
    
    def get_provider_name(self) -> str:
        return "my-provider"
```

Register in `__init__.py`:

```python
try:
    from .my_provider import MyProvider
    LLMProviderRegistry.register("my-provider", MyProvider)
except ImportError:
    pass
```

**Usage:**
```bash
python -c "from src.llm.providers import LLMProviderRegistry; \
           llm = LLMProviderRegistry.create('my-provider', model='my-model')"
```

#### Adding a New Strategy

Create `src/strategies/my_strategy.py`:

```python
from .base import PlanGenerationStrategy

class MyStrategy(PlanGenerationStrategy):
    def generate_plan(self, domain_pddl, problem_pddl, valid_operators):
        # Your implementation
        pass
```

Register in `factory.py`:

```python
try:
    from .my_strategy import MyStrategy
    StrategyRegistry.register("my-strategy", MyStrategy)
except ImportError:
    pass
```

**Usage:**
```bash
python src/cli/main.py --strategy my-strategy ...
```

#### Adding a New Prompt Template

```python
from src.llm.prompts import PromptRegistry, PromptTemplate, PromptStyle

# Create template
my_template = PromptTemplate(
    style=PromptStyle.ZERO_SHOT,
    system_prompt="You are...",
    user_template="Domain: {domain}\nProblem: {problem}"
)

# Register
PromptRegistry.register("my-template", my_template)

# Use
template = PromptRegistry.get("my-template")
messages = template.format(domain=..., problem=...)
```

## Component Architecture

```
src/
├── llm/
│   ├── providers/          # LLM provider implementations
│   │   ├── __init__.py     # LLMProviderRegistry
│   │   ├── base.py         # BaseLLMProvider interface
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── ollama_provider.py
│   │   ├── huggingface_provider.py
│   │   └── azure_provider.py
│   └── prompts.py          # Prompt templates & registry
│
├── strategies/             # Planning strategies
│   ├── __init__.py
│   ├── factory.py          # StrategyRegistry
│   ├── base.py             # PlanGenerationStrategy interface
│   ├── no_validation.py    # ✅ Implemented
│   ├── autoregressive.py   # 📝 Placeholder
│   ├── repair.py           # 📝 Placeholder
│   └── randomized.py       # 📝 Placeholder
│
└── utils/
    ├── experiments.py      # ExperimentTracker
    └── config_manager.py   # ConfigManager
```

## Available Components

### LLM Providers

| Provider | Status | Usage |
|----------|--------|-------|
| **OpenAI** | ✅ Ready | `provider='openai', model='gpt-4'` |
| **Anthropic** | ✅ Ready | `provider='anthropic', model='claude-3-5-sonnet-20241022'` |
| **Ollama** | ✅ Ready | `provider='ollama', model='llama3:8b'` |
| **HuggingFace** | ✅ Ready | `provider='huggingface', model='...'` |
| **Azure OpenAI** | ✅ Ready | `provider='azure', model='gpt-4'` |

### Planning Strategies

| Strategy | Status | Description |
|----------|--------|-------------|
| **no-val** | ✅ Implemented | Open-loop without validation |
| **autoregressive** | 📝 Placeholder | Step-by-step generation |
| **repair** | 📝 Placeholder | Generate + validate + repair loop |
| **randomized** | 📝 Placeholder | Multiple candidates with selection |

### Prompt Templates

| Template | Style | Use Case |
|----------|-------|----------|
| **zero-shot** | Direct | Standard planning |
| **few-shot** | Examples | With demonstrations |
| **chain-of-thought** | Reasoning | Step-by-step thinking |
| **react** | Iterative | Reason + Act + Observe |
| **repair** | Feedback | Error correction |
| **autoregressive** | Sequential | One action at a time |

## Configuration System

### YAML Configs

Create `configs/my_experiment.yaml`:

```yaml
experiment_name: my-experiment
strategy: no-val
llm:
  provider: openai
  model: gpt-4
  temperature: 0.0
planning:
  max_steps: 50
  validate: true
```

Load and use:

```python
from src.utils.config_manager import ConfigManager

config = ConfigManager.load_config("configs/my_experiment.yaml")
```

### Parameter Sweeps

```python
from src.utils.config_manager import ConfigManager

base = {
    "strategy": "no-val",
    "llm": {"provider": "openai"}
}

sweep = {
    "llm.model": ["gpt-3.5-turbo", "gpt-4"],
    "llm.temperature": [0.0, 0.5, 1.0],
}

configs = ConfigManager.generate_sweep(base, sweep)
# Returns 6 configs (2 models × 3 temperatures)
```

## Experiment Tracking

```python
from src.utils.experiments import ExperimentTracker

tracker = ExperimentTracker("experiments")

# Log experiment
tracker.log_run(
    experiment_id="exp001",
    config={"strategy": "no-val", "model": "gpt-4"},
    results={"plan_length": 10, "valid": True, "time_seconds": 5.2},
    metadata={"domain": "gripper", "problem": "prob01"}
)

# Analyze results
summary = tracker.summarize()
df = tracker.get_dataframe()
tracker.export_csv("results.csv")
```

## Quick Start Examples

### 1. Using Different LLM Providers

```python
from src.llm.providers import LLMProviderRegistry

# OpenAI
llm = LLMProviderRegistry.create("openai", model="gpt-4", api_key="...")

# Anthropic
llm = LLMProviderRegistry.create("anthropic", model="claude-3-5-sonnet-20241022", api_key="...")

# Ollama (local)
llm = LLMProviderRegistry.create("ollama", model="llama3:8b", url="http://localhost:11434")

# Use any provider the same way
response = llm.generate("Hello!")
```

### 2. Using Different Strategies

```python
from src.strategies.factory import StrategyRegistry
from src.llm.providers import LLMProviderRegistry

llm = LLMProviderRegistry.create("ollama", model="llama3:8b")

# Create strategy
strategy = StrategyRegistry.create("no-val", llm=llm, config=config)

# Generate plan
plan = strategy.generate_plan(domain_pddl, problem_pddl, operators)
```

### 3. Custom Prompts

```python
from src.llm.prompts import PromptTemplate, PromptStyle, PromptRegistry

# Create custom template
custom = PromptTemplate(
    style=PromptStyle.CHAIN_OF_THOUGHT,
    system_prompt="Think carefully about each step.",
    user_template="Problem: {problem}\n\nSolve step-by-step."
)

# Register it
PromptRegistry.register("my-custom-cot", custom)

# Use it
template = PromptRegistry.get("my-custom-cot")
messages = template.format(problem="...")
response = llm.chat(messages)
```

## Adding New Features Checklist

### New LLM Provider
- [ ] Create `src/llm/providers/my_provider.py`
- [ ] Inherit from `BaseLLMProvider`
- [ ] Implement `chat()` and `get_provider_name()`
- [ ] Register in `__init__.py`
- [ ] Add to `requirements.txt` as optional
- [ ] Test with simple example

### New Strategy
- [ ] Create `src/strategies/my_strategy.py`
- [ ] Inherit from `PlanGenerationStrategy`
- [ ] Implement `generate_plan()`
- [ ] Register in `factory.py`
- [ ] Add tests
- [ ] Document usage

### New Prompt Template
- [ ] Define template in `src/llm/prompts.py` or custom file
- [ ] Register with `PromptRegistry`
- [ ] Add examples
- [ ] Document use cases

## Best Practices

1. **Use registries** - Don't modify factory code, just register new components
2. **Placeholders** - Create placeholder implementations with TODOs for future work
3. **Optional imports** - Wrap provider imports in try-except for optional dependencies
4. **Type hints** - Use typing for better IDE support
5. **Documentation** - Add docstrings with examples
6. **Testing** - Create tests for new components
7. **Config files** - Use YAML for experiment configurations
8. **Track experiments** - Log all runs for reproducibility

## Research Workflow

```bash
# 1. Create experiment config
python -c "from src.utils.config_manager import save_example_configs; save_example_configs()"

# 2. Edit config
vim configs/my_experiment.yaml

# 3. Run experiments
python src/cli/main.py --config configs/my_experiment.yaml --batch

# 4. Analyze results
python -c "
from src.utils.experiments import ExperimentTracker
tracker = ExperimentTracker()
print(tracker.summarize())
tracker.export_csv('results.csv')
"

# 5. Visualize (add your own plotting code)
python scripts/plot_results.py results.csv
```

## Next Steps

1. **Implement placeholder strategies** - Fill in autoregressive, repair, randomized
2. **Add more providers** - Cohere, AI21, local models
3. **Create visualization tools** - Plot success rates, plan lengths
4. **Add few-shot examples** - Create example databases
5. **Parallel execution** - Multi-threaded batch processing
6. **Web interface** - Optional Gradio/Streamlit UI

## Questions?

- **How do I add a new LLM?** → See "Adding a New LLM Provider" above
- **How do I try different prompts?** → Use `PromptRegistry` and `PromptTemplate`
- **How do I track my experiments?** → Use `ExperimentTracker`
- **Where are the placeholders?** → `src/strategies/{autoregressive,repair,randomized}.py`
- **Can I use local models?** → Yes! Use Ollama or HuggingFace providers
