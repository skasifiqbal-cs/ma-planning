# Quick Start: Adding New Components

## 🚀 Add a New LLM Provider (5 minutes)

**1. Create provider file:**
```bash
cat > src/llm/providers/gemini_provider.py << 'EOF'
from .base import BaseLLMProvider
from typing import List, Dict

class GeminiProvider(BaseLLMProvider):
    def __init__(self, model="gemini-pro", api_key=None, **kwargs):
        super().__init__(model, **kwargs)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Convert messages to Gemini format
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        response = self.client.generate_content(prompt)
        return response.text
    
    def get_provider_name(self) -> str:
        return "gemini"
EOF
```

**2. Register it:**
```bash
# Add to src/llm/providers/__init__.py (in _register_providers function)
# Add these lines before the last line:
```
```python
try:
    from .gemini_provider import GeminiProvider
    LLMProviderRegistry.register("gemini", GeminiProvider)
except ImportError:
    pass
```

**3. Use it:**
```python
from src.llm.providers import LLMProviderRegistry
llm = LLMProviderRegistry.create("gemini", model="gemini-pro", api_key="...")
response = llm.generate("Hello!")
```

## 🎯 Add a New Planning Strategy (10 minutes)

**1. Create strategy file:**
```bash
cat > src/strategies/beam_search.py << 'EOF'
from .base import PlanGenerationStrategy
from typing import List

class BeamSearchStrategy(PlanGenerationStrategy):
    def __init__(self, llm, config, beam_width=3):
        self.llm = llm
        self.config = config
        self.beam_width = beam_width
    
    def generate_plan(self, domain_pddl, problem_pddl, valid_operators):
        # TODO: Implement beam search
        # 1. Generate beam_width initial actions
        # 2. For each, generate beam_width next actions
        # 3. Keep top beam_width candidates
        # 4. Repeat until goal or max steps
        raise NotImplementedError("Implement beam search here")
EOF
```

**2. Register it:**
```bash
# Add to src/strategies/factory.py (_register_strategies function):
```
```python
try:
    from .beam_search import BeamSearchStrategy
    StrategyRegistry.register("beam-search", BeamSearchStrategy)
except ImportError:
    pass
```

**3. Use it:**
```bash
python src/cli/main.py --strategy beam-search --domain-dir ... --problem-file ...
```

## 📝 Add a Custom Prompt (2 minutes)

**In your script:**
```python
from src.llm.prompts import PromptRegistry, PromptTemplate, PromptStyle

# Define template
my_prompt = PromptTemplate(
    style=PromptStyle.FEW_SHOT,
    system_prompt="You are a world-class planner.",
    user_template="Domain: {domain}\nProblem: {problem}\n\nGenerate plan:",
    examples=[
        {
            "input": "Move block A to B",
            "output": "(pick A)\n(move A B)\n(place A)"
        }
    ]
)

# Register
PromptRegistry.register("my-few-shot", my_prompt)

# Use anywhere
template = PromptRegistry.get("my-few-shot")
messages = template.format(domain="...", problem="...")
```

## 🧪 Run an Experiment

**1. Create config:**
```yaml
# configs/my_exp.yaml
experiment_name: my-experiment
strategy: no-val
llm:
  provider: ollama
  model: llama3:8b
  temperature: 0.0
planning:
  max_steps: 50
  validate: true
```

**2. Run:**
```bash
# Single problem
python src/cli/main.py \
  --config configs/my_exp.yaml \
  --domain-dir centralized/gripper \
  --problem-file prob01.pddl

# Batch
python src/cli/main.py \
  --config configs/my_exp.yaml \
  --domain-dir centralized/gripper \
  --batch
```

**3. Analyze:**
```python
from src.utils.experiments import ExperimentTracker

tracker = ExperimentTracker()
print(tracker.summarize())
df = tracker.get_dataframe()
df.to_csv("results.csv")
```

## 📊 Parameter Sweep

```python
from src.utils.config_manager import ConfigManager

base = {
    "strategy": "no-val",
    "llm": {"provider": "ollama", "model": "llama3:8b"},
    "planning": {"max_steps": 50, "validate": True}
}

sweep = {
    "llm.temperature": [0.0, 0.3, 0.7, 1.0],
    "planning.max_steps": [30, 50, 100],
}

configs = ConfigManager.generate_sweep(base, sweep)
print(f"Generated {len(configs)} configurations")

# Save each config
for i, config in enumerate(configs):
    ConfigManager.save_config(config, f"configs/sweep_{i:03d}.yaml")
```

## 🔍 Check Available Components

```python
from src.llm.providers import LLMProviderRegistry
from src.strategies.factory import StrategyRegistry
from src.llm.prompts import PromptRegistry

print("LLM Providers:", LLMProviderRegistry.list_providers())
print("Strategies:", StrategyRegistry.list_strategies())
print("Prompts:", PromptRegistry.list_templates())
```

## 💡 Common Patterns

### Using any LLM provider
```python
from src.llm.providers import LLMProviderRegistry

# All providers have the same interface
for provider_name in ["openai", "anthropic", "ollama"]:
    llm = LLMProviderRegistry.create(
        provider_name,
        model="appropriate-model-name"
    )
    response = llm.generate("Test prompt")
```

### Trying multiple strategies
```python
from src.strategies.factory import StrategyRegistry

strategies = ["no-val", "autoregressive", "repair"]
for strat_name in strategies:
    try:
        strategy = StrategyRegistry.create(strat_name, llm=llm, config=config)
        plan = strategy.generate_plan(domain, problem, operators)
        print(f"{strat_name}: {len(plan)} actions")
    except NotImplementedError:
        print(f"{strat_name}: Not implemented yet")
```

### Logging experiments
```python
from src.utils.experiments import ExperimentTracker
import time

tracker = ExperimentTracker("experiments")

start = time.time()
plan = strategy.generate_plan(...)
elapsed = time.time() - start

tracker.log_run(
    experiment_id=f"exp_{strategy}_{model}",
    config={
        "strategy": strategy_name,
        "model": model_name,
        "temperature": 0.0,
    },
    results={
        "plan_length": len(plan),
        "time_seconds": elapsed,
        "valid": True,  # After validation
    },
    metadata={
        "domain": "gripper",
        "problem": "prob01",
    }
)
```

## 🛠️ Development Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt  # or requirements-minimal.txt

# 2. Run tests
python tests/test_structure.py
./tests/smoke_test.sh

# 3. Try an example
python tests/test_planning.py

# 4. Add your component (see sections above)

# 5. Test it
python -c "
from src.llm.providers import LLMProviderRegistry
print(LLMProviderRegistry.list_providers())
"

# 6. Run your experiment
python src/cli/main.py --strategy your-strategy ...

# 7. Track results
python -c "
from src.utils.experiments import ExperimentTracker
tracker = ExperimentTracker()
print(tracker.summarize())
"
```

## 📚 File Locations

- **Add LLM provider**: `src/llm/providers/my_provider.py`
- **Add strategy**: `src/strategies/my_strategy.py`
- **Add prompt**: Define in `src/llm/prompts.py` or register dynamically
- **Experiment configs**: `configs/my_config.yaml`
- **Results**: `experiments/experiments.jsonl` (auto-created)

## 🎓 Learning Examples

All placeholder strategies include detailed implementation skeletons:
- `src/strategies/autoregressive.py` - Step-by-step planning
- `src/strategies/repair.py` - Validation-based repair
- `src/strategies/randomized.py` - Multiple candidates

Each has:
- ✅ Docstrings explaining the approach
- ✅ TODO checklist
- ✅ Code skeleton showing structure
- ✅ Example implementation in comments

## Need Help?

See full docs:
- **EXTENSIBLE_ARCHITECTURE.md** - Complete architecture guide
- **TESTING.md** - Testing guide
- **QUICK_REFERENCE.md** - Usage examples
- **PROJECT_STRUCTURE.md** - Directory organization
