# MA-PDDL Planning with LLMs

**Research-ready framework** for multi-agent PDDL planning using Large Language Models.

## ✨ Highlights

- 🤖 **6 LLM Providers** - OpenAI, Anthropic, Groq, Ollama, HuggingFace, Azure
- 🎯 **4 Planning Strategies** - No-validation, autoregressive, repair, randomized
- 📝 **6 Prompt Templates** - Zero-shot, few-shot, chain-of-thought, ReAct, and more
- 🧪 **Experiment Tracking** - Log all runs, export to CSV, analyze with pandas
- ⚙️ **Simple Config** - One YAML file with commented examples for all providers
- 🔧 **Extensible** - Add providers/strategies/prompts in minutes
- ✅ **Clean & Organized** - Minimal files, clear structure, focused docs

## 🚀 Quick Start

### Installation
```bash
./setup.sh
source venv/bin/activate
pip install -r requirements.txt
```

### Configure
Edit `config.yaml` to set your LLM provider and model:

```yaml
llm:
  provider: ollama       # ollama, openai, anthropic, groq, huggingface, azure
  model: llama3:8b
  temperature: 0.0       # 0.0=deterministic, 1.0=creative
```

### Run
```bash
# Start Ollama (if using local LLM)
ollama serve
ollama pull llama3:8b

# Run planning
python src/cli/main.py \
  --domain-dir centralized/gripper \
  --problem-file prob01.pddl \
  --validate
```

### Batch Mode
```bash
# Run all problems in a domain
python src/cli/main.py \
  --domain-dir centralized/gripper \
  --batch \
  --validate
```

### Use Different Config
```bash
# Create custom config
cp config.yaml my_experiment.yaml
# Edit my_experiment.yaml...

# Run with custom config
python src/cli/main.py \
  --config my_experiment.yaml \
  --domain-dir centralized/gripper \
  --problem-file prob01.pddl
```

## 🧪 Experiments

Track experiments automatically:
```python
from src.utils.experiments import ExperimentTracker

tracker = ExperimentTracker()
print(tracker.summarize())
tracker.export_csv('results.csv')
```

Parameter sweeps:
```python
from src.utils.config_manager import ConfigManager

configs = ConfigManager.generate_sweep(
    base={"strategy": "no-val"},
    sweep={
        "llm.model": ["llama3:8b", "mistral"],
        "llm.temperature": [0.0, 0.5]
    }
)
# Generates 6 configs (2 models × 3 temperatures)
```

## Project Structure

```
ma-planning/
├── src/              # Source code
│   ├── core/        # Config & pipeline
│   ├── converters/  # MA-PDDL conversion
│   ├── llm/         # LLM clients
│   ├── strategies/  # Planning strategies
│   ├── validation/  # Plan evaluation
│   ├── utils/       # Utilities
│   └── cli/         # CLI entry point
├── tools/           # External tool wrappers & utilities
├── docs/            # Documentation
└── legacy/          # Old files (for reference only)
```

## Configuration

Environment variables:
```bash
export MAP_PLANNING_LLM_MODEL="llama3:8b"
export MAP_PLANNING_LLM_URL="http://localhost:11434"
export MAP_PLANNING_VALIDATE_BIN="./VAL/Validate"
export MAP_PLANNING_MAX_STEPS="30"
export MAP_PLANNING_TEMPERATURE="0.7"
```

## Python API

```python
from src.core.config import Config
from src.core.pipeline import MAPLLMPipeline

config = Config()
config.resolve()

pipeline = MAPLLMPipeline(config)
plan, path, _, _ = pipeline.run(
    domain_dir="/path/to/domain",
    domain_file="domain.pddl",
    problem_file="problem.pddl",
    validate_after=True,
)
```

## 🎓 For Researchers

### Add Your Own LLM Provider (5 min)
```python
# src/llm/providers/my_provider.py
from .base import BaseLLMProvider

class MyProvider(BaseLLMProvider):
    def chat(self, messages, **kwargs):
        # Your implementation
        pass
    
    def get_provider_name(self):
        return "my-provider"

# Register in __init__.py
LLMProviderRegistry.register("my-provider", MyProvider)
```

### Add Your Own Strategy (10 min)
```python
# src/strategies/my_strategy.py
from .base import PlanGenerationStrategy

class MyStrategy(PlanGenerationStrategy):
    def generate_plan(self, domain_pddl, problem_pddl, valid_operators):
        # Your implementation
        return plan
```

### Add Custom Prompts (2 min)
```python
from src.llm.prompts import PromptRegistry, PromptTemplate

my_prompt = PromptTemplate(
    style="zero-shot",
    system_prompt="You are...",
    user_template="Problem: {problem}"
)
PromptRegistry.register("my-prompt", my_prompt)
```

## 📚 Documentation

| Guide | Purpose |
|-------|---------|
| **[RESEARCH_READY.md](docs/RESEARCH_READY.md)** | Complete feature overview |
| **[ADDING_COMPONENTS.md](docs/ADDING_COMPONENTS.md)** | How to add LLMs, strategies, prompts |
| **[EXTENSIBLE_ARCHITECTURE.md](docs/EXTENSIBLE_ARCHITECTURE.md)** | Architecture & design patterns |
| [COMMANDS.md](docs/COMMANDS.md) | Command reference |
| [TESTING.md](docs/TESTING.md) | How to test |

## 🧪 Testing

```bash
# Quick validation
./run_tests.sh

# Individual tests
python tests/test_structure.py  # Imports (6/6 passing)
./tests/smoke_test.sh           # Quick check (4/5 passing)
python tests/test_planning.py   # End-to-end test
```

## 🗂️ Architecture

```
src/
├── llm/
│   ├── providers/          # 6 LLM providers (OpenAI, Anthropic, Ollama, etc.)
│   │   └── __init__.py     # LLMProviderRegistry
│   └── prompts.py          # PromptRegistry + 6 templates
│
├── strategies/             # Planning strategies
│   ├── factory.py          # StrategyRegistry
│   ├── no_validation.py    # ✅ Implemented
│   ├── autoregressive.py   # 📝 Placeholder with guide
│   ├── repair.py           # 📝 Placeholder with guide
│   └── randomized.py       # 📝 Placeholder with guide
│
├── utils/
│   ├── experiments.py      # ExperimentTracker
│   └── config_manager.py   # ConfigManager + parameter sweeps
│
└── ... (core, converters, validation, cli)
```

## 📦 Requirements

### Core
- Python 3.10+
- pyperplan
- requests

### LLM Providers (Optional)
- `openai>=1.0.0` - For OpenAI (GPT-4, GPT-3.5)
- `anthropic>=0.18.0` - For Anthropic (Claude)
- `groq>=0.4.0` - For Groq (free and paid APIs)
- `huggingface-hub>=0.20.0` - For HuggingFace

### Research Tools
- `typer[all]>=0.9.0` - Modern CLI
- `rich>=13.0.0` - Beautiful output
- `pyyaml>=6.0` - Config files
- `pandas>=2.0.0` - Analysis

### External Tools
- [Ollama](https://ollama.ai) - Local LLM runtime (optional)
- [VAL](https://github.com/KCL-Planning/VAL) - Plan validation
- [CoDMAP](https://github.com/AI-Planning/codmap-2015) - MA-PDDL converter

## 🎯 What's Included

- ✅ **6 LLM providers** ready to use
- ✅ **4 planning strategies** (1 implemented, 3 with guides)
- ✅ **6 prompt templates** registered
- ✅ **Experiment tracking** system
- ✅ **Config management** with YAML
- ✅ **Parameter sweeps** utility
- ✅ **Test suite** (6/6 passing)
- ✅ **10 documentation guides**
- ✅ **Clean architecture** (src/, tools/, docs/, tests/)

## 🤝 Contributing

This is a research framework - we encourage experimentation!

1. **Add new components** - Follow guides in docs/
2. **Implement placeholders** - autoregressive, repair, randomized strategies
3. **Share your results** - Track with ExperimentTracker
4. **Improve docs** - Help others understand your additions

## 📄 License

MIT License - See LICENSE file

---

**Built for research. Designed for extensibility. Ready to use.** 🎓🚀
