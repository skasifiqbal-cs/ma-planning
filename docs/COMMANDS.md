# Command Reference

## ⚙️ Configuration

Edit `config.yaml` to change LLM provider, model, or settings:

```yaml
llm:
  provider: ollama           # ollama, openai, anthropic, huggingface, azure
  model: llama3:8b
  temperature: 0.0           # 0.0 = deterministic, 1.0 = creative
```

Use different config:
```bash
python src/cli/main.py --config my_config.yaml --domain-dir ...
```

## 🚀 Quick Commands

### Check What's Available
```bash
# List LLM providers
python -c "from src.llm.providers import LLMProviderRegistry; print(LLMProviderRegistry.list_providers())"

# List strategies
python -c "from src.strategies.factory import StrategyRegistry; print(StrategyRegistry.list_strategies())"

# List prompt templates
python -c "from src.llm.prompts import PromptRegistry; print(PromptRegistry.list_templates())"
```

### Test LLM Providers
```bash
# Ollama (local)
python -c "from src.llm.providers import LLMProviderRegistry; llm = LLMProviderRegistry.create('ollama', model='llama3:8b'); print(llm.generate('Hello!'))"

# OpenAI
export OPENAI_API_KEY="your-key"
python -c "from src.llm.providers import LLMProviderRegistry; llm = LLMProviderRegistry.create('openai', model='gpt-4'); print(llm.generate('Hello!'))"

# Anthropic
export ANTHROPIC_API_KEY="your-key"
python -c "from src.llm.providers import LLMProviderRegistry; llm = LLMProviderRegistry.create('anthropic', model='claude-3-5-sonnet-20241022'); print(llm.generate('Hello!'))"
```

### Generate Example Configs
```bash
python -c "from src.utils.config_manager import save_example_configs; save_example_configs()"
# Creates configs/zero_shot_gpt4.yaml, configs/repair_claude.yaml, etc.
```

### Track Experiments
```bash
# View summary
python -c "from src.utils.experiments import ExperimentTracker; print(ExperimentTracker().summarize())"

# Export to CSV
python -c "from src.utils.experiments import ExperimentTracker; ExperimentTracker().export_csv('results.csv')"

# View as DataFrame
python -c "from src.utils.experiments import ExperimentTracker; print(ExperimentTracker().get_dataframe())"
```

### Parameter Sweeps
```bash
# Generate configs for sweep
python -c "
from src.utils.config_manager import ConfigManager
configs = ConfigManager.generate_sweep(
    base={'strategy': 'no-val', 'llm': {'provider': 'ollama'}},
    sweep={'llm.model': ['llama3:8b', 'mistral'], 'llm.temperature': [0.0, 0.5, 1.0]}
)
for i, c in enumerate(configs):
    ConfigManager.save_config(c, f'configs/sweep_{i:03d}.yaml')
print(f'Generated {len(configs)} configs')
"
```

### Run Tests
```bash
# All tests
./run_tests.sh

# Structure test only
python tests/test_structure.py

# Smoke test only
./tests/smoke_test.sh

# Planning test
python tests/test_planning.py
```

## 📝 One-Liners

### Check installation
```bash
python -c "from src.core.config import Config; c = Config(); c.resolve(); print('✓ Everything configured')"
```

### Test full pipeline
```bash
python -c "
from src.core.config import Config
from src.core.pipeline import MAPLLMPipeline
config = Config()
config.resolve()
pipeline = MAPLLMPipeline(config)
print('✓ Pipeline ready')
"
```

### Validate a plan
```bash
python -c "
from tools.val_wrapper import VALWrapper
val = VALWrapper()
is_valid, output = val.validate('domain.pddl', 'problem.pddl', 'plan.txt')
print(f'Valid: {is_valid}')
"
```

### Run CoDMAP planner
```bash
python -c "
from tools.codmap_wrapper import run_codmap_simple
plan = run_codmap_simple('domain.pddl', 'problem.pddl')
print(f'CoDMAP generated {len(plan)} actions')
"
```

## 🔧 Development

### Add and test new provider
```bash
# 1. Create provider file
cat > src/llm/providers/test_provider.py << 'EOF'
from .base import BaseLLMProvider
class TestProvider(BaseLLMProvider):
    def chat(self, messages, **kwargs):
        return "Test response"
    def get_provider_name(self):
        return "test"
EOF

# 2. Test it
python -c "
from src.llm.providers.test_provider import TestProvider
from src.llm.providers import LLMProviderRegistry
LLMProviderRegistry.register('test', TestProvider)
llm = LLMProviderRegistry.create('test', model='test-model')
print(llm.generate('Hello'))
"
```

### Add and test new strategy
```bash
# 1. Create strategy file
cat > src/strategies/test_strategy.py << 'EOF'
from .base import PlanGenerationStrategy
class TestStrategy(PlanGenerationStrategy):
    def generate_plan(self, domain_pddl, problem_pddl, valid_operators):
        return ['(action1)', '(action2)']
EOF

# 2. Test it
python -c "
from src.strategies.test_strategy import TestStrategy
from src.strategies.factory import StrategyRegistry
StrategyRegistry.register('test', TestStrategy)
strategy = StrategyRegistry.create('test', llm=None, config=None)
plan = strategy.generate_plan('', '', [])
print(f'Generated {len(plan)} actions')
"
```

## 🎯 Research Workflow Commands

### Setup new experiment
```bash
# Create experiment directory
mkdir -p experiments/exp_$(date +%Y%m%d)
cd experiments/exp_$(date +%Y%m%d)

# Create config
cat > config.yaml << 'EOF'
experiment_name: my_experiment
strategy: no-val
llm:
  provider: ollama
  model: llama3:8b
  temperature: 0.0
planning:
  max_steps: 50
  validate: true
EOF

# Run
python ../../src/cli/main.py --config config.yaml --batch
```

### Batch process domains
```bash
# Run on all domains in a directory
for domain in /path/to/domains/*/ ; do
    echo "Processing $domain"
    python src/cli/main.py --domain-dir "$domain" --batch
done
```

### Compare strategies
```bash
# Run same problem with different strategies
for strategy in no-val autoregressive repair randomized; do
    echo "Testing $strategy"
    python src/cli/main.py \
        --strategy "$strategy" \
        --domain-dir centralized/gripper \
        --problem-file prob01.pddl \
        2>&1 | grep -E "(Plan|Error)"
done
```

### Export results for analysis
```bash
# Export to multiple formats
python -c "
from src.utils.experiments import ExperimentTracker
tracker = ExperimentTracker()
tracker.export_csv('results.csv')
df = tracker.get_dataframe()
df.to_json('results.json', orient='records', lines=True)
df.to_excel('results.xlsx', index=False)  # Requires openpyxl
print(f'Exported {len(df)} experiments')
"
```

## 🐛 Debugging

### Check configuration
```bash
python -c "
from src.core.config import Config
c = Config()
c.resolve()
print('VAL:', c.resolved_val_bin)
print('Converter:', c.resolved_converter_script)
print('Python2:', c.resolved_python_cmd)
"
```

### Test LLM connection
```bash
# Ollama
curl http://localhost:11434/api/tags

# OpenAI
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Check imports
```bash
python -c "
import sys
from src.llm.providers import LLMProviderRegistry
from src.strategies.factory import StrategyRegistry
from src.llm.prompts import PromptRegistry
print('All imports successful!')
print(f'Python: {sys.version}')
print(f'Providers: {LLMProviderRegistry.list_providers()}')
print(f'Strategies: {StrategyRegistry.list_strategies()}')
"
```

## 💡 Tips

### Run with verbose output
```bash
python src/cli/main.py --verbose ...
```

### Time your experiments
```bash
time python src/cli/main.py ...
```

### Log output to file
```bash
python src/cli/main.py ... 2>&1 | tee experiment.log
```

### Run in background
```bash
nohup python src/cli/main.py ... > output.log 2>&1 &
```

### Monitor progress
```bash
watch -n 1 'tail -20 output.log'
```

## 📊 Analysis

### Success rate by strategy
```bash
python -c "
from src.utils.experiments import ExperimentTracker
import pandas as pd
df = ExperimentTracker().get_dataframe()
if not df.empty and 'config_strategy' in df and 'result_valid' in df:
    print(df.groupby('config_strategy')['result_valid'].agg(['sum', 'count', 'mean']))
"
```

### Average plan length by model
```bash
python -c "
from src.utils.experiments import ExperimentTracker
import pandas as pd
df = ExperimentTracker().get_dataframe()
if not df.empty and 'config_model' in df and 'result_plan_length' in df:
    print(df.groupby('config_model')['result_plan_length'].agg(['mean', 'min', 'max']))
"
```

### Filter experiments
```bash
python -c "
from src.utils.experiments import ExperimentTracker
tracker = ExperimentTracker()
valid_plans = tracker.filter_experiments(valid_only=True)
print(f'Found {len(valid_plans)} valid plans')
print(valid_plans[['experiment_id', 'result_plan_length']])
"
```
