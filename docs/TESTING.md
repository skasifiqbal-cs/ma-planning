# Testing Guide

## Quick Tests

### 1. Structure Test (Fast)
Verifies all modules can be imported and configuration works:
```bash
python3 tests/test_structure.py
```

**Expected output:** All 6 tests should pass ✓

### 2. Smoke Test (Fast)
Quick check that everything is working:
```bash
./tests/smoke_test.sh
```

**Expected output:** All basic tests pass ✓

### 3. Planning Test (Requires LLM)
Test actual planning with LLM:
```bash
python3 tests/test_planning.py
```

**Expected output:** Generates a plan and saves it

## Full Integration Tests

### Test with Sample Domain

If you have test data available:

```bash
# Single problem
python3 src/cli/main.py \
  --domain-dir centralized/gripper \
  --domain-file domain \
  --problem-file prob01 \
  --validate

# Batch mode
python3 src/cli/main.py \
  --domain-dir centralized/gripper \
  --domain-file domain \
  --batch \
  --validate
```

### Test with Your Data

```bash
# Replace with your actual paths
python3 src/cli/main.py \
  --domain-dir /path/to/your/domain \
  --domain-file domain.pddl \
  --problem-file problem.pddl \
  --mode no-val \
  --validate
```

## Testing Different Components

### Test Configuration
```bash
python3 -c "
from src.core.config import Config
c = Config()
c.resolve()
print('Config OK')
print(f'LLM: {c.llm_model} @ {c.llm_url}')
print(f'VAL: {c.resolved_val_bin}')
"
```

### Test LLM Client
```bash
python3 -c "
from src.llm.client import LLMClient
llm = LLMClient(model='llama3:8b', url='http://localhost:11434')
response = llm.chat('Say hello')
print(f'LLM responded: {response}')
"
```

**Note:** Requires Ollama running (`ollama serve`)

### Test Strategy
```bash
python3 -c "
from src.strategies import StrategyFactory
from src.llm.client import LLMClient
from src.core.config import Config

config = Config()
llm = LLMClient(model='llama3:8b', url='http://localhost:11434')
strategy = StrategyFactory.create('no-val', llm, config)
print(f'Strategy created: {type(strategy).__name__}')
"
```

### Test Validation
```bash
python3 -c "
from src.validation.evaluator import PlanEvaluator
eval = PlanEvaluator('./VAL/build/linux64/Release/bin/Validate')
print('Evaluator ready')
"
```

### Test Tool Wrappers
```bash
python3 -c "
from tools.val_wrapper import VALWrapper
from tools.codmap_wrapper import CoDMAPWrapper
val = VALWrapper()
print(f'VAL path: {val.val_path}')
"
```

## Common Test Scenarios

### 1. Fresh Install Test
After running `./setup.sh`:
```bash
source venv/bin/activate
python3 tests/test_structure.py
./tests/smoke_test.sh
```

### 2. After Code Changes
```bash
# Run structure test
python3 tests/test_structure.py

# Run quick smoke test
./tests/smoke_test.sh

# Test actual planning if needed
python3 tests/test_planning.py
```

### 3. Before Committing
```bash
# Check structure
python3 tests/test_structure.py

# Check CLI still works
python3 src/cli/main.py --help

# Optional: Run full test
./tests/smoke_test.sh
```

## Troubleshooting Tests

### "No module named 'src'"
```bash
# Make sure you're in the project root
cd /home/rr/ma-planning
python3 tests/test_structure.py
```

### "Validate binary not found"
```bash
# Check if VAL is built
ls VAL/build/linux64/Release/bin/Validate

# If not, build it
cd VAL && make && cd ..

# Or set the path
export MAP_PLANNING_VALIDATE_BIN="./VAL/build/linux64/Release/bin/Validate"
```

### "LLM connection failed"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull a model
ollama pull llama3:8b
```

### "Converter script not found"
```bash
# Check if codmap-2015 exists
ls codmap-2015/competition/centalized/ma-to-pddl.py

# Clone if needed
git clone https://github.com/AI-Planning/codmap-2015.git
```

## Test Outputs

### Structure Test Output
```
✓ PASS: Imports
✓ PASS: Config
✓ PASS: Config Resolution
✓ PASS: LLM Client
✓ PASS: Strategy Factory
✓ PASS: Tool Wrappers
Total: 6/6 tests passed
```

### Smoke Test Output
```
✓ Structure test passed
✓ Configuration resolved
✓ CLI is working
✓ Found centralized test data
All basic tests passed!
```

### Planning Test Output
```
Using sample domain: /path/to/domain
1. Initializing configuration...
   ✓ Configuration resolved
2. Creating planning pipeline...
   ✓ Pipeline created
3. Running planning...
   ✓ Planning complete!
   Generated 5 actions
   Plan saved to: results/domain/problem.plan
```

## Automated Testing

Create a test script for CI/CD:

```bash
#!/bin/bash
# run_all_tests.sh

set -e

echo "Running all tests..."

# Structure test
echo "1. Structure test..."
python3 tests/test_structure.py

# Smoke test
echo "2. Smoke test..."
./tests/smoke_test.sh

# CLI test
echo "3. CLI help test..."
python3 src/cli/main.py --help > /dev/null

echo "All tests passed!"
```

## Performance Testing

Time how long planning takes:

```bash
time python3 src/cli/main.py \
  --domain-dir /path/to/domain \
  --problem-file problem.pddl \
  --mode no-val
```

## Coverage Testing

If you install pytest-cov:

```bash
pip install pytest pytest-cov
pytest tests/ --cov=src --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html
```
