# MA-PDDL Planning with LLMs

Research framework for comparing planning approaches (LLM-based, classical, hybrid) on multi-agent PDDL benchmarks.

## Setup

### Option A — Docker (recommended, no local dependencies)

```bash
git clone <repo-url>
cd ma-planning
cp .env.example .env       # then add your API key
docker compose build       # builds VAL and installs Python deps (~2 min first time)
```

Run experiments:
```bash
docker compose run --rm plan plan --domain-dir domains/unfactored/rovers --problem-file p10 --validate
docker compose run --rm plan experiment --domains rovers logistics00
docker compose run --rm plan analyze
```

Results write to `./results/` on your host. `config.yaml` is mounted directly — edit it without rebuilding.

**Requirements:** Docker with Compose plugin. Nothing else.

---

### Option B — Local (Python venv)

```bash
git clone <repo-url>
cd ma-planning
sudo apt install cmake g++     # Ubuntu/Debian — needed to build VAL
bash setup.sh                  # creates venv, installs deps, builds VAL
source venv/bin/activate
```

Edit `.env` and add your API key:
```bash
GROQ_API_KEY=gsk_...   # free at https://console.groq.com/keys
```

**Requirements:** Python ≥ 3.10, `cmake`, `g++`.

## Usage

All workflows go through `run.py`:

```bash
# Single problem
python run.py plan --domain-dir domains/unfactored/rovers --problem-file p10 --validate

# All problems in one domain
python run.py batch --domain-dir domains/unfactored/rovers --validate

# Full experiment: subset of domains
python run.py experiment --domains rovers logistics00 --max-problems 18

# Full experiment: all domains in a directory
python run.py experiment

# Resume a crashed experiment (skips already-completed problems)
python run.py experiment --resume --run-id 20260608_143022_llmmodulo_groq-llama70b_r2_fs0

# Classical planners (separate tools)
python experiments/run_fmap_batch.py --domain logistics00 --num-problems 20
python experiments/run_maplan_batch.py --domain blocks --num-problems 20

# Aggregate all results to CSV
python run.py analyze
```

Each `batch` / `experiment` run writes to a self-contained directory:
```
results/20260608_143022_llmmodulo_groq-llama70b_r2_fs0/
├── config.yaml          ← exact config snapshot
├── run_meta.json        ← git commit, strategy, model, timestamp
├── rovers/
│   ├── eval.json        ← per-problem results + aggregate stats
│   └── log.json         ← LLM prompts, raw outputs, validation attempts
└── logistics00/
    └── ...
```

## Project Structure

```
ma-planning/
├── run.py              ← single entry point for all workflows
├── config.yaml         ← central configuration
├── .env.example        ← API key template (copy to .env)
├── src/                ← library code only
│   ├── core/           # Config & pipeline
│   ├── strategies/     # Planning strategies
│   ├── llm/            # LLM provider abstraction
│   ├── validation/     # VAL wrapper & plan evaluator
│   ├── converters/     # MA-PDDL → centralized PDDL
│   └── evaluation/     # Batch evaluator & logger
├── experiments/        # Analysis scripts (aggregate_results.py, build_few_shot.py, etc.)
├── domains/
│   ├── unfactored/     # CoDMAP unfactored MA-PDDL benchmarks (LLM input, 12 domains)
│   └── factored/       # CoDMAP factored benchmarks (FMAP experiments)
├── centralized/        # Converter output cache (VAL input; never read by LLMs)
├── results/            # Experiment outputs — one dir per run, config snapshot inside
├── VAL/                # Bundled VAL validator source + binary
├── pyperplan/          # Bundled pyperplan classical planner
└── codmap-2015/        # Bundled CoDMAP MA-PDDL → centralized converter
```

## Configuration

All settings in `config.yaml`. Key sections:

```yaml
llm:
  provider: groq          # groq | openai | anthropic | google | ollama | deepseek
  model: llama-3.3-70b-versatile
  temperature: 0.0

planning:
  strategy: llm-modulo    # llm-modulo | llm-repair | llm-merge | base

llm-modulo:
  backprompt_max_retries: 2

few_shot:
  enabled: false
  count: 2                # examples per domain injected into the prompt
  examples_file: results/few_shot_examples.json
```

Override via environment variables (prefix `MAP_PLANNING_`):
```bash
export MAP_PLANNING_LLM_MODEL=llama3:8b
export MAP_PLANNING_BACKPROMPT_MAX_RETRIES=3
```

## Planning Strategies

| Strategy | Key | Description |
|----------|-----|-------------|
| LLM-Modulo | `llm-modulo` | Generate → validate with VAL → backprompt with errors → retry |
| LLM-Repair | `llm-repair` | Replace invalid actions with nearest valid ones via embedding similarity |
| LLM-Merge  | `llm-merge`  | Per-agent subproblems solved by pyperplan, merged by LLM |
| Base       | `base`       | LLM generates plan, no validation (open-loop baseline) |

Old names (`val-feedback`, `repair`, `decomposition`, `no-val`) still work as aliases.

## Result Schema

### eval.json (per-problem results)

Each entry in `results[]` includes:

| Field | Description |
|-------|-------------|
| `problem` | Problem stem |
| `status` | `"success"` or `"error"` |
| `num_actions` | Plan length |
| `validation_passed` | Boolean |
| `execution_time` | Seconds |
| `num_attempts` | Validation attempts made (llm-modulo only; 0 for other strategies) |
| `prompt_tokens` | Tokens in prompt(s) sent to LLM |
| `completion_tokens` | Tokens in LLM response(s) |
| `total_tokens` | Total tokens consumed |

The `summary` block adds: `total_retries`, `avg_retries`, `total_prompt_tokens`, `total_completion_tokens`, `total_tokens` across the domain.

### log.json (full trace)

Each problem entry includes LLM prompt messages, raw output, final plan, validation result, timing, and (for llm-modulo) `validation_attempts` — the full per-attempt plan and VAL output.

### run_meta.json

Written alongside `config.yaml` in every run directory:
```json
{
  "run_id": "20260608_143022_llmmodulo_groq-llama70b_r2_fs2",
  "git_commit": "abc1234",
  "timestamp": "2026-06-08T14:30:22",
  "strategy": "llm-modulo",
  "model": "groq/llama-3.3-70b-versatile",
  "provider": "groq",
  "backprompt_max_retries": 2,
  "few_shot_example_count": 2
}
```

### domain_variance_report.csv

Generated by `python run.py analyze`. Columns: Date, Run ID, Model, Domain, N Problems, Coverage, Total Time (s), Avg Plan Length, Avg Retries, Total Tokens, Prompt Tokens, Completion Tokens.

## Few-Shot Workflow

1. **Run zero-shot** to collect solved plans:
   ```bash
   python run.py experiment
   ```

2. **Build example file** — picks the `count` shortest solved problems *per domain*:
   ```bash
   python experiments/build_few_shot.py --count 2 --domains rovers logistics00
   # writes results/few_shot_examples.json
   ```

3. **Enable few-shot** in `config.yaml`:
   ```yaml
   few_shot:
     enabled: true
     count: 2
     examples_file: results/few_shot_examples.json
   ```

4. **Run experiment** — the runner automatically excludes example problems from the test set per domain (no manual `--max-problems` adjustment needed):
   ```bash
   python run.py experiment
   ```

Examples are injected as alternating `user`/`assistant` turns before the actual problem, with the system prompt explaining their role.

## Extending

**Add a provider:**
```python
# src/llm/providers/my_provider.py
from .base import BaseLLMProvider

class MyProvider(BaseLLMProvider):
    def chat(self, messages, **kwargs): ...
    def get_provider_name(self): return "my-provider"
```
Register in `src/llm/provider_factory.py`.

**Add a strategy:**
```python
# src/strategies/my_strategy.py
from .base import PlanGenerationStrategy

class MyStrategy(PlanGenerationStrategy):
    def generate_plan(self, ma_domain_file, ma_problem_file,
                      ground_domain_file, ground_problem_file, max_steps): ...
```
Register in `src/strategies/factory.py`.

## Python API

```python
from src.core.config import Config
from src.core.pipeline import MAPLLMPipeline

config = Config.from_yaml("config.yaml")
config.resolve()

pipeline = MAPLLMPipeline(config)
plan, plan_path, domain_path, problem_path, raw_output = pipeline.run(
    domain_dir="domains/unfactored/rovers",
    domain_file="domain",
    problem_file="p01",
    mode="llm-modulo",
    validate_after=True,
)

# For llm-modulo: number of validation attempts made
num_attempts = pipeline.num_attempts       # validation loops (llm-modulo)
total_tokens = pipeline.total_tokens      # {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
```
