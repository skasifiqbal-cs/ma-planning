# Configuration Guide

## Quick Start

Edit `config.yaml` to change your LLM provider or model:

```yaml
llm:
  provider: ollama       # Change this line
  model: llama3:8b       # Change this line
  temperature: 0.0       # 0.0 = consistent, 1.0 = creative
```

## Available Providers

| Provider | Set in config.yaml | Environment Variable |
|----------|-------------------|---------------------|
| **Ollama** (local) | `provider: ollama` | None |
| **OpenAI** | `provider: openai` | `OPENAI_API_KEY` |
| **Anthropic** | `provider: anthropic` | `ANTHROPIC_API_KEY` |
| **HuggingFace** | `provider: huggingface` | `HUGGINGFACE_API_KEY` |
| **Azure OpenAI** | `provider: azure` | `AZURE_OPENAI_API_KEY` |

## Examples

### Use GPT-4
1. Edit `config.yaml`:
   ```yaml
   provider: openai
   model: gpt-4
   ```
2. Set API key: `export OPENAI_API_KEY="sk-..."`
3. Run: `python src/cli/main.py --domain-dir ... --problem-file ...`

### Use Claude
1. Edit `config.yaml`:
   ```yaml
   provider: anthropic
   model: claude-3-5-sonnet-20241022
   ```
2. Set API key: `export ANTHROPIC_API_KEY="sk-ant-..."`
3. Run: `python src/cli/main.py --domain-dir ... --problem-file ...`

### Use Local Ollama
1. Start Ollama: `ollama serve`
2. Pull model: `ollama pull llama3:8b`
3. Config already set to use Ollama by default
4. Run: `python src/cli/main.py --domain-dir ... --problem-file ...`

## Multiple Configs

Create different config files for different experiments:

```bash
# Create custom config
cp config.yaml experiment1.yaml
# Edit experiment1.yaml...

# Use it
python src/cli/main.py --config experiment1.yaml --domain-dir ...
```

## Tips

- **Temperature**: Start with 0.0 for reproducibility, try 0.5-0.7 for variety
- **Validation**: Set `validate: true` to check plans with VAL
- **Batch mode**: Use `--batch` to run all problems in a domain
- **Environment variables**: Set API keys as env vars instead of hardcoding
