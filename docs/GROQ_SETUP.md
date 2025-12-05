# Using Groq API with MA-PDDL Planning

This guide explains how to use the free and paid Groq APIs with the MA-PDDL planning framework.

## What is Groq?

Groq provides fast API access to powerful open-source models like Mixtral and Llama 3:
- **Free tier**: Yes, with generous limits
- **Speed**: Significantly faster than other providers (50+ tokens/sec)
- **Models**: Mixtral 8x7B, Llama 2 70B, Llama 3 70B, Llama 3 8B
- **Cost**: Very competitive (free tier available)
- **Website**: https://groq.com

## Quick Start

### 1. Get a Free API Key

Visit https://console.groq.com/keys and:
1. Sign up (free account required)
2. Create an API key
3. Copy the key

### 2. Set the API Key

```bash
export GROQ_API_KEY='gsk_...'  # Your API key
```

### 3. Update config.yaml

```yaml
llm:
  provider: groq
  model: mixtral-8x7b-32768
  temperature: 0.0
```

### 4. Run Planning

```bash
python src/cli/main.py \
  --domain-dir centralized/gripper \
  --problem-file prob01.pddl
```

## Available Groq Models

All models are available in the free tier with rate limits:

| Model | Identifier | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| **Mixtral 8x7B** | `mixtral-8x7b-32768` | ⚡⚡⚡ Fastest | Good | Recommended for most tasks |
| **Llama 3 70B** | `llama-3-70b-8192` | ⚡⚡ Fast | Excellent | Best quality for complex planning |
| **Llama 3 8B** | `llama-3-8b-8192` | ⚡⚡⚡ Fastest | Good | Lightweight, quick responses |
| **Llama 2 70B** | `llama2-70b-4096` | ⚡ Slower | Very Good | Alternative to Llama 3 |

## Configuration Examples

### Mixtral (Recommended - Best Speed/Quality Balance)

```yaml
llm:
  provider: groq
  model: mixtral-8x7b-32768
  temperature: 0.0
  max_tokens: 1024
```

### Llama 3 70B (Best Quality)

```yaml
llm:
  provider: groq
  model: llama-3-70b-8192
  temperature: 0.0
  max_tokens: 2048
```

### Llama 3 8B (Fastest)

```yaml
llm:
  provider: groq
  model: llama-3-8b-8192
  temperature: 0.0
  max_tokens: 512
```

## Advanced Usage

### Custom Temperature

```yaml
llm:
  provider: groq
  model: mixtral-8x7b-32768
  temperature: 0.3  # More creative (0.0 = deterministic, 1.0 = random)
```

### Batch Processing

Run all problems in a domain with Groq:

```bash
python src/cli/main.py \
  --config groq-config.yaml \
  --domain-dir centralized/gripper \
  --batch \
  --validate
```

### Python API

```python
from src.llm.provider_factory import ProviderFactory

# Create Groq provider
groq = ProviderFactory.create(
    provider="groq",
    model="mixtral-8x7b-32768",
    temperature=0.0,
)

# Send messages
messages = [
    {"role": "system", "content": "You are a planning expert."},
    {"role": "user", "content": "Generate a plan for..."}
]
response = groq.chat(messages)
print(response)
```

## Rate Limits

Free tier rate limits (per minute):
- **RPM**: 30 requests per minute
- **TPM**: 6,000 tokens per minute

For higher limits or custom rates, check https://console.groq.com

## Troubleshooting

### "api_key must be set"

```bash
# Make sure the environment variable is set correctly
export GROQ_API_KEY='gsk_...'

# Verify it's set
echo $GROQ_API_KEY
```

### "Rate limit exceeded"

Use smaller models or reduce batch size:
- Switch to `llama-3-8b-8192` for faster requests
- Reduce `max_tokens` in config
- Add delays between batch requests

### "Connection timeout"

The Groq servers are usually very responsive. If you get timeouts:
1. Check your internet connection
2. Try again (temporary network issue)
3. Check Groq status page: https://status.groq.com

## Performance Comparison

Typical response times on Groq (vs local Ollama):

| Model | Groq | Ollama (llama3:8b) |
|-------|------|-------------------|
| Mixtral 8x7B | 2-3s | 15-20s |
| Llama 3 70B | 3-5s | N/A (too large) |
| Llama 3 8B | 1-2s | 5-8s |

Times are approximate for typical 500-1000 token outputs.

## Cost Estimate

Free tier: **No cost** (up to rate limits)
- 30 requests/minute, 6,000 tokens/minute is quite generous!

Paid tier (if you exceed limits):
- Around $0.001-0.01 per 1M tokens depending on model
- Significantly cheaper than GPT-4

## Next Steps

1. **Try different models** - Test which works best for your planning tasks
2. **Experiment with temperature** - 0.0 for deterministic, 0.5 for balanced
3. **Measure performance** - Use ExperimentTracker to log all results
4. **Compare with other providers** - Try OpenAI, Anthropic, or Ollama

## Example: Full Experiment

```yaml
# groq-experiment.yaml
llm:
  provider: groq
  model: mixtral-8x7b-32768
  temperature: 0.0
  max_tokens: 1024

planning:
  strategy: no-val
  max_steps: 50
  validate: true
```

```bash
export GROQ_API_KEY='gsk_...'

python src/cli/main.py \
  --config groq-experiment.yaml \
  --domain-dir centralized/gripper \
  --batch \
  --validate
```

## Support

- Groq Docs: https://console.groq.com/docs
- Status: https://status.groq.com
- Community: https://groq.com/community

---

**Enjoy fast, free LLM inference for PDDL planning!** 🚀
