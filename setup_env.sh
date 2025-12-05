#!/bin/bash
# Setup script for API keys from .env file
# Run this before using the planning framework

# Load environment variables from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "✓ Loaded API keys from .env"
else
    echo "⚠ .env file not found"
    echo "  Copy .env.example to .env and add your API keys"
    echo ""
    echo "  cp .env.example .env"
    echo "  # Edit .env and add your API keys"
    exit 1
fi

# Verify Groq API key is set (most common use case)
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠ Warning: GROQ_API_KEY not set in .env"
    echo "  Get your key at: https://console.groq.com/keys"
else
    echo "✓ GROQ_API_KEY is set"
fi

# Optional: verify other API keys
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✓ OPENAI_API_KEY is set"
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✓ ANTHROPIC_API_KEY is set"
fi

if [ -n "$HUGGINGFACE_API_KEY" ]; then
    echo "✓ HUGGINGFACE_API_KEY is set"
fi

echo ""
echo "Ready to use MA-PDDL planning framework!"
