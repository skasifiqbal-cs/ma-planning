#!/bin/bash
# Installation script for MA-Planning

set -e

echo "========================================="
echo "MA-Planning Setup Script"
echo "========================================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install package in editable mode
echo "Installing ma-planning in editable mode..."
pip install -e .

# Clone and build VAL if not present
if [ ! -d "VAL" ]; then
    echo "Cloning VAL..."
    git clone https://github.com/KCL-Planning/VAL.git
    cd VAL
    echo "Building VAL..."
    make
    cd ..
else
    echo "VAL already exists, skipping..."
fi

# Clone CoDMAP if not present
if [ ! -d "codmap-2015" ]; then
    echo "Cloning CoDMAP..."
    git clone https://github.com/AI-Planning/codmap-2015.git
    cd codmap-2015
    echo "Building CoDMAP..."
    make
    cd ..
else
    echo "CoDMAP already exists, skipping..."
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p centralized
mkdir -p results
mkdir -p plans
mkdir -p tools

# Setup environment variables from .env
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠ Please edit .env and add your API keys"
fi

# Source .env file
set -a
[ -f .env ] && source .env
set +a

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "API Key Setup:"
echo "  1. Edit .env and add your GROQ_API_KEY (https://console.groq.com/keys)"
echo "  2. Optional: Add OPENAI_API_KEY, ANTHROPIC_API_KEY, etc."
echo "  3. Run: source .env"
echo ""
echo "To run planning with Groq, use:"
echo "  python src/cli/main.py --domain-dir centralized/gripper --problem-file prob01.pddl"
echo ""
echo "To run planning with Ollama (local):"
echo "  1. Update config.yaml: provider: ollama"
echo "  2. Run: ollama serve &"
echo "  3. Run: python src/cli/main.py --domain-dir centralized/gripper --problem-file prob01.pddl"
echo ""
