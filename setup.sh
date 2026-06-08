#!/bin/bash
# Setup script for MA-PDDL Planning framework
# Run once after cloning: bash setup.sh

set -e
cd "$(dirname "$0")"

echo "=== MA-PDDL Planning Setup ==="

# ── Python venv ────────────────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "[1/4] Python dependencies installed."

# ── Build VAL validator ────────────────────────────────────────────────────────
VAL_BIN="VAL/build/bin/Validate"
if [ ! -f "$VAL_BIN" ]; then
    echo "[2/4] Building VAL validator (requires cmake, g++)..."
    if ! command -v cmake &>/dev/null; then
        echo "  ERROR: cmake not found. Install with: sudo apt install cmake g++"
        exit 1
    fi
    mkdir -p VAL/build
    cmake -S VAL -B VAL/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_DEFAULT_CMP0057=NEW 2>/dev/null
    cmake --build VAL/build --config Release --parallel
    echo "[2/4] VAL built at $VAL_BIN"
else
    echo "[2/4] VAL already built."
fi

# ── Directories ────────────────────────────────────────────────────────────────
mkdir -p results
echo "[3/4] Directories ready."

# ── API keys ───────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[4/4] Created .env from .env.example — edit it to add your API keys."
else
    echo "[4/4] .env already exists."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your API key (e.g. GROQ_API_KEY)"
echo "  2. source venv/bin/activate"
echo "  3. python run.py plan --domain-dir centralized/rovers --problem-file p01 --validate"
echo ""
echo "Quick test (no API key needed — checks imports and config):"
echo "  python -c \"from src.core.config import Config; c = Config.from_yaml(); print('Config OK:', c.strategy)\""
