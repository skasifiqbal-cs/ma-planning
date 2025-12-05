#!/bin/bash
# Quick smoke test for MA-PDDL planning

set -e

echo "========================================="
echo "MA-PDDL Smoke Test"
echo "========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test 1: Structure test
echo ""
echo "Test 1: Verifying structure..."
if python3 tests/test_structure.py > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Structure test passed"
else
    echo -e "${RED}✗${NC} Structure test failed"
    python3 tests/test_structure.py
    exit 1
fi

# Test 2: Config resolution
echo ""
echo "Test 2: Testing configuration..."
if python3 -c "from src.core.config import Config; c = Config(); c.resolve(); print('Config OK')" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Configuration resolved"
else
    echo -e "${RED}✗${NC} Configuration failed"
    python3 -c "from src.core.config import Config; c = Config(); c.resolve()"
    exit 1
fi

# Test 3: CLI help
echo ""
echo "Test 3: Testing CLI..."
if python3 src/cli/main.py --help > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} CLI is working"
else
    echo -e "${RED}✗${NC} CLI failed"
    python3 src/cli/main.py --help
    exit 1
fi

# Test 4: LLM eval help
echo ""
echo "Test 4: Testing LLM evaluation script..."
if [ -f "tools/llm_eval.py" ]; then
    if python3 tools/llm_eval.py --help > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} LLM evaluation script is working"
    else
        echo -e "${RED}✗${NC} LLM evaluation script failed"
        python3 tools/llm_eval.py --help
        exit 1
    fi
else
    echo -e "${RED}✗${NC} LLM evaluation script not found"
fi

# Test 5: Check if sample domain exists
echo ""
echo "Test 5: Looking for test data..."
if [ -d "centralized" ] && [ "$(ls -A centralized)" ]; then
    echo -e "${GREEN}✓${NC} Found centralized test data"
    SAMPLE_DOMAIN=$(find centralized -name "*.pddl" -type f | head -1)
    if [ -n "$SAMPLE_DOMAIN" ]; then
        echo "  Sample: $SAMPLE_DOMAIN"
    fi
else
    echo -e "${RED}⚠${NC}  No centralized test data (run planning once to generate)"
fi

echo ""
echo "========================================="
echo -e "${GREEN}All basic tests passed!${NC}"
echo "========================================="
echo ""
echo "To run a full planning test, try:"
echo "  python3 src/cli/main.py \\"
echo "    --domain-dir /path/to/domain \\"
echo "    --domain-file domain.pddl \\"
echo "    --problem-file problem.pddl \\"
echo "    --validate"
echo ""
