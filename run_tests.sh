#!/bin/bash
# Comprehensive test runner for MA-Planning

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================"
echo "  MA-Planning Test Suite"
echo "========================================"
echo ""

# Test 1: Structure
echo -e "${YELLOW}[1/4] Running structure test...${NC}"
python3 tests/test_structure.py
echo -e "${GREEN}✓ Structure test passed${NC}"
echo ""

# Test 2: Configuration
echo -e "${YELLOW}[2/4] Testing configuration resolution...${NC}"
python3 -c "
from src.core.config import Config
c = Config()
c.resolve()
print('✓ Config resolved successfully')
print(f'  - LLM: {c.llm_model} @ {c.llm_url}')
print(f'  - VAL: {c.resolved_val_bin}')
print(f'  - Converter: {c.resolved_converter_script}')
"
echo ""

# Test 3: CLI
echo -e "${YELLOW}[3/4] Testing CLI interface...${NC}"
python3 src/cli/main.py --help > /dev/null
echo -e "${GREEN}✓ CLI working${NC}"
echo ""

# Test 4: Planning (optional, requires LLM)
echo -e "${YELLOW}[4/4] Testing actual planning...${NC}"
if python3 tests/test_planning.py 2>&1 | grep -q "Planning complete"; then
    echo -e "${GREEN}✓ Planning test passed${NC}"
else
    echo -e "${RED}✗ Planning test failed (requires Ollama running)${NC}"
    echo "  Start Ollama with: ollama serve"
fi
echo ""

echo "========================================"
echo -e "${GREEN}All tests completed!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "  - Run your own domain: python3 src/cli/main.py --domain-dir /path/to/domain --problem-file problem.pddl"
echo "  - See docs/TESTING.md for more test options"
echo "  - See docs/QUICK_REFERENCE.md for usage examples"
