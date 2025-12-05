#!/usr/bin/env python3
"""Quick test to verify the refactored structure works."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.core.config import Config

        print("✓ src.core.config")
    except Exception as e:
        print(f"✗ src.core.config: {e}")
        return False

    try:
        from src.core.pipeline import MAPLLMPipeline

        print("✓ src.core.pipeline")
    except Exception as e:
        print(f"✗ src.core.pipeline: {e}")
        return False

    try:
        from src.converters.ma_converter import MAPDDLConverter

        print("✓ src.converters.ma_converter")
    except Exception as e:
        print(f"✗ src.converters.ma_converter: {e}")
        return False

    try:
        from src.llm.client import LLMClient

        print("✓ src.llm.client")
    except Exception as e:
        print(f"✗ src.llm.client: {e}")
        return False

    try:
        from src.validation.evaluator import PlanEvaluator

        print("✓ src.validation.evaluator")
    except Exception as e:
        print(f"✗ src.validation.evaluator: {e}")
        return False

    try:
        from src.strategies import StrategyFactory

        print("✓ src.strategies.factory")
    except Exception as e:
        print(f"✗ src.strategies.factory: {e}")
        return False

    try:
        from src.utils.grounding import ActionGrounder

        print("✓ src.utils.grounding")
    except Exception as e:
        print(f"✗ src.utils.grounding: {e}")
        return False

    try:
        from src.utils.parsing import parse_actions_no_validation

        print("✓ src.utils.parsing")
    except Exception as e:
        print(f"✗ src.utils.parsing: {e}")
        return False

    return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")

    try:
        from src.core.config import Config

        config = Config()
        print(f"✓ Config created")
        print(f"  LLM Model: {config.llm_model}")
        print(f"  LLM URL: {config.llm_url}")
        print(f"  Max Steps: {config.max_steps}")
        print(f"  Temperature: {config.temperature}")
        return True
    except Exception as e:
        print(f"✗ Config failed: {e}")
        return False


def test_config_resolve():
    """Test configuration resolution."""
    print("\nTesting configuration resolution...")

    try:
        from src.core.config import Config

        config = Config()
        config.resolve()
        print("✓ Config resolved successfully")
        print(f"  Converter: {config.resolved_converter_script}")
        print(f"  Python: {config.resolved_python_cmd}")
        print(f"  Validate: {config.resolved_val_bin}")
        return True
    except Exception as e:
        print(f"✗ Config resolution failed: {e}")
        print("  Note: This is expected if VAL or converter are not installed")
        return False


def test_llm_client():
    """Test LLM client creation."""
    print("\nTesting LLM client...")

    try:
        from src.llm.client import LLMClient

        client = LLMClient(
            model="llama3:8b",
            url="http://localhost:11434",
            temperature=0.7,
        )
        print("✓ LLM client created")
        print(f"  Model: {client.model}")
        print(f"  URL: {client.url}")
        return True
    except Exception as e:
        print(f"✗ LLM client failed: {e}")
        return False


def test_strategy_factory():
    """Test strategy factory."""
    print("\nTesting strategy factory...")

    try:
        from src.strategies import StrategyFactory
        from src.llm.client import LLMClient
        from src.core.config import Config

        config = Config()
        llm = LLMClient(model="llama3:8b", url="http://localhost:11434")

        strategy = StrategyFactory.create(mode="no-val", llm=llm, config=config)
        print("✓ Strategy created")
        print(f"  Type: {type(strategy).__name__}")
        return True
    except Exception as e:
        print(f"✗ Strategy factory failed: {e}")
        return False


def test_tool_wrappers():
    """Test tool wrappers."""
    print("\nTesting tool wrappers...")

    try:
        from tools.val_wrapper import VALWrapper
        from tools.codmap_wrapper import CoDMAPWrapper

        print("✓ Tool wrappers imported")
        return True
    except Exception as e:
        print(f"✗ Tool wrappers failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MA-PDDL Refactored Structure Test")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Config Resolution", test_config_resolve()))
    results.append(("LLM Client", test_llm_client()))
    results.append(("Strategy Factory", test_strategy_factory()))
    results.append(("Tool Wrappers", test_tool_wrappers()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Structure is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("   Some failures are expected if external tools are not installed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
