#!/usr/bin/env python3
"""
Example: Run MA-PDDL planning on a sample problem.

This script demonstrates how to use the planning pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Config
from src.core.pipeline import MAPLLMPipeline


def main():
    """Run a sample planning task."""

    # Check if we have sample data
    sample_domains = [
        "/home/rr/ma-planning/centralized/gripper",
        "/home/rr/Downloads/pddl-data-master/codmap-2015/unfactored/rovers",
        "/home/rr/Downloads/pddl-data-master/ipc-2000/blocks",
    ]

    domain_dir = None
    for d in sample_domains:
        if Path(d).exists():
            domain_dir = d
            break

    if not domain_dir:
        print("No sample domain found. Please provide a domain directory:")
        print("  python3 src/cli/main.py --domain-dir /path/to/domain --help")
        return 1

    print(f"Using sample domain: {domain_dir}")
    print()

    # Initialize config
    print("1. Initializing configuration...")
    config = Config()
    try:
        config.resolve()
        print("   ✓ Configuration resolved")
    except Exception as e:
        print(f"   ✗ Configuration failed: {e}")
        print("   Run ./setup.sh to install dependencies")
        return 1

    # Create pipeline
    print("\n2. Creating planning pipeline...")
    pipeline = MAPLLMPipeline(config)
    print("   ✓ Pipeline created")

    # Find a problem file
    domain_path = Path(domain_dir)
    problem_files = list(domain_path.glob("prob*.pddl")) or list(
        domain_path.glob("p*.pddl")
    )

    if not problem_files:
        print(f"   ✗ No problem files found in {domain_dir}")
        return 1

    problem_file = problem_files[0].name
    print(f"   Using problem: {problem_file}")

    # Run planning
    print("\n3. Running planning...")
    print("   (This will call the LLM, may take a moment...)")

    try:
        plan, plan_path, _, _ = pipeline.run(
            domain_dir=str(domain_path),
            domain_file="domain",
            problem_file=problem_file,
            mode="no-val",
            max_steps=10,  # Limit to 10 steps for quick test
            validate_after=False,  # Skip validation for quick test
        )

        print(f"   ✓ Planning complete!")
        print(f"   Generated {len(plan)} actions")
        print(f"   Plan saved to: {plan_path}")

        if plan:
            print("\n   First few actions:")
            for i, action in enumerate(plan[:5]):
                print(f"     {i+1}. {action}")

        return 0

    except Exception as e:
        print(f"   ✗ Planning failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
