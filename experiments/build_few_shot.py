#!/usr/bin/env python3
"""Build a reusable few-shot example file from solved MA-PLAN runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


add_repo_root_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a JSON file of few-shot examples from solved MA-PLAN plans"
    )
    parser.add_argument(
        "--unfactored-root",
        type=Path,
        default=Path(__file__).parent.parent / "domains" / "unfactored",
        help="Root containing unfactored MA-PDDL domain/problem files",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root containing MA-PLAN output plans (results/<domain>/<problem>.plan)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "few_shot_examples.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of shortest solved problems to include",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Optional subset of domain directory names to consider",
    )
    parser.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        help="Store raw PDDL instead of compressed PDDL in the examples",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.unfactored_root.is_dir():
        raise SystemExit(f"Unfactored root not found: {args.unfactored_root}")
    if not args.results_root.is_dir():
        raise SystemExit(f"Results root not found: {args.results_root}")

    from src.utils.few_shot_examples import write_few_shot_examples_file

    output_path = write_few_shot_examples_file(
        output_file=args.output,
        unfactored_root=args.unfactored_root,
        results_root=args.results_root,
        count=args.count,
        selected_domains=args.domains,
        compress_inputs=args.compress,
    )

    print(f"Wrote few-shot examples to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())