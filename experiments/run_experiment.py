#!/usr/bin/env python3
"""Unified experiment runner for all backends: llm, fmap, maplan.

Usage:
    # LLM batch over all domains
    python experiments/run_experiment.py --backend llm --root-dir centralized/

    # Classical planner batch (fmap or maplan)
    python experiments/run_experiment.py --backend fmap --domain rovers
    python experiments/run_experiment.py --backend maplan --domain rovers --num-problems 20

    # Specific domains with LLM backend
    python experiments/run_experiment.py --backend llm --domains rovers logistics00 --max-problems 20
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for LLM and classical planning backends"
    )
    parser.add_argument(
        "--backend",
        choices=["llm", "fmap", "maplan"],
        required=True,
        help="Planning backend to use",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Config YAML (LLM backend only)",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Root dir with domain subdirs (LLM backend; default depends on backend)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Single domain name (classical backends)",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Subset of domain names (LLM backend)",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=20,
        help="Number of problems per domain (classical backends)",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=18,
        help="Max problems per domain (LLM backend)",
    )
    parser.add_argument(
        "--few-shot-example-count",
        type=int,
        default=0,
        help="Few-shot example count (LLM backend, 0=zero-shot)",
    )
    parser.add_argument(
        "--mode",
        default="val-feedback",
        help="Planning strategy (LLM backend)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    return parser.parse_args()


def run_llm(args: argparse.Namespace) -> int:
    root = args.root_dir or Path("/home/rr/Downloads/pddl-data-master/codmap-2015/unfactored/")
    cmd = [
        sys.executable,
        "experiments/run_llm_batch.py",
        "--root-dir", str(root),
        "--config", str(args.config),
        "--mode", args.mode,
        "--max-problems", str(args.max_problems),
        "--few-shot-example-count", str(args.few_shot_example_count),
    ]
    if args.domains:
        cmd.extend(["--domains"] + args.domains)
    if args.dry_run:
        cmd.append("--dry-run")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


def run_fmap(args: argparse.Namespace) -> int:
    if not args.domain:
        print("--domain required for fmap backend", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        "experiments/run_fmap_batch.py",
        "--domain", args.domain,
        "--num-problems", str(args.num_problems),
    ]
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


def run_maplan(args: argparse.Namespace) -> int:
    if not args.domain:
        print("--domain required for maplan backend", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        "experiments/run_maplan_batch.py",
        "--domain", args.domain,
        "--num-problems", str(args.num_problems),
    ]
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


def main() -> int:
    args = parse_args()
    runners = {"llm": run_llm, "fmap": run_fmap, "maplan": run_maplan}
    return runners[args.backend](args)


if __name__ == "__main__":
    raise SystemExit(main())
