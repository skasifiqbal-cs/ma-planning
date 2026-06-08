#!/usr/bin/env python3
"""Run pyperplan on decomposed single-agent problems."""

import argparse
import sys
from pathlib import Path


def add_pyperplan_to_path():
    repo_root = Path(__file__).resolve().parents[1]
    pyperplan_path = repo_root / "pyperplan"
    if pyperplan_path.exists() and str(pyperplan_path) not in sys.path:
        sys.path.insert(0, str(pyperplan_path))


def load_planner():
    try:
        from pyperplan import planner
    except Exception as exc:
        raise RuntimeError(
            "pyperplan import failed. Ensure pyperplan is available."
        ) from exc
    return planner


def find_agent_dirs(base_dir: Path):
    if not base_dir.exists():
        raise FileNotFoundError(f"Decomposed base dir not found: {base_dir}")
    return [p for p in base_dir.iterdir() if p.is_dir()]


def find_domain_problem(agent_dir: Path):
    domain_candidates = sorted(agent_dir.glob("domain*.pddl"))
    problem_candidates = sorted(agent_dir.glob("problem*.pddl"))
    if not domain_candidates:
        raise FileNotFoundError(f"No domain*.pddl in {agent_dir}")
    if not problem_candidates:
        raise FileNotFoundError(f"No problem*.pddl in {agent_dir}")

    domain_file = domain_candidates[0]
    problem_file = problem_candidates[0]

    return domain_file, problem_file


def solve_with_pyperplan(
    planner, domain_file: Path, problem_file: Path, search: str, heuristic: str = None
):
    search_alg = planner.SEARCHES[search]
    heuristic_class = planner.HEURISTICS.get(heuristic) if heuristic else None
    plan = planner.search_plan(
        domain_file=str(domain_file),
        problem_file=str(problem_file),
        search=search_alg,
        heuristic_class=heuristic_class,
        use_preferred_ops=False,
    )
    if not plan:
        return []
    return [str(action) for action in plan]


def main():
    parser = argparse.ArgumentParser(
        description="Run pyperplan on decomposed agent problems."
    )
    parser.add_argument(
        "--decomp-dir",
        default="/tmp/ma-decomp-test/probBLOCKS-10-0",
        help="Directory containing per-agent subfolders",
    )
    parser.add_argument(
        "--search",
        default="bfs",
        help="pyperplan search algorithm (e.g., bfs, astar, gbf)",
    )
    parser.add_argument(
        "--heuristic",
        default=None,
        help="pyperplan heuristic (e.g., hff, hadd, hmax, blind) - required for astar/gbf",
    )
    args = parser.parse_args()

    add_pyperplan_to_path()
    planner = load_planner()

    base_dir = Path(args.decomp_dir)
    agent_dirs = find_agent_dirs(base_dir)

    if not agent_dirs:
        print(f"No agent directories found in {base_dir}")
        return 1

    print(f"Found {len(agent_dirs)} agent directories in {base_dir}")

    for agent_dir in sorted(agent_dirs):
        agent_name = agent_dir.name
        domain_file, problem_file = find_domain_problem(agent_dir)
        print(f"\n[AGENT] {agent_name}")
        print(f"  Domain:  {domain_file}")
        print(f"  Problem: {problem_file}")

        try:
            plan = solve_with_pyperplan(
                planner, domain_file, problem_file, args.search, args.heuristic
            )
        except Exception as exc:
            print(f"  ✗ pyperplan error: {exc}")
            continue

        if not plan:
            print("  ✗ No plan found")
            continue

        print(f"  ✓ Plan length: {len(plan)}")
        for idx, action in enumerate(plan[:10], 1):
            print(f"    {idx}. {action}")
        if len(plan) > 10:
            print(f"    ... ({len(plan) - 10} more actions)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
