#!/usr/bin/env python3
"""Aggregate coverage across batch planning runs.

This script scans batch report files and computes mean coverage for a set of
runs, optionally filtered by domain, model, and an arbitrary substring match.
By default, no runs are excluded.

It supports two report shapes:
- ``batch_eval_*.json`` reports with a top-level ``summary`` field.
- ``batch_log_*.json`` reports with a top-level ``problems`` field.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class RunCoverage:
    path: Path
    domain: str
    model: str
    coverage_percent: float
    errors: int
    total: int
    source: str


def iter_report_files(root: Path, source: str) -> Iterable[Path]:
    if source == "batch_eval":
        yield from sorted(root.rglob("batch_eval_*.json"))
    elif source == "batch_log":
        yield from sorted(root.rglob("batch_log_*.json"))
    else:
        yield from sorted(root.rglob("batch_eval_*.json"))
        yield from sorted(root.rglob("batch_log_*.json"))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _compute_from_summary(data: dict[str, Any], path: Path) -> Optional[RunCoverage]:
    summary = data.get("summary")
    metadata = data.get("metadata", {})
    if not isinstance(summary, dict):
        return None

    domain = _as_text(metadata.get("domain", path.parent.name))
    model = _as_text(metadata.get("model", ""))
    errors = int(summary.get("errors", 0) or 0)

    if "coverage_percent" in summary:
        coverage = float(summary["coverage_percent"])
        total = int(summary.get("total_instances", summary.get("total_problems", 0)) or 0)
    else:
        total = int(summary.get("total_problems", summary.get("total_instances", 0)) or 0)
        if total <= 0:
            return None
        if "completed" in summary:
            solved = int(summary.get("completed", 0) or 0)
        elif "solved_instances" in summary:
            solved = int(summary.get("solved_instances", 0) or 0)
        else:
            solved = int(summary.get("passed", 0) or 0)
        coverage = (solved / total) * 100.0 if total else 0.0

    return RunCoverage(
        path=path,
        domain=domain,
        model=model,
        coverage_percent=coverage,
        errors=errors,
        total=total,
        source="batch_eval",
    )


def _compute_from_problems(data: dict[str, Any], path: Path) -> Optional[RunCoverage]:
    problems = data.get("problems")
    metadata = data.get("metadata", {})
    if not isinstance(problems, dict):
        return None

    total = len(problems)
    if total <= 0:
        return None

    covered = 0
    errors = 0
    for problem in problems.values():
        if not isinstance(problem, dict):
            continue
        final_plan = problem.get("final_plan") or []
        if final_plan:
            covered += 1
        validation = problem.get("validation", {})
        message = ""
        if isinstance(validation, dict):
            message = _as_text(validation.get("message", ""))
        if message.lower().startswith("error:"):
            errors += 1

    coverage = (covered / total) * 100.0
    return RunCoverage(
        path=path,
        domain=_as_text(metadata.get("domain", path.parent.name)),
        model=_as_text(metadata.get("model", "")),
        coverage_percent=coverage,
        errors=errors,
        total=total,
        source="batch_log",
    )


def load_run(path: Path) -> Optional[RunCoverage]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    run = _compute_from_summary(data, path)
    if run is not None:
        return run
    return _compute_from_problems(data, path)


def matches_filters(
    run: RunCoverage,
    *,
    domain: Optional[str],
    model: Optional[str],
    contains: Optional[str],
) -> bool:
    haystack = " | ".join([run.domain, run.model, str(run.path), run.source]).lower()
    if domain and domain.lower() not in run.domain.lower():
        return False
    if model and model.lower() not in run.model.lower():
        return False
    if contains and contains.lower() not in haystack:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean coverage across batch runs"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing batch report files",
    )
    parser.add_argument(
        "--domain",
        help="Only include runs for this domain (substring match)",
    )
    parser.add_argument(
        "--model",
        help="Only include runs for this model (substring match)",
    )
    parser.add_argument(
        "--contains",
        help="Only include runs whose path, domain, model, or source contains this string",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=None,
        help="Optionally ignore runs with more than this many errors",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Write the included runs to a CSV file",
    )
    parser.add_argument(
        "--source",
        choices=["batch_eval", "batch_log", "both"],
        default="batch_eval",
        help="Which report type to scan",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.results_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Results root not found: {root}")

    runs = []
    for path in iter_report_files(root, args.source):
        run = load_run(path)
        if run is None:
            continue
        if matches_filters(
            run,
            domain=args.domain,
            model=args.model,
            contains=args.contains,
        ):
            runs.append(run)

    if args.max_errors is None:
        included = list(runs)
        excluded = []
    else:
        included = [run for run in runs if run.errors <= args.max_errors]
        excluded = [run for run in runs if run.errors > args.max_errors]

    mean_coverage = (
        sum(run.coverage_percent for run in included) / len(included)
        if included
        else 0.0
    )

    if args.csv_out:
        csv_path = args.csv_out.resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "path",
                    "domain",
                    "model",
                    "source",
                    "coverage_percent",
                    "errors",
                    "total",
                ],
            )
            writer.writeheader()
            for run in included:
                writer.writerow(
                    {
                        "path": str(run.path),
                        "domain": run.domain,
                        "model": run.model,
                        "source": run.source,
                        "coverage_percent": f"{run.coverage_percent:.4f}",
                        "errors": run.errors,
                        "total": run.total,
                    }
                )

    if args.json:
        print(
            json.dumps(
                {
                    "results_root": str(root),
                    "filters": {
                        "domain": args.domain,
                        "model": args.model,
                        "contains": args.contains,
                        "max_errors": args.max_errors,
                    },
                    "matched_runs": len(runs),
                    "included_runs": len(included),
                    "excluded_runs": len(excluded),
                    "mean_coverage_percent": round(mean_coverage, 4),
                    "runs": [
                        {
                            "path": str(run.path),
                            "domain": run.domain,
                            "model": run.model,
                            "coverage_percent": round(run.coverage_percent, 4),
                            "errors": run.errors,
                            "total": run.total,
                            "source": run.source,
                        }
                        for run in included
                    ],
                },
                indent=2,
            )
        )
        return 0

    if args.csv_out:
        print(f"CSV written to: {csv_path}")

    print(f"Results root: {root}")
    print(
        f"Filters: domain={args.domain or '*'} model={args.model or '*'} contains={args.contains or '*'} max_errors={'none' if args.max_errors is None else f'<={args.max_errors}'}"
    )
    print(f"Matched runs: {len(runs)}")
    print(f"Included runs: {len(included)}")
    print(f"Excluded runs: {len(excluded)}")
    print(f"Mean coverage: {mean_coverage:.2f}%")
    if included:
        print("")
        for run in included:
            print(
                f"{run.domain:<14} {run.model:<24} {run.coverage_percent:>7.2f}% errors={run.errors:<3} {run.path}"
            )
    if excluded:
        print("")
        print("Excluded due to error count:")
        for run in excluded:
            print(
                f"{run.domain:<14} {run.model:<24} {run.coverage_percent:>7.2f}% errors={run.errors:<3} {run.path}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())