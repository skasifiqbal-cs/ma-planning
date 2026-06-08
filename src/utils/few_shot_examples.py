"""Utilities for building few-shot examples from solved MA-PDDL runs.

The few-shot problems come from the unfactored MA-PDDL benchmark tree, and the
solutions come from MA-PLAN result files under ``results/<domain>/<problem>.plan``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .parsing import compress_pddl


@dataclass(frozen=True)
class FewShotCandidate:
    domain: str
    problem: str
    problem_size: int
    plan_length: int
    domain_file: Path
    problem_file: Path
    plan_file: Path


def iter_domain_problem_pairs(
    unfactored_root: Path,
    results_root: Path,
    selected_domains: Optional[Sequence[str]] = None,
) -> Iterable[FewShotCandidate]:
    selected = set(selected_domains) if selected_domains else None

    for domain_dir in sorted(unfactored_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain_name = domain_dir.name
        if selected is not None and domain_name not in selected:
            continue

        domain_file = domain_dir / "domain.pddl"
        if not domain_file.is_file():
            continue

        for problem_file in sorted(domain_dir.glob("*.pddl")):
            if problem_file.name == "domain.pddl":
                continue
            problem_name = problem_file.stem
            plan_file = results_root / domain_name / f"{problem_name}.plan"
            if not plan_file.is_file() or plan_file.stat().st_size == 0:
                continue

            plan_lines = _read_plan_lines(plan_file)
            if not plan_lines:
                continue

            yield FewShotCandidate(
                domain=domain_name,
                problem=problem_name,
                problem_size=problem_file.stat().st_size,
                plan_length=len(plan_lines),
                domain_file=domain_file,
                problem_file=problem_file,
                plan_file=plan_file,
            )


def build_few_shot_examples(
    unfactored_root: str | Path,
    results_root: str | Path,
    count: int = 3,
    selected_domains: Optional[Sequence[str]] = None,
    compress_inputs: bool = True,
) -> List[Dict[str, str]]:
    """Build few-shot examples — picks `count` shortest solved problems per domain.

    Selecting per-domain guarantees every domain gets `count` examples when
    loaded with domain filtering, rather than pulling N globally-smallest
    problems that may all come from the same domain.
    """
    unfactored_root = Path(unfactored_root)
    results_root = Path(results_root)

    all_candidates = list(
        iter_domain_problem_pairs(unfactored_root, results_root, selected_domains)
    )

    # Group by domain, pick N shortest per domain
    by_domain: Dict[str, List[FewShotCandidate]] = {}
    for c in all_candidates:
        by_domain.setdefault(c.domain, []).append(c)

    per_domain_sorted = []
    for domain_candidates in by_domain.values():
        domain_candidates.sort(key=lambda c: (c.problem_size, c.plan_length, c.problem))
        per_domain_sorted.extend(domain_candidates[: max(0, count)])

    examples: List[Dict[str, str]] = []
    for candidate in per_domain_sorted:
        domain_text = candidate.domain_file.read_text(encoding="utf-8")
        problem_text = candidate.problem_file.read_text(encoding="utf-8")
        if compress_inputs:
            domain_text = compress_pddl(
                domain_text, strip_comments=True, compact_whitespace=True
            )
            problem_text = compress_pddl(
                problem_text, strip_comments=True, compact_whitespace=True
            )

        examples.append(
            {
                "domain": candidate.domain,
                "problem": candidate.problem,
                "problem_size": str(candidate.problem_size),
                "plan_length": str(candidate.plan_length),
                "input": (
                    f"Domain: {domain_text}\n\n"
                    f"Problem: {problem_text}\n\n"
                    "Generate a plan to achieve the goal. Output actions one per line as: "
                    "(action_name agent_name arg1 arg2 ...)"
                ),
                "output": "\n".join(_read_plan_lines(candidate.plan_file)),
            }
        )

    return examples


def write_few_shot_examples_file(
    output_file: str | Path,
    unfactored_root: str | Path,
    results_root: str | Path,
    count: int = 2,
    selected_domains: Optional[Sequence[str]] = None,
    compress_inputs: bool = True,
) -> Path:
    """Build examples and write them to a JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = build_few_shot_examples(
        unfactored_root=unfactored_root,
        results_root=results_root,
        count=count,
        selected_domains=selected_domains,
        compress_inputs=compress_inputs,
    )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "unfactored_root": str(Path(unfactored_root).resolve()),
        "results_root": str(Path(results_root).resolve()),
        "example_count": len(examples),
        "examples": examples,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _read_plan_lines(plan_file: Path) -> List[str]:
    lines = []
    for line in plan_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines
