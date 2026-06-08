#!/usr/bin/env python3
"""Batch-run centralized CoDMAP benchmarks with MAPlan in multi-agent unfactored mode.

For each domain under a centralized benchmark root (contains `domain.pddl`, `problem*.pddl`,
and corresponding `problem*.addl` files), this script:
 - detects matching .addl files for each problem
 - translates domain+problem+addl to `problem.proto` using the provided translator
 - runs the MAPlan planner in unfactored multi-agent mode against the produced `problem.proto`
 - validates the plan with VAL if available
 - saves artifacts and produces a batch-style text summary

The script is resumable: if a problem has an existing validated plan, it is skipped.

Usage example:
 python tools/evaluate_maplan_batch.py \
   --maplan-root /home/rr/maplan \
   --bench-root /home/rr/ma-planning/centralized \
   --val-bin /home/rr/ma-planning/VAL/build/linux64/Release/bin/Validate

"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import json
import time

DEFAULT_MAPLAN_ROOT = Path("/home/rr/maplan")
DEFAULT_TRANSLATOR = Path("/home/rr/maplan/third-party/translate/translate.py")
DEFAULT_PLANNER = Path("/home/rr/maplan/bin/search")
DEFAULT_BENCH_ROOT = Path("/home/rr/ma-planning/centralized")
DEFAULT_OUTPUT_ROOT = Path("results_maplan")
DEFAULT_LOGS_ROOT = Path("logs")
DEFAULT_VAL_BIN = Path("/home/rr/ma-planning/VAL/build/linux64/Release/bin/Validate")
DEFAULT_TIMEOUT_TRANSLATE = 360
DEFAULT_TIMEOUT_PLAN = 3600
DEFAULT_MEMORY_MB = 8192  # 8 GB


@dataclass
class ProblemResult:
    domain: str
    problem: str
    plan_path: Optional[str]
    log_path: str
    translation_success: bool
    planning_success: bool
    validation_success: bool
    plan_found: bool
    plan_length: int
    runtime_seconds: float
    failure_stage: str  # "none", "translation", "planning", "validation"
    error_message: Optional[str]


def discover_domains(bench_root: Path, selected: List[str] | None = None) -> List[Path]:
    domains: List[Path] = []
    for d in sorted(bench_root.iterdir()):
        if d.is_dir():
            if selected and d.name not in selected:
                continue
            domains.append(d)
    return domains


def find_matching_addl(domain_dir: Path, problem_file: Path) -> Optional[Path]:
    """Find .addl file matching the problem stem."""
    stem = problem_file.stem
    addl = domain_dir / f"{stem}.addl"
    return addl if addl.exists() else None


def collect_problem_files(domain_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    """Collect (problem_file, addl_file) pairs. Only include problems with matching .addl."""
    problems = [
        p for p in sorted(domain_dir.glob("*.pddl")) if p.name.lower() != "domain.pddl"
    ]
    result = []
    for prob in problems:
        addl = find_matching_addl(domain_dir, prob)
        if addl:
            result.append((prob, addl))
    return result


def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def translate(
    domain_file: Path,
    problem_file: Path,
    addl_file: Path,
    translator: Path,
    work_dir: Path,
    timeout: int | None,
) -> Tuple[bool, str, str]:
    """Run translation with .addl file, produce `problem.proto` in work_dir.
    Returns (success, stdout, stderr)
    """
    out_file = work_dir / "problem.proto"
    cmd = [
        sys.executable,
        str(translator),
        "--proto",
        "--output",
        str(out_file),
        str(domain_file),
        str(problem_file),
        str(addl_file),
    ]
    try:
        env = dict(os.environ)
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        stdout = _coerce_text(proc.stdout)
        stderr = _coerce_text(proc.stderr)
        success = out_file.exists() and out_file.stat().st_size > 0
        return (success, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        return (False, _coerce_text(exc.stdout), _coerce_text(exc.stderr))
    except Exception as exc:
        return (False, "", str(exc))


def plan_proto(
    proto_path: Path,
    planner: Path,
    work_dir: Path,
    timeout: int | None,
    memory_mb: int | None = None,
) -> Tuple[bool, str, str, int]:
    """Run MAPlan planner in unfactored multi-agent mode on `problem.proto`.
    Returns (solved, stdout, stderr, plan_length)
    """
    out_plan = work_dir / "plan.out"
    cmd = [
        str(planner),
        "--ma-unfactor",
        "-p",
        str(proto_path.name),
        "-H",
        "ma-ff",
        "-s",
        "lazy",
        "-o",
        str(out_plan.name),
    ]
    if memory_mb:
        cmd.extend(["--max-mem", str(memory_mb)])
    if timeout:
        cmd.extend(["--max-time", str(timeout)])
    try:
        proc = subprocess.run(
            cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout
        )
        stdout = _coerce_text(proc.stdout)
        stderr = _coerce_text(proc.stderr)
        solved = out_plan.exists() and out_plan.stat().st_size > 0
        plan_len = 0
        if solved:
            # Try to extract plan length from output
            for line in stdout.split("\n"):
                if "Plan Length" in line or "Plan Cost" in line:
                    try:
                        plan_len = int(line.split(":")[-1].strip())
                    except:
                        pass
        return (solved, stdout, stderr, plan_len)
    except subprocess.TimeoutExpired as exc:
        return (False, _coerce_text(exc.stdout), _coerce_text(exc.stderr), 0)
    except Exception as exc:
        return (False, "", str(exc), 0)


def validate_plan(
    plan_file: Path,
    domain_file: Path,
    problem_file: Path,
    val_bin: Path,
    timeout: int | None = 30,
) -> Tuple[bool, str, str]:
    """Validate plan with VAL. Returns (valid, stdout, stderr)."""
    if not val_bin or not val_bin.exists():
        return (
            True,
            "VAL not available, skipping validation",
            "",
        )  # Skip if VAL not available

    cmd = [str(val_bin), str(domain_file), str(problem_file), str(plan_file)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout = _coerce_text(proc.stdout)
        stderr = _coerce_text(proc.stderr)
        # VAL outputs "Plan valid" or "Plan failed"
        valid = "valid" in stdout.lower() and proc.returncode == 0
        return (valid, stdout, stderr)
    except subprocess.TimeoutExpired:
        return (False, "", "VAL timeout")
    except Exception as exc:
        return (False, "", str(exc))


def run_problem(
    domain_dir: Path,
    domain_file: Path,
    problem_file: Path,
    addl_file: Path,
    translator: Path,
    planner: Path,
    val_bin: Optional[Path],
    output_root: Path,
    logs_root: Path,
    timeout_translate: int,
    timeout_plan: int,
    memory_mb: int,
) -> ProblemResult:
    domain = domain_dir.name
    problem_stem = problem_file.stem
    out_dir = output_root / domain
    log_dir = logs_root / domain
    ensure_dirs(out_dir)
    ensure_dirs(log_dir)
    plan_dest = out_dir / f"{problem_stem}.plan"
    log_dest = log_dir / f"{problem_stem}.log"
    validation_marker = out_dir / f"{problem_stem}.validated"

    # Resumable: skip if already validated
    if validation_marker.exists():
        plan_path = (
            str(plan_dest)
            if plan_dest.exists() and plan_dest.stat().st_size > 0
            else None
        )
        return ProblemResult(
            domain=domain,
            problem=problem_stem,
            plan_path=plan_path,
            log_path=str(log_dest),
            translation_success=True,
            planning_success=plan_path is not None,
            validation_success=True,
            plan_found=plan_path is not None,
            plan_length=0,
            runtime_seconds=0.0,
            failure_stage="none",
            error_message=None,
        )

    start = time.time()
    with tempfile.TemporaryDirectory(prefix=f"maplan_{domain}_{problem_stem}_") as td:
        work_dir = Path(td)
        # Copy files into work dir
        local_domain = work_dir / domain_file.name
        local_problem = work_dir / problem_file.name
        local_addl = work_dir / addl_file.name
        shutil.copy2(domain_file, local_domain)
        shutil.copy2(problem_file, local_problem)
        shutil.copy2(addl_file, local_addl)

        log_lines = []
        translation_success = False
        planning_success = False
        validation_success = False
        plan_length = 0
        plan_found = False
        failure_stage = "none"
        error_msg = None

        # === TRANSLATION ===
        log_lines.append("=== TRANSLATION ===")
        t_success, t_stdout, t_stderr = translate(
            local_domain,
            local_problem,
            local_addl,
            translator,
            work_dir,
            timeout_translate,
        )
        log_lines.append(t_stdout)
        if t_stderr:
            log_lines.append("STDERR: " + t_stderr)

        if not t_success:
            failure_stage = "translation"
            error_msg = "Translation failed"
            log_lines.append(error_msg)
        else:
            translation_success = True

            # === PLANNING ===
            log_lines.append("=== PLANNING ===")
            proto = work_dir / "problem.proto"
            p_success, p_stdout, p_stderr, plan_len = plan_proto(
                proto, planner, work_dir, timeout_plan, memory_mb
            )
            log_lines.append(p_stdout)
            if p_stderr:
                log_lines.append("STDERR: " + p_stderr)

            if not p_success:
                failure_stage = "planning"
                error_msg = "Planning failed or timed out"
                log_lines.append(error_msg)
            else:
                planning_success = True
                plan_found = True
                plan_length = plan_len
                out_plan = work_dir / "plan.out"
                shutil.copy2(out_plan, plan_dest)

                # === VALIDATION ===
                if val_bin and val_bin.exists():
                    log_lines.append("=== VALIDATION ===")
                    val_success, val_stdout, val_stderr = validate_plan(
                        plan_dest, local_domain, local_problem, val_bin
                    )
                    log_lines.append(val_stdout)
                    if val_stderr:
                        log_lines.append("STDERR: " + val_stderr)

                    if val_success:
                        validation_success = True
                        validation_marker.touch()
                    else:
                        failure_stage = "validation"
                        error_msg = "Plan validation failed"
                        log_lines.append(error_msg)
                else:
                    # No VAL available; mark as validated anyway
                    validation_success = True
                    validation_marker.touch()

        duration = time.time() - start
        combined_log = "\n".join(log_lines)
        log_dest.write_text(combined_log, encoding="utf-8")

        return ProblemResult(
            domain=domain,
            problem=problem_stem,
            plan_path=str(plan_dest) if plan_found else None,
            log_path=str(log_dest),
            translation_success=translation_success,
            planning_success=planning_success,
            validation_success=validation_success,
            plan_found=plan_found,
            plan_length=plan_length,
            runtime_seconds=duration,
            failure_stage=failure_stage,
            error_message=error_msg,
        )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-run MAPlan over centralized benchmarks with .addl files."
    )
    parser.add_argument(
        "--maplan-root",
        type=Path,
        default=DEFAULT_MAPLAN_ROOT,
        help="Path to MAPlan root",
    )
    parser.add_argument(
        "--translator",
        type=Path,
        default=DEFAULT_TRANSLATOR,
        help="Path to translate.py",
    )
    parser.add_argument(
        "--planner", type=Path, default=DEFAULT_PLANNER, help="Path to planner binary"
    )
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=DEFAULT_BENCH_ROOT,
        help="Centralized benchmark root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where to save plans",
    )
    parser.add_argument(
        "--logs-root", type=Path, default=DEFAULT_LOGS_ROOT, help="Where to save logs"
    )
    parser.add_argument(
        "--val-bin",
        type=Path,
        default=DEFAULT_VAL_BIN,
        help="Path to VAL Validate binary (optional)",
    )
    parser.add_argument(
        "--domains",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated list of domains to run",
    )
    parser.add_argument(
        "--timeout-translate",
        type=int,
        default=DEFAULT_TIMEOUT_TRANSLATE,
        help="Timeout for translation in seconds",
    )
    parser.add_argument(
        "--timeout-plan",
        type=int,
        default=DEFAULT_TIMEOUT_PLAN,
        help="Timeout for planning in seconds",
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=DEFAULT_MEMORY_MB,
        help="Memory limit for planner in MB",
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="Optional JSON report output path"
    )
    parser.add_argument(
        "--text-report",
        type=Path,
        default=None,
        help="Optional batch-style text summary output path",
    )

    args = parser.parse_args(argv)

    maplan_root = args.maplan_root.resolve()
    translator = args.translator
    planner = args.planner
    bench_root = args.bench_root.resolve()
    output_root = args.output_root.resolve()
    logs_root = args.logs_root.resolve()
    val_bin = args.val_bin if args.val_bin and args.val_bin.exists() else None
    timeout_translate = args.timeout_translate
    timeout_plan = args.timeout_plan
    memory_mb = args.memory_mb
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # basic validations
    if not maplan_root.exists():
        print(f"MAPlan root not found: {maplan_root}")
        return 2
    if not translator.exists():
        print(f"Translator not found: {translator}")
        return 2
    if not planner.exists():
        print(f"Planner not found: {planner}")
        return 2
    if not bench_root.exists():
        print(f"Benchmark root not found: {bench_root}")
        return 2

    ensure_dirs(output_root)
    ensure_dirs(logs_root)

    domains = discover_domains(bench_root, args.domains)
    results: List[ProblemResult] = []

    for domain_dir in domains:
        domain_file = domain_dir / "domain.pddl"
        if not domain_file.is_file():
            print(f"Skipping {domain_dir.name}: domain.pddl missing")
            continue
        problems_with_addl = collect_problem_files(domain_dir)
        if not problems_with_addl:
            print(f"No problems with matching .addl files found under {domain_dir}")
            continue
        print(f"Running {len(problems_with_addl)} problems in domain {domain_dir.name}")
        for prob, addl in problems_with_addl:
            print(f"-> {domain_dir.name}/{prob.name}")
            res = run_problem(
                domain_dir,
                domain_file,
                prob,
                addl,
                translator,
                planner,
                val_bin,
                output_root,
                logs_root,
                timeout_translate,
                timeout_plan,
                memory_mb,
            )
            results.append(res)
            status = (
                "PASS"
                if res.validation_success
                else (
                    res.failure_stage.upper()
                    if res.failure_stage != "none"
                    else "SKIPPED"
                )
            )
            print(
                f"   {status} | plan_len={res.plan_length} | runtime={res.runtime_seconds:.2f}s | log={res.log_path}"
            )

    # JSON report
    if args.report:
        report_path: Path = args.report
        report_path.write_text(
            json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
        )
        print(f"JSON report written to {report_path}")

    summary_path = args.text_report or output_root / f"batch_eval_maplan_{batch_id}.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        render_text_report(
            results=results,
            report_path=summary_path,
            maplan_root=maplan_root,
            bench_root=bench_root,
            translator=translator,
            planner=planner,
            val_bin=val_bin,
            timeout_translate=timeout_translate,
            timeout_plan=timeout_plan,
            memory_mb=memory_mb,
        ),
        encoding="utf-8",
    )
    print(f"Text summary written to {summary_path}")

    # summary
    total = len(results)
    translated = sum(1 for r in results if r.translation_success)
    planned = sum(1 for r in results if r.planning_success)
    validated = sum(1 for r in results if r.validation_success)
    failures = total - validated
    print("\n==== SUMMARY ====\n")
    print(f"Total problems: {total}")
    print(f"Translated: {translated}")
    print(f"Planned: {planned}")
    print(f"Validated: {validated}")
    print(f"Failed: {failures}")
    print(f"Coverage: {(validated/total*100) if total > 0 else 0:.1f}%")

    return 0


def render_text_report(
    results: List[ProblemResult],
    report_path: Path,
    maplan_root: Path,
    bench_root: Path,
    translator: Path,
    planner: Path,
    val_bin: Optional[Path],
    timeout_translate: int,
    timeout_plan: int,
    memory_mb: int,
) -> str:
    total = len(results)
    translated = sum(1 for r in results if r.translation_success)
    planned = sum(1 for r in results if r.planning_success)
    validated = sum(1 for r in results if r.validation_success)
    failed = total - validated
    plan_total = sum(r.plan_length for r in results)
    avg_plan = (plan_total / planned) if planned else 0.0
    avg_runtime = sum(r.runtime_seconds for r in results) / total if total else 0.0

    lines = []
    lines.append("=" * 80)
    lines.append("MAPlan BATCH EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Report:        {report_path}")
    lines.append(f"Generated:     {datetime.now().isoformat()}")
    lines.append(f"MAPlan root:   {maplan_root}")
    lines.append(f"Bench root:    {bench_root}")
    lines.append(f"Translator:    {translator}")
    lines.append(f"Planner:       {planner}")
    lines.append(f"VAL:           {val_bin if val_bin else 'not available'}")
    lines.append(f"Translate tm:   {timeout_translate}s")
    lines.append(f"Plan tm:        {timeout_plan}s")
    lines.append(f"Memory limit:   {memory_mb} MB")
    lines.append("")
    lines.append("-" * 80)
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Total Problems:      {total}")
    lines.append(f"Translated:          {translated}")
    lines.append(f"Planned:             {planned}")
    lines.append(f"Validated:           {validated}")
    lines.append(f"Failed:              {failed}")
    lines.append(f"Plan Length Total:    {plan_total}")
    lines.append(f"Avg Plan Length:      {avg_plan:.2f}")
    lines.append(f"Avg Runtime/Problem:  {avg_runtime:.2f}s")
    lines.append("")
    lines.append("-" * 80)
    lines.append("PER-PROBLEM RESULTS")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        f"{'Domain':<14} {'Problem':<28} {'Trans':<7} {'Plan':<7} {'Valid':<8} {'Len':>5} {'Time':>8}  {'Error Type'}"
    )
    lines.append("-" * 80)
    for r in results:
        error_type = r.failure_stage if r.failure_stage != "none" else "ok"
        lines.append(
            f"{r.domain:<14} {r.problem:<28} {str(r.translation_success):<7} {str(r.planning_success):<7} {str(r.validation_success):<8} {r.plan_length:>5} {r.runtime_seconds:>7.2f}s  {error_type}"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
