#!/usr/bin/env python3
"""Batch-evaluate FMAP on factored CoDMAP instances.

This script runs the FMAP jar over each factored multi-agent instance under
`codmap-2015/factored/<domain>/prob.../`, validates the produced plan with VAL,
and writes batch-style summary reports similar to the LLM planning pipeline.

Outputs are written under:
  results/fmap/<domain>/...
or, for batch runs:
  results/fmap/batch_eval_<run-id>.json
  results/fmap/batch_eval_<run-id>.txt

The summary includes:
- total time
- average time per instance
- average plan length
- coverage (plans found)
- validation coverage / pass rate
- per-instance results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


add_repo_root_to_path()

from src.validation.evaluator import (  # noqa: E402
    PlanEvaluator,
    extract_failure_reason,
)

DEFAULT_FMAP_ROOT = Path("/home/rr/Downloads/altorler-fmap-2ce663469695")
DEFAULT_FACTORED_ROOT = Path("/home/rr/Downloads/pddl-data-master/codmap-2015/factored")
DEFAULT_VALIDATION_ROOT = Path("/home/rr/ma-planning/centralized")
DEFAULT_OUTPUT_ROOT = Path("fmap_results")
DEFAULT_VAL_BIN = Path("/home/rr/ma-planning/VAL/build/linux64/Release/bin/Validate")
DEFAULT_SEARCH = 0
DEFAULT_HEURISTIC = 2
DEFAULT_TIMEOUT = 5400
DEFAULT_HOST = "127.0.0.1"
DEFAULT_JAVA_XMX = "16g"
DEFAULT_JAVA_XMS = "16g"
PLAN_LINE_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)?:\s*)?\([^\n]+\)\s*(?:\[[^\]]+\])?\s*$")
STEP_LINE_RE = re.compile(r"^\s*(?P<step>\d+):\s*(?P<action>.*\S)\s*$")


@dataclass
class InstanceResult:
    domain: str
    problem: str
    instance_dir: str
    status: str
    plan_found: bool
    validation_passed: bool
    validation_message: str
    validation_reason: str
    plan_length: int
    agent_count: int
    runtime_seconds: float
    timed_out: bool
    error_message: Optional[str]
    plan_file: Optional[str]
    stdout_file: Optional[str]
    stderr_file: Optional[str]
    timestamp: str


@dataclass
class BatchSummary:
    total_instances: int = 0
    solved_instances: int = 0
    validated_instances: int = 0
    failed_instances: int = 0
    timeout_instances: int = 0
    total_plan_length: int = 0
    total_runtime_seconds: float = 0.0
    avg_runtime_per_instance: float = 0.0
    avg_runtime_per_solved_instance: float = 0.0
    avg_plan_length: float = 0.0
    coverage_percent: float = 0.0
    validation_coverage_percent: float = 0.0
    validation_pass_rate_percent: float = 0.0


class FMAPBatchEvaluator:
    def __init__(
        self,
        fmap_root: Path,
        factored_root: Path,
        output_root: Path,
        val_bin: Optional[Path],
        validation_root: Path,
        search: int,
        heuristic: int,
        timeout: int,
        host: str = DEFAULT_HOST,
        validate: bool = True,
        java_xmx: str = DEFAULT_JAVA_XMX,
        java_xms: str = DEFAULT_JAVA_XMS,
    ):
        self.fmap_root = fmap_root.resolve()
        self.fmap_jar = (self.fmap_root / "FMAP.jar").resolve()
        self.factored_root = factored_root.resolve()
        self.output_root = output_root.resolve()
        self.val_bin = (
            val_bin.resolve() if val_bin else None
        ) or DEFAULT_VAL_BIN.resolve()
        self.validation_root = validation_root.resolve()
        self.search = search
        self.heuristic = heuristic
        self.timeout = timeout
        self.host = host

        self.validate = validate
        self.java_xmx = java_xmx
        self.java_xms = java_xms

        if not self.fmap_jar.is_file():
            raise FileNotFoundError(f"FMAP.jar not found: {self.fmap_jar}")
        # Validation is optional. Only require the Validate binary and
        # validation input files if validation is enabled.
        if self.validate:
            if not self.val_bin or not Path(self.val_bin).exists():
                raise FileNotFoundError(
                    f"Validate binary not found: {self.val_bin or '(not provided)'}"
                )
            if not self.validation_root.exists():
                raise FileNotFoundError(
                    f"Validation root not found: {self.validation_root}"
                )

        self.results: List[InstanceResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def discover_instances(
        self, selected_domains: Optional[Sequence[str]] = None
    ) -> List[Tuple[str, Path]]:
        domains = []
        if selected_domains:
            for domain in selected_domains:
                domain_dir = self.factored_root / domain
                if domain_dir.is_dir():
                    domains.append((domain, domain_dir))
        else:
            for domain_dir in sorted(self.factored_root.iterdir()):
                if domain_dir.is_dir():
                    domains.append((domain_dir.name, domain_dir))
        instances: List[Tuple[str, Path]] = []
        for domain_name, domain_dir in domains:
            for instance_dir in sorted(domain_dir.iterdir()):
                # Accept any directory as a valid factored instance. Some
                # datasets use names like p01/p02, others use prob*/pfile*.
                # Previously we filtered by prefixes; allow all directories
                # so users can point to arbitrary factored instance layouts.
                if instance_dir.is_dir():
                    instances.append((domain_name, instance_dir))
        return instances

    def run_batch(
        self,
        selected_domains: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Tuple[BatchSummary, Path, Path]:
        self.start_time = time.time()
        self.results = []

        instances = self.discover_instances(selected_domains)
        if limit is not None:
            instances = instances[: max(0, limit)]

        if not instances:
            raise RuntimeError("No factored FMAP instances found")

        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = f"fmap-s{self.search}-h{self.heuristic}"
        report_dir = self.output_root
        report_dir.mkdir(parents=True, exist_ok=True)
        json_report = report_dir / f"batch_eval_{model_slug}_{batch_id}.json"
        txt_report = report_dir / f"batch_eval_{model_slug}_{batch_id}.txt"

        print(f"[FMAP] Running batch over {len(instances)} instances")
        print(f"[FMAP] FMAP root: {self.fmap_root}")
        print(f"[FMAP] Factored root: {self.factored_root}")
        print(f"[FMAP] Output root: {self.output_root}")
        print(f"[FMAP] Validation: {'ON' if self.validate else 'OFF'}")
        print(
            f"[FMAP] Search={self.search} Heuristic={self.heuristic} Timeout={self.timeout}s"
        )

        for domain_name, instance_dir in instances:
            result = self.run_instance(domain_name, instance_dir)
            self.results.append(result)
            status = (
                "PASS"
                if result.validation_passed
                else ("SOLVED" if result.plan_found else result.status.upper())
            )
            print(
                f"[{domain_name}/{instance_dir.name}] {status} | plan_len={result.plan_length} | time={result.runtime_seconds:.2f}s"
            )

        self.end_time = time.time()
        summary = self.compute_summary()
        self.write_reports(json_report, txt_report, summary)
        return summary, json_report, txt_report

    def run_instance(self, domain_name: str, instance_dir: Path) -> InstanceResult:
        problem_name = instance_dir.name
        timestamp = datetime.now().isoformat()
        instance_output_dir = (self.output_root / domain_name / problem_name).resolve()
        instance_output_dir.mkdir(parents=True, exist_ok=True)
        # validation_domain_file and validation_problem_file are resolved
        # only when validation is enabled and a plan is found.

        domain_problem_pairs = self._collect_agent_files(instance_dir)
        if not domain_problem_pairs:
            return InstanceResult(
                domain=domain_name,
                problem=problem_name,
                instance_dir=str(instance_dir),
                status="error",
                plan_found=False,
                validation_passed=False,
                validation_message="No agent files found",
                validation_reason="",
                plan_length=0,
                agent_count=0,
                runtime_seconds=0.0,
                timed_out=False,
                error_message="No agent domain/problem files found",
                plan_file=None,
                stdout_file=None,
                stderr_file=None,
                timestamp=timestamp,
            )

        start = time.time()
        timed_out = False
        error_message = None
        stdout_text = ""
        stderr_text = ""
        plan_lines: List[str] = []
        validation_passed = False
        validation_message = "not validated"
        validation_reason = ""
        plan_file = instance_output_dir / "fmap.plan"
        validation_plan_file = instance_output_dir / "centralized_for_val.plan"
        validation_file = instance_output_dir / "validation.txt"
        stdout_file = instance_output_dir / "stdout.txt"
        stderr_file = instance_output_dir / "stderr.txt"
        agent_list_file = instance_output_dir / "agent-list.txt"

        try:
            self._write_agent_list(
                agent_list_file, [agent for agent, _, _ in domain_problem_pairs]
            )
            cmd = self._build_command(domain_problem_pairs, agent_list_file)
            completed = subprocess.run(
                cmd,
                cwd=str(self.fmap_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout_text = completed.stdout or ""
            stderr_text = completed.stderr or ""
            plan_lines = self.extract_plan_lines(stdout_text)
            if not plan_lines:
                # Some FMAP runs may emit the plan to stderr or via mixed logs.
                plan_lines = self.extract_plan_lines(stderr_text)
            plan_file.write_text(
                ("\n".join(plan_lines) + "\n") if plan_lines else "", encoding="utf-8"
            )
            if not plan_lines:
                error_message = "No plan lines extracted from FMAP output"
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            error_message = f"Timeout after {self.timeout}s"
            stdout_text = self._coerce_text(exc.stdout)
            stderr_text = self._coerce_text(exc.stderr)
            plan_lines = self.extract_plan_lines(
                stdout_text
            ) or self.extract_plan_lines(stderr_text)
            plan_file.write_text(
                ("\n".join(plan_lines) + "\n") if plan_lines else "", encoding="utf-8"
            )
        except Exception as exc:
            error_message = str(exc)

        stdout_file.write_text(stdout_text, encoding="utf-8")
        stderr_file.write_text(stderr_text, encoding="utf-8")

        plan_found = bool(plan_lines)
        runtime = time.time() - start
        status = (
            "success" if plan_found and not timed_out and not error_message else "error"
        )

        # Save per-agent plans for observability. Try to split extracted plan by agent name;
        # otherwise, write the merged plan into each agent file so we still keep a record.
        agents_dir = instance_output_dir / "agents"
        agents_dir.mkdir(exist_ok=True)
        # Initialize per-agent lists
        per_agent_lines: Dict[str, List[str]] = {
            agent: [] for agent, _, _ in domain_problem_pairs
        }
        if plan_lines:
            for pl in plan_lines:
                assigned = False
                for agent in per_agent_lines:
                    # Heuristic: if the agent name appears as a separate token in the line, assign it
                    tokens = re.findall(r"[A-Za-z0-9_\-]+", pl)
                    if agent in tokens:
                        per_agent_lines[agent].append(pl)
                        assigned = True
                        break
                if not assigned:
                    # If no agent matched, leave it unassigned; we'll duplicate merged plan below.
                    continue
        # Write per-agent plan files; if empty, copy merged plan as placeholder
        for agent, lines in per_agent_lines.items():
            agent_file = agents_dir / f"{agent}.plan"
            if lines:
                agent_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                # fallback: copy merged plan (if any) so there is always a per-agent plan file
                if plan_found:
                    agent_file.write_text(
                        "\n".join(plan_lines) + "\n", encoding="utf-8"
                    )
                else:
                    agent_file.write_text("", encoding="utf-8")

        if plan_found:
            if self.validate:
                # Resolve validation inputs lazily to avoid requiring validation
                # files when validation is disabled.
                validation_domain_file, validation_problem_file = (
                    self._resolve_validation_inputs(domain_name, problem_name)
                )
                sorted_plan_lines = self._sort_plan_lines(plan_lines)
                validation_plan_file.write_text(
                    ("\n".join(sorted_plan_lines) + "\n") if sorted_plan_lines else "",
                    encoding="utf-8",
                )
                evaluator = PlanEvaluator(
                    validate_bin=str(self.val_bin),
                    timeout=120,
                    extra_flags=["-v"],
                    print_on_pass=False,
                    print_on_fail=False,
                )
                validation_passed, _, val_out, val_err = evaluator.evaluate_detailed(
                    str(validation_domain_file),
                    str(validation_problem_file),
                    str(validation_plan_file),
                )
                validation_message = "VALID" if validation_passed else "INVALID"
                validation_reason = (
                    extract_failure_reason(val_out, val_err)
                    if not validation_passed
                    else ""
                )
                validation_file.write_text(
                    "\n".join(
                        [
                            f"Validation domain: {validation_domain_file}",
                            f"Validation problem: {validation_problem_file}",
                            f"FMAP plan file: {plan_file}",
                            f"Validation plan file: {validation_plan_file}",
                            "",
                            (val_out or "").strip(),
                            (val_err or "").strip(),
                        ]
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                print(f"[{domain_name}/{problem_name}] Plan stored at: {plan_file}")
                print(
                    f"[{domain_name}/{problem_name}] Validation stored at: {validation_file}"
                )
            else:
                validation_passed = False
                validation_message = "SKIPPED"
                validation_reason = ""
                validation_file.write_text("Validation skipped\n", encoding="utf-8")
            print("; Solution plan - CoDMAP Distributed format")
            print("; -----------------------------------------")
            for line in plan_lines:
                print(line)
        else:
            validation_passed = False
            validation_message = "ERROR" if error_message else "not_planned"
            validation_reason = ""
            validation_file.write_text(
                error_message or "No plan found", encoding="utf-8"
            )

        return InstanceResult(
            domain=domain_name,
            problem=problem_name,
            instance_dir=str(instance_dir),
            status=status,
            plan_found=plan_found,
            validation_passed=validation_passed,
            validation_message=validation_message,
            validation_reason=validation_reason,
            plan_length=len(plan_lines),
            agent_count=len(domain_problem_pairs),
            runtime_seconds=runtime,
            timed_out=timed_out,
            error_message=error_message,
            plan_file=str(plan_file) if plan_found else None,
            stdout_file=str(stdout_file),
            stderr_file=str(stderr_file),
            timestamp=timestamp,
        )

    def compute_summary(self) -> BatchSummary:
        summary = BatchSummary()
        summary.total_instances = len(self.results)
        summary.solved_instances = sum(1 for r in self.results if r.plan_found)
        summary.validated_instances = sum(
            1 for r in self.results if r.validation_passed
        )
        summary.timeout_instances = sum(1 for r in self.results if r.timed_out)
        summary.failed_instances = summary.total_instances - summary.solved_instances
        summary.total_plan_length = sum(
            r.plan_length for r in self.results if r.plan_found
        )
        summary.total_runtime_seconds = sum(r.runtime_seconds for r in self.results)

        if summary.total_instances > 0:
            summary.avg_runtime_per_instance = (
                summary.total_runtime_seconds / summary.total_instances
            )
            summary.coverage_percent = (
                summary.solved_instances / summary.total_instances
            ) * 100.0
            summary.validation_coverage_percent = (
                summary.validated_instances / summary.total_instances
            ) * 100.0
        if summary.solved_instances > 0:
            summary.avg_plan_length = (
                summary.total_plan_length / summary.solved_instances
            )
            summary.avg_runtime_per_solved_instance = (
                summary.total_runtime_seconds / summary.solved_instances
            )
            summary.validation_pass_rate_percent = (
                summary.validated_instances / summary.solved_instances
            ) * 100.0
        return summary

    def write_reports(
        self, json_report: Path, txt_report: Path, summary: BatchSummary
    ) -> None:
        report = {
            "metadata": {
                "tool": "FMAP batch evaluator",
                "fmap_root": str(self.fmap_root),
                "factored_root": str(self.factored_root),
                "search": self.search,
                "heuristic": self.heuristic,
                "timeout_seconds": self.timeout,
                "validate": self.validate,
                "generated_at": datetime.now().isoformat(),
            },
            "summary": asdict(summary),
            "results": [asdict(r) for r in self.results],
        }
        json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        txt_report.write_text(self._render_text_report(report), encoding="utf-8")

    def _render_text_report(self, report: Dict) -> str:
        summary = report["summary"]
        lines = []
        lines.append("=" * 80)
        lines.append("FMAP BATCH EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        meta = report["metadata"]
        lines.append(f"FMAP root:      {meta['fmap_root']}")
        lines.append(f"Factored root:   {meta['factored_root']}")
        lines.append(f"Search / H:     {meta['search']} / {meta['heuristic']}")
        lines.append(f"Timeout:        {meta['timeout_seconds']}s")
        lines.append(f"Validation:     {'on' if meta['validate'] else 'off'}")
        lines.append(f"Generated:      {meta['generated_at']}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Total Instances:            {summary['total_instances']}")
        lines.append(f"Solved Instances:           {summary['solved_instances']}")
        lines.append(f"Coverage:                   {summary['coverage_percent']:.2f}%")
        lines.append(f"Validated Instances:        {summary['validated_instances']}")
        lines.append(
            f"Validation Coverage:        {summary['validation_coverage_percent']:.2f}%"
        )
        lines.append(
            f"Validation Pass Rate:       {summary['validation_pass_rate_percent']:.2f}%"
        )
        lines.append(f"Timeouts:                   {summary['timeout_instances']}")
        lines.append(f"Total Plan Length:          {summary['total_plan_length']}")
        lines.append(f"Avg Plan Length:            {summary['avg_plan_length']:.2f}")
        lines.append(
            f"Total Runtime:              {summary['total_runtime_seconds']:.2f}s"
        )
        lines.append(
            f"Avg Runtime / Instance:     {summary['avg_runtime_per_instance']:.2f}s"
        )
        lines.append(
            f"Avg Runtime / Solved:       {summary['avg_runtime_per_solved_instance']:.2f}s"
        )
        lines.append("")
        lines.append("-" * 80)
        lines.append("PER-INSTANCE RESULTS")
        lines.append("-" * 80)
        lines.append("")
        lines.append(
            f"{'Domain':<14} {'Problem':<24} {'Status':<10} {'Len':>6} {'Valid':<8} {'Time':>8}"
        )
        lines.append("-" * 80)
        for result in report["results"]:
            status = "✓" if result["plan_found"] else "✗"
            valid = "VALID" if result["validation_passed"] else result["validation_message"]
            lines.append(
                f"{result['domain']:<14} {result['problem']:<24} {status:<10} {result['plan_length']:>6} {valid:<8} {result['runtime_seconds']:>7.2f}s"
            )
        failed_validations = [
            result
            for result in report["results"]
            if result["plan_found"] and not result["validation_passed"]
        ]
        if failed_validations:
            lines.append("")
            lines.append("VALIDATION FAILURES")
            lines.append("-" * 80)
            lines.append("")
            for result in failed_validations:
                lines.append(f"{result['domain']}/{result['problem']}")
                reason = result.get("validation_reason", "")
                if reason:
                    lines.append("  Reason:")
                    for line in reason.splitlines():
                        lines.append(f"    {line}")
                lines.append("")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _collect_agent_files(instance_dir: Path) -> List[Tuple[str, Path, Path]]:
        domain_files = sorted(instance_dir.glob("domain-*.pddl"))
        pairs = []
        for domain_file in domain_files:
            agent = domain_file.stem.split("-", 1)[1]
            problem_file = instance_dir / f"problem-{agent}.pddl"
            if problem_file.is_file():
                pairs.append((agent, domain_file, problem_file))
        return pairs

    def _write_agent_list(self, agent_list_file: Path, agents: Sequence[str]) -> None:
        with agent_list_file.open("w", encoding="utf-8") as f:
            for agent in agents:
                f.write(f"{agent} {self.host}\n")

    @staticmethod
    def _coerce_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _sort_plan_lines(plan_lines: Sequence[str]) -> List[str]:
        numbered_actions: List[Tuple[int, str]] = []
        fallback_actions: List[str] = []
        for line in plan_lines:
            match = STEP_LINE_RE.match(line)
            if match:
                numbered_actions.append((int(match.group("step")), match.group("action")))
            else:
                fallback_actions.append(line)
        if numbered_actions:
            numbered_actions.sort(key=lambda item: item[0])
            return [action for _, action in numbered_actions]
        return list(fallback_actions)

    def _build_command(
        self,
        domain_problem_pairs: Sequence[Tuple[str, Path, Path]],
        agent_list_file: Path,
    ) -> List[str]:
        cmd = [
            "java",
            f"-Xms{self.java_xms}",
            f"-Xmx{self.java_xmx}",
            "-jar",
            str(self.fmap_jar),
        ]
        for agent, domain_file, problem_file in domain_problem_pairs:
            cmd.extend([agent, str(domain_file), str(problem_file)])
        cmd.append(str(agent_list_file))
        cmd.extend(["-s", str(self.search), "-h", str(self.heuristic)])
        return cmd

    def _resolve_validation_inputs(
        self, domain_name: str, problem_name: str
    ) -> Tuple[Path, Path]:
        validation_domain_dir = self.validation_root / domain_name
        validation_domain_file = validation_domain_dir / "domain.pddl"
        validation_problem_file = validation_domain_dir / f"{problem_name}.pddl"

        if validation_domain_file.is_file() and validation_problem_file.is_file():
            return validation_domain_file, validation_problem_file

        alt_problem_candidates = [
            validation_domain_dir / f"{problem_name.lower()}.pddl",
            validation_domain_dir / f"{problem_name.upper()}.pddl",
        ]
        for candidate in alt_problem_candidates:
            if validation_domain_file.is_file() and candidate.is_file():
                return validation_domain_file, candidate

        raise FileNotFoundError(
            f"Validation files not found for {domain_name}/{problem_name}: "
            f"expected {validation_domain_file} and {validation_problem_file}"
        )

    @staticmethod
    def extract_plan_lines(output: str) -> List[str]:
        """Extract PDDL plan lines from FMAP output."""
        if not output:
            return []

        lines = output.splitlines()
        plan_lines: List[str] = []
        in_plan = False
        markers = (
            "solution found",
            "solution plan",
            "plan found",
            "final plan",
            "plan length",
        )

        for line in lines:
            lower = line.lower()
            if any(marker in lower for marker in markers):
                in_plan = True
                continue

            stripped = line.strip()
            if not stripped:
                continue

            if in_plan:
                if PLAN_LINE_RE.match(stripped):
                    plan_lines.append(stripped)

        if plan_lines:
            return plan_lines

        # Fallback: collect any CoDMAP-like action lines from the whole output.
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if PLAN_LINE_RE.match(stripped):
                plan_lines.append(stripped)

        return plan_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate FMAP on factored CoDMAP instances. All plans are automatically validated with VAL."
    )
    parser.add_argument(
        "--fmap-root",
        type=Path,
        default=DEFAULT_FMAP_ROOT,
        help="Path to FMAP repository root containing FMAP.jar",
    )
    parser.add_argument(
        "--factored-root",
        type=Path,
        default=DEFAULT_FACTORED_ROOT,
        help="Path to codmap-2015/factored",
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=DEFAULT_VALIDATION_ROOT,
        help="Path to centralized VAL-compatible benchmark files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where reports and per-instance artifacts are written",
    )
    parser.add_argument(
        "--val-bin",
        type=Path,
        default=DEFAULT_VAL_BIN,
        help="Path to VAL Validate binary",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Optional subset of domains to evaluate (e.g. blocksworld logistics00)",
    )
    parser.add_argument(
        "--search",
        type=int,
        default=DEFAULT_SEARCH,
        help="FMAP search method id (-s). Default: 0",
    )
    parser.add_argument(
        "--heuristic",
        type=int,
        default=DEFAULT_HEURISTIC,
        help="FMAP heuristic id (-h). Default: 2",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-instance timeout in seconds",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="IP address to write into the temporary FMAP agent-list file",
    )
    parser.add_argument(
        "--java-xmx",
        default=DEFAULT_JAVA_XMX,
        help="Maximum JVM heap for FMAP, e.g. 8g, 16g, 24g (default: 16g)",
    )
    parser.add_argument(
        "--java-xms",
        default=DEFAULT_JAVA_XMS,
        help="Initial JVM heap for FMAP, e.g. 8g, 16g, 24g (default: 16g)",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Disable VAL validation (default: enabled)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of instances to evaluate (useful for smoke tests)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluator = FMAPBatchEvaluator(
        fmap_root=args.fmap_root,
        factored_root=args.factored_root,
        output_root=args.output_root,
        val_bin=args.val_bin,
        validation_root=args.validation_root,
        search=args.search,
        heuristic=args.heuristic,
        timeout=args.timeout,
        host=args.host,
        java_xmx=args.java_xmx,
        java_xms=args.java_xms,
        validate=args.validate,
    )

    summary, json_report, txt_report = evaluator.run_batch(args.domains, args.limit)

    print("")
    print("=" * 80)
    print("FMAP BATCH COMPLETE - ALL INSTANCES VALIDATED")
    print("=" * 80)
    print(f"Solved instances: {summary.solved_instances} / {summary.total_instances}")
    print(
        f"Plans validated: {summary.validated_instances} / {summary.solved_instances}"
    )
    print(f"Valid plans: {summary.validated_instances}")
    print(f"Invalid plans: {summary.failed_instances}")
    print(f"Validation pass rate: {summary.validation_pass_rate_percent:.2f}%")
    print(f"")
    print(f"Plan statistics:")
    print(f"  Total plan length: {summary.total_plan_length}")
    print(f"  Avg plan length: {summary.avg_plan_length:.2f}")
    print(f"")
    print(f"Performance:")
    print(f"  Total runtime: {summary.total_runtime_seconds:.2f}s")
    print(f"  Avg per instance: {summary.avg_runtime_per_instance:.2f}s")
    print(f"  Avg per solved: {summary.avg_runtime_per_solved_instance:.2f}s")
    print(f"")
    print(f"Reports:")
    print(f"  JSON: {json_report}")
    print(f"  Text: {txt_report}")
    print(f"  Plans: {evaluator.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
