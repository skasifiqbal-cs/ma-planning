"""Comprehensive batch evaluation for planning experiments."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class BatchEvaluator:
    """
    Tracks and evaluates batch planning experiments.

    Creates model-specific evaluation reports with:
    - Per-problem results
    - Aggregate statistics
    - Success/failure analysis
    - Performance metrics
    """

    def __init__(self, domain_name: str, model_name: str, output_dir: str = "results"):
        """
        Initialize batch evaluator.

        Args:
            domain_name: Name of the planning domain (e.g., "driverlog")
            model_name: LLM model name (e.g., "openai/gpt-oss-120b")
            output_dir: Base directory for results
        """
        self.domain_name = domain_name
        self.model_name = model_name
        self.output_dir = Path(output_dir)

        # Clean model name for filename (replace / with -)
        self.model_slug = model_name.replace("/", "-").replace(":", "-")

        # Results storage
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Statistics
        self.stats = {
            "total_problems": 0,
            "completed": 0,
            "errors": 0,
            "passed": 0,
            "failed": 0,
            "not_validated": 0,
            "total_actions": 0,
            "avg_actions": 0.0,
            "total_time": 0.0,
            "avg_time_per_problem": 0.0,
            "pass_rate_percent": 0.0,
            "total_retries": 0,
            "avg_retries": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
        }

    def start_batch(self):
        """Mark start of batch evaluation."""
        self.start_time = time.time()
        self.results = []

    def add_result(
        self,
        problem_name: str,
        status: str,
        num_actions: int,
        validation_passed: bool,
        validation_message: str,
        execution_time: float,
        validation_reason: str = "",
        error_message: Optional[str] = None,
        num_attempts: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ):
        """
        Record result for a single problem.

        Args:
            problem_name: Problem identifier
            status: "success" or "error"
            num_actions: Number of actions in generated plan
            validation_passed: Whether plan passed validation
            validation_message: Validation status message
            execution_time: Time taken for this problem (seconds)
            error_message: Error message if status is "error"
            num_attempts: Number of validation attempts (llm-modulo only)
        """
        result = {
            "problem": problem_name,
            "status": status,
            "num_actions": num_actions,
            "validation_passed": validation_passed,
            "validation_message": validation_message,
            "validation_reason": validation_reason,
            "execution_time": execution_time,
            "error_message": error_message,
            "num_attempts": num_attempts,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "timestamp": datetime.now().isoformat(),
        }

        self.results.append(result)

        # Update running statistics
        self.stats["total_problems"] += 1
        if status == "success":
            self.stats["completed"] += 1
            self.stats["total_actions"] += num_actions
        else:
            self.stats["errors"] += 1

        if validation_passed:
            self.stats["passed"] += 1
        elif validation_message == "FAILED":
            self.stats["failed"] += 1
        elif validation_message in ("not validated", "ERROR"):
            self.stats["not_validated"] += 1

        self.stats["total_retries"] += num_attempts
        self.stats["total_prompt_tokens"] += prompt_tokens
        self.stats["total_completion_tokens"] += completion_tokens
        self.stats["total_tokens"] += total_tokens

    def end_batch(self):
        """Mark end of batch and compute final statistics."""
        self.end_time = time.time()

        # Compute aggregate statistics
        if self.stats["total_problems"] > 0:
            self.stats["total_time"] = self.end_time - self.start_time
            self.stats["avg_time_per_problem"] = (
                self.stats["total_time"] / self.stats["total_problems"]
            )

        if self.stats["completed"] > 0:
            self.stats["avg_actions"] = (
                self.stats["total_actions"] / self.stats["completed"]
            )

        validated_count = self.stats["passed"] + self.stats["failed"]
        if validated_count > 0:
            self.stats["pass_rate_percent"] = round(
                (self.stats["passed"] / validated_count) * 100, 2
            )

        if self.stats["total_problems"] > 0:
            self.stats["avg_retries"] = round(
                self.stats["total_retries"] / self.stats["total_problems"], 2
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return self.stats.copy()

    def generate_report(self) -> Path:
        """
        Generate comprehensive evaluation report.

        Creates two files:
        1. JSON file with detailed results
        2. Text file with human-readable summary

        Returns:
            Path to the JSON report file
        """
        # Ensure output directory exists
        domain_dir = self.output_dir / self.domain_name
        domain_dir.mkdir(parents=True, exist_ok=True)

        json_file = domain_dir / "eval.json"
        txt_file = domain_dir / "eval.txt"

        report_data = {
            "metadata": {
                "domain": self.domain_name,
                "model": self.model_name,
                "generated_at": datetime.now().isoformat(),
            },
            "summary": self.stats,
            "results": self.results,
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        self._write_text_report(txt_file, report_data)

        return json_file

    def _write_text_report(self, filepath: Path, report_data: Dict):
        """Generate human-readable text report."""
        with open(filepath, "w", encoding="utf-8") as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("BATCH EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Metadata
            meta = report_data["metadata"]
            f.write(f"Domain:      {meta['domain']}\n")
            f.write(f"Model:       {meta['model']}\n")
            f.write(f"Generated:   {meta['generated_at']}\n")
            f.write("\n")

            # Summary Statistics
            f.write("-" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            stats = report_data["summary"]
            f.write(f"Total Problems:           {stats['total_problems']}\n")
            f.write(f"  Completed:              {stats['completed']}\n")
            f.write(f"  Errors:                 {stats['errors']}\n")
            f.write(f"\nValidation Results:\n")
            f.write(f"  Passed:                 {stats['passed']}\n")
            f.write(f"  Failed:                 {stats['failed']}\n")
            f.write(f"  Not Validated:          {stats['not_validated']}\n")
            f.write(f"  Pass Rate:              {stats['pass_rate_percent']}%\n")
            f.write(f"\nPlan Quality:\n")
            f.write(f"  Total Actions:          {stats['total_actions']}\n")
            f.write(f"  Avg Actions/Problem:    {stats['avg_actions']:.2f}\n")
            if stats.get("total_retries", 0) > 0:
                f.write(f"  Avg Retries/Problem:    {stats['avg_retries']:.2f}\n")
            f.write(f"\nPerformance:\n")
            f.write(f"  Total Time:             {stats['total_time']:.2f}s\n")
            f.write(f"  Avg Time/Problem:       {stats['avg_time_per_problem']:.2f}s\n")
            f.write("\n")

            # Per-Problem Results
            f.write("-" * 80 + "\n")
            f.write("PER-PROBLEM RESULTS\n")
            f.write("-" * 80 + "\n\n")

            # Header row
            f.write(
                f"{'Problem':<20} {'Status':<10} {'Actions':>8} {'Valid':>10} {'Time':>8}\n"
            )
            f.write("-" * 80 + "\n")

            for result in report_data["results"]:
                problem = result["problem"][:19]  # Truncate if too long
                status = "✓" if result["status"] == "success" else "✗ ERROR"
                actions = result["num_actions"]
                valid = (
                    "✓ PASS"
                    if result["validation_passed"]
                    else result["validation_message"]
                )
                exec_time = result["execution_time"]

                f.write(
                    f"{problem:<20} {status:<10} {actions:>8} {valid:>10} {exec_time:>7.2f}s\n"
                )

            # Failures section (if any)
            failures = [r for r in report_data["results"] if r["status"] == "error"]
            if failures:
                f.write("\n")
                f.write("-" * 80 + "\n")
                f.write("ERRORS\n")
                f.write("-" * 80 + "\n\n")

                for fail in failures:
                    f.write(f"Problem: {fail['problem']}\n")
                    f.write(f"  Error: {fail['error_message']}\n")
                    f.write("\n")

            # Validation failures (if any)
            val_failures = [
                r for r in report_data["results"] if r["validation_message"] == "FAILED"
            ]
            if val_failures:
                f.write("-" * 80 + "\n")
                f.write("VALIDATION FAILURES\n")
                f.write("-" * 80 + "\n\n")

                for fail in val_failures:
                    f.write(f"Problem: {fail['problem']}\n")
                    f.write(f"  Actions: {fail['num_actions']}\n")
                    f.write(f"  Time: {fail['execution_time']:.2f}s\n")
                    reason = fail.get("validation_reason", "")
                    if reason:
                        f.write("  Reason:\n")
                        for line in reason.splitlines():
                            f.write(f"    {line}\n")
                    f.write("\n")

            # Footer
            f.write("=" * 80 + "\n")
            f.write(f"Report saved: {filepath}\n")
            f.write("=" * 80 + "\n")
