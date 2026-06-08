"""Plan evaluator using VAL (Validate) tool."""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional


class PlanEvaluator:
    """Wrapper around external Validate binary for plan validation."""

    def __init__(
        self,
        validate_bin: str,
        timeout: int = 120,
        print_on_fail: bool = True,
        print_on_pass: bool = False,
        max_output_chars: int = 2000,
        extra_flags: Optional[List[str]] = None,
    ):
        """
        Initialize plan evaluator.

        Args:
            validate_bin: Path to Validate binary
            timeout: Validation timeout in seconds
            print_on_fail: Print output on validation failure
            print_on_pass: Print output on validation success
            max_output_chars: Maximum output characters to print
            extra_flags: Additional flags for Validate (e.g., ["-v"])
        """
        self.validate_bin = validate_bin
        self.timeout = timeout
        self.print_on_fail = print_on_fail
        self.print_on_pass = print_on_pass
        self.max_output_chars = max_output_chars
        self.extra_flags = extra_flags or []

    def evaluate(self, domain_file: str, problem_file: str, plan_file: str) -> bool:
        """
        Evaluate a plan file.

        Args:
            domain_file: Path to domain PDDL file
            problem_file: Path to problem PDDL file
            plan_file: Path to plan file

        Returns:
            True if plan is valid, False otherwise
        """
        ok, rc, out, err = self._run(domain_file, problem_file, plan_file)

        if (ok and self.print_on_pass) or (not ok and self.print_on_fail):
            status = "PASS" if ok else f"FAIL (rc={rc})"
            print(f"[EVAL {status}] {plan_file}")
            if out:
                print("[EVAL STDOUT]")
                print(out[: self.max_output_chars])
            if err:
                print("[EVAL STDERR]")
                print(err[: self.max_output_chars])

        return ok

    def evaluate_detailed(
        self, domain_file: str, problem_file: str, plan_file: str
    ) -> Tuple[bool, int, str, str]:
        """
        Evaluate a plan and return detailed results.

        Returns:
            Tuple of (success, return_code, stdout, stderr)
        """
        return self._run(domain_file, problem_file, plan_file)

    def evaluate_prefix(
        self,
        domain_file: str,
        problem_file: str,
        action_lines: List[str],
        keep_temp: bool = False,
    ) -> Tuple[bool, int, str, str, str]:
        """
        Validate a plan prefix (partial plan).

        Args:
            domain_file: Path to domain PDDL file
            problem_file: Path to problem PDDL file
            action_lines: List of action strings
            keep_temp: Keep temporary plan file after validation

        Returns:
            Tuple of (success, return_code, stdout, stderr, temp_plan_path)
        """
        fd = tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".plan", encoding="utf-8"
        )
        try:
            with fd:
                for a in action_lines:
                    fd.write(a.strip() + "\n")
            temp_path = fd.name
            ok, rc, out, err = self._run(domain_file, problem_file, temp_path)
            return ok, rc, out, err, temp_path
        finally:
            if not keep_temp:
                try:
                    Path(fd.name).unlink(missing_ok=True)
                except Exception:
                    pass

    def _run(
        self, domain_file: str, problem_file: str, plan_file: str
    ) -> Tuple[bool, int, str, str]:
        """
        Run Validate binary.

        Returns:
            Tuple of (success, return_code, stdout, stderr)
        """
        d, p, plan = Path(domain_file), Path(problem_file), Path(plan_file)

        for x, label in [(d, "Domain"), (p, "Problem"), (plan, "Plan")]:
            if not x.is_file():
                msg = f"[EVAL ERROR] {label} file missing: {x}"
                print(msg)
                return False, -1, msg, ""

        cmd = [self.validate_bin] + self.extra_flags + [str(d), str(p), str(plan)]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutError:
            msg = f"[EVAL ERROR] Validation timed out for {plan}"
            return False, -1, msg, ""
        except Exception as e:
            msg = f"[EVAL ERROR] Exception during validation: {e}"
            return False, -1, msg, ""

        rc = proc.returncode
        out = proc.stdout or ""
        err = proc.stderr or ""

        # Validate can return 0 even for failed plans, so check output text
        # Look for success/failure indicators in stdout
        ok = rc == 0
        if ok:
            # Check for failure indicators in output
            out_lower = out.lower()
            if any(
                indicator in out_lower
                for indicator in [
                    "failed plans:",
                    "bad plan description",
                    "plan failed",
                    "goal not satisfied",
                    "plan invalid",
                ]
            ):
                ok = False
            # Also check for explicit success indicators
            elif "plan executed successfully" in out_lower and "goal" in out_lower:
                ok = True

        return ok, rc, out, err


def extract_failure_reason(stdout: str, stderr: str, max_lines: int = 4) -> str:
    """Return the trailing non-empty VAL output lines as a compact reason."""
    lines = []
    for chunk in (stdout, stderr):
        if not chunk:
            continue
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped:
                lines.append(stripped)

    if not lines:
        return ""

    return "\n".join(lines[-max_lines:])
