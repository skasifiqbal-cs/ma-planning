import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional


class PlanEvaluator:
    """
    Wrapper around external Validate binary.

    Core behavior (unchanged from your original):
      - evaluate(): returns True only if exit code == 0 (goal satisfied).

    Optional extras:
      - evaluate_detailed(): returns (ok, rc, stdout, stderr).
      - evaluate_prefix(): validate an in-memory list of actions (prefix testing).
      - Configurable output printing, truncation, and optional Validate flags.
    """

    def __init__(
        self,
        validate_bin: str,
        timeout: int = 120,
        print_on_fail: bool = True,
        print_on_pass: bool = False,
        max_output_chars: int = 2000,
        extra_flags: Optional[List[str]] = None,  # e.g. ["-S"] or ["-v"]
    ):
        self.validate_bin = validate_bin
        self.timeout = timeout
        self.print_on_fail = print_on_fail
        self.print_on_pass = print_on_pass
        self.max_output_chars = max_output_chars
        self.extra_flags = extra_flags or []

    # -------- Original-style simple evaluation --------
    def evaluate(self, domain_file: str, problem_file: str, plan_file: str) -> bool:
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

    # -------- Detailed evaluation (structured) --------
    def evaluate_detailed(
        self, domain_file: str, problem_file: str, plan_file: str
    ) -> Tuple[bool, int, str, str]:
        return self._run(domain_file, problem_file, plan_file)

    # -------- Prefix / in-memory plan validation --------
    def evaluate_prefix(
        self,
        domain_file: str,
        problem_file: str,
        action_lines: List[str],
        keep_temp: bool = False,
    ) -> Tuple[bool, int, str, str, str]:
        """
        Validate a prefix (list of action strings).
        Returns (ok, rc, stdout, stderr, temp_plan_path).
        If keep_temp is False, the temp file is deleted afterwards.
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

    # -------- Internal runner --------
    def _run(
        self, domain_file: str, problem_file: str, plan_file: str
    ) -> Tuple[bool, int, str, str]:
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
        ok = rc == 0  # STRICT: only rc==0 is success (goal satisfied)

        return ok, rc, out, err
