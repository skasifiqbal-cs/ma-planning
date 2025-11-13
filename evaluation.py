import subprocess
from pathlib import Path
from typing import Optional


class PlanEvaluator:
    """
    Wraps external Validate binary:
    Usage: Validate <domain.pddl> <problem.pddl> <plan_file>
    Returns True if validation succeeds (exit code 0), False otherwise.
    """

    def __init__(self, validate_bin: str, timeout: int = 120):
        self.validate_bin = validate_bin
        self.timeout = timeout

    def evaluate(self, domain_file: str, problem_file: str, plan_file: str) -> bool:
        domain_path = Path(domain_file)
        problem_path = Path(problem_file)
        plan_path = Path(plan_file)

        if not domain_path.is_file():
            print(f"[EVAL ERROR] Domain file missing: {domain_path}")
            return False
        if not problem_path.is_file():
            print(f"[EVAL ERROR] Problem file missing: {problem_path}")
            return False
        if not plan_path.is_file():
            print(f"[EVAL ERROR] Plan file missing: {plan_path}")
            return False

        cmd = [
            self.validate_bin,
            str(domain_path),
            str(problem_path),
            str(plan_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutError:
            print(f"[EVAL ERROR] Validation timed out for {plan_file}")
            return False
        except Exception as e:
            print(f"[EVAL ERROR] Exception during validation: {e}")
            return False

        if proc.returncode == 0:
            return True

        print(f"[EVAL FAIL] {plan_file} -> returncode {proc.returncode}")
        if proc.stdout:
            print(f"[EVAL STDOUT]\n{proc.stdout[:500]}")
        if proc.stderr:
            print(f"[EVAL STDERR]\n{proc.stderr[:500]}")
        return False
