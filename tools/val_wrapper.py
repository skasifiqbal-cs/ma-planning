"""Wrapper for VAL (Validate) tool."""

import subprocess
from pathlib import Path
from typing import Optional, Tuple


class VALWrapper:
    """Wrapper for the VAL (Validate) PDDL validation tool."""

    def __init__(self, val_path: Optional[str] = None):
        """
        Initialize VAL wrapper.

        Args:
            val_path: Path to Validate binary. If None, searches in common locations.
        """
        self.val_path = self._find_validate(val_path)
        if not self.val_path:
            raise FileNotFoundError("Validate binary not found")

    def _find_validate(self, val_path: Optional[str]) -> Optional[str]:
        """Find Validate binary."""
        if val_path and Path(val_path).is_file():
            return str(Path(val_path).resolve())

        # Check VAL directory in project
        project_val = Path(__file__).parent.parent / "VAL" / "Validate"
        if project_val.is_file():
            return str(project_val.resolve())

        # Check PATH
        import shutil

        cmd = shutil.which("Validate")
        if cmd:
            return cmd

        return None

    def validate(
        self,
        domain_file: str,
        problem_file: str,
        plan_file: str,
        verbose: bool = False,
        timeout: int = 120,
    ) -> Tuple[bool, str, str]:
        """
        Validate a PDDL plan.

        Args:
            domain_file: Path to domain PDDL file
            problem_file: Path to problem PDDL file
            plan_file: Path to plan file
            verbose: Enable verbose output
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        cmd = [self.val_path]
        if verbose:
            cmd.append("-v")
        cmd.extend([domain_file, problem_file, plan_file])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            success = result.returncode == 0
            return success, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Validation timed out"
        except Exception as e:
            return False, "", f"Error running Validate: {e}"

    def parse(
        self,
        domain_file: str,
        problem_file: str,
        timeout: int = 30,
    ) -> Tuple[bool, str, str]:
        """
        Parse PDDL files without validating a plan.

        Args:
            domain_file: Path to domain PDDL file
            problem_file: Path to problem PDDL file
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        # VAL's Parser binary if available
        parser_path = Path(self.val_path).parent / "Parser"
        if not parser_path.is_file():
            return False, "", "Parser binary not found"

        cmd = [str(parser_path), domain_file, problem_file]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            success = result.returncode == 0
            return success, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Parsing timed out"
        except Exception as e:
            return False, "", f"Error running Parser: {e}"
