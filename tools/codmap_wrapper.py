"""Wrapper for CoDMAP planner."""

import subprocess
from pathlib import Path
from typing import Optional, Tuple, List


class CoDMAPWrapper:
    """Wrapper for the CoDMAP multi-agent planner."""

    def __init__(self, codmap_path: Optional[str] = None):
        """
        Initialize CoDMAP wrapper.

        Args:
            codmap_path: Path to codmap binary. If None, searches in common locations.
        """
        self.codmap_path = self._find_codmap(codmap_path)
        if not self.codmap_path:
            raise FileNotFoundError("CoDMAP binary not found")

    def _find_codmap(self, codmap_path: Optional[str]) -> Optional[str]:
        """Find CoDMAP binary."""
        if codmap_path and Path(codmap_path).is_file():
            return str(Path(codmap_path).resolve())

        # Check codmap-2015 directory in project
        project_codmap = Path(__file__).parent.parent / "codmap-2015" / "codmap"
        if project_codmap.is_file():
            return str(project_codmap.resolve())

        # Check PATH
        import shutil

        cmd = shutil.which("codmap")
        if cmd:
            return cmd

        return None

    def plan(
        self,
        domain_files: List[str],
        problem_files: List[str],
        output_file: str,
        timeout: int = 300,
        additional_args: Optional[List[str]] = None,
    ) -> Tuple[bool, str, str]:
        """
        Run CoDMAP planner.

        Args:
            domain_files: List of domain PDDL files (one per agent)
            problem_files: List of problem PDDL files (one per agent)
            output_file: Path to save the output plan
            timeout: Timeout in seconds
            additional_args: Additional command-line arguments

        Returns:
            Tuple of (success, stdout, stderr)
        """
        cmd = [self.codmap_path]

        # Add domain files
        for df in domain_files:
            cmd.extend(["-d", df])

        # Add problem files
        for pf in problem_files:
            cmd.extend(["-p", pf])

        # Output file
        cmd.extend(["-o", output_file])

        # Additional arguments
        if additional_args:
            cmd.extend(additional_args)

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
            return False, "", "Planning timed out"
        except Exception as e:
            return False, "", f"Error running CoDMAP: {e}"


def run_codmap_simple(
    domain_dir: str,
    output_file: str,
    codmap_path: Optional[str] = None,
    timeout: int = 300,
) -> Tuple[bool, str, str]:
    """
    Simple wrapper to run CoDMAP on a directory with auto-discovery of files.

    Args:
        domain_dir: Directory containing Domain*.pddl and Problem*.pddl files
        output_file: Path to save the output plan
        codmap_path: Optional path to CoDMAP binary
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, stdout, stderr)
    """
    wrapper = CoDMAPWrapper(codmap_path)

    # Discover domain and problem files
    domain_dir_path = Path(domain_dir)
    domain_files = sorted(domain_dir_path.glob("Domain*.pddl"))
    problem_files = sorted(domain_dir_path.glob("Problem*.pddl"))

    if not domain_files:
        return False, "", f"No Domain*.pddl files found in {domain_dir}"
    if not problem_files:
        return False, "", f"No Problem*.pddl files found in {domain_dir}"

    return wrapper.plan(
        [str(f) for f in domain_files],
        [str(f) for f in problem_files],
        output_file,
        timeout=timeout,
    )
