"""MA-PDDL to centralized PDDL converter."""

from pathlib import Path
import subprocess


class MAPDDLConverter:
    """Converts MA-PDDL files to centralized PDDL using external converter script."""

    def __init__(self, converter_script: str, python_cmd: str, centralized_root: str):
        """
        Initialize converter.

        Args:
            converter_script: Path to ma-to-pddl.py converter script
            python_cmd: Python command to run the converter
            centralized_root: Root directory for centralized output files
        """
        self.converter_script = Path(converter_script)
        self.python_cmd = python_cmd
        self.centralized_root = Path(centralized_root)

        if not self.converter_script.is_file():
            raise FileNotFoundError(f"Converter script not found: {converter_script}")

    def convert(
        self, domain_dir: str, domain_file: str, problem_file: str
    ) -> tuple[str, str]:
        """
        Convert MA-PDDL to centralized PDDL.

        Args:
            domain_dir: Directory containing MA-PDDL files
            domain_file: Domain file name (with or without .pddl extension)
            problem_file: Problem file name (with or without .pddl extension)

        Returns:
            Tuple of (centralized_domain_path, centralized_problem_path)
        """
        # Output centralized files under centralized_root/<domain-name>
        out_dir = self.centralized_root / Path(domain_dir).name
        out_dir.mkdir(parents=True, exist_ok=True)

        # The converter script always adds .pddl extension, so we need to pass only the stem
        domain_stem = (
            Path(domain_file).stem if Path(domain_file).suffix else domain_file
        )
        problem_stem = (
            Path(problem_file).stem if Path(problem_file).suffix else problem_file
        )

        cmd = [
            self.python_cmd,
            str(self.converter_script),
            str(domain_dir),
            domain_stem,
            problem_stem,
            str(out_dir),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Converter failed: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}"
            )

        domain_pddl = out_dir / f"{domain_stem}.pddl"
        problem_pddl = out_dir / f"{problem_stem}.pddl"

        if not domain_pddl.is_file():
            raise FileNotFoundError(f"Expected output not found: {domain_pddl}")
        if not problem_pddl.is_file():
            raise FileNotFoundError(f"Expected output not found: {problem_pddl}")

        return str(domain_pddl), str(problem_pddl)
