from pathlib import Path
import subprocess


class MAPDDLConverter:
    def __init__(self, converter_script, python_cmd, centralized_root):
        self.converter_script = Path(converter_script)
        self.python_cmd = python_cmd
        self.centralized_root = Path(centralized_root)

    def convert(self, domain_dir, domain_file, problem_file):
        # Output centralized files directly under centralized_root/<domain-name>
        out_dir = self.centralized_root / Path(domain_dir).name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_cmd,
            str(self.converter_script),
            str(domain_dir),
            domain_file,
            problem_file,
            str(out_dir),
        ]
        subprocess.run(cmd, check=True)

        domain_pddl = out_dir / f"{Path(domain_file).stem}.pddl"
        problem_pddl = out_dir / f"{Path(problem_file).stem}.pddl"
        return str(domain_pddl), str(problem_pddl)
