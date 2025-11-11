from pathlib import Path
import subprocess


class MAPDDLConverter:
    def __init__(self, converter_script, python_cmd, centralized_root):
        self.converter_script = Path(converter_script)
        self.python_cmd = python_cmd
        self.centralized_root = Path(centralized_root)

    def convert(self, domain_dir, domain_file, problem_file):
        # domain_file and problem_file are base names like "domain", "p10"
        out_dir = self.centralized_root / Path(domain_dir).name / "files"
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
        domain_pddl = out_dir / f"{domain_file}.pddl"
        problem_pddl = out_dir / f"{problem_file}.pddl"
        return str(domain_pddl), str(problem_pddl)
