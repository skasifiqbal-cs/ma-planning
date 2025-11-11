from pathlib import Path
import subprocess

class MAPDDLConverter:
    def __init__(self, converter_script, python_cmd="python2"):
        self.converter_script = Path(converter_script)
        self.python_cmd = python_cmd

    def convert(self, domain_dir, domain_file, problem_file, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_cmd,
            str(self.converter_script),
            str(domain_dir),
            domain_file,
            problem_file,
            str(out_dir)
        ]
        subprocess.run(cmd, check=True)
        # Return paths to converted PDDL files
        domain_pddl = out_dir / domain_file
        problem_pddl = out_dir / problem_file
        return str(domain_pddl), str(problem_pddl)