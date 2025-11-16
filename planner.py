from pathlib import Path
from typing import Tuple, List

from centralize_ma import MAPDDLConverter
from llm_prompt import LLMPrompt
from strategies import OpenLoopNoValidationStrategy


class MAPLLMPipeline:
    """
    Pipeline that:
      1) Reads unfactored MA-PDDL domain/problem for prompting the LLM.
      2) Converts to centralized PDDL for grounding/validation.
      3) Prompts the LLM to produce a sequential plan with agent as first arg.
      4) Saves plan to results/<domain>/<problem>.plan
    """

    def __init__(self, config):
        self.config = config
        # Converter for centralized PDDL (used for grounding + validation)
        self.converter = MAPDDLConverter(
            converter_script=config.converter["converter_script"],
            python_cmd=config.converter["python_cmd"],
            centralized_root=config.centralized_root,
        )
        # LLM client and strategy
        self.llm = LLMPrompt(
            model=config.llm_model,
            url=config.llm_url,
            temperature=config.temperature,
            debug=config.debug,
        )
        self.strategy = OpenLoopNoValidationStrategy(
            llm=self.llm,
            debug=config.debug,
            show_prompt=getattr(config, "debug_print_prompt", False),
            show_response=getattr(config, "debug_print_response", True),
        )

    def _resolve_input_file(self, base_dir: Path, base_name: str) -> Path:
        """
        Locate MA-PDDL input file given base name that may or may not have .pddl.
        Tries '<base_name>' then '<base_name>.pddl'.
        """
        p = base_dir / base_name
        if p.is_file():
            return p
        p2 = base_dir / f"{base_name}.pddl"
        if p2.is_file():
            return p2
        raise FileNotFoundError(f"Input file not found: {p} or {p2}")

    def run(
        self,
        domain_dir: str,
        domain_file: str,
        problem_file: str,
        mode: str = "no-val",
        max_steps: int = 0,
    ) -> Tuple[List[str], Path, Path, Path]:
        """
        Returns:
          plan (list of actions),
          plan_path,
          centralized_domain_path,
          centralized_problem_path
        """
        base_dir = Path(domain_dir)
        domain_name = base_dir.name

        # 1) MA-PDDL input files (for the prompt)
        ma_domain_path = self._resolve_input_file(base_dir, domain_file)
        ma_problem_path = self._resolve_input_file(base_dir, problem_file)

        # 2) Centralize (for grounding + validation)
        centralized_domain, centralized_problem = self.converter.convert(
            domain_dir, domain_file, problem_file
        )
        centralized_domain_path = Path(centralized_domain)
        centralized_problem_path = Path(centralized_problem)

        # 3) Generate plan using MA-PDDL prompt, but ground against centralized PDDL
        plan = self.strategy.generate_plan(
            ma_domain_file=str(ma_domain_path),
            ma_problem_file=str(ma_problem_path),
            ground_domain_file=str(centralized_domain_path),
            ground_problem_file=str(centralized_problem_path),
            max_steps=max_steps or self.config.max_steps,
        )

        # 4) Save plan
        results_dir = Path(self.config.results_root) / domain_name
        results_dir.mkdir(parents=True, exist_ok=True)
        prob_stem = (
            Path(problem_file).stem if Path(problem_file).suffix else problem_file
        )
        plan_path = results_dir / f"{prob_stem}.plan"
        with plan_path.open("w", encoding="utf-8") as f:
            for a in plan:
                f.write(a.strip() + "\n")

        if self.config.debug:
            print("[RESULT] Final generated plan:")
            for i, a in enumerate(plan[:50]):
                print(f"{i}: {a}")
            print(f"[INFO] Plan saved to: {plan_path}")

        return plan, plan_path, centralized_domain_path, centralized_problem_path
