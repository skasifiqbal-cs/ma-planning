import logging
import os
from pathlib import Path
from centralize_ma import MAPDDLConverter
from llm_prompt import LLMPrompt
from validation import Validator

from strategies import (
    PlanGenerationStrategy,
    OpenLoopNoValidationStrategy,
    SoftValidationAutoregressiveStrategy,
)


class MAPLLMPipeline:
    def __init__(self, config):
        self.config = config
        self.converter = MAPDDLConverter(
            config.converter["converter_script"],
            config.converter["python_cmd"],
            config.centralized_root,
        )
        self.prompt = LLMPrompt(config)
        self.validator = Validator(config.val_bin)

    def run(self, domain_dir, domain_file, problem_file, mode="no-val", max_steps=None):
        if not max_steps:
            max_steps = self.config.max_steps

        # Centralize the PDDL domain/problem files
        centralized_domain, centralized_problem = self.converter.convert(
            domain_dir, domain_file, problem_file
        )
        print("[INFO] Centralized files:", centralized_domain, centralized_problem)

        # Choose strategy and generate plan
        strategy = self._select_strategy(mode)
        plan = strategy.generate_plan(
            centralized_domain, centralized_problem, max_steps
        )

        # Print plan
        print("\n[RESULT] Final generated plan:")
        for i, action in enumerate(plan):
            print(f"{i}: {action}")

        # Save plan
        out_path = self._save_plan(domain_dir, problem_file, plan)
        print(f"[INFO] Plan saved to: {out_path}")

        # Return for callers (e.g., main.py batch/validate)
        return plan, out_path, centralized_domain, centralized_problem

    def _select_strategy(self, mode: str) -> PlanGenerationStrategy:
        debug = getattr(self.config, "debug", False)
        if mode == "no-val":
            return OpenLoopNoValidationStrategy(self.prompt, debug=debug)
        if mode == "soft-val-ar":
            return SoftValidationAutoregressiveStrategy(self.prompt, debug=debug)
        raise NotImplementedError(f"Unknown mode: {mode}")

    def _save_plan(
        self, domain_dir: str, problem_file_arg: str, plan: list[str]
    ) -> Path:
        domain_dir_name = Path(domain_dir).name
        out_dir = Path("results") / domain_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        problem_stem = Path(problem_file_arg).stem
        out_path = out_dir / f"{problem_stem}.plan"
        content = "\n".join(plan)
        if content and not content.endswith("\n"):
            content += "\n"
        out_path.write_text(content)
        return out_path
