from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

from centralize_ma import MAPDDLConverter
from llm_prompt import LLMPrompt
from strategies import (
    OpenLoopNoValidationStrategy,
    LLM4PDDLZeroShotAutoregressiveStrategy,
    OpenLoopSimilarityRepairStrategy,
    OpenLoopRandomizedStrategy,
)
from evaluation import PlanEvaluator


class MAPLLMPipeline:
    def __init__(self, config):
        self.config = config
        self.converter = MAPDDLConverter(
            converter_script=config.converter["converter_script"],
            python_cmd=config.converter["python_cmd"],
            centralized_root=config.centralized_root,
        )
        self.llm = LLMPrompt(
            model=config.llm_model,
            url=config.llm_url,
            temperature=config.temperature,
            max_tokens=getattr(config, "max_tokens", 128),
            debug=config.debug,
        )

    def _resolve_input_file(self, base_dir: Path, base_name: str) -> Path:
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
        validate_after: bool = False,
        print_eval_fail: bool = True,
        print_eval_pass: bool = False,
    ) -> Tuple[List[str], Path, Path, Path]:
        base_dir = Path(domain_dir)
        domain_name = base_dir.name

        ma_domain_path = self._resolve_input_file(base_dir, domain_file)
        ma_problem_path = self._resolve_input_file(base_dir, problem_file)

        centralized_domain, centralized_problem = self.converter.convert(
            domain_dir, domain_file, problem_file
        )
        centralized_domain_path = Path(centralized_domain)
        centralized_problem_path = Path(centralized_problem)

        effective_max = max_steps or self.config.max_steps
        if mode == "soft-val-ar":
            strategy = LLM4PDDLZeroShotAutoregressiveStrategy(
                llm=self.llm,
                config=self.config,
                embed_model=getattr(
                    self.config, "embed_model", "paraphrase-MiniLM-L6-v2"
                ),
            )

        elif mode == "open-loop-repair":
            strategy = OpenLoopSimilarityRepairStrategy(
                llm=self.llm,
                config=self.config,
                embed_model=getattr(
                    self.config, "embed_model", "paraphrase-MiniLM-L6-v2"
                ),
                # prevent_backtrack=True,  # toggle if desired
            )

        elif mode == "open-loop-rand":
            strategy = OpenLoopRandomizedStrategy(
                llm=self.llm,
                config=self.config,
                embed_model=getattr(
                    self.config, "embed_model", "paraphrase-MiniLM-L6-v2"
                ),
            )

        else:
            strategy = OpenLoopNoValidationStrategy(
                llm=self.llm,
                config=self.config,
            )

        # Strategy now returns CANONICAL (inverted) plan
        canonical_plan = strategy.generate_plan(
            ma_domain_file=str(ma_domain_path),
            ma_problem_file=str(ma_problem_path),
            ground_domain_file=str(centralized_domain_path),
            ground_problem_file=str(centralized_problem_path),
            max_steps=effective_max,
        )

        results_dir = Path(self.config.results_root) / domain_name
        results_dir.mkdir(parents=True, exist_ok=True)
        prob_stem = (
            Path(problem_file).stem if Path(problem_file).suffix else problem_file
        )
        plan_path = results_dir / f"{prob_stem}.plan"
        with plan_path.open("w", encoding="utf-8") as f:
            for a in canonical_plan:
                f.write(a + "\n")

        if self.config.debug:
            print("[RESULT] Final canonical plan (inverted randomization):")
            for i, a in enumerate(canonical_plan[:100]):
                print(f"{i}: {a}")
            print(f"[INFO] Plan saved to: {plan_path}")

        if validate_after:
            evaluator = PlanEvaluator(
                validate_bin=self.config.val_bin,
                timeout=120,
            )
            ok = evaluator.evaluate(
                str(centralized_domain_path),
                str(centralized_problem_path),
                str(plan_path),
            )
            if self.config.debug:
                print(f"[VALIDATE] {plan_path.name}: {'PASSED' if ok else 'FAILED'}")

        return (
            canonical_plan,
            plan_path,
            centralized_domain_path,
            centralized_problem_path,
        )
