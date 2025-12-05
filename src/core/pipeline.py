"""Main pipeline for MA-PDDL planning with LLMs."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

from ..converters.ma_converter import MAPDDLConverter
from ..llm.provider_factory import ProviderFactory
from ..strategies import StrategyFactory
from ..validation.evaluator import PlanEvaluator


class MAPLLMPipeline:
    """Main pipeline for multi-agent PDDL planning with LLMs."""

    def __init__(self, config):
        """Initialize pipeline with configuration."""
        self.config = config
        self.converter = MAPDDLConverter(
            converter_script=config.resolved_converter_script,
            python_cmd=config.resolved_python_cmd,
            centralized_root=config.centralized_root,
        )
        self.llm = ProviderFactory.create(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            url=config.llm_url,  # Pass URL for Ollama and other providers that need it
            api_key=config.llm_api_key,  # Pass API key for OpenAI, Groq, Anthropic, etc.
        )

    def run(
        self,
        domain_dir: str,
        domain_file: str,
        problem_file: str,
        mode: str = "no-val",
        max_steps: int = 0,
        validate_after: bool = False,
    ) -> Tuple[List[str], Path, Path, Path]:
        """
        Run the planning pipeline.

        Args:
            domain_dir: Directory containing domain and problem files
            domain_file: Domain file name
            problem_file: Problem file name
            mode: Planning mode (no-val, soft-val-ar, open-loop-repair, etc.)
            max_steps: Maximum plan length
            validate_after: Whether to validate the plan

        Returns:
            Tuple of (plan_actions, plan_path, centralized_domain_path, centralized_problem_path)
        """
        base_dir = Path(domain_dir)
        domain_name = base_dir.name

        # Resolve input files
        ma_domain_path = self._resolve_input_file(base_dir, domain_file)
        ma_problem_path = self._resolve_input_file(base_dir, problem_file)

        # Convert to centralized PDDL
        centralized_domain, centralized_problem = self.converter.convert(
            domain_dir, domain_file, problem_file
        )
        centralized_domain_path = Path(centralized_domain)
        centralized_problem_path = Path(centralized_problem)

        # Create strategy and generate plan
        effective_max = max_steps or self.config.max_steps
        strategy = StrategyFactory.create(
            mode=mode,
            llm=self.llm,
            config=self.config,
        )

        plan = strategy.generate_plan(
            ma_domain_file=str(ma_domain_path),
            ma_problem_file=str(ma_problem_path),
            ground_domain_file=str(centralized_domain_path),
            ground_problem_file=str(centralized_problem_path),
            max_steps=effective_max,
        )

        # Save plan
        results_dir = Path(self.config.results_root) / domain_name
        results_dir.mkdir(parents=True, exist_ok=True)
        prob_stem = (
            Path(problem_file).stem if Path(problem_file).suffix else problem_file
        )
        plan_path = results_dir / f"{prob_stem}.plan"

        with plan_path.open("w", encoding="utf-8") as f:
            for action in plan:
                f.write(action + "\n")

        if self.config.debug:
            print(f"[PIPELINE] Plan saved to: {plan_path}")

        # Validate if requested
        if validate_after:
            evaluator = PlanEvaluator(
                validate_bin=self.config.resolved_val_bin,
                timeout=120,
                extra_flags=["-v"],
                print_on_pass=True,
                print_on_fail=True,
            )
            ok = evaluator.evaluate(
                str(centralized_domain_path),
                str(centralized_problem_path),
                str(plan_path),
            )
            if self.config.debug:
                print(f"[PIPELINE] Validation: {'PASSED' if ok else 'FAILED'}")

        return plan, plan_path, centralized_domain_path, centralized_problem_path

    def _resolve_input_file(self, base_dir: Path, base_name: str) -> Path:
        """Resolve input file path, checking with and without .pddl extension."""
        p = base_dir / base_name
        if p.is_file():
            return p
        p2 = base_dir / f"{base_name}.pddl"
        if p2.is_file():
            return p2
        raise FileNotFoundError(f"Input file not found: {p} or {p2}")
