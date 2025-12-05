"""Command-line interface for MA-PDDL planning with Typer."""

import sys
import logging
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import typer
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback message
    print(
        "Warning: typer and rich not installed. Install with: pip install typer[all] rich"
    )

from src.core.config import Config
from src.core.pipeline import MAPLLMPipeline
from src.validation.evaluator import PlanEvaluator


# Setup logging
def setup_logging(verbose: bool = False):
    """Configure logging with rich handler if available."""
    level = logging.DEBUG if verbose else logging.INFO

    if RICH_AVAILABLE:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=Console(), rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# Initialize logger
logger = logging.getLogger(__name__)


def run_single(
    config,  # Config object
    domain_dir: str,
    domain_file: str,
    problem_file: str,
    mode: str,
    max_steps: Optional[int],
    validate: bool,
) -> None:
    """Run planning for a single problem."""
    logger.info(f"Planning for problem: {problem_file}")

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing pipeline...", total=None)
            pipeline = MAPLLMPipeline(config)
    else:
        pipeline = MAPLLMPipeline(config)

    plan, plan_path, centralized_domain, centralized_problem = pipeline.run(
        domain_dir,
        domain_file,
        problem_file,
        mode=mode,
        max_steps=max_steps,
        validate_after=validate,
    )

    logger.info(f"✓ Plan saved to: {plan_path}")
    logger.info(f"✓ Generated {len(plan)} actions")


# Initialize console
console = Console() if RICH_AVAILABLE else None


def discover_problems(domain_dir: str) -> list[str]:
    """Discover problem files in a domain directory."""
    root = Path(domain_dir)
    if not root.is_dir():
        raise ValueError(f"Domain directory not found: {domain_dir}")

    probs = []
    for f in sorted(root.iterdir()):
        if not f.is_file():
            continue
        name = f.name
        if name in ("domain", "domain.pddl"):
            continue
        if name.startswith("p") or f.suffix == ".pddl":
            probs.append(name)


def run_batch(
    config,  # Config object
    domain_dir: str,
    domain_file: str,
    mode: str,
    max_steps: Optional[int],
    validate: bool,
    problems: Optional[List[str]] = None,
) -> None:
    """Run planning for multiple problems in batch mode."""
    pipeline = MAPLLMPipeline(config)

    all_problems = discover_problems(domain_dir)
    if problems:
        provided = set(Path(p).stem for p in problems)
        all_problems = [p for p in all_problems if Path(p).stem in provided]

    if not all_problems:
        logger.warning("No problems discovered")
        return

    logger.info(f"Discovered {len(all_problems)} problems")

    evaluator = PlanEvaluator(config.resolved_val_bin) if validate else None
    passed = 0
    total = 0
    results = []

    for prob in all_problems:
        prob_base = Path(prob).stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Problem: {prob_base}")
        logger.info(f"{'='*60}")

        try:
            plan, plan_path, centralized_domain, centralized_problem = pipeline.run(
                domain_dir,
                domain_file,
                prob_base,
                mode=mode,
                max_steps=max_steps,
            )

            valid = None
            if validate and evaluator:
                total += 1
                valid = evaluator.evaluate(
                    str(centralized_domain),
                    str(centralized_problem),
                    str(plan_path),
                )
                if valid:
                    passed += 1
                    logger.info(f"✓ Plan validated successfully")
                else:
                    logger.error(f"✗ Plan validation failed")

            results.append(
                {
                    "problem": prob_base,
                    "plan_path": str(plan_path),
                    "actions": len(plan) if plan else 0,
                    "valid": valid,
                }
            )
            logger.info(f"✓ Plan saved to: {plan_path}")
            logger.info(f"✓ Generated {len(plan)} actions")

        except Exception as e:
            logger.error(f"✗ Planning failed for {prob_base}: {e}")
            results.append(
                {"problem": prob_base, "plan_path": None, "actions": 0, "valid": False}
            )


if RICH_AVAILABLE:
    # Modern Typer CLI
    app = typer.Typer(
        name="ma-planning",
        help="Multi-Agent PDDL Planning with LLMs",
        add_completion=True,
    )

    @app.command()
    def plan(
        config_file: Optional[Path] = typer.Option(
            "config.yaml",
            "--config",
            help="Configuration YAML file",
        ),
        domain_dir: Path = typer.Option(
            ...,
            "--domain-dir",
            "-d",
            help="Directory containing MA-PDDL domain files",
            exists=True,
        ),
        problem_file: Optional[str] = typer.Option(
            None,
            "--problem-file",
            "-p",
            help="Problem file (for single problem mode)",
        ),
        domain_file: str = typer.Option(
            "domain",
            "--domain-file",
            help="Domain file base name",
        ),
        mode: str = typer.Option(
            "no-val",
            "--mode",
            "-m",
            help="Planning strategy",
        ),
        max_steps: Optional[int] = typer.Option(
            None,
            "--max-steps",
            help="Maximum planning steps",
        ),
        validate: bool = typer.Option(
            False,
            "--validate/--no-validate",
            help="Validate generated plans",
        ),
        validate_bin: Optional[str] = typer.Option(
            None,
            "--validate-bin",
            help="Path to VAL Validate binary",
        ),
        batch: bool = typer.Option(
            False,
            "--batch",
            help="Batch mode: solve all problems in directory",
        ),
        problems: Optional[List[str]] = typer.Option(
            None,
            "--problems",
            help="Subset of problems for batch mode",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Verbose output",
        ),
    ):
        """Generate multi-agent PDDL plans using LLMs."""
        # Setup logging
        setup_logging(verbose)

        # Load config from YAML file
        config = Config.from_yaml(str(config_file))

        # Override with CLI args if provided
        if max_steps is not None:
            config.max_steps = max_steps
        if validate_bin:
            config.val_bin = validate_bin

        try:
            config.resolve()
            if verbose:
                logger.info(f"Using: {config.llm_model} @ temp={config.temperature}")
        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            raise typer.Exit(1)

        # Run planning
        try:
            if batch:
                run_batch(
                    config,
                    str(domain_dir),
                    domain_file,
                    mode,
                    max_steps,
                    validate,
                    problems,
                )
            else:
                if not problem_file:
                    logger.error("--problem-file required for non-batch mode")
                    raise typer.Exit(1)
                run_single(
                    config,
                    str(domain_dir),
                    domain_file,
                    problem_file,
                    mode,
                    max_steps,
                    validate,
                )
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            if verbose:
                logger.exception("Full traceback:")
            raise typer.Exit(1)

    def main():
        """Entry point for Typer CLI."""
        app()

else:
    # Fallback to basic argparse if typer not available
    import argparse

    def main():
        """Fallback CLI entry point."""
        parser = argparse.ArgumentParser(description="MA-PDDL Planning with LLMs")
        parser.add_argument("--config", default="config.yaml", help="Config YAML file")
        parser.add_argument("--domain-dir", required=True, help="Domain directory")
        parser.add_argument("--domain-file", default="domain", help="Domain file")
        parser.add_argument("--problem-file", help="Problem file")
        parser.add_argument("--batch", action="store_true", help="Batch mode")
        parser.add_argument("--problems", nargs="*", help="Problem subset")
        parser.add_argument("--validate", action="store_true", help="Validate plans")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")

        args = parser.parse_args()
        setup_logging(args.verbose)

        # Load config from YAML
        config = Config.from_yaml(args.config)

        try:
            config.resolve()
            if args.verbose:
                print(f"Using: {config.llm_model} @ temp={config.temperature}")
        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            sys.exit(1)

        if args.batch:
            run_batch(
                config,
                args.domain_dir,
                args.domain_file,
                config.strategy,
                config.max_steps,
                args.validate,
                args.problems,
            )
        else:
            if not args.problem_file:
                logger.error("--problem-file required")
                sys.exit(1)
            run_single(
                config,
                args.domain_dir,
                args.domain_file,
                args.problem_file,
                config.strategy,
                config.max_steps,
                args.validate,
            )


if __name__ == "__main__":
    main()
