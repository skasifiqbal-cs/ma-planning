#!/usr/bin/env python3
"""MA-PDDL Planning — single entry point for all workflows.

  python run.py plan       --domain-dir /path/to/unfactored/rovers --problem-file p01 --validate
  python run.py batch      --domain-dir /path/to/unfactored/rovers --validate
  python run.py experiment --domains rovers logistics00   # root-dir from config.yaml paths.domain_root
  python run.py analyze
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import Config
from src.core.pipeline import MAPLLMPipeline
from src.evaluation.batch_evaluator import BatchEvaluator
from src.evaluation.batch_logger import BatchLogger
from src.validation.evaluator import PlanEvaluator, extract_failure_reason

app = typer.Typer(name="ma-planning", help="Multi-Agent PDDL Planning with LLMs", add_completion=False)
console = Console()
logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _make_run_id(config: Config) -> str:
    """Build a human-readable run identifier encoding all key parameters."""
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy = config.strategy.replace("-", "")[:14]
    provider = config.llm_provider[:6]
    model_slug = config.llm_model.split("/")[-1].split(":")[0][:14].replace("-", "")
    retries = f"r{config.backprompt_max_retries}"
    fewshot = f"fs{config.few_shot_example_count}"
    return f"{date}_{strategy}_{provider}-{model_slug}_{retries}_{fewshot}"


def _load_completed(domain_run_dir: Path) -> set:
    """Return problem stems already recorded in an existing eval.json."""
    eval_file = domain_run_dir / "eval.json"
    if not eval_file.is_file():
        return set()
    try:
        import json
        data = json.loads(eval_file.read_text(encoding="utf-8"))
        return {r["problem"] for r in data.get("results", []) if isinstance(r, dict) and r.get("problem")}
    except Exception:
        return set()


def _few_shot_excluded(examples_file: str, domain_name: str) -> set:
    """Return problem stems used as few-shot examples for this domain."""
    import json
    path = Path(examples_file)
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        examples = data.get("examples", data) if isinstance(data, dict) else data
        return {
            ex["problem"]
            for ex in examples
            if isinstance(ex, dict) and ex.get("domain") == domain_name and ex.get("problem")
        }
    except Exception:
        return set()


def _discover_problems(domain_dir: Path) -> List[str]:
    probs = []
    for f in sorted(domain_dir.iterdir()):
        if not f.is_file():
            continue
        if f.name in ("domain", "domain.pddl"):
            continue
        if f.name.startswith("p") or f.suffix == ".pddl":
            probs.append(f.name)
    return probs


def _save_run_meta(run_dir: Path, run_id: str, config: Config, config_file: Path) -> None:
    """Write config snapshot + run_meta.json to run_dir for reproducibility."""
    import json, shutil
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_file, run_dir / "config.yaml")
    meta = {
        "run_id": run_id,
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(),
        "strategy": config.strategy,
        "model": config.llm_model,
        "provider": config.llm_provider,
        "backprompt_max_retries": config.backprompt_max_retries,
        "few_shot_example_count": config.few_shot_example_count,
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _load_config(config_file: Path, max_steps: Optional[int] = None, validate_bin: Optional[str] = None) -> Config:
    cfg = Config.from_yaml(str(config_file))
    if max_steps is not None:
        cfg.max_steps = max_steps
    if validate_bin:
        cfg.val_bin = validate_bin
    cfg.resolve()
    return cfg


def _run_domain(
    config: Config,
    domain_dir: Path,
    run_dir: Path,
    domain_file: str,
    mode: str,
    validate: bool,
    problems: Optional[List[str]] = None,
    resume: bool = False,
) -> dict:
    """Run batch planning for one domain, writing results to run_dir/domain_name/."""
    domain_name = domain_dir.name
    domain_run_dir = run_dir / domain_name
    domain_run_dir.mkdir(parents=True, exist_ok=True)

    all_problems = _discover_problems(domain_dir)
    if problems:
        target = {Path(p).stem for p in problems}
        all_problems = [p for p in all_problems if Path(p).stem in target]

    if config.use_few_shot and config.few_shot_examples_file:
        excluded = _few_shot_excluded(config.few_shot_examples_file, domain_name)
        if excluded:
            logger.info(f"[{domain_name}] Excluding {len(excluded)} few-shot example(s): {sorted(excluded)}")
            all_problems = [p for p in all_problems if Path(p).stem not in excluded]

    if resume:
        completed = _load_completed(domain_run_dir)
        if completed:
            logger.info(f"[{domain_name}] Resuming — skipping {len(completed)} already-done: {sorted(completed)}")
            all_problems = [p for p in all_problems if Path(p).stem not in completed]

    if not all_problems:
        logger.warning(f"[{domain_name}] No problems found")
        return {}

    batch_eval = BatchEvaluator(domain_name, config.llm_model, output_dir=str(run_dir))
    batch_logger = BatchLogger(domain_name, config.llm_model, output_dir=str(run_dir))
    evaluator = PlanEvaluator(config.resolved_val_bin) if validate else None
    batch_eval.start_batch()

    for prob in all_problems:
        prob_base = Path(prob).stem
        prob_start = time.time()

        logger.info(f"[{domain_name}] {prob_base}")
        batch_logger.start_problem(prob_base)

        status = "success"
        error_msg = None
        validation_msg = "not validated"
        validation_passed = False
        validation_reason = ""
        num_actions = 0
        num_attempts = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        try:
            pipeline = MAPLLMPipeline(config)
            # Override results root so plans land in the run dir
            config_copy = config
            config_copy.results_root = str(run_dir)

            plan, plan_path, c_domain, c_problem, raw_output = pipeline.run(
                domain_dir=str(domain_dir),
                domain_file=domain_file,
                problem_file=prob_base,
                mode=mode,
            )

            num_attempts = getattr(pipeline, "num_attempts", 0)
            val_attempts = getattr(pipeline, "validation_attempts", [])
            tok = getattr(pipeline, "total_tokens", {})
            prompt_tokens = tok.get("prompt_tokens", 0)
            completion_tokens = tok.get("completion_tokens", 0)
            total_tokens = tok.get("total_tokens", 0)

            try:
                msgs = getattr(pipeline, "last_prompt_messages", None)
                if msgs:
                    batch_logger.log_prompt(prob_base, msgs)
            except Exception:
                pass

            num_actions = len(plan) if plan else 0
            batch_logger.log_llm_output(prob_base, raw_output, plan)
            batch_logger.log_final_plan(prob_base, plan, 0)
            if val_attempts:
                batch_logger.log_validation_attempts(prob_base, val_attempts)

            if validate and evaluator:
                valid, _, val_out, val_err = evaluator.evaluate_detailed(
                    str(c_domain), str(c_problem), str(plan_path)
                )
                validation_passed = valid
                validation_msg = "PASSED" if valid else "FAILED"
                validation_reason = extract_failure_reason(val_out, val_err) if not valid else ""
                batch_logger.log_validation_result(prob_base, valid, f"Plan validation {validation_msg}", validation_reason)
                if valid:
                    logger.info(f"  ✓ valid  ({num_actions} actions)")
                else:
                    logger.warning(f"  ✗ invalid ({num_actions} actions)")
            else:
                logger.info(f"  → {num_actions} actions")

        except KeyboardInterrupt:
            status = "skipped"
            error_msg = "Skipped by user"
            batch_logger.log_error(prob_base, error_msg)
            logger.warning(f"  ⊘ skipped")

        except Exception as e:
            status = "error"
            error_msg = str(e)
            batch_logger.log_error(prob_base, error_msg)
            logger.error(f"  ✗ error: {error_msg}")

        prob_time = time.time() - prob_start
        batch_logger.end_problem(prob_base)
        batch_eval.add_result(
            problem_name=prob_base,
            status=status,
            num_actions=num_actions,
            validation_passed=validation_passed,
            validation_message=validation_msg,
            execution_time=prob_time,
            validation_reason=validation_reason,
            error_message=error_msg,
            num_attempts=num_attempts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    batch_eval.end_batch()
    batch_eval.generate_report()
    batch_logger.generate_log_report()

    summary = batch_eval.get_summary()
    logger.info(
        f"[{domain_name}] done — {summary['passed']}/{summary['total_problems']} valid"
        f"  ({summary['pass_rate_percent']}%)"
    )
    return summary


# ── Subcommands ────────────────────────────────────────────────────────────────

@app.command()
def plan(
    domain_dir: Path = typer.Option(..., "--domain-dir", "-d", help="Domain directory"),
    problem_file: str = typer.Option(..., "--problem-file", "-p", help="Problem file stem"),
    domain_file: str = typer.Option("domain", "--domain-file", help="Domain file base name"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Planning strategy (overrides config)"),
    config_file: Path = typer.Option(Path("config.yaml"), "--config"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps"),
    validate: bool = typer.Option(False, "--validate/--no-validate"),
    validate_bin: Optional[str] = typer.Option(None, "--validate-bin"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Plan a single problem."""
    _setup_logging(verbose)
    try:
        cfg = _load_config(config_file, max_steps, validate_bin)
    except Exception as e:
        logger.error(f"Config error: {e}")
        raise typer.Exit(1)

    effective_mode = mode or cfg.strategy
    run_id = _make_run_id(cfg)
    run_dir = Path(cfg.results_root) / run_id

    if verbose:
        logger.info(f"Run: {run_id}")
        logger.info(f"Model: {cfg.llm_model} | Strategy: {effective_mode}")

    try:
        pipeline = MAPLLMPipeline(cfg)
        cfg.results_root = str(run_dir)
        plan_actions, plan_path, c_domain, c_problem, _ = pipeline.run(
            domain_dir=str(domain_dir),
            domain_file=domain_file,
            problem_file=problem_file,
            mode=effective_mode,
            validate_after=validate,
        )
        logger.info(f"Plan: {plan_path}  ({len(plan_actions)} actions)")
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        if verbose:
            logger.exception("")
        raise typer.Exit(1)


@app.command()
def batch(
    domain_dir: Path = typer.Option(..., "--domain-dir", "-d", help="Domain directory"),
    domain_file: str = typer.Option("domain", "--domain-file"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m"),
    config_file: Path = typer.Option(Path("config.yaml"), "--config"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps"),
    validate: bool = typer.Option(False, "--validate/--no-validate"),
    problems: Optional[List[str]] = typer.Option(None, "--problems"),
    resume: bool = typer.Option(False, "--resume", help="Skip already-completed problems in an existing run dir"),
    run_id_override: Optional[str] = typer.Option(None, "--run-id", help="Resume into a specific existing run directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run all problems in one domain."""
    _setup_logging(verbose)
    try:
        cfg = _load_config(config_file, max_steps)
    except Exception as e:
        logger.error(f"Config error: {e}")
        raise typer.Exit(1)

    effective_mode = mode or cfg.strategy
    run_id = run_id_override or _make_run_id(cfg)
    run_dir = Path(cfg.results_root) / run_id

    _save_run_meta(run_dir, run_id, cfg, config_file)

    logger.info(f"Run: {run_id}")
    _run_domain(cfg, domain_dir, run_dir, domain_file, effective_mode, validate, problems, resume=resume)


@app.command()
def experiment(
    root_dir: Optional[Path] = typer.Option(
        None,
        "--root-dir",
        help="Root dir containing domain subdirs (unfactored MA-PDDL). Falls back to paths.domain_root in config.yaml.",
    ),
    domain_file: str = typer.Option("domain.pddl", "--domain-file"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m"),
    config_file: Path = typer.Option(Path("config.yaml"), "--config"),
    domains: Optional[List[str]] = typer.Option(None, "--domains", help="Domain subset"),
    max_problems: int = typer.Option(18, "--max-problems"),
    validate: bool = typer.Option(True, "--validate/--no-validate"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    resume: bool = typer.Option(False, "--resume", help="Skip already-completed problems; requires --run-id"),
    run_id_override: Optional[str] = typer.Option(None, "--run-id", help="Resume into a specific existing run directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run all problems across multiple domains (full experiment)."""
    _setup_logging(verbose)
    try:
        cfg = _load_config(config_file)
    except Exception as e:
        logger.error(f"Config error: {e}")
        raise typer.Exit(1)

    if root_dir is None:
        if cfg.domain_root:
            root_dir = Path(cfg.domain_root)
        else:
            logger.error("No --root-dir given and paths.domain_root not set in config.yaml")
            raise typer.Exit(1)

    effective_mode = mode or cfg.strategy
    run_id = run_id_override or _make_run_id(cfg)
    run_dir = Path(cfg.results_root) / run_id

    domain_dirs = sorted(
        d for d in root_dir.iterdir()
        if d.is_dir() and (domains is None or d.name in set(domains))
    )
    if not domain_dirs:
        logger.error(f"No domain dirs found under {root_dir}")
        raise typer.Exit(1)

    logger.info(f"Run:     {run_id}")
    logger.info(f"Model:   {cfg.llm_model}")
    logger.info(f"Strategy: {effective_mode}")
    logger.info(f"Domains: {len(domain_dirs)}")

    if dry_run:
        for d in domain_dirs:
            logger.info(f"  would run: {d.name}")
        return

    _save_run_meta(run_dir, run_id, cfg, config_file)

    failed = []
    started = time.time()

    for i, domain_dir in enumerate(domain_dirs, 1):
        logger.info(f"\n[{i}/{len(domain_dirs)}] {domain_dir.name}")

        all_probs = _discover_problems(domain_dir)
        probs_to_run = all_probs[:max_problems] if max_problems > 0 else all_probs

        try:
            _run_domain(cfg, domain_dir, run_dir, domain_file, effective_mode, validate, probs_to_run, resume=resume)
        except Exception as e:
            logger.error(f"Domain {domain_dir.name} failed: {e}")
            failed.append(domain_dir.name)

    elapsed = time.time() - started
    logger.info(f"\nDone — {len(domain_dirs) - len(failed)}/{len(domain_dirs)} domains"
                f"  ({elapsed:.0f}s)  results → {run_dir}")
    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")


@app.command()
def analyze(
    results_dir: Path = typer.Option(Path("results"), "--results-dir"),
):
    """Aggregate all experiment results to results/domain_variance_report.csv."""
    import subprocess
    ret = subprocess.call([sys.executable, "experiments/aggregate_results.py"])
    raise typer.Exit(ret)


def main():
    app()


if __name__ == "__main__":
    main()
