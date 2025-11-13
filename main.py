import argparse
import json
import sys
from pathlib import Path

from config import Config
from planner import MAPLLMPipeline
from evaluation import PlanEvaluator


def _stem(name: str) -> str:
    p = Path(name)
    return p.stem if p.suffix else p.name


def _discover_problems(domain_dir: str) -> list[str]:
    root = Path(domain_dir)
    if not root.is_dir():
        raise ValueError(f"Domain directory not found: {domain_dir}")
    probs: list[str] = []
    for f in sorted(root.iterdir()):
        if not f.is_file():
            continue
        name = f.name
        if name in ("domain", "domain.pddl"):
            continue
        if name.startswith("p") or f.suffix == ".pddl":
            probs.append(name)
    return probs


def run_single(config: Config, args) -> None:
    pipeline = MAPLLMPipeline(config)
    plan, plan_path, centralized_domain, centralized_problem = pipeline.run(
        args.domain_dir,
        args.domain_file,
        args.problem_file,
        mode=args.mode,
        max_steps=args.max_steps,
    )

    if args.validate:
        evaluator = PlanEvaluator(config.val_bin)
        ok = evaluator.evaluate(centralized_domain, centralized_problem, str(plan_path))
        print(f"[VALIDATE] {Path(plan_path).name}: {'PASSED' if ok else 'FAILED'}")


def run_batch(config: Config, args) -> None:
    pipeline = MAPLLMPipeline(config)

    problems = _discover_problems(args.domain_dir)
    if args.problems:
        provided = set(_stem(p) for p in args.problems)
        problems = [p for p in problems if _stem(p) in provided]
    if not problems:
        print("[INFO] No problems discovered.")
        return

    print(
        f"[INFO] Discovered {len(problems)} problems: {[ _stem(p) for p in problems ]}"
    )

    domain_name = Path(args.domain_dir).name
    results_dir = Path("results") / domain_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Centralized locations (created by converter)
    centralized_root = Path(config.centralized_root)
    centralized_dir = centralized_root / domain_name
    centralized_domain = centralized_dir / f"{_stem(args.domain_file)}.pddl"

    evaluator = PlanEvaluator(config.val_bin) if args.validate else None

    rows: list[str] = []
    passed = 0
    total = 0

    for prob in problems:
        prob_base = _stem(prob)
        print(f"\n=== Solving problem: {prob_base} ===")
        plan, plan_path, _, centralized_problem = pipeline.run(
            args.domain_dir,
            args.domain_file,
            prob_base,
            mode=args.mode,
            max_steps=args.max_steps,
        )

        status = ""
        num_actions = len(plan)

        if args.validate and evaluator:
            ok = evaluator.evaluate(
                str(centralized_domain), str(centralized_problem), str(plan_path)
            )
            status = "PASSED" if ok else "FAILED"
            total += 1
            if ok:
                passed += 1
            print(f"[EVAL] {prob_base}: {status} (actions={num_actions})")

        rows.append(
            f"{prob_base},{num_actions},{1 if status=='PASSED' else 0},{plan_path}"
        )

    if args.validate and total > 0:
        coverage = (passed / total) * 100
        print(f"\n=== COVERAGE SUMMARY ===")
        print(f"Passed: {passed} / {total}  ({coverage:.2f}%)")

        csv_path = results_dir / "summary.csv"
        json_path = results_dir / "summary.json"

        with csv_path.open("w", encoding="utf-8") as f:
            f.write("problem,num_actions,passed,plan_file\n")
            for r in rows:
                f.write(r + "\n")
            f.write(
                f"TOTAL,{sum(int(r.split(',')[1]) for r in rows)},{passed}/{total},coverage={coverage:.2f}%\n"
            )

        summary_obj = {
            "domain": domain_name,
            "mode": args.mode,
            "total_problems": total,
            "passed": passed,
            "coverage_percent": coverage,
            "results": [
                {
                    "problem": r.split(",")[0],
                    "num_actions": int(r.split(",")[1]),
                    "passed": bool(int(r.split(",")[2])),
                    "plan_file": r.split(",")[3],
                }
                for r in rows
            ],
        }
        with json_path.open("w", encoding="utf-8") as jf:
            jf.write(json.dumps(summary_obj, indent=2))

        print(f"[INFO] Summary CSV: {csv_path}")
        print(f"[INFO] Summary JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain-dir", required=True, help="Path to benchmark domain directory"
    )
    parser.add_argument(
        "--domain-file",
        default="domain",
        help="Domain file base name (e.g., 'domain' or 'domain.pddl')",
    )
    parser.add_argument(
        "--problem-file",
        help="Single problem base name (e.g., 'p10' or 'p10.pddl') for non-batch mode",
    )
    parser.add_argument(
        "--mode",
        default="no-val",
        help="Planning mode: 'no-val' (open-loop, skip invalid) or 'soft-val-ar' (future)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max plan length (truncate if needed)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Solve all problems sequentially in the domain",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        help="Optional subset of problems to run in batch mode (e.g., p10 p11)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated plans using the Validate binary",
    )
    parser.add_argument(
        "--validate-bin",
        default=None,
        help="Path to Validate binary (overrides Config.val_bin)",
    )
    args = parser.parse_args()

    # Build Config (adapt to your Config implementation)
    config = Config()
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.validate_bin:
        config.val_bin = args.validate_bin

    if args.batch:
        run_batch(config, args)
    else:
        if not args.problem_file:
            print(
                "Error: --problem-file is required for non-batch mode", file=sys.stderr
            )
            sys.exit(2)
        run_single(config, args)


if __name__ == "__main__":
    main()
