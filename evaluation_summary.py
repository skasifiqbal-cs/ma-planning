from pathlib import Path
from typing import Iterable, Any


def write_eval_summary(
    domain_dir: str, eval_results: Iterable[Any], extension: str = "txt"
) -> Path:
    """
    Write evaluation summary to eval/<domain-name>.<extension>.
    eval_results: iterable of dicts OR objects with 'passed' attribute.
    """
    total = 0
    valid = 0
    for r in eval_results:
        total += 1
        if isinstance(r, dict):
            passed = bool(r.get("passed"))
        else:
            passed = bool(getattr(r, "passed", False))
        if passed:
            valid += 1
    coverage = (valid / total) if total else 0.0

    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / f"{Path(domain_dir).name}.{extension}"

    lines = [
        f"Domain: {Path(domain_dir).name}",
        f"Total problems: {total}",
        f"Valid plans: {valid}",
        f"Coverage: {valid}/{total} ({coverage:.2%})",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[INFO] Evaluation summary written to: {out_path}")
    return out_path
