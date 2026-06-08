#!/usr/bin/env python3
"""Aggregate all experiment results into results/domain_variance_report.csv.

Handles two result layouts:

  New (run_id dirs):   results/{run_id}/{domain}/eval.json
  Old (flat):          results/{domain}/batch_eval_{model}_{date}_{time}.json

Run from repo root:
  python experiments/aggregate_results.py
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_RUN_ID_RE = re.compile(r"^(\d{8})_(\d{6})_(.+)$")
_OLD_EVAL_RE = re.compile(r"^batch_eval_(.*?)_(\d{8})_(\d{6})\.json$")


def _as_float(v):
    return float(v) if isinstance(v, (int, float)) else None


def _first_float(d, keys):
    for k in keys:
        v = _as_float(d.get(k)) if isinstance(d, dict) else None
        if v is not None:
            return v
    return None


def _parse_eval(path: Path) -> dict | None:
    """Parse one eval file and return a flat record dict, or None on error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    results_list = data.get("results", [])
    if isinstance(results_list, dict):
        results_list = list(results_list.values())

    summary = data.get("summary", {})
    meta = data.get("metadata", {})

    total_time = _first_float(summary, ["total_time", "total_runtime_seconds"])
    avg_plan = _first_float(summary, ["avg_actions", "avg_plan_length"])
    avg_retries = _first_float(summary, ["avg_retries"])
    total_tokens = _first_float(summary, ["total_tokens"])
    total_prompt_tokens = _first_float(summary, ["total_prompt_tokens"])
    total_completion_tokens = _first_float(summary, ["total_completion_tokens"])

    passed = 0
    total = 0
    time_sum = 0.0
    plan_sum = 0.0
    plan_count = 0

    for res in results_list:
        if not isinstance(res, dict):
            continue
        total += 1
        if res.get("validation_passed", False):
            passed += 1
        rt = _first_float(res, ["execution_time", "runtime_seconds"])
        if rt is not None:
            time_sum += rt
        pv = _first_float(res, ["num_actions", "plan_length"])
        if pv is not None:
            plan_sum += pv
            plan_count += 1

    if total == 0:
        return None

    if total_time is None and time_sum > 0:
        total_time = time_sum
    if avg_plan is None and plan_count > 0:
        avg_plan = plan_sum / plan_count

    return {
        "passed": passed,
        "total": total,
        "total_time": total_time,
        "avg_plan": avg_plan,
        "avg_retries": avg_retries,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "model": meta.get("model", "unknown"),
    }


def aggregate(results_root: Path = Path("results")) -> Path:
    """Scan results_root and write domain_variance_report.csv. Returns output path."""
    records: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print(f"Scanning {results_root} ...")

    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue

        run_id_match = _RUN_ID_RE.match(child.name)

        if run_id_match:
            date, _time, run_suffix = run_id_match.groups()
            for domain_dir in sorted(child.iterdir()):
                if not domain_dir.is_dir():
                    continue
                eval_file = domain_dir / "eval.json"
                if not eval_file.is_file():
                    continue
                rec = _parse_eval(eval_file)
                if rec:
                    rec["run_id"] = child.name
                    records[date][child.name][domain_dir.name].append(rec)
        else:
            domain_name = child.name
            for eval_file in sorted(child.glob("batch_eval_*.json")):
                m = _OLD_EVAL_RE.match(eval_file.name)
                if not m or eval_file.name.endswith("_latest.json"):
                    continue
                model, date, run_time = m.groups()
                run_key = f"{date}_{run_time}_{model}"
                rec = _parse_eval(eval_file)
                if rec:
                    rec.setdefault("model", model)
                    rec["run_id"] = run_key
                    records[date][run_key][domain_name].append(rec)

    output_file = results_root / "domain_variance_report.csv"
    rows_written = 0

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date",
            "Run ID",
            "Model",
            "Domain",
            "N Problems",
            "Coverage",
            "Total Time (s)",
            "Avg Plan Length",
            "Avg Retries",
            "Total Tokens",
            "Prompt Tokens",
            "Completion Tokens",
        ])

        for date in sorted(records):
            for run_id in sorted(records[date]):
                for domain in sorted(records[date][run_id]):
                    recs = records[date][run_id][domain]
                    total = sum(r["total"] for r in recs)
                    passed = sum(r["passed"] for r in recs)
                    coverage = passed / total if total > 0 else 0.0
                    times = [r["total_time"] for r in recs if r["total_time"] is not None]
                    plans = [r["avg_plan"] for r in recs if r["avg_plan"] is not None]
                    retries = [r["avg_retries"] for r in recs if r["avg_retries"] is not None]
                    tokens = [r["total_tokens"] for r in recs if r["total_tokens"] is not None]
                    prompt_tok = [r["total_prompt_tokens"] for r in recs if r["total_prompt_tokens"] is not None]
                    compl_tok = [r["total_completion_tokens"] for r in recs if r["total_completion_tokens"] is not None]
                    model = recs[0].get("model", run_id)

                    writer.writerow([
                        date,
                        run_id,
                        model,
                        domain,
                        total,
                        f"{coverage:.3f}",
                        f"{sum(times):.1f}" if times else "",
                        f"{sum(plans)/len(plans):.1f}" if plans else "",
                        f"{sum(retries)/len(retries):.2f}" if retries else "",
                        int(sum(tokens)) if tokens else "",
                        int(sum(prompt_tok)) if prompt_tok else "",
                        int(sum(compl_tok)) if compl_tok else "",
                    ])
                    rows_written += 1

    print(f"Written {rows_written} rows → {output_file}")
    return output_file


def main() -> int:
    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    if not results_root.is_dir():
        print(f"Results directory not found: {results_root}", file=sys.stderr)
        return 1
    aggregate(results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
