import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

Message = Dict[str, str]


def build_messages(domain_text: str, problems: List[Tuple[str, str]]) -> List[Message]:
    system = (
        "You are an expert multi-agent PDDL planner. "
        "Return EXACTLY one JSON object: "
        '{"plan":[{"time":int,"agent":str,"action":str,"args":[str]}],"notes":str}. '
        "No extra text."
    )
    problems_blob = "Problems (multi-agent; one file per agent):\n" + "\n\n".join(
        [f"--- {fname} ---\n{content}" for fname, content in problems]
    )
    user = (
        f"Domain:\n{domain_text}\n\n"
        f"{problems_blob}\n\n"
        "Semantics:\n"
        "- Actions with the same integer 'time' execute concurrently.\n"
        "- Preconditions are checked on the pre-step state; effects apply simultaneously.\n"
        "- The plan must satisfy the joint goals across all agents.\n\n"
        "Task: Return a valid JSON plan for the combined multi-agent problem."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_json(s: str) -> Any:
    s = s.strip()
    if s.startswith("```"):
        lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
        s = "\n".join(lines).strip()
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i : j + 1]
    return json.loads(s)


def chat_ollama(model: str, messages: List[Message], temperature: float) -> str:
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "top_p": 0.9},
    }
    resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def collect_domain_text(pfile_dir: Path) -> str:
    parts = []
    for f in sorted(pfile_dir.glob("Domain*.pddl")):
        parts.append(f.read_text(encoding="utf-8"))
    agents_dir = pfile_dir / "agents"
    if agents_dir.is_dir():
        for f in sorted(agents_dir.rglob("*.pddl")):
            parts.append(f.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def collect_all_problems(pfile_dir: Path) -> List[Tuple[str, str]]:
    problems = []
    for f in sorted(pfile_dir.glob("Problem*.pddl")):
        problems.append((f.name, f.read_text(encoding="utf-8")))
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MA-PDDL evaluator (Ollama only). One joint run per Pfile* dir."
    )
    ap.add_argument(
        "--model", required=True, help="Ollama model name (e.g. llama3.1:8b-instruct)"
    )
    ap.add_argument(
        "--data-dir",
        default="depots",
        help="root dataset dir containing Pfile* folders",
    )
    ap.add_argument(
        "--pfile-pattern", default="Pfile*", help="glob pattern for problem set dirs"
    )
    ap.add_argument("--out", default="results/run.jsonl")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    base = Path(args.data_dir)
    if not base.exists():
        raise SystemExit(f"data dir not found: {base}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pfile_dirs = [p for p in sorted(base.glob(args.pfile_pattern)) if p.is_dir()]
    if not pfile_dirs:
        raise SystemExit(f"No Pfile* directories found under {base}")

    for pdir in pfile_dirs:
        domain_text = collect_domain_text(pdir)
        if not domain_text.strip():
            print(f"Warning: no Domain*.pddl found in {pdir}; skipping")
            continue

        problems = collect_all_problems(pdir)
        if not problems:
            print(f"Warning: no Problem*.pddl in {pdir}; skipping")
            continue

        messages = build_messages(domain_text, problems)

        try:
            content = chat_ollama(args.model, messages, args.temperature)
            try:
                parsed = normalize_json(content)
            except Exception as e:
                parsed = {"error": str(e), "raw": content}
        except Exception as e:
            parsed = {"error": f"chat error: {e}"}

        record = {
            "backend": "ollama",
            "model": args.model,
            "pfile": pdir.name,
            "domain_files": [p.name for p in sorted(pdir.glob("Domain*.pddl"))],
            "problem_files": [fname for fname, _ in problems],
            "temperature": args.temperature,
            "output": parsed,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(f"[{pdir.name}] -> joint multi-agent plan saved")

    print(f"All done. Results appended to {out_path}")


if __name__ == "__main__":
    main()
