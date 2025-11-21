from typing import List, Protocol, Set, Dict
from pathlib import Path
import re
import difflib

from ground_actions import ActionGrounder
from plan_parser import parse_actions_no_validation
from llm_prompt import LLMPrompt
from evaluation import PlanEvaluator

try:
    from sentence_transformers import SentenceTransformer, util as st_util

    _HAS_ST = True
except Exception:
    _HAS_ST = False


def _sanitize_action(a: str) -> str:
    """Ensure exactly one parenthesized action and normalize whitespace."""
    s = (a or "").strip()
    m = re.search(r"\([^\(\)]+\)", s)
    if m:
        s = m.group(0)
    else:
        toks = s.split()
        if not toks:
            return ""
        s = "(" + " ".join(toks) + ")"
    return re.sub(r"\s+", " ", s).strip()


def _op_symbol(a: str) -> str:
    s = a.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s.split()[0] if s else ""


def _unwrap_all(a: str) -> str:
    """Remove all outer pairs of parentheses and trim."""
    s = a.strip()
    while s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        if not inner:
            break
        s = inner
    return re.sub(r"\s+", " ", s).strip()


def _single_wrap(a: str) -> str:
    """Wrap once with parentheses after unwrapping; collapse whitespace."""
    inner = _unwrap_all(a)
    return f"({inner})" if inner else ""


def _extract_first_action(raw: str) -> str:
    """Extract first parenthesized chunk or fallback tokens until ')'. Returns single-wrapped."""
    raw = raw.strip()
    m = re.search(r"\([^\(\)]+\)", raw)
    if m:
        return _single_wrap(m.group(0))
    # Fallback
    line = raw.splitlines()[0] if raw else ""
    if ")" in line:
        prefix = line.split(")", 1)[0]
    else:
        prefix = line
    toks = prefix.split()
    if not toks:
        return ""
    return _single_wrap("(" + " ".join(toks) + ")")


def _canonical(a: str) -> str:
    """Return canonical action string with no outer parens."""
    return _unwrap_all(a)


def _similarity_best(query: str, pool: List[str], embedder=None) -> str:
    if not pool:
        return ""
    if embedder is not None:
        q = embedder.encode([query], convert_to_tensor=True)[0]
        embs = embedder.encode(pool, convert_to_tensor=True, show_progress_bar=False)
        sims = st_util.cos_sim(q, embs)[0].cpu().tolist()
        return pool[max(range(len(pool)), key=lambda i: sims[i])]
    scored = [(difflib.SequenceMatcher(None, query, p).ratio(), p) for p in pool]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


class PlanGenerationStrategy(Protocol):
    def generate_plan(
        self,
        ma_domain_file: str,
        ma_problem_file: str,
        ground_domain_file: str,
        ground_problem_file: str,
        max_steps: int,
    ) -> List[str]: ...


class OpenLoopNoValidationStrategy:
    """
    no-val: Prompt with MA-PDDL; parse; filter by operator names; return.
    Actions must be: (operator agent ...other args...)
    Debug printing is controlled ONLY by config flags.
    """

    def __init__(self, llm: LLMPrompt, config):
        self._llm = llm
        self._cfg = config  # uses: debug, debug_print_prompt, debug_print_response

    def generate_plan(
        self,
        ma_domain_file: str,
        ma_problem_file: str,
        ground_domain_file: str,
        ground_problem_file: str,
        max_steps: int,
    ) -> List[str]:
        domain_txt = Path(ma_domain_file).read_text(encoding="utf-8")
        problem_txt = Path(ma_problem_file).read_text(encoding="utf-8")

        grounder = ActionGrounder(ground_domain_file, ground_problem_file)
        task = grounder.task
        op_symbols: Set[str] = {op.name.split()[0].lstrip("(") for op in task.operators}

        prompt = self._build_ma_prompt(domain_txt, problem_txt)

        if self._cfg.debug and getattr(self._cfg, "debug_print_prompt", True):
            print("\n===== LLM PROMPT BEGIN =====")
            print(prompt)
            print("===== LLM PROMPT END =====\n")

        response = self._llm.chat(prompt) or ""

        if self._cfg.debug and getattr(self._cfg, "debug_print_response", True):
            print("[DEBUG] Open-loop LLM raw response (full):")
            print(response)

        plan = parse_actions_no_validation(
            response, valid_ground_ops=op_symbols, disable_name_checks=False
        )
        plan = [_sanitize_action(p) for p in plan]

        if max_steps and len(plan) > max_steps:
            plan = plan[:max_steps]
        return plan

    def _build_ma_prompt(self, domain_txt: str, problem_txt: str) -> str:
        instr = (
            "You are an expert multi-agent planner working with MA-PDDL.\n"
            "Output a valid sequential plan as grounded actions, one per line.\n"
            "CRITICAL FORMAT: (operator agent ...other arguments...)\n"
            "Use ONLY operator names defined in the DOMAIN below.\n"
            "The agent argument must ALWAYS be the first after the operator.\n"
            "Do NOT invent operators (e.g., move_to, collect_soil_sample); use the domain's exact names.\n"
            "Output ONLY the actions, one per line. No explanations."
        )
        return (
            f"{instr}\n\n"
            f"DOMAIN:\n{domain_txt}\n\n"
            f"PROBLEM:\n{problem_txt}\n\n"
            f"PLAN:"
        )


class LLM4PDDLAutoregressiveStrategy:
    """
    Reproduces llm4pddl autoregressive core:
      - Single growing prompt.
      - One action per iteration (stop token ')').
      - Canonical operator names (no outer parentheses) for applicability lookup.
      - Similarity repair only when proposed action inapplicable or unknown.
      - Wrap actions exactly once when storing.
    """

    def __init__(
        self, llm: LLMPrompt, config, embed_model: str = "paraphrase-MiniLM-L6-v2"
    ):
        self._llm = llm
        self._cfg = config
        self._use_embed = _HAS_ST
        self._embedder = None
        if self._use_embed:
            try:
                self._embedder = SentenceTransformer(embed_model)
                if self._cfg.debug:
                    print(f"[AR] Loaded embedding model: {embed_model}")
            except Exception as e:
                if self._cfg.debug:
                    print("[AR] Embedder load failed; using difflib:", e)
                self._use_embed = False

    def generate_plan(
        self,
        ma_domain_file: str,
        ma_problem_file: str,
        ground_domain_file: str,
        ground_problem_file: str,
        max_steps: int,
    ) -> List[str]:
        domain_txt = Path(ma_domain_file).read_text(encoding="utf-8")
        problem_txt = Path(ma_problem_file).read_text(encoding="utf-8")

        # Ground centralized domain
        grounder = ActionGrounder(ground_domain_file, ground_problem_file)
        task = grounder.task

        # Canonical mapping: "navigate rover0 wp4 wp6" -> operator
        ground_ops: Dict[str, object] = {op.name.strip(): op for op in task.operators}

        def applicable(canon: str, facts) -> bool:
            return canon in ground_ops and ground_ops[canon].applicable(facts)

        prompt = (
            "You are a MA-PDDL planning assistant.\n"
            "Return ONE grounded action, end at ')'. No explanations.\n\n"
            f"DOMAIN:\n{domain_txt}\n\nPROBLEM:\n{problem_txt}\n\nPLAN:\n"
        )

        plan: List[str] = []
        current_facts = task.initial_state

        for step in range(1, (max_steps or 999999) + 1):
            if self._cfg.debug and self._cfg.debug_print_prompt:
                print(f"\n===== STEP {step} PROMPT (TAIL) =====")
                tail = prompt.split("PLAN:", 1)[-1]
                print("...PLAN:" + tail[-1000:])
                print("=====================================\n")

            sys = "Output ONE grounded action and stop at the first ')'."
            raw = self._llm.chat(prompt, extra_system=sys, stop=")") or ""
            if not raw.strip().endswith(")"):
                raw = raw.strip() + ")"

            if self._cfg.debug and self._cfg.debug_print_response:
                print(f"[DEBUG] Raw step {step}:\n{raw}")

            # Extract first action
            action_paren = _extract_first_action(raw)
            if not action_paren:
                if self._cfg.debug:
                    print("[AR] Could not extract an action; stopping.")
                break

            canon = _canonical(action_paren)

            if applicable(canon, current_facts):
                accepted_canon = canon
                repair_reason = None
            else:
                # Build list of current applicable canonical actions
                current_applicable = [
                    a for a, op in ground_ops.items() if op.applicable(current_facts)
                ]
                if not current_applicable:
                    if self._cfg.debug:
                        print("[AR] Dead end (no applicable actions). Stopping.")
                    break
                accepted_canon = _similarity_best(
                    canon,
                    current_applicable,
                    self._embedder if self._use_embed else None,
                )
                repair_reason = "inapplicable-or-unknown"

            # Skip consecutive exact duplicate
            if plan and _canonical(plan[-1]) == accepted_canon:
                if self._cfg.debug:
                    print("[AR] Skipping exact duplicate.")
                continue

            # Apply
            op_obj = ground_ops[accepted_canon]
            current_facts = op_obj.apply(current_facts)

            final_action = _single_wrap(accepted_canon)
            plan.append(final_action)
            prompt += "\n" + final_action

            if self._cfg.debug and repair_reason:
                print(f"[AR] Repaired '{canon}' -> '{final_action}' ({repair_reason})")

            # Goal check (pyperplan stores goal as set/list of facts)
            try:
                if all(g in current_facts for g in task.goal):
                    if self._cfg.debug:
                        print("[AR] Goal satisfied; stopping.")
                    break
            except Exception:
                pass

            if max_steps and len(plan) >= max_steps:
                break

        return plan
