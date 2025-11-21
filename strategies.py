import random
from random import Random
from typing import List, Protocol, Set, Dict
from pathlib import Path
import os
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
    raw = raw.strip()
    m = re.search(r"\([^\(\)]+\)", raw)
    if m:
        return _single_wrap(m.group(0))
    line = raw.splitlines()[0] if raw else ""
    if ")" in line:
        prefix = line.split(")", 1)[0]
    else:
        prefix = line
    toks = prefix.strip().split()
    if not toks:
        return ""
    return _single_wrap("(" + " ".join(toks) + ")")


def _single_wrap(a: str) -> str:
    inner = a.strip()
    while inner.startswith("(") and inner.endswith(")"):
        tmp = inner[1:-1].strip()
        if not tmp:
            break
        inner = tmp
    inner = re.sub(r"\s+", " ", inner)
    return f"({inner})" if inner else ""


def _canonical(a: str) -> str:
    s = a.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return re.sub(r"\s+", " ", s)


def _similarity_best(query: str, pool: List[str], embedder=None) -> str:
    if not pool:
        return ""
    if embedder is not None:
        qvec = embedder.encode([query], convert_to_tensor=True)[0]
        pvecs = embedder.encode(pool, convert_to_tensor=True, show_progress_bar=False)
        sims = st_util.cos_sim(qvec, pvecs)[0].cpu().tolist()
        return pool[max(range(len(pool)), key=lambda i: sims[i])]
    scored = [
        (difflib.SequenceMatcher(None, query, cand).ratio(), cand) for cand in pool
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _random_aliases(names: List[str], rng: Random) -> Dict[str, str]:
    subs: Dict[str, str] = {}
    charset = "abcdefghijklmnopqrstuvwxyz0123456789"
    for name in sorted(names):
        token = rng.choice("abcdefghijklmnopqrstuvwxyz")
        token += "".join(rng.choice(charset) for _ in range(5))
        subs[name] = token
    return subs


def _apply_subs(text: str, subs: Dict[str, str]) -> str:
    if not subs:
        return text
    pattern = re.compile(
        r"\b("
        + "|".join(re.escape(k) for k in sorted(subs, key=len, reverse=True))
        + r")\b"
    )
    return pattern.sub(lambda m: subs[m.group(0)], text)


def _invert_action(
    action_paren: str, op_subs: Dict[str, str], obj_subs: Dict[str, str]
) -> str:
    canon = _canonical(action_paren)
    toks = canon.split()
    if not toks:
        return canon
    op, args = toks[0], toks[1:]
    rev_op = {v: k for k, v in op_subs.items()}
    rev_obj = {v: k for k, v in obj_subs.items()}
    if op in rev_op:
        op = rev_op[op]
    args = [rev_obj.get(a, a) for a in args]
    return " ".join([op] + args)


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


# ---------- Zero-Shot llm4pddl-style Autoregressive with Randomization ----------


class LLM4PDDLZeroShotAutoregressiveStrategy:
    """
    Zero-shot llm4pddl-style autoregressive with optional randomization:
      - Randomization shown only to LLM.
      - Canonical plan is stored & returned.
      - Full prompt printed each step.
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

        self._rand_ops_flag = os.getenv(
            "MAP_PLANNING_RANDOMIZE_OPERATOR_NAMES", "0"
        ).lower() in {"1", "true", "yes", "on"}
        self._rand_objs_flag = os.getenv(
            "MAP_PLANNING_RANDOMIZE_OBJECT_NAMES", "0"
        ).lower() in {"1", "true", "yes", "on"}
        seed_env = os.getenv("MAP_PLANNING_RANDOM_SEED", "0")
        try:
            self._rand_seed = int(seed_env)
        except ValueError:
            self._rand_seed = 0
        self._rng = Random(self._rand_seed)

        self._op_subs: Dict[str, str] = {}
        self._obj_subs: Dict[str, str] = {}

    def _build_initial_prompt(self, domain_txt: str, problem_txt: str) -> str:
        return (
            "Q:\n"
            "DOMAIN:\n"
            f"{domain_txt}\n\n"
            "PROBLEM:\n"
            f"{problem_txt}\n\n"
            "A:\n"
        )

    def _create_randomizations(self, ground_ops: Dict[str, object]):
        if self._rand_ops_flag:
            op_names = sorted({op.split()[0] for op in ground_ops.keys()})
            self._op_subs = _random_aliases(op_names, self._rng)
        else:
            self._op_subs = {}
        if self._rand_objs_flag:
            obj_names = set()
            for canon in ground_ops.keys():
                toks = canon.split()
                for t in toks[1:]:
                    obj_names.add(t)
            self._obj_subs = _random_aliases(sorted(obj_names), self._rng)
        else:
            self._obj_subs = {}

    def _randomize_action_for_prompt(self, canonical_action: str) -> str:
        toks = canonical_action.split()
        if not toks:
            return canonical_action
        op = toks[0]
        args = toks[1:]
        op_out = self._op_subs.get(op, op)
        args_out = [self._obj_subs.get(a, a) for a in args]
        return _single_wrap(f"{op_out} " + " ".join(args_out))

    def _invert_llm_action(self, llm_action_paren: str) -> str:
        return _invert_action(llm_action_paren, self._op_subs, self._obj_subs)

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

        ground_ops: Dict[str, object] = {op.name.strip(): op for op in task.operators}

        def applicable(c: str, facts) -> bool:
            return c in ground_ops and ground_ops[c].applicable(facts)

        self._create_randomizations(ground_ops)
        domain_txt_rand = _apply_subs(
            _apply_subs(domain_txt, self._op_subs), self._obj_subs
        )
        problem_txt_rand = _apply_subs(
            _apply_subs(problem_txt, self._op_subs), self._obj_subs
        )

        prompt = self._build_initial_prompt(domain_txt_rand, problem_txt_rand)

        plan_display: List[str] = []  # randomized (shown to LLM)
        plan_canonical: List[str] = []  # original (returned/saved)
        current_facts = task.initial_state

        for step in range(1, (max_steps or 999999) + 1):
            if self._cfg.debug and self._cfg.debug_print_prompt:
                print(f"\n===== STEP {step} FULL PROMPT =====")
                print(prompt)
                print("===== END STEP PROMPT =====\n")

            sys = "Return ONE grounded PDDL plan action (using shown names) and stop at ')'. No commentary."
            raw = self._llm.chat(prompt, extra_system=sys, stop=")") or ""
            if not raw.strip().endswith(")"):
                raw = raw.strip() + ")"

            if self._cfg.debug and self._cfg.debug_print_response:
                print(f"[DEBUG] Raw step {step}:\n{raw}")

            act_rand = _extract_first_action(raw)
            if not act_rand:
                if self._cfg.debug:
                    print("[AR] No action parsed; stopping.")
                break

            canon = _canonical(self._invert_llm_action(act_rand))

            if applicable(canon, current_facts):
                accepted_canon = canon
                repaired = False
            else:
                candidates = [
                    a for a, op in ground_ops.items() if op.applicable(current_facts)
                ]
                if not candidates:
                    if self._cfg.debug:
                        print("[AR] No applicable actions; stopping.")
                    break
                accepted_canon = _similarity_best(
                    canon, candidates, self._embedder if self._use_embed else None
                )
                repaired = True

            if plan_canonical and plan_canonical[-1] == accepted_canon:
                if self._cfg.debug:
                    print("[AR] Skipping duplicate.")
                continue

            op_obj = ground_ops[accepted_canon]
            current_facts = op_obj.apply(current_facts)

            plan_canonical.append(accepted_canon)
            randomized_action = self._randomize_action_for_prompt(accepted_canon)
            plan_display.append(randomized_action)
            prompt += randomized_action + "\n"

            if self._cfg.debug and repaired:
                print(
                    f"[AR] Repaired '{canon}' -> '{accepted_canon}' (shown as {randomized_action})"
                )

            try:
                if all(g in current_facts for g in task.goal):
                    if self._cfg.debug:
                        print("[AR] Goal satisfied; stopping.")
                    break
            except Exception:
                pass

            if max_steps and len(plan_canonical) >= max_steps:
                break

        # Return canonical actions, single-wrapped, suitable for saving & validation
        return [_single_wrap(a) for a in plan_canonical]
