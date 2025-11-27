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


class OpenLoopRandomizedStrategy:
    """
    Single-shot zero-shot open-loop planning with optional randomization.

    Steps:
      1. Randomize domain/problem text (prompt only).
      2. Single LLM call to produce multi-line plan.
      3. Parse each '(...)' action.
      4. Invert randomization to canonical names.
      5. If operator is grounded AND applicable in current state -> accept & apply.
         Otherwise skip.
      6. Early stop if goal satisfied.
      7. Return canonical single-wrapped actions.

    Environment variables:
      MAP_PLANNING_RANDOMIZE_OPERATOR_NAMES=1|0
      MAP_PLANNING_RANDOMIZE_OBJECT_NAMES=1|0
      MAP_PLANNING_RANDOM_SEED=<int>

    No similarity repair, no backtrack suppression.
    """

    def __init__(
        self, llm: LLMPrompt, config, embed_model: str = "paraphrase-MiniLM-L6-v2"
    ):
        self._llm = llm
        self._cfg = config

        # Randomization flags
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

        # Subs
        self._op_subs: Dict[str, str] = {}
        self._obj_subs: Dict[str, str] = {}

    def _create_randomizations(self, ground_ops: Dict[str, object]):
        if self._rand_ops_flag:
            ops = sorted({c.split()[0] for c in ground_ops.keys()})
            self._op_subs = _random_aliases(ops, self._rng)
        else:
            self._op_subs = {}
        if self._rand_objs_flag:
            objs = set()
            for canon in ground_ops.keys():
                for t in canon.split()[1:]:
                    objs.add(t)
            self._obj_subs = _random_aliases(sorted(objs), self._rng)
        else:
            self._obj_subs = {}

    def _randomize_text(self, text: str) -> str:
        return _apply_subs(_apply_subs(text, self._op_subs), self._obj_subs)

    def _invert_llm_action(self, act_paren: str) -> str:
        return _invert_action(act_paren, self._op_subs, self._obj_subs)

    def generate_plan(
        self,
        ma_domain_file: str,
        ma_problem_file: str,
        ground_domain_file: str,
        ground_problem_file: str,
        max_steps: int,
    ) -> List[str]:

        # Read original domain/problem
        domain_txt = Path(ma_domain_file).read_text(encoding="utf-8")
        problem_txt = Path(ma_problem_file).read_text(encoding="utf-8")

        # Ground domain/problem (centralized or original fallback)
        grounder = ActionGrounder(ground_domain_file, ground_problem_file)
        task = grounder.task
        ground_ops: Dict[str, object] = {op.name.strip(): op for op in task.operators}

        def applicable(c: str, facts) -> bool:
            return c in ground_ops and ground_ops[c].applicable(facts)

        # Build randomization maps
        self._create_randomizations(ground_ops)

        # Prompt
        prompt = (
            "You are a zero-shot PDDL planner. Produce a valid sequential plan.\n"
            "Format: one grounded action per line: (operator arg1 arg2 ...).\n"
            "No explanations.\n\n"
            f"DOMAIN:\n{self._randomize_text(domain_txt)}\n\n"
            f"PROBLEM:\n{self._randomize_text(problem_txt)}\n\n"
            "PLAN:\n"
        )

        if self._cfg.debug and getattr(self._cfg, "debug_print_prompt", True):
            print("\n===== OPEN-LOOP RANDOMIZED PROMPT =====")
            print(prompt)
            print("===== END PROMPT =====\n")

        raw = self._llm.chat(prompt) or ""
        if self._cfg.debug and getattr(self._cfg, "debug_print_response", True):
            print("[DEBUG] Raw open-loop response:\n" + raw)

        # Collect all parenthesized segments
        raw_actions = re.findall(r"\([^\(\)]+\)", raw)
        if self._cfg.debug:
            print(f"[OL-RAND] Extracted {len(raw_actions)} raw action strings.")

        current_facts = task.initial_state
        plan: List[str] = []
        steps = 0

        for act in raw_actions:
            if max_steps and steps >= max_steps:
                break
            steps += 1

            canon = _canonical(self._invert_llm_action(act))
            if not canon:
                continue

            if not applicable(canon, current_facts):
                # Skip invalid/inapplicable
                if self._cfg.debug:
                    print(f"[OL-RAND] Skipped (inapplicable/unknown): {canon}")
                continue

            # Apply
            op_obj = ground_ops[canon]
            current_facts = op_obj.apply(current_facts)
            plan.append(_single_wrap(canon))

            if self._cfg.debug:
                print(f"[OL-RAND] Step {steps}: accepted -> {plan[-1]}")

            # Early goal stop
            try:
                if all(g in current_facts for g in task.goal):
                    if self._cfg.debug:
                        print("[OL-RAND] Goal satisfied; truncating remainder.")
                    break
            except Exception:
                pass

        return plan


# ---------- Zero-Shot llm4pddl-style Autoregressive with Randomization ----------


class LLM4PDDLZeroShotAutoregressiveStrategy:
    """
    Zero-shot llm4pddl-style autoregressive:
      - Optional randomization (operator/object tokens to the LLM).
      - One action per step (stop at ')').
      - Similarity repair only when proposed action is unknown or inapplicable.
      - Early stop on goal satisfaction.
      - NO progress-op preference, NO backtrack suppression, NO few-shot.
      - Returns canonical actions.
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
                    print("[AR] Embedder load failed; difflib fallback:", e)
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
            "DOMAIN:\n" + domain_txt + "\n\n"
            "PROBLEM:\n" + problem_txt + "\n\n"
            "A:\n"
        )

    def _create_randomizations(self, ground_ops: Dict[str, object]):
        if self._rand_ops_flag:
            ops = sorted({g.split()[0] for g in ground_ops.keys()})
            self._op_subs = _random_aliases(ops, self._rng)
        else:
            self._op_subs = {}
        if self._rand_objs_flag:
            objs = set()
            for c in ground_ops.keys():
                for t in c.split()[1:]:
                    objs.add(t)
            self._obj_subs = _random_aliases(sorted(objs), self._rng)
        else:
            self._obj_subs = {}

    def _randomize_text(self, text: str) -> str:
        return _apply_subs(_apply_subs(text, self._op_subs), self._obj_subs)

    def _randomize_action_for_prompt(self, canon: str) -> str:
        toks = canon.split()
        if not toks:
            return canon
        op = self._op_subs.get(toks[0], toks[0])
        args = [self._obj_subs.get(a, a) for a in toks[1:]]
        return _single_wrap(" ".join([op] + args))

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
        prompt = self._build_initial_prompt(
            self._randomize_text(domain_txt),
            self._randomize_text(problem_txt),
        )

        canonical_plan: List[str] = []
        current_facts = task.initial_state

        for step in range(1, (max_steps or 999999) + 1):
            if self._cfg.debug and self._cfg.debug_print_prompt:
                print(f"\n===== STEP {step} FULL PROMPT =====")
                print(prompt)
                print("===== END STEP PROMPT =====\n")

            sys = "Return ONE grounded PDDL action (using shown tokens) and stop at ')'. No commentary."
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
                accepted = canon
                repaired = False
            else:
                # pure llm4pddl-style: repair only when invalid/inapplicable
                candidates = [
                    a for a, op in ground_ops.items() if op.applicable(current_facts)
                ]
                if not candidates:
                    if self._cfg.debug:
                        print("[AR] No applicable actions; stopping.")
                    break
                accepted = _similarity_best(
                    canon, candidates, self._embedder if self._use_embed else None
                )
                repaired = True

            # Skip exact duplicate
            if canonical_plan and _canonical(canonical_plan[-1]) == accepted:
                if self._cfg.debug:
                    print("[AR] Skipping duplicate.")
                continue

            # Apply
            current_facts = ground_ops[accepted].apply(current_facts)
            final = _single_wrap(accepted)
            canonical_plan.append(final)
            prompt += self._randomize_action_for_prompt(accepted) + "\n"

            if self._cfg.debug and repaired:
                print(f"[AR] Repaired '{canon}' -> '{accepted}'")

            # Early stop on goal
            try:
                if all(g in current_facts for g in task.goal):
                    if self._cfg.debug:
                        print("[AR] Goal satisfied; stopping.")
                    break
            except Exception:
                pass

            if max_steps and len(canonical_plan) >= max_steps:
                break

        return canonical_plan


class OpenLoopSimilarityRepairStrategy:
    """
    Single-shot plan generation + post-hoc repair (no heuristics):
      - Randomization (prompt only, reversible) via env:
          MAP_PLANNING_RANDOMIZE_OPERATOR_NAMES=1|0
          MAP_PLANNING_RANDOMIZE_OBJECT_NAMES=1|0
          MAP_PLANNING_RANDOM_SEED=<int>
      - Single LLM call to produce multi-line plan
      - For each line: invert aliases -> canonical; if applicable -> accept; else repair to most similar applicable
      - Early stop on goal satisfaction
      - Returns canonical plan (single-wrapped) suitable for validation
    """

    def __init__(
        self, llm: LLMPrompt, config, embed_model: str = "paraphrase-MiniLM-L6-v2"
    ):
        self._llm = llm
        self._cfg = config

        # Embedding similarity (optional)
        self._use_embed = _HAS_ST
        self._embedder = None
        if self._use_embed:
            try:
                self._embedder = SentenceTransformer(embed_model)
                if self._cfg.debug:
                    print(f"[REPAIR] Loaded embedding model: {embed_model}")
            except Exception as e:
                if self._cfg.debug:
                    print("[REPAIR] Embed model load failed; difflib fallback:", e)
                self._use_embed = False

        # Randomization flags
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

    def _create_randomizations(self, ground_ops: Dict[str, object]):
        if self._rand_ops_flag:
            ops = sorted({g.split()[0] for g in ground_ops.keys()})
            self._op_subs = _random_aliases(ops, self._rng)
        else:
            self._op_subs = {}
        if self._rand_objs_flag:
            objs = set()
            for c in ground_ops.keys():
                toks = c.split()
                for t in toks[1:]:
                    objs.add(t)
            self._obj_subs = _random_aliases(sorted(objs), self._rng)
        else:
            self._obj_subs = {}

    def _randomize_text(self, text: str) -> str:
        return _apply_subs(_apply_subs(text, self._op_subs), self._obj_subs)

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

        # Read visible (MA) domain/problem
        domain_txt = Path(ma_domain_file).read_text(encoding="utf-8")
        problem_txt = Path(ma_problem_file).read_text(encoding="utf-8")

        # Ground centralized (or original fallback) domain/problem
        grounder = ActionGrounder(ground_domain_file, ground_problem_file)
        task = grounder.task
        ground_ops: Dict[str, object] = {op.name.strip(): op for op in task.operators}

        def applicable(canon: str, facts) -> bool:
            return canon in ground_ops and ground_ops[canon].applicable(facts)

        # Randomize prompt text (LLM-facing only)
        self._create_randomizations(ground_ops)
        prompt = (
            "You are an expert PDDL planner.\n"
            "Produce a valid sequential plan (one grounded action per line).\n"
            "Format: (operator arg1 arg2 ...)\n"
            "No explanations.\n\n"
            f"DOMAIN:\n{self._randomize_text(domain_txt)}\n\n"
            f"PROBLEM:\n{self._randomize_text(problem_txt)}\n\nPLAN:\n"
        )

        if self._cfg.debug and getattr(self._cfg, "debug_print_prompt", True):
            print("\n===== OPEN-LOOP REPAIR PROMPT =====")
            print(prompt)
            print("===== END PROMPT =====\n")

        raw = self._llm.chat(prompt) or ""
        if self._cfg.debug and getattr(self._cfg, "debug_print_response", True):
            print("[DEBUG] Raw open-loop response:\n" + raw)

        raw_actions = re.findall(r"\([^\(\)]+\)", raw)
        if self._cfg.debug:
            print(f"[REPAIR] Extracted {len(raw_actions)} raw action strings.")

        current_facts = task.initial_state
        repaired_plan: List[str] = []
        steps = 0

        for raw_act in raw_actions:
            if max_steps and steps >= max_steps:
                break
            steps += 1

            # Invert randomization to canonical before checks
            canon = _canonical(self._invert_llm_action(raw_act))
            if not canon:
                continue

            if applicable(canon, current_facts):
                accepted = canon
                status = "ok"
            else:
                # NO heuristics: repair among all applicable actions (no category preference)
                applicable_pool = [
                    a for a, op in ground_ops.items() if op.applicable(current_facts)
                ]
                if not applicable_pool:
                    if self._cfg.debug:
                        print("[REPAIR] Dead end; stopping.")
                    break
                accepted = _similarity_best(
                    canon, applicable_pool, self._embedder if self._use_embed else None
                )
                status = "repaired"

            # Apply
            op_obj = ground_ops.get(accepted)
            if not op_obj:
                if self._cfg.debug:
                    print(f"[REPAIR] Missing operator object for {accepted}")
                continue

            current_facts = op_obj.apply(current_facts)
            repaired_plan.append(_single_wrap(accepted))

            if self._cfg.debug:
                print(f"[REPAIR] Step {steps}: {status} -> {repaired_plan[-1]}")

            # Early stop on goal
            try:
                if all(g in current_facts for g in task.goal):
                    if self._cfg.debug:
                        print("[REPAIR] Goal satisfied; stopping.")
                    break
            except Exception:
                pass

        return repaired_plan
