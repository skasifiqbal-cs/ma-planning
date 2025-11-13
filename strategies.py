from typing import List, Optional, Protocol
from pathlib import Path

from ground_actions import ActionGrounder
from plan_parser import parse_actions_no_validation
from llm_prompt import LLMPrompt


class PlanGenerationStrategy(Protocol):
    def generate_plan(
        self, domain_file: str, problem_file: str, max_steps: int
    ) -> List[str]: ...


class OpenLoopNoValidationStrategy:
    """
    True llm4pddl-style No Validation (open-loop):
    - Build a single prompt (DOMAIN + PROBLEM + instruction).
    - Ask the LLM for the entire plan in one response.
    - Parse the response to extract actions and skip invalid ones (no retries, no applicability checks).
    - Truncate to max_steps if provided.
    """

    def __init__(self, llm: LLMPrompt, debug: bool = False):
        self._llm = llm
        self._debug = debug

    def generate_plan(
        self, domain_file: str, problem_file: str, max_steps: int
    ) -> List[str]:
        domain_txt = Path(domain_file).read_text()
        problem_txt = Path(problem_file).read_text()

        # Ground once to get the full set of grounded operator names (for name/type filtering only).
        grounder = ActionGrounder(domain_file, problem_file)
        task = grounder.task
        ground_ops = {op.name for op in task.operators}

        prompt = self._build_full_plan_prompt(domain_txt, problem_txt)
        if self._debug:
            print("\n===== LLM OPEN-LOOP PROMPT BEGIN =====")
            print(prompt)
            print("===== LLM OPEN-LOOP PROMPT END =====\n")

        response = self._llm.chat(prompt)
        if response is None:
            if self._debug:
                print("[DEBUG] LLM returned empty response for open-loop plan.")
            return []

        if self._debug:
            print("[DEBUG] Open-loop LLM raw response (full):")
            print(response)

        # Parse the entire response and skip invalid actions by name/type (no applicability).
        plan = parse_actions_no_validation(
            response, valid_ground_ops=ground_ops, disable_name_checks=False
        )

        # Truncate to max_steps if needed.
        if max_steps and len(plan) > max_steps:
            plan = plan[:max_steps]

        if self._debug:
            print("\n[DEBUG] Parsed plan actions (full):")
            for i, a in enumerate(plan):
                print(f"{i}: {a}")
            print()

        return plan

    def _build_full_plan_prompt(self, domain_txt: str, problem_txt: str) -> str:
        instr = (
            "Return a COMPLETE plan as a sequence of grounded actions, one per line.\n"
            "Action format: (operator arg1 arg2 ...)\n"
            "Do NOT explain. Do NOT output anything else. Only the actions."
        )
        return (
            f"{instr}\n\n"
            f"DOMAIN:\n{domain_txt}\n\n"
            f"PROBLEM:\n{problem_txt}\n\n"
            f"PLAN:"
        )


class SoftValidationAutoregressiveStrategy:
    """
    Placeholder for a Soft Validation (autoregressive) strategy:
    - One action at a time.
    - On invalid action, replace with nearest applicable via embeddings (Sentence-BERT).
    - Append action to the prompt and re-query.
    """

    def __init__(self, llm: LLMPrompt, debug: bool = False):
        self._llm = llm
        self._debug = debug

    def generate_plan(
        self, domain_file: str, problem_file: str, max_steps: int
    ) -> List[str]:
        raise NotImplementedError(
            "SoftValidationAutoregressiveStrategy is not implemented yet."
        )
