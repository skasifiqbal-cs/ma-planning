from typing import List, Protocol, Set
from pathlib import Path

from ground_actions import ActionGrounder
from plan_parser import parse_actions_no_validation
from llm_prompt import LLMPrompt


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
    Open-loop plan generation for MA-PDDL:
      - Prompt uses unfactored MA-PDDL domain/problem.
      - Grounding/Filtering uses centralized PDDL (pyperplan).
      - Output is a sequential list of grounded actions, one per line.
      - Each action must be: (operator agent ...other arguments...)
    """

    def __init__(
        self,
        llm: LLMPrompt,
        debug: bool = False,
        show_prompt: bool = False,
        show_response: bool = True,
    ):
        self._llm = llm
        self._debug = debug
        self._show_prompt = show_prompt
        self._show_response = show_response

    def generate_plan(
        self,
        ma_domain_file: str,
        ma_problem_file: str,
        ground_domain_file: str,
        ground_problem_file: str,
        max_steps: int,
    ) -> List[str]:
        # Load MA-PDDL text for the prompt
        domain_txt = Path(ma_domain_file).read_text(encoding="utf-8")
        problem_txt = Path(ma_problem_file).read_text(encoding="utf-8")

        # Ground operator names from centralized PDDL
        grounder = ActionGrounder(ground_domain_file, ground_problem_file)
        task = grounder.task
        op_names: Set[str] = {op.name for op in task.operators}

        # Build MA prompt
        prompt = self._build_ma_prompt(domain_txt, problem_txt)

        # IMPORTANT: print the FULL prompt in debug (restore earlier behavior)
        if self._debug and self._show_prompt:
            print("\n===== LLM PROMPT BEGIN =====")
            print(prompt)
            print("===== LLM PROMPT END =====\n")

        # LLM call
        response = self._llm.chat(prompt)

        if self._debug and self._show_response:
            print("[DEBUG] Open-loop LLM raw response (full):")
            print(response or "")

        # Parse and keep actions whose operator name exists in the centralized ground set
        plan = parse_actions_no_validation(
            response or "", valid_ground_ops=op_names, disable_name_checks=False
        )

        # Truncate if needed
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
