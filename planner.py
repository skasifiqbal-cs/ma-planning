import logging
import re
from pathlib import Path
from centralize_ma import MAPDDLConverter
from llm_prompt import LLMPrompt
from ground_actions import ActionGrounder
from validation import Validator


class MAPLLMPipeline:
    def __init__(self, config):
        self.config = config
        self.converter = MAPDDLConverter(
            config.converter["converter_script"],
            config.converter["python_cmd"],
            config.centralized_root,
        )
        self.prompt = LLMPrompt(config)
        self.validator = Validator(config.val_bin)

    def run(self, domain_dir, domain_file, problem_file, mode="no-val", max_steps=None):
        if not max_steps:
            max_steps = self.config.max_steps

        centralized_domain, centralized_problem = self.converter.convert(
            domain_dir, domain_file, problem_file
        )
        print("[INFO] Centralized files:", centralized_domain, centralized_problem)

        if mode == "no-val":
            plan = self._autoregressive_plan_no_val(
                centralized_domain, centralized_problem, max_steps
            )
        else:
            raise NotImplementedError("Only 'no-val' mode implemented.")

        print("\n[RESULT] Final generated plan:")
        for i, act in enumerate(plan):
            print(f"{i}: {act}")

    def _print_prompt(self, text: str, tag: str):
        if getattr(self.config, "debug", False):
            print(f"\n===== {tag} BEGIN =====")
            print(text)
            print(f"===== {tag} END =====\n")

    def _autoregressive_plan_no_val(
        self, domain_file: str, problem_file: str, max_steps: int
    ):
        grounder = ActionGrounder(domain_file, problem_file)
        task = grounder.task
        ground_ops = {op.name: op for op in task.operators}

        current_facts = task.initial_state
        plan: list[str] = []

        for step in range(max_steps):
            applicable = [
                name for name, op in ground_ops.items() if op.applicable(current_facts)
            ]
            if not applicable:
                print("[INFO] No applicable actions remain. Stopping.")
                break

            user_prompt = self._build_llm_prompt(
                domain_file, problem_file, plan, applicable, retry=False
            )
            self._print_prompt(user_prompt, f"LLM PROMPT (step {step})")
            raw_response = self.prompt.chat(user_prompt)

            if not raw_response:
                print("[WARN] LLM returned empty response. Stopping.")
                break

            action = LLMPrompt.extract_first_action(raw_response)
            logging.debug(f"[DEBUG] Raw LLM response: {raw_response}")
            logging.debug(f"[DEBUG] Parsed action: {action}")

            # Validate grounding & applicability
            if (
                not action
                or action not in ground_ops
                or not ground_ops[action].applicable(current_facts)
            ):
                print(f"[WARN] Invalid action from LLM: {action or raw_response!r}")
                # Retry ONCE with explicit invalid notice
                retry_prompt = self._build_llm_prompt(
                    domain_file,
                    problem_file,
                    plan,
                    applicable,
                    retry=True,
                    invalid_action=action or raw_response,
                )
                self._print_prompt(retry_prompt, f"LLM PROMPT RETRY (step {step})")
                retry_response = self.prompt.chat(retry_prompt)
                if not retry_response:
                    print("[ERROR] Retry also returned empty. Aborting.")
                    break
                action = LLMPrompt.extract_first_action(retry_response)
                logging.debug(f"[DEBUG] Retry raw response: {retry_response}")
                logging.debug(f"[DEBUG] Retry parsed action: {action}")
                if (
                    not action
                    or action not in ground_ops
                    or not ground_ops[action].applicable(current_facts)
                ):
                    print(
                        f"[ERROR] Retry produced invalid action: {action or retry_response!r}. Aborting."
                    )
                    break

            # Apply valid action
            ground_op = ground_ops[action]
            current_facts = ground_op.apply(current_facts)
            plan.append(action)
            print(f"[STEP {step}] Added action: {action}")

        return plan

    def _build_llm_prompt(
        self,
        domain_file: str,
        problem_file: str,
        plan: list[str],
        applicable_actions: list[str],
        retry: bool,
        invalid_action: str | None = None,
    ) -> str:
        # Minimize size: we don't need the entire domain every step; truncate.
        domain_txt = Path(domain_file).read_text()
        # domain_txt = ""

        problem_txt = Path(problem_file).read_text()
        # Optional truncation if huge
        # max_chars = 4000
        # if len(domain_txt) > max_chars:
        #     domain_txt = domain_txt[:max_chars] + "\n... [truncated]"
        # if len(problem_txt) > max_chars:
        #     problem_txt = problem_txt[:max_chars] + "\n... [truncated]"

        plan_section = "\n".join(plan) if plan else "(empty)"
        applicable_section = "\n".join(applicable_actions)

        base_instr = (
            "Return EXACTLY ONE next grounded action which is valid from the domain description.\n"
            "Format: (operator arg1 arg2 ...)\n"
            "Do NOT explain. Do NOT output anything else."
        )

        if retry and invalid_action:
            instr = f"{invalid_action} is wrong try something else\n{base_instr}"
        else:
            instr = base_instr

        # prompt = (
        #     f"{instr}\n\nDOMAIN:\n{domain_txt}\n\nPROBLEM:\n{problem_txt}\n\n"
        #     f"CURRENT PLAN:\n{plan_section}\n\nAPPLICABLE ACTIONS:\n{applicable_section}\n\nNEXT ACTION:"
        # )

        prompt = (
            f"{instr}\n\nDOMAIN:\n{domain_txt}\n\nPROBLEM:\n{problem_txt}\n\n"
            f"CURRENT PLAN:\n{plan_section}\n\nNEXT ACTION:"
        )
        return prompt
