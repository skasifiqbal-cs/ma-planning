from centralize_ma import MAPDDLConverter
from llm_prompt import LLMPrompt  # Updated import
from ground_actions import ActionGrounder
from validation import Validator
from pathlib import Path
import logging


class MAPLLMPipeline:
    def __init__(self, config):
        self.config = config
        self.converter = MAPDDLConverter(
            config.converter["converter_script"],
            config.converter["python_cmd"],
            config.centralized_root,
        )
        # Initialize LLMPrompt with the Config instance
        self.prompt = LLMPrompt(config)
        self.validator = Validator(config.val_bin)

    def run(self, domain_dir, domain_file, problem_file, mode="no-val", max_steps=None):
        if not max_steps:
            max_steps = self.config.max_steps

        # Centralize the PDDL domain/problem files
        centralized_domain, centralized_problem = self.converter.convert(
            domain_dir, domain_file, problem_file
        )
        print("[INFO] Centralized files:", centralized_domain, centralized_problem)

        # Handle no-val separately from other modes
        if mode == "no-val":
            plan = self._autoregressive_plan_no_val(
                centralized_domain, centralized_problem, max_steps
            )
        else:
            raise NotImplementedError(
                "[ERROR] Only 'no-val' mode is implemented right now."
            )

        # Final validated plan
        print("\n[RESULT] Final generated plan:")
        for i, action in enumerate(plan):
            print(f"{i}: {action}")

    def _prompt_autoregressive(self, prompt: str, task: dict, max_loops: int) -> list:
        """Prompt the LLM for one action at a time, handling plain text responses."""
        # Ground the task and initialize operators
        grounder = ActionGrounder(task["domain_file"], task["problem_file"])
        pyperplan_task = grounder.task
        ground_ops = {o.name: o for o in pyperplan_task.operators}
        current_facts = pyperplan_task.initial_state

        plan = []  # A growing list of valid actions
        sep = "\n"  # Default separator for prompt updates

        # Iteratively query the LLM for max_loops times
        for _ in range(max_loops):
            # Query the LLM for the next action
            response = self._query_llm(prompt)
            if not response:
                print("[WARN] LLM returned an empty response.")
                break

            # Parse action from plain text response
            action = response.strip()
            logging.info(f"[DEBUG] LLM Suggested Action: {action}")

            # Validate Action Applicability
            if action not in ground_ops or not ground_ops[action].applicable(
                current_facts
            ):
                print(f"[WARN] Action '{action}' is invalid. Prompting LLM to redo.")
                prompt = (
                    prompt
                    + f"\n[NOTE]: Action '{action}' is invalid. Please suggest a valid next action."
                )
                response = self._query_llm(prompt)  # Ask the LLM to retry
                if not response or response.strip() not in ground_ops:
                    print("[ERROR] LLM failed to generate a valid action. Exiting.")
                    break
                action = response.strip()

            # Apply the action and update the prompt
            ground_op = ground_ops[action]
            current_facts = ground_op.apply(current_facts)
            plan.append(action)  # Add to the plan
            prompt += sep + action  # Update the prompt with the latest action

        return plan

    def _autoregressive_plan_no_val(
        self, domain_file: str, problem_file: str, max_steps: int
    ):
        """Simple autoregressive plan generation for no-val mode."""
        task = {"domain_file": domain_file, "problem_file": problem_file}
        prompt = self._prepare_prompt(domain_file, problem_file, [])
        return self._prompt_autoregressive(prompt, task, max_steps)

    def _prepare_prompt(self, domain_file: str, problem_file: str, plan: list) -> str:
        """Prepare the LLM prompt based on the domain/problem context and the current validated plan."""
        plan_str = "\n".join(f"{i}: {action}" for i, action in enumerate(plan))
        domain_content = Path(domain_file).read_text()
        problem_content = Path(problem_file).read_text()

        prompt = (
            "You are a PDDL planner assisting in generating a valid task plan.\n"
            "DOMAIN:\n"
            + domain_content
            + "\nPROBLEM:\n"
            + problem_content
            + "\nCURRENT PLAN:\n"
            + plan_str
            + "\nNEXT ACTION:"
        )
        return prompt

    def _query_llm(self, prompt: str) -> str:
        """Query the LLM for the next action, handling plain text responses."""
        response = self.prompt.prompt(prompt)
        if response:
            return response.strip()  # Process plain text response
        return ""
