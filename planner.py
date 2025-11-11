from centralize_ma import MAPDDLConverter
from llm_prompt import LLMPrompt
from ground_actions import ActionGrounder
from validation import Validator, SoftValidator


class MAPLLMPipeline:
    def __init__(self, config):
        self.config = config
        self.converter = MAPDDLConverter(
            config.converter["converter_script"],
            config.converter["python_cmd"],
            config.centralized_root,
        )
        self.prompt = LLMPrompt(config.llm_model)
        self.validator = Validator(config.val_bin)
        self.soft_validator = SoftValidator(config.val_bin, config.embed_model)

    def run(self, domain_dir, domain_file, problem_file, mode=None, max_steps=None):
        if mode is None:
            mode = self.config.validation_mode
        if max_steps is None:
            max_steps = self.config.max_steps

        print(
            f"[INFO] Starting run for domain: {domain_dir}, files: {domain_file}, {problem_file}, mode: {mode}, max_steps: {max_steps}"
        )

        # 1. Centralize
        dom_pddl, prob_pddl = self.converter.convert(
            domain_dir, domain_file, problem_file
        )
        print(f"[DEBUG] Centralized files -> domain: {dom_pddl}, problem: {prob_pddl}")

        # 2. Ground actions
        grounder = ActionGrounder(dom_pddl, prob_pddl)
        print(f"[DEBUG] Initial state: {grounder.state}")
        print(f"[DEBUG] Number of operators: {len(grounder.operators)}")
        if len(grounder.operators) > 0:
            print("[DEBUG] First 3 operator representations for inspection:")
            for op in grounder.operators[:3]:
                print("   [OP] Attributes:", vars(op))

        plan_lines = []
        for step in range(max_steps):
            print(f"[STEP {step}] Current state: {grounder.state}")
            messages = self.build_messages(dom_pddl, prob_pddl, plan_lines)
            print(f"[DEBUG] Sending messages to LLM: {messages}")
            llm_output = self.prompt.prompt(messages)
            print(f"[DEBUG] LLM output: {llm_output}")
            if llm_output.strip().upper().startswith("DONE"):
                print(f"[INFO] LLM indicated DONE at step {step}.")
                break
            llm_action_str = self.extract_action(llm_output)
            print(f"[DEBUG] Extracted action: {llm_action_str}")
            if mode == "soft-val":
                chosen = self.soft_validator.correct_action(
                    llm_action_str, grounder.operators, grounder.state
                )
                print(f"[DEBUG] Soft-val chosen action: {chosen}")
                if not chosen:
                    print(
                        f"[WARN] Step {step}: No applicable actions for soft validation, terminating plan."
                    )
                    break
            else:  # strict or no-val
                chosen = llm_action_str
                print(f"[DEBUG] No-val chosen action: {chosen}")
            plan_lines.append(chosen)
            # Apply action to state
            action_applied = False
            for op in grounder.operators:
                if hasattr(op, "name"):
                    op_str = (
                        op.name
                    )  # Operators already include arguments in name format e.g., (pick-up a3 b)
                else:
                    op_str = str(
                        op
                    )  # Fallback to raw string rendering if name is unavailable
                print(f"[DEBUG] Comparing op_str='{op_str}' to chosen='{chosen}'")
                if op_str == chosen and op.applicable(grounder.state):
                    grounder.apply_action(op)
                    action_applied = True
                    print(f"[INFO] Applied action: {op_str}")
                    break
            if not action_applied:
                print(
                    f"[WARN] Step {step}: Action '{chosen}' was not applicable or not found among grounded operators. Skipping."
                )

        print("\n[RESULT] Final plan:")
        for line in plan_lines:
            print(line)

    # Define the build_messages method
    def build_messages(self, dom_pddl, prob_pddl, plan_lines):
        # Load domain and problem file contents for context
        with open(dom_pddl, "r") as domain_file:
            domain_content = domain_file.read()
        with open(prob_pddl, "r") as problem_file:
            problem_content = problem_file.read()

        # Build a prompt for the LLM
        plan_str = "\n".join(f"{i} {line}" for i, line in enumerate(plan_lines))
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert multi-agent planner. Your task is to generate "
                    "an optimal plan for the given domain and problem in PDDL.\n"
                    "Instructions:\n"
                    "- Provide the plan step-by-step in the following format:\n"
                    "  <time step as integer> action(<args> ...)\n"
                    "- Ensure each step is valid based on the PDDL domain and problem files.\n"
                    "- Respond with only the next step in the plan, or indicate DONE if the plan is complete."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"DOMAIN:\n{domain_content}\n"
                    f"PROBLEM:\n{problem_content}\n"
                    f"CURRENT PLAN:\n{plan_str}\n"
                    "NEXT STEP (in format: <time step as integer> action(<args> ...), or DONE):"
                ),
            },
        ]

    def extract_action(self, llm_output):
        import re

        line = llm_output.strip().splitlines()[0].strip()
        m = re.search(r"\([^)]+\)", line)
        return m.group(0) if m else line
