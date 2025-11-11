from centralize_ma import MAPDDLConverter
from llm_prompt import LLMPrompt
from ground_actions import ActionGrounder
from validation import Validator, SoftValidator
from config import Config

class MAPLLMPipeline:
    def __init__(self, config):
        self.config = config
        self.converter = MAPDDLConverter(**config.converter)
        self.prompt = LLMPrompt(config.llm_model)
        self.validator = Validator(config.val_bin)
        self.soft_validator = SoftValidator(config.val_bin)

    def run(self, domain_dir, domain_file, problem_file, out_dir, mode="soft-val", max_steps=30):
        # 1. Centralize
        dom_pddl, prob_pddl = self.converter.convert(domain_dir, domain_file, problem_file, out_dir)
        # 2. Ground actions
        grounder = ActionGrounder(dom_pddl, prob_pddl)
        plan_lines = []
        for step in range(max_steps):
            messages = self.build_messages(dom_pddl, prob_pddl, plan_lines)
            llm_output = self.prompt.prompt(messages)
            if llm_output.strip().upper().startswith("DONE"):
                break
            llm_action_str = self.extract_action(llm_output)
            if mode == "soft-val":
                # Choose best corrected applicable action
                chosen = self.soft_validator.correct_action(llm_action_str, grounder.operators, grounder.state)
                if not chosen:
                    break
            else: # strict or no-val
                chosen = llm_action_str
            plan_lines.append(chosen)
            # Apply action to state
            for op in grounder.operators:
                if f"({op.name} {' '.join(op.args)})" == chosen and op.applicable(grounder.state):
                    grounder.apply_action(op)
                    break
            if self.validator.validate_plan(dom_pddl, prob_pddl, plan_lines):
                continue
        # Output plan at the end

    def build_messages(self, dom_pddl, prob_pddl, plan_lines):
        # You might want to load the original MAPDDL as context!
        # Template; adjust as needed
        plan_str = "\n".join(plan_lines)
        return [
            {"role": "system", "content": "You are a PDDL planner..."},
            {"role": "user", "content": f"DOMAIN:\n{dom_pddl}\n\nPROBLEM:\n{prob_pddl}\nCURRENT PLAN:\n{plan_str}\nNEXT ACTION:"}
        ]

    def extract_action(self, llm_output):
        line = llm_output.strip().splitlines()[0].strip()
        m = re.search(r"\([^)]+\)", line)
        return m.group(0) if m else line
