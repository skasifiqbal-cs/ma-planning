import subprocess
from sentence_transformers import SentenceTransformer
import numpy as np

class Validator:
    def __init__(self, val_bin="Validate"):
        self.val_bin = val_bin

    def validate_plan(self, domain_pddl, problem_pddl, plan_lines):
        from tempfile import NamedTemporaryFile
        temp_plan = NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8", suffix=".plan")
        temp_plan.write("\n".join(plan_lines) + "\n")
        temp_plan.flush()
        temp_plan_name = temp_plan.name
        temp_plan.close()
        cmd = [self.val_bin, "-v", domain_pddl, problem_pddl, temp_plan_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return "Plan valid" in result.stdout

class SoftValidator(Validator):
    def __init__(self, val_bin="Validate", embed_model="paraphrase-MiniLM-L6-v2"):
        super().__init__(val_bin)
        self.embedder = SentenceTransformer(embed_model)

    def correct_action(self, llm_action_str, grounded_actions, state):
        # Find best match among applicable actions
        candidates = [f"({op.name} {' '.join(op.args)})" for op in grounded_actions if op.applicable(state)]
        if not candidates:
            return None
        llm_emb = self.embedder.encode([llm_action_str])[0]
        c_embs = self.embedder.encode(candidates)
        sims = [np.dot(llm_emb, ce)/(np.linalg.norm(llm_emb)*np.linalg.norm(ce)) for ce in c_embs]
        best = candidates[np.argmax(sims)]
        return best