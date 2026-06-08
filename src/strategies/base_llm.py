"""Open-loop no-validation strategy with configurable action validation."""

import re
from pathlib import Path
from typing import List, Set


class BaseStrategy:
    """
    Strategy that uses action_validation flag to control behavior:
    - "raw": Use all LLM-generated actions as-is
    - "filter": Remove actions with unsatisfied preconditions
    - "repair": Replace invalid actions with closest valid operator (via similarity)
    """

    def __init__(self, llm, config, **kwargs):
        self.llm = llm
        self.config = config
        # Support both old and new config formats
        self.action_validation = getattr(config, "action_validation", "raw")
        # Store raw LLM output for logging
        self.last_raw_output = None

    def generate_plan(
        self,
        ma_domain_file: str,
        ma_problem_file: str,
        ground_domain_file: str,
        ground_problem_file: str,
        max_steps: int,
    ) -> List[str]:
        """Generate plan using open-loop no-validation approach."""
        from ..utils.grounding import ActionGrounder
        from ..utils.parsing import compress_pddl
        from ..llm.prompts import PlanningPrompts

        # Read MA-PDDL files
        domain_txt = Path(ma_domain_file).read_text(encoding="utf-8")
        problem_txt = Path(ma_problem_file).read_text(encoding="utf-8")

        # Compress PDDL to reduce token count
        compress = getattr(self.config, "compress_pddl", True)
        if compress:
            domain_txt = compress_pddl(
                domain_txt, strip_comments=True, compact_whitespace=True
            )
            problem_txt = compress_pddl(
                problem_txt, strip_comments=True, compact_whitespace=True
            )

        # Try to ground; if domain has unsupported keywords (e.g., :functions), skip grounding
        grounder = None
        valid_ops = set()
        try:
            grounder = ActionGrounder(ground_domain_file, ground_problem_file)
            valid_ops = {self._op_name(op) for op in grounder.operators}
        except Exception as e:
            if self.config.debug:
                print(f"[STRATEGY] Skipping pyperplan grounding: {str(e)[:100]}")
                print("[STRATEGY] Continuing without operator validation...")
            # Will proceed with raw LLM output and action extraction only

        few_shot_examples = []
        if getattr(self.config, "use_few_shot", False):
            few_shot_examples = PlanningPrompts.load_few_shot_examples(
                getattr(self.config, "few_shot_examples_file", None),
                getattr(self.config, "few_shot_example_count", None),
            )

        # Build prompt, optionally with few-shot examples
        template = PlanningPrompts.zero_shot_planning(few_shot_examples)

        messages = template.format(
            debug=getattr(self.config, "debug_prompts", False),
            domain=domain_txt,
            problem=problem_txt,
        )

        if self.config.debug:
            print("[STRATEGY] Using BaseStrategy")
            print(f"[STRATEGY] Number of valid operators: {len(valid_ops)}")
            if few_shot_examples:
                print(f"[STRATEGY] Few-shot examples loaded: {len(few_shot_examples)}")

        # Query LLM with the full prompt, including optional few-shot examples.
        raw_response = self.llm.chat(messages)
        # Store raw output for logging
        self.last_raw_output = raw_response

        if not raw_response:
            if self.config.debug:
                print("[STRATEGY] LLM returned empty response")
            return []

        # Show raw LLM output
        if self.config.debug:
            print("\n" + "=" * 70)
            print("[LLM RAW OUTPUT]")
            print("=" * 70)
            # Limit to first 1500 chars for readability
            preview = raw_response[:1500] if len(raw_response) > 1500 else raw_response
            print(preview)
            if len(raw_response) > 1500:
                print(f"\n... (truncated, total {len(raw_response)} chars) ...")

            # Check if output was truncated/cut off mid-action
            if raw_response.rstrip().endswith(("(", "e", "a", "t", "l", "d", "k")) or (
                raw_response.count("(") > raw_response.count(")") + 1
            ):
                print("\n⚠ WARNING: Output appears to be cut off mid-action!")
                print("  The LLM likely ran out of tokens (max_tokens limit reached)")
                print("  Consider increasing max_tokens in config.yaml")
            print("=" * 70 + "\n")

        # Parse actions - first extract all parenthesized expressions
        from ..utils.parsing import extract_parenthesized_actions

        all_extracted = extract_parenthesized_actions(raw_response)

        if self.config.debug:
            print(
                f"[STRATEGY] Extracted {len(all_extracted)} parenthesized expressions from LLM output"
            )
            if all_extracted:
                print(f"[STRATEGY] First few extracted: {all_extracted[:5]}")

        # If grounding failed, return raw actions (no validation/filtering/repair possible)
        if grounder is None:
            if self.config.debug:
                print(
                    f"[STRATEGY] No grounder available; returning {len(all_extracted)} raw actions"
                )
            return all_extracted

        # Build a map from action string to operator object for quick lookup
        op_map = {self._op_name(op): op for op in grounder.operators}

        if self.config.debug:
            print(f"[DEBUG] valid_ops sample: {list(valid_ops)[:3]}")
            print(f"[DEBUG] op_map sample: {list(op_map.keys())[:3]}")

        # Parse actions using action_validation mode
        if self.action_validation == "filter":
            # Filter mode: remove actions with unsatisfied preconditions
            plan = []
            actions_rejected_reason = []

            for action_str in all_extracted:
                # Find matching operator
                op = op_map.get(action_str)
                if not op:
                    actions_rejected_reason.append((action_str, "not_in_grounded_ops"))
                    continue

                # Check if applicable in current state
                if op.applicable(grounder.state):
                    plan.append(action_str)
                    grounder.apply_action(op)
                    if self.config.debug and len(plan) <= 5:
                        print(f"[FILTER] Action {len(plan)}: {action_str} ✓ applied")
                else:
                    actions_rejected_reason.append(
                        (action_str, "preconditions_not_met")
                    )
                    if self.config.debug:
                        print(
                            f"[FILTER] Action rejected: {action_str} (preconditions not satisfied)"
                        )

            if self.config.debug:
                total_invalid = len(all_extracted) - len(plan)
                if total_invalid > 0:
                    print(
                        f"\n[STRATEGY] ⚠ Filtered out {total_invalid} invalid actions"
                    )
                print(
                    f"[STRATEGY] ✓ Final plan: {len(plan)} valid & applicable actions"
                )
                if plan:
                    print(f"[STRATEGY] First 3 actions: {plan[:3]}")
                    if len(plan) > 1:
                        print(f"[STRATEGY] Last action: {plan[-1]}")

        elif self.action_validation == "repair":
            # Repair mode: use similarity to replace invalid actions
            # Delegate to similarity-based repair strategy
            return self._repair_with_similarity(grounder, all_extracted, max_steps)
        else:
            # Raw mode: use all extracted actions as-is
            plan = all_extracted

            if self.config.debug:
                print(
                    f"[STRATEGY] ✓ Final plan: {len(plan)} actions (raw mode - no filtering)"
                )
                if plan:
                    print(f"[STRATEGY] First 3 actions: {plan[:3]}")
                    if len(plan) > 1:
                        print(f"[STRATEGY] Last action: {plan[-1]}")

        if max_steps > 0:
            plan = plan[:max_steps]

        return plan

    def _repair_with_similarity(self, grounder, all_extracted, max_steps):
        """Repair actions using similarity-based matching."""
        from .llm_repair import LLMRepairStrategy

        strategy = LLMRepairStrategy(
            llm=self.llm,
            config=self.config,
            embed_model=getattr(self.config, "embed_model", "paraphrase-MiniLM-L6-v2"),
        )

        # Generate plan using similarity repair
        # Note: We pass a dummy generate_plan call - reuse the repair logic directly
        return strategy._repair_actions_inline(
            grounder, all_extracted, self.config.debug, max_steps
        )

    def _op_name(self, op) -> str:
        """Extract operator name from pyperplan operator."""
        name = str(op.name) if hasattr(op, "name") else str(op)
        return name.strip()
