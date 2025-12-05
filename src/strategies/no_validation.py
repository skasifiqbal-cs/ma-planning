"""Open-loop no-validation strategy."""

import re
from pathlib import Path
from typing import List, Set


class OpenLoopNoValidationStrategy:
    """
    No-validation strategy: prompt with MA-PDDL, parse, filter by operator names.
    Actions must be: (operator agent ...other args...)

    Can optionally filter by state (preconditions) using state_based_validation flag.
    """

    def __init__(self, llm, config, state_based_validation=False):
        self.llm = llm
        self.config = config
        self.state_based_validation = state_based_validation

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
        from ..utils.parsing import parse_actions_no_validation
        from ..llm.prompts import PromptRegistry

        # Read MA-PDDL files
        domain_txt = Path(ma_domain_file).read_text(encoding="utf-8")
        problem_txt = Path(ma_problem_file).read_text(encoding="utf-8")

        # Ground to get valid operator names
        grounder = ActionGrounder(ground_domain_file, ground_problem_file)
        valid_ops = {self._op_name(op) for op in grounder.operators}

        # Build prompt using registry with debug flag
        template = PromptRegistry.get("zero-shot")

        messages = template.format(
            debug=getattr(self.config, "debug_prompts", False),
            domain=domain_txt,
            problem=problem_txt,
        )

        if self.config.debug:
            print("[STRATEGY] Using OpenLoopNoValidationStrategy")
            print(f"[STRATEGY] Number of valid operators: {len(valid_ops)}")

        # Extract user message content (skip system message)
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        # Query LLM - reconstruct messages for provider
        provider_messages = [
            {
                "role": "system",
                "content": "You are an expert PDDL planning agent. Output only valid PDDL actions.",
            },
            {"role": "user", "content": user_message},
        ]
        raw_response = self.llm.chat(provider_messages)
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

        # Build a map from action string to operator object for quick lookup
        op_map = {self._op_name(op): op for op in grounder.operators}

        if self.config.debug:
            print(f"[DEBUG] valid_ops sample: {list(valid_ops)[:3]}")
            print(f"[DEBUG] op_map sample: {list(op_map.keys())[:3]}")

        # Parse actions using either state-based or name-based filtering
        if self.state_based_validation:
            # State-aware filtering: check preconditions for each action
            plan = []
            actions_rejected_reason = []  # Track why actions were rejected

            for action_str in all_extracted:
                # Skip if doesn't match valid operator
                if action_str not in valid_ops:
                    actions_rejected_reason.append((action_str, "not_in_grounded_ops"))
                    continue

                # Find the operator
                op = op_map.get(action_str)
                if not op:
                    actions_rejected_reason.append((action_str, "operator_not_found"))
                    continue

                # Check if applicable in current state
                if op.applicable(grounder.state):
                    plan.append(action_str)
                    grounder.apply_action(op)
                    if self.config.debug and len(plan) <= 5:
                        print(f"[STATE] Action {len(plan)}: {action_str} ✓ applied")
                else:
                    actions_rejected_reason.append(
                        (action_str, "preconditions_not_met")
                    )
                    if self.config.debug:
                        print(
                            f"[STATE] Action rejected: {action_str} (preconditions not satisfied)"
                        )

            if self.config.debug:
                total_invalid = len(all_extracted) - len(plan)
                if total_invalid > 0:
                    print(f"\n[STRATEGY] ⚠ Rejected {total_invalid} actions:")

                    # Group by rejection reason
                    by_reason = {}
                    for action_str, reason in actions_rejected_reason:
                        if reason not in by_reason:
                            by_reason[reason] = []
                        by_reason[reason].append(action_str)

                    reason_names = {
                        "not_in_grounded_ops": "Not in valid grounded operators",
                        "operator_not_found": "Operator lookup failed",
                        "preconditions_not_met": "Preconditions not satisfied in current state",
                    }

                    for reason, actions_list in by_reason.items():
                        reason_name = reason_names.get(reason, reason)
                        print(f"  {reason_name}: {len(actions_list)}")
                        for action_str in actions_list[:3]:
                            print(f"    - {action_str}")
                        if len(actions_list) > 3:
                            print(f"    ... and {len(actions_list) - 3} more")

                print(
                    f"\n[STRATEGY] ✓ Final plan: {len(plan)} valid & applicable actions"
                )
                if plan:
                    print(f"[STRATEGY] First 3 actions: {plan[:3]}")
                    if len(plan) > 1:
                        print(f"[STRATEGY] Last action: {plan[-1]}")
        else:
            # Simple name-based filtering (legacy behavior) - accept all extracted actions
            # without checking against grounded operators
            plan = all_extracted

            if self.config.debug:
                invalid_count = len(all_extracted) - len(plan)
                if invalid_count > 0:
                    print(
                        f"[STRATEGY] ⚠ Filtered out {invalid_count} invalid actions (not in grounded operators)"
                    )
                print(
                    f"[STRATEGY] ✓ Final plan: {len(plan)} valid actions (state-based validation disabled)"
                )
                if plan:
                    print(f"[STRATEGY] First 3 actions: {plan[:3]}")
                    if len(plan) > 1:
                        print(f"[STRATEGY] Last action: {plan[-1]}")

        if max_steps > 0:
            plan = plan[:max_steps]

        return plan

    def _op_name(self, op) -> str:
        """Extract operator name from pyperplan operator."""
        name = str(op.name) if hasattr(op, "name") else str(op)
        return name.strip()
