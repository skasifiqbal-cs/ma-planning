"""Action repair strategy using cosine similarity to fix invalid actions."""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from functools import lru_cache


class LLMRepairStrategy:
    """
    Repair invalid actions by finding closest valid grounded operator using cosine similarity.

    Process:
    1. Extract actions from LLM output
    2. Check each action against current state
    3. If preconditions not met, find closest valid operator using embeddings
    4. Replace with closest valid action
    5. Apply and continue
    """

    def __init__(
        self, llm, config, embed_model: str = "paraphrase-MiniLM-L6-v2", **kwargs
    ):
        """
        Initialize repair strategy.

        Args:
            llm: LLM client
            config: Configuration object
            embed_model: Name of pretrained transformer model for embeddings
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.llm = llm
        self.config = config
        self.embed_model_name = embed_model
        self._embedder = None
        self._embedding_cache: Dict[str, any] = {}
        # Store raw LLM output for logging
        self.last_raw_output = None

    @property
    def embedder(self):
        """Lazy load the embedder."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.embed_model_name)
            except ImportError:
                raise ImportError(
                    f"sentence-transformers not installed. "
                    f"Install with: pip install sentence-transformers"
                )
        return self._embedder

    def _repair_actions_inline(
        self, grounder, all_extracted, debug, max_steps
    ) -> List[str]:
        """
        Repair actions using similarity without re-running full generation.
        Used when action_validation mode is set to 'repair'.
        """
        # Precompute embeddings for all grounded operators
        if debug:
            print("[STRATEGY] Computing embeddings for grounded operators...")
        operator_embeddings = self._get_operator_embeddings(grounder.operators)

        # Repair and apply actions sequentially
        plan = []
        repairs_made = []

        for i, action_str in enumerate(all_extracted):
            if len(plan) >= max_steps > 0:
                if debug:
                    print(f"[STRATEGY] Reached max_steps limit ({max_steps})")
                break

            # Try to find matching operator
            matching_op = self._find_matching_operator(action_str, grounder.operators)

            if matching_op and matching_op.applicable(grounder.state):
                # Action is valid in current state
                plan.append(action_str)
                grounder.apply_action(matching_op)
                if debug and len(plan) <= 5:
                    print(f"[REPAIR] Action {len(plan)}: {action_str} ✓ (direct match)")
            else:
                # Find closest valid action using similarity
                best_op, similarity = self._find_closest_applicable_operator(
                    action_str, grounder.operators, grounder.state, operator_embeddings
                )

                if best_op:
                    # Use the repaired action
                    repaired_action = self._op_to_action_str(best_op)
                    plan.append(repaired_action)
                    repairs_made.append(
                        {
                            "original": action_str,
                            "repaired": repaired_action,
                            "similarity": similarity,
                        }
                    )
                    grounder.apply_action(best_op)
                    if debug:
                        print(
                            f"[REPAIR] Action {len(plan)}: {repaired_action} "
                            f"(repaired from {action_str}, sim={similarity:.3f})"
                        )
                else:
                    # No applicable action found - skip
                    if debug:
                        print(
                            f"[REPAIR] Action skipped: {action_str} (no applicable operators)"
                        )

        if debug:
            print(f"\n[STRATEGY] ✓ Final plan: {len(plan)} actions")
            if repairs_made:
                print(f"[STRATEGY] ✓ Repairs made: {len(repairs_made)}")
                for repair in repairs_made:
                    print(
                        f"  - {repair['original']} → {repair['repaired']} "
                        f"(similarity: {repair['similarity']:.3f})"
                    )
            if plan:
                print(f"[STRATEGY] First 3 actions: {plan[:3]}")
                if len(plan) > 1:
                    print(f"[STRATEGY] Last action: {plan[-1]}")

        return plan

    def generate_plan(
        self,
        ma_domain_file: str,
        ma_problem_file: str,
        ground_domain_file: str,
        ground_problem_file: str,
        max_steps: int,
    ) -> List[str]:
        """Generate plan using similarity-based action repair."""
        from ..utils.grounding import ActionGrounder
        from ..utils.parsing import extract_parenthesized_actions, compress_pddl
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

        # Check if we should skip pyperplan grounding
        skip_pyperplan = getattr(self.config, "skip_pyperplan", False)

        if skip_pyperplan:
            # Skip pyperplan - just get LLM output and return raw actions
            if self.config.debug:
                print("\n" + "=" * 70)
                print(
                    "[STRATEGY] Using LLMRepairStrategy (PYPERPLAN SKIPPED)"
                )
                print("[STRATEGY] Numeric domain mode: LLM → VAL validation only")
                print("=" * 70)

            # Build prompt
            few_shot_examples = []
            if getattr(self.config, "use_few_shot", False):
                few_shot_examples = PlanningPrompts.load_few_shot_examples(
                    getattr(self.config, "few_shot_examples_file", None),
                    getattr(self.config, "few_shot_example_count", None),
                )

            template = PlanningPrompts.zero_shot_planning(few_shot_examples)
            messages = template.format(
                debug=getattr(self.config, "debug_prompts", False),
                domain=domain_txt,
                problem=problem_txt,
            )

            # Get LLM output
            raw_output = self.llm.chat(messages)

            # Extract actions
            actions = extract_parenthesized_actions(raw_output)

            if self.config.debug:
                print(f"\n[STRATEGY] Extracted {len(actions)} actions from LLM")
                print(f"[STRATEGY] Actions will be validated by VAL only")

            return actions

        # Ground to get valid operators (normal pyperplan mode)
        grounder = ActionGrounder(ground_domain_file, ground_problem_file)

        if self.config.debug:
            print("\n" + "=" * 70)
            print("[STRATEGY] Using LLMRepairStrategy")
            print(f"[STRATEGY] Total grounded operators: {len(grounder.operators)}")
            print(f"[STRATEGY] Embedding model: {self.embed_model_name}")
            print("=" * 70)

        # Build prompt
        few_shot_examples = []
        if getattr(self.config, "use_few_shot", False):
            few_shot_examples = PlanningPrompts.load_few_shot_examples(
                getattr(self.config, "few_shot_examples_file", None),
                getattr(self.config, "few_shot_example_count", None),
            )

        template = PlanningPrompts.zero_shot_planning(few_shot_examples)
        messages = template.format(
            debug=getattr(self.config, "debug_prompts", False),
            domain=domain_txt,
            problem=problem_txt,
        )

        # Display formatted prompt if debug
        if self.config.debug:
            print("\n" + "=" * 70)
            print("SYSTEM MESSAGE")
            print("=" * 70)
            for msg in messages:
                if msg.get("role") == "system":
                    system_content = msg.get("content", "")
                    preview = (
                        system_content[:500]
                        if len(system_content) > 500
                        else system_content
                    )
                    print(preview)
                    if len(system_content) > 500:
                        print(f"\n... (system message truncated) ...\n")
                    break

            print("\n" + "=" * 70)
            print("USER MESSAGE")
            print("=" * 70)
            for msg in messages:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    preview = (
                        user_content[:1000]
                        if len(user_content) > 1000
                        else user_content
                    )
                    print(preview)
                    if len(user_content) > 1000:
                        print(
                            f"\n... (user message truncated, total {len(user_content)} chars) ...\n"
                        )
                    break
            print("=" * 70)

        # Extract user message
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        # Query LLM
        provider_messages = [
            {
                "role": "system",
                "content": "You are an expert PDDL planning agent. Output only valid action sequences.",
            },
            {"role": "user", "content": user_message},
        ]
        raw_response = self.llm.chat(provider_messages)
        # Store raw output for logging
        self.last_raw_output = raw_response

        if not raw_response:
            if self.config.debug:
                print("[STRATEGY] LLM returned empty response")
            return []

        if self.config.debug:
            print("\n" + "=" * 70)
            print("LLM RAW OUTPUT")
            print("=" * 70)
            preview = raw_response[:1500] if len(raw_response) > 1500 else raw_response
            print(preview)
            if len(raw_response) > 1500:
                print(f"\n... (truncated, total {len(raw_response)} chars) ...")
            print("=" * 70)

        # Extract actions
        all_extracted = extract_parenthesized_actions(raw_response)
        if self.config.debug:
            print(
                f"\n[STRATEGY] Extracted {len(all_extracted)} actions from LLM output"
            )

        # Precompute embeddings for all grounded operators
        if self.config.debug:
            print("[STRATEGY] Computing embeddings for grounded operators...")
        operator_embeddings = self._get_operator_embeddings(grounder.operators)

        # Repair and apply actions sequentially with state tracking
        plan = []
        repairs_made = []

        # Show initial state - all applicable actions from initial state
        show_operators = self.config.debug and getattr(
            self.config, "show_applicable_operators", True
        )
        if show_operators:
            print("\n" + "╔" + "═" * 68 + "╗")
            print("║" + " " * 15 + "STATE 0 - INITIAL STATE" + " " * 30 + "║")
            print("║" + " " * 10 + "Available Grounded Actions:" + " " * 31 + "║")
            print("╠" + "═" * 68 + "╣")
            applicable_ops = [
                op for op in grounder.operators if op.applicable(grounder.state)
            ]
            for j, op in enumerate(applicable_ops[:15], 1):
                op_str = self._op_to_action_str(op)
                op_display = f"  {j}. {op_str}"
                padding = " " * (68 - len(op_display))
                print("║" + op_display + padding + "║")
            if len(applicable_ops) > 15:
                remaining = f"  ... +{len(applicable_ops) - 15} more"
                padding = " " * (68 - len(remaining))
                print("║" + remaining + padding + "║")
            total_str = f"  Total: {len(applicable_ops)} operators"
            padding = " " * (68 - len(total_str))
            print("╚" + "═" * 68 + "╝")
            print()

        for i, action_str in enumerate(all_extracted):
            if len(plan) >= max_steps > 0:
                if self.config.debug:
                    print(f"[STRATEGY] Reached max_steps limit ({max_steps})")
                break

            # Try to find matching operator
            matching_op = self._find_matching_operator(action_str, grounder.operators)

            if matching_op and matching_op.applicable(grounder.state):
                # Action is valid in current state
                plan.append(action_str)
                grounder.apply_action(matching_op)
                if self.config.debug:
                    print(f"[REPAIR] Action {len(plan)}: {action_str} ✓ (direct match)")
                    # Show next state only every 3 actions to reduce noise
                    if len(plan) % 3 == 0:
                        self._print_applicable_actions(grounder, len(plan))
            else:
                # Find closest valid action using similarity
                best_op, similarity = self._find_closest_applicable_operator(
                    action_str, grounder.operators, grounder.state, operator_embeddings
                )

                # Skip repair if the best match is 100% identical (similarity = 1.0)
                # This prevents unnecessary token-expensive replacements
                if best_op and similarity >= 0.999:
                    # Action is essentially identical to a valid operator, use it without repair
                    plan.append(action_str)
                    grounder.apply_action(best_op)
                    if self.config.debug:
                        print(
                            f"[REPAIR] Action {len(plan)}: {action_str} ✓ "
                            f"(100% match with operator, no repair needed, sim={similarity:.3f})"
                        )
                        # Show next state only every 3 actions
                        if len(plan) % 3 == 0:
                            self._print_applicable_actions(grounder, len(plan))
                elif best_op:
                    # Use the repaired action
                    repaired_action = self._op_to_action_str(best_op)
                    plan.append(repaired_action)
                    repairs_made.append(
                        {
                            "original": action_str,
                            "repaired": repaired_action,
                            "similarity": similarity,
                        }
                    )
                    grounder.apply_action(best_op)
                    if self.config.debug:
                        print(
                            f"[REPAIR] Action {len(plan)}: {repaired_action} "
                            f"(repaired from '{action_str}', sim={similarity:.3f})"
                        )
                        # Show next state
                        self._print_applicable_actions(grounder, len(plan))
                else:
                    # No applicable action found - skip
                    if self.config.debug:
                        print(
                            f"[REPAIR] Action {len(plan) + 1} skipped: {action_str} (no applicable operators)"
                        )

        if self.config.debug:
            print("\n" + "╔" + "═" * 68 + "╗")
            print("║" + f" FINAL PLAN SUMMARY".ljust(69) + "║")
            print("╠" + "═" * 68 + "╣")
            print("║" + f" ✓ Plan: {len(plan)} actions".ljust(69) + "║")
            if repairs_made:
                print(
                    "║"
                    + f" ✓ Repairs: {len(repairs_made)} actions repaired".ljust(69)
                    + "║"
                )
                print("╠" + "═" * 68 + "╣")
                for repair in repairs_made:
                    orig = repair["original"][:30]
                    repaired = repair["repaired"][:30]
                    sim = repair["similarity"]
                    line = f" • {orig} → {repaired} ({sim:.3f})"
                    print("║" + line.ljust(69) + "║")
            print("╚" + "═" * 68 + "╝")

        return plan

    def _print_applicable_actions(self, grounder, state_num: int):
        """Print available grounded actions after state update."""
        show_operators = getattr(self.config, "show_applicable_operators", True)
        if not show_operators:
            return

        applicable_ops = [
            op for op in grounder.operators if op.applicable(grounder.state)
        ]

        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + f" STATE {state_num} - After Action {state_num}".ljust(69) + "║")
        print("║" + " Available Grounded Actions:".ljust(69) + "║")
        print("╠" + "═" * 68 + "╣")

        for j, op in enumerate(applicable_ops[:12], 1):
            op_str = self._op_to_action_str(op)
            op_display = f"  {j}. {op_str}"
            padding = " " * (68 - len(op_display))
            print("║" + op_display + padding + "║")

        if len(applicable_ops) > 12:
            remaining = f"  ... +{len(applicable_ops) - 12} more"
            padding = " " * (68 - len(remaining))
            print("║" + remaining + padding + "║")

        print("╚" + "═" * 68 + "╝")
        print()

    def _get_operator_embeddings(self, operators) -> Dict[int, any]:
        """Compute and cache embeddings for all operators."""
        embeddings = {}
        operator_texts = []

        # Create text representations for embedding
        for i, op in enumerate(operators):
            action_str = self._op_to_action_str(op)
            operator_texts.append(action_str)

        # Batch embed all operators
        embedding_list = self.embedder.encode(operator_texts, convert_to_tensor=True)

        for i, op in enumerate(operators):
            embeddings[id(op)] = embedding_list[i]

        return embeddings

    def _find_matching_operator(self, action_str: str, operators) -> Optional[any]:
        """Find an operator that exactly matches the action string."""
        action_str_normalized = action_str.strip()
        for op in operators:
            if self._op_to_action_str(op).strip() == action_str_normalized:
                return op
        return None

    def _extract_action_name(self, action_str: str) -> str:
        """Extract action/predicate name from action string like '(walk driver1 s0 s1)'."""
        # Remove parentheses and split
        cleaned = action_str.strip().strip("()").strip()
        parts = cleaned.split()
        if parts:
            return parts[0].lower()  # Return predicate name in lowercase
        return ""

    def _find_closest_applicable_operator(
        self,
        action_str: str,
        operators,
        current_state,
        embeddings: Dict[int, any],
        top_k: int = 1,
    ) -> Tuple[Optional[any], float]:
        """
        Find closest operator that:
        1. Is applicable in current state (preconditions satisfied)
        2. Has the same action name as the LLM output
        3. Has highest semantic similarity among matching candidates

        This ensures we only consider valid, grounded actions for the current state.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Extract the action name from LLM output
        llm_action_name = self._extract_action_name(action_str)
        if not llm_action_name:
            return None, 0.0

        try:
            action_embedding = self.embedder.encode(action_str, convert_to_tensor=True)
        except Exception as e:
            if self.config.debug:
                print(f"[REPAIR] Error embedding action: {e}")
            return None, 0.0

        # Find operators that:
        # 1. Are applicable in current state (preconditions satisfied)
        # 2. Have the same action name as the LLM output
        applicable_ops = [
            op
            for op in operators
            if op.applicable(current_state)
            and self._extract_action_name(self._op_to_action_str(op)) == llm_action_name
        ]

        if not applicable_ops:
            return None, 0.0

        # Compute similarities to applicable operators with same name
        best_op = None
        best_similarity = -1.0

        for op in applicable_ops:
            op_embedding = embeddings.get(id(op))
            if op_embedding is None:
                continue

            # Convert to numpy for similarity computation
            try:
                if hasattr(action_embedding, "cpu"):
                    action_emb_np = action_embedding.cpu().numpy()
                else:
                    action_emb_np = np.array(action_embedding)

                if hasattr(op_embedding, "cpu"):
                    op_emb_np = op_embedding.cpu().numpy()
                else:
                    op_emb_np = np.array(op_embedding)

                similarity = float(
                    cosine_similarity(
                        action_emb_np.reshape(1, -1), op_emb_np.reshape(1, -1)
                    )[0, 0]
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_op = op
            except Exception as e:
                if self.config.debug:
                    print(f"[REPAIR] Error computing similarity: {e}")
                continue

        return best_op, best_similarity

    def _op_to_action_str(self, op) -> str:
        """Convert operator to action string."""
        return str(op.name) if hasattr(op, "name") else str(op)
