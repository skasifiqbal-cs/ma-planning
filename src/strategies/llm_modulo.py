"""VAL feedback strategy using validation errors to improve LLM plans through backprompting."""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tempfile import NamedTemporaryFile

from ..validation.evaluator import PlanEvaluator
from ..llm.prompts import PlanningPrompts
from ..utils.parsing import compress_pddl


class LLMModuloStrategy:
    """
    Planning strategy that uses VAL validation feedback for iterative improvement.

    Process:
    1. Generate initial plan with LLM
    2. Validate with VAL (verbose mode)
    3. If validation fails, extract error details and backprompt LLM
    4. Repeat until valid plan or max retries reached
    """

    def __init__(self, llm, config, **kwargs):
        """
        Initialize VAL feedback strategy.

        Args:
            llm: LLM client
            config: Configuration object
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.llm = llm
        self.config = config

        # Initialize VAL evaluator with verbose output and continue-on-error
        self.evaluator = PlanEvaluator(
            validate_bin=config.resolved_val_bin,
            timeout=120,
            extra_flags=["-v", "-c"],  # Enable verbose output and continue execution
            print_on_fail=False,  # We'll handle output ourselves
            print_on_pass=False,
        )

        # Get backprompting configuration
        self.max_retries = getattr(config, "backprompt_max_retries", 2)
        self.stop_on_empty_plan = getattr(config, "stop_on_empty_plan", False)

        # Track metrics
        self.last_raw_output = None
        self.validation_attempts = []  # List of (attempt_num, plan, validation_output)

        # Debug settings
        self.debug_detailed = getattr(config, "debug_val_feedback", False)
        self.few_shot_examples = []
        # Domain name may be passed via kwargs (from pipeline)
        self.domain_name = kwargs.get("domain_name") if kwargs else None
        if getattr(self.config, "use_few_shot", False):
            # Load examples, filtering by domain when domain_name is available
            self.few_shot_examples = PlanningPrompts.load_few_shot_examples(
                getattr(self.config, "few_shot_examples_file", None),
                getattr(self.config, "few_shot_example_count", None),
                domain=self.domain_name,
            )

    def generate_plan(
        self,
        ma_domain_file: str,
        ma_problem_file: str,
        ground_domain_file: str,
        ground_problem_file: str,
        max_steps: int,
    ) -> List[str]:
        """
        Generate plan using VAL feedback strategy.

        Returns:
            List of action strings forming the plan
        """
        debug = getattr(self.config, "debug_val_feedback", False)

        if debug:
            print(
                f"\n[VAL_FEEDBACK] Starting plan generation (max_retries={self.max_retries})"
            )
            if self.debug_detailed:
                print("[DEBUG] Detailed validation feedback enabled")
            if self.few_shot_examples:
                print(
                    f"[DEBUG] Few-shot examples loaded: {len(self.few_shot_examples)}"
                )

        # Read domain and problem for prompting
        domain_content = self._read_and_compress_pddl(ground_domain_file)
        problem_content = self._read_and_compress_pddl(ground_problem_file)

        # Generate initial plan
        current_attempt = 0
        plan = None
        validation_output = ""

        while current_attempt <= self.max_retries:
            if debug:
                print(
                    f"\n[VAL_FEEDBACK] Attempt {current_attempt + 1}/{self.max_retries + 1}"
                )
                if self.debug_detailed:
                    print("=" * 50)

            # Generate plan (initial or backprompt)
            if current_attempt == 0:
                # Initial attempt
                plan = self._generate_initial_plan(
                    domain_content, problem_content, max_steps, debug
                )
            else:
                # Backprompt with validation feedback
                plan = self._generate_backprompt_plan(
                    domain_content,
                    problem_content,
                    plan,
                    validation_output,
                    max_steps,
                    debug,
                )

            if not plan:
                if debug:
                    print(
                        f"[VAL_FEEDBACK] No plan generated in attempt {current_attempt + 1}"
                    )
                if self.stop_on_empty_plan:
                    if debug:
                        print(
                            "[VAL_FEEDBACK] stop_on_empty_plan enabled; aborting further retries"
                        )
                    return []
                current_attempt += 1
                continue

            # Validate the plan
            is_valid, validation_output = self._validate_plan_with_feedback(
                ground_domain_file, ground_problem_file, plan, debug
            )

            # Store attempt results
            self.validation_attempts.append(
                (current_attempt + 1, plan.copy(), validation_output)
            )

            if is_valid:
                if debug:
                    print(
                        f"[VAL_FEEDBACK] ✓ Valid plan found in attempt {current_attempt + 1} ({len(plan)} actions)"
                    )
                return plan

            if debug:
                print(f"[VAL_FEEDBACK] ✗ Plan invalid in attempt {current_attempt + 1}")

                if self.debug_detailed and current_attempt < self.max_retries:
                    print(f"[BACKPROMPT] Preparing retry with validation feedback...")

            current_attempt += 1

        # If we reach here, all attempts failed
        if debug:
            print(
                f"[VAL_FEEDBACK] All {self.max_retries + 1} attempts failed, returning best attempt"
            )

        # Return the plan from the last attempt (even if invalid)
        return plan if plan else []

    def _generate_initial_plan(
        self, domain_content: str, problem_content: str, max_steps: int, debug: bool
    ) -> List[str]:
        """Generate initial plan using standard prompt."""
        prompt_template = PlanningPrompts.zero_shot_planning(self.few_shot_examples)
        messages = prompt_template.format(
            domain=domain_content,
            problem=problem_content,
        )

        # Record the formatted prompt for logging/inspection
        try:
            self.last_prompt_messages = messages.copy()
        except Exception:
            self.last_prompt_messages = messages

        if debug and getattr(self.config, "debug_prompts", False):
            print(f"[DEBUG] Initial prompt: {len(str(messages))} characters")
            if getattr(self.config, "show_prompts", False):
                print("\n" + "-" * 40)
                print("[INITIAL PROMPT]")
                print("-" * 40)
                for msg in messages:
                    print(f"{msg['role'].upper()}: {msg['content'][:200]}...")
                print("-" * 40)

        response = self.llm.chat(messages)
        self.last_raw_output = response

        if getattr(self.config, "show_llm_output", False):
            print("\\n[LLM_OUTPUT] Initial generation:")
            print(response)

        # Extract actions from response
        return self._extract_actions(response, debug)

    def _generate_backprompt_plan(
        self,
        domain_content: str,
        problem_content: str,
        previous_plan: List[str],
        validation_output: str,
        max_steps: int,
        debug: bool,
    ) -> List[str]:
        """Generate plan using validation feedback in backprompt."""
        # Extract only relevant error information to reduce tokens (if enabled)
        if getattr(self.config, "condense_val_errors", True):
            condensed_errors = self._extract_val_errors(validation_output)

            if debug:
                original_len = len(validation_output)
                condensed_len = len(condensed_errors)
                reduction_pct = (
                    (1 - condensed_len / original_len) * 100 if original_len > 0 else 0
                )
                print(
                    f"[TOKEN_REDUCTION] VAL output: {original_len} → {condensed_len} chars ({reduction_pct:.1f}% reduction)"
                )

            validation_feedback = condensed_errors
        else:
            validation_feedback = validation_output

        # Create backprompt with validation feedback
        prompt_template = PlanningPrompts.val_feedback_backprompt(
            self.few_shot_examples
        )

        # Format previous plan as string
        previous_plan_str = "\n".join(
            f"{i+1}. {action}" for i, action in enumerate(previous_plan)
        )

        messages = prompt_template.format(
            domain=domain_content,
            problem=problem_content,
            previous_plan=previous_plan_str,
            validation_output=validation_feedback,
        )

        if debug and getattr(self.config, "debug_prompts", False):
            print(f"[DEBUG] Backprompt: {len(str(messages))} characters")
            if getattr(self.config, "show_prompts", False):
                print("\n" + "-" * 40)
                print("[BACKPROMPT]")
                print("-" * 40)
                for msg in messages:
                    print(f"{msg['role'].upper()}: {msg['content'][:200]}...")
                print("-" * 40)

        response = self.llm.chat(messages)
        self.last_raw_output = response

        if getattr(self.config, "show_llm_output", False):
            print("\n[LLM_OUTPUT] Backprompt response:")
            print(response)

        # Extract actions from response
        return self._extract_actions(response, debug)

    def _validate_plan_with_feedback(
        self, domain_file: str, problem_file: str, plan: List[str], debug: bool
    ) -> Tuple[bool, str]:
        """
        Validate plan and return detailed feedback.

        Returns:
            Tuple of (is_valid, validation_output)
        """
        if not plan:
            return False, "No plan to validate"

        # Write plan to temporary file
        with NamedTemporaryFile(
            mode="w", suffix=".plan", delete=False, encoding="utf-8"
        ) as f:
            for action in plan:
                f.write(action.strip() + "\n")
            temp_plan_file = f.name

        try:
            # Run validation with detailed output
            is_valid, return_code, stdout, stderr = self.evaluator.evaluate_detailed(
                domain_file, problem_file, temp_plan_file
            )

            # Combine stdout and stderr for comprehensive feedback
            validation_output = ""
            if stdout:
                validation_output += "VALIDATION OUTPUT:\n" + stdout
            if stderr:
                validation_output += "\nVALIDATION ERRORS:\n" + stderr

            if debug:
                status = "VALID" if is_valid else "INVALID"
                print(
                    f"[VAL_FEEDBACK] Validation: {status} (return code: {return_code})"
                )

                # Show key validation information without full dump
                if not is_valid and stdout:
                    # Extract and show the key error
                    lines = stdout.split("\n")
                    for i, line in enumerate(lines):
                        if "Plan failed because of" in line:
                            print(f"[VAL_ERROR] {line}")
                            # Show the problematic action if available
                            if i + 1 < len(lines) and lines[i + 1].strip().startswith(
                                "("
                            ):
                                print(f"[VAL_ACTION] {lines[i + 1].strip()}")
                            break
                        elif "unsatisfied precondition" in line:
                            print(f"[VAL_ERROR] {line}")
                        elif "Plan Repair Advice:" in line and i + 2 < len(lines):
                            advice = lines[i + 2].strip()
                            if advice:
                                print(f"[VAL_ADVICE] {advice}")

                elif is_valid and self.debug_detailed:
                    print(
                        "[VAL_SUCCESS] Plan executed successfully and achieves all goals"
                    )

            return is_valid, validation_output

        finally:
            # Clean up temporary file
            try:
                Path(temp_plan_file).unlink(missing_ok=True)
            except Exception:
                pass

    def _extract_actions(self, response: str, debug: bool) -> List[str]:
        """Extract action list from LLM response."""
        actions = []

        # Look for numbered actions or plan format
        patterns = [
            r"^\s*\d+[\.\:\)]\s*\(([^)]+)\)\s*$",  # "1. (action params)"
            r"^\s*\(([^)]+)\)\s*$",  # "(action params)"
        ]

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    action = f"({match.group(1)})"
                    actions.append(action)
                    break

        if debug:
            print(f"[EXTRACT] Found {len(actions)} actions")

        return actions

    def _extract_val_errors(self, val_output: str) -> str:
        """
        Extract only the relevant error information from VAL output.

        This significantly reduces token usage by filtering out verbose type-checking
        and plan validation details, keeping only:
        - Failed actions with unsatisfied preconditions
        - Repair advice
        - Goal satisfaction status

        Args:
            val_output: Full VAL validation output

        Returns:
            Condensed error information
        """
        if not val_output:
            return ""

        lines = val_output.split("\n")
        relevant_lines = []
        in_error_section = False
        in_repair_advice = False
        skip_next = 0

        for i, line in enumerate(lines):
            # Skip already processed lines
            if skip_next > 0:
                skip_next -= 1
                continue

            line_lower = line.lower()

            # Skip type-checking verbosity
            if "type-checking" in line_lower or "...action passes" in line_lower:
                continue

            # Skip plan header (we already have the plan)
            if "checking plan:" in line_lower or "plan to validate:" in line_lower:
                continue
            if line.strip().startswith("Plan size:"):
                continue
            if re.match(r"^\d+:\s*$", line.strip()):
                continue

            # Skip detailed state changes (Checking next happening, Deleting, Adding)
            if any(
                x in line_lower
                for x in ["checking next happening", "deleting (", "adding ("]
            ):
                continue

            # Capture error sections
            if "plan failed because" in line_lower:
                in_error_section = True
                relevant_lines.append(line)
                # Get the next line which usually has the action
                if i + 1 < len(lines):
                    relevant_lines.append(lines[i + 1])
                    skip_next = 1
                continue

            # Capture unsatisfied precondition errors
            if "unsatisfied precondition" in line_lower:
                in_error_section = True
                relevant_lines.append(line)
                # Capture repair advice that follows
                j = i + 1
                while j < len(lines) and j < i + 10:  # Look ahead up to 10 lines
                    next_line = lines[j]
                    if next_line.strip().startswith("(") and "Set" in next_line:
                        relevant_lines.append(next_line)
                    elif not next_line.strip():
                        break
                    j += 1
                continue

            # Capture Plan Repair Advice section
            if "plan repair advice:" in line_lower:
                in_repair_advice = True
                relevant_lines.append("\n" + line)
                continue

            # In repair advice section
            if in_repair_advice:
                if line.strip() and not line.strip().startswith("Failed plans:"):
                    relevant_lines.append(line)
                elif line.strip().startswith("Failed plans:"):
                    in_repair_advice = False
                continue

            # Capture goal failure
            if "goal not satisfied" in line_lower:
                relevant_lines.append("\n" + line)
                # Get goal requirements
                j = i + 1
                while j < len(lines) and j < i + 20:
                    next_line = lines[j]
                    if "Set (" in next_line or "Follow each" in next_line:
                        relevant_lines.append(next_line)
                    elif next_line.strip() == ")":
                        relevant_lines.append(next_line)
                        break
                    elif not next_line.strip():
                        break
                    j += 1
                continue

            # Capture plan execution status
            if "plan executed successfully" in line_lower or "plan valid" in line_lower:
                relevant_lines.append(line)
                continue

        condensed = "\n".join(relevant_lines)

        # If we didn't extract anything meaningful, return a summary
        if not condensed.strip():
            if (
                "plan valid" in val_output.lower()
                and "plan executed successfully" in val_output.lower()
            ):
                return (
                    "Plan valid - all actions executed successfully and goal satisfied."
                )
            else:
                return "Validation failed - see detailed output for errors."

        return condensed.strip()

    def _read_and_compress_pddl(self, file_path: str) -> str:
        """Read and optionally compress PDDL file."""
        with open(file_path, "r") as f:
            content = f.read()
        if getattr(self.config, "compress_pddl", True):
            content = compress_pddl(content)
        return content

    def get_validation_history(self) -> List[Tuple[int, List[str], str]]:
        """Return history of validation attempts for debugging."""
        return self.validation_attempts.copy()
