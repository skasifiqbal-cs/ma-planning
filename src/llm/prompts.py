"""Prompt template system for various planning approaches."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum


class PromptStyle(str, Enum):
    """Different prompting approaches."""

    ZERO_SHOT = "zero-shot"
    FEW_SHOT = "few-shot"
    CHAIN_OF_THOUGHT = "chain-of-thought"
    REACT = "react"
    TREE_OF_THOUGHTS = "tree-of-thoughts"
    LEAST_TO_MOST = "least-to-most"
    SELF_CONSISTENCY = "self-consistency"


class PromptTemplate:
    """Base template for LLM prompts."""

    def __init__(
        self,
        style: PromptStyle,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Initialize prompt template.

            user_template: User message template with {placeholders}
            examples: Example interactions for few-shot learning
        """
        self.style = style
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.examples = examples or []

    def format(self, **kwargs) -> List[Dict[str, str]]:
        """
        Format template with provided variables.

        Returns:
            List of chat messages
        """
        messages = []

        # Add system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add examples for few-shot (interleaved user/assistant turns)
        if self.examples and self.style == PromptStyle.FEW_SHOT:
            for example in self.examples:
                messages.append({"role": "user", "content": example["input"]})
                messages.append({"role": "assistant", "content": example["output"]})

        # Add user message
        if self.user_template:
            messages.append(
                {"role": "user", "content": self.user_template.format(**kwargs)}
            )

        # Debug: print if requested (extract debug flag if present)
        debug = kwargs.pop("debug", False)
        if debug:
            self._print_debug(messages)

        return messages

    @staticmethod
    def _print_debug(messages: List[Dict[str, str]]):
        """Print formatted prompt for debugging."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                panel = Panel(content, title=f"[bold cyan]{role}[/bold cyan]")
                console.print(panel)
        except ImportError:
            # Fallback if rich not available
            for msg in messages:
                print(f"\n{'='*60}")
                print(f"{msg.get('role', 'unknown').upper()}:")
                print(f"{'='*60}")
                print(msg.get("content", ""))


class PlanningPrompts:
    """Collection of planning-specific prompt templates."""

    @staticmethod
    def zero_shot_planning(
        examples: Optional[List[Dict[str, str]]] = None,
    ) -> PromptTemplate:
        """Zero-shot planning prompt with preconditions and effects."""
        style = PromptStyle.FEW_SHOT if examples else PromptStyle.ZERO_SHOT
        system_base = (
            "You are an expert PDDL planning agent. Generate valid action sequences.\n\n"
            "Key rules:\n"
            "1. Actions require preconditions to be satisfied in the current state\n"
            "2. Actions modify state via add/delete effects\n"
            "3. Track state progression: later actions depend on earlier effects\n"
            "4. For multi-agent problems: Include the agent name after the action name"
        )
        if examples:
            system_base += (
                f"\n\nThe conversation so far contains {len(examples)} example "
                "problem(s) and their reference plans. Use them to understand the "
                "expected format and action style, then solve the new problem."
            )
        return PromptTemplate(
            style=style,
            system_prompt=system_base,
            user_template=(
                "Domain: {domain}\n\n"
                "Problem: {problem}\n\n"
                "Generate a plan to achieve the goal. Output actions one per line as: (action_name agent_name arg1 arg2 ...)\n"
                # "For example: (drive-truck d1 truck1 loc1 loc2) or (load-package p1 loc1 truck1)"
            ),
            examples=examples,
        )

    @staticmethod
    def few_shot_planning(examples: List[Dict[str, str]]) -> PromptTemplate:
        """Few-shot planning with examples."""
        return PlanningPrompts.zero_shot_planning(examples=examples)

    @staticmethod
    def chain_of_thought_planning() -> PromptTemplate:
        """Chain-of-thought planning prompt with detailed reasoning."""
        return PromptTemplate(
            style=PromptStyle.CHAIN_OF_THOUGHT,
            system_prompt=(
                "You are an expert AI planning agent. Think step-by-step about the "
                "planning problem, carefully reasoning about preconditions and effects.\n\n"
                "REASONING FRAMEWORK:\n"
                "- PRECONDITIONS: What facts must be true for this action to apply?\n"
                "- EFFECTS: What facts become true/false after this action?\n"
                "- STATE TRACKING: Keep track of the current state as you plan\n"
                "- DEPENDENCY ANALYSIS: Which actions must come before others?\n"
                "- AGENT ASSIGNMENT: In multi-agent problems, assign actions to the correct agent"
            ),
            user_template=(
                "Domain: {domain}\n\n"
                "Problem: {problem}\n\n"
                "Analyze this step-by-step:\n"
                "1. GOAL ANALYSIS: List all goal facts that must be true\n"
                "2. OPERATOR MATCHING: Which operators produce each goal fact? (check their positive effects)\n"
                "3. PRECONDITION CHECKING: For each selected operator, verify:\n"
                "   - Are its preconditions met in the initial state?\n"
                "   - If not, which operators can establish these preconditions?\n"
                "4. EFFECT ANALYSIS: After each action, what facts change?\n"
                "   - Which facts are added? (positive effects)\n"
                "   - Which facts are removed? (negative effects)\n"
                "5. DEPENDENCY ORDERING: Order actions considering:\n"
                "   - Actions that establish preconditions should come first\n"
                "   - Avoid actions that remove facts needed by later actions\n"
                "6. STATE VERIFICATION: Trace through the state after each action\n"
                "   - Confirm all intermediate states are valid\n"
                "   - Confirm final state satisfies all goal facts\n"
                "7. AGENT ASSIGNMENT: For multi-agent domains, identify which agent performs each action\n\n"
                "Output format: (action_name agent_name arg1 arg2 ...)\n"
                "Example: (drive-truck d1 truck1 loc1 loc2) or (board-truck d1 truck1 loc1)\n\n"
                "Provide your detailed reasoning, then output the final plan."
            ),
        )

    @staticmethod
    def react_planning() -> PromptTemplate:
        """ReAct (Reasoning + Acting) planning prompt."""
        return PromptTemplate(
            style=PromptStyle.REACT,
            system_prompt=(
                "You are an AI planning agent using the ReAct framework. "
                "For each step: (1) Think about what to do, (2) Act by selecting "
                "an action, (3) Observe the result."
            ),
            user_template=(
                "Domain: {domain}\n\n"
                "Problem: {problem}\n\n"
                "Current State: {current_state}\n\n"
                "Use this format:\n"
                "Thought: [your reasoning]\n"
                "Action: (action_name arg1 arg2 ...)\n"
                "Observation: [expected state after action]\n\n"
                "Continue until the goal is reached."
            ),
        )

    @staticmethod
    def repair_planning(
        examples: Optional[List[Dict[str, str]]] = None,
    ) -> PromptTemplate:
        """Plan repair prompt with error analysis."""
        style = PromptStyle.FEW_SHOT if examples else PromptStyle.ZERO_SHOT
        return PromptTemplate(
            style=style,
            system_prompt=(
                "You are an expert AI planning agent specialized in fixing invalid plans. "
                "Analyze validation errors and generate a corrected plan.\n\n"
                "COMMON PLANNING ERRORS:\n"
                "1. VIOLATED PRECONDITIONS: An action was taken but its preconditions weren't met\n"
                "2. MISSING PREREQUISITE ACTIONS: Goal facts need intermediate actions first\n"
                "3. DESTRUCTIVE EFFECTS: An action removed facts needed later in the plan\n"
                "4. INCOMPLETE GOAL: The plan doesn't achieve all goal facts\n"
                "5. ORDERING ISSUES: Actions are in the wrong sequence\n"
                "6. AGENT ASSIGNMENT ERRORS: Wrong agent assigned to an action in multi-agent problems"
            ),
            user_template=(
                "Domain: {domain}\n\n"
                "Problem: {problem}\n\n"
                "Previous Plan:\n{previous_plan}\n\n"
                "Validation Errors:\n{errors}\n\n"
                "ERROR ANALYSIS:\n"
                "1. Identify what caused each error (violated precondition, missing action, agent mismatch, etc.)\n"
                "2. Trace through the state: which facts were true/false at each step?\n"
                "3. Identify which operators should have been used instead\n"
                "4. Ensure all preconditions are met before each action\n"
                "5. Ensure no essential facts are removed before they're used\n"
                "6. Verify correct agent assignment for each action\n\n"
                "Generate a corrected plan that fixes these errors. Output format: (action_name agent_name arg1 arg2 ...), one per line."
            ),
        )

    @staticmethod
    def autoregressive_step() -> PromptTemplate:
        """Autoregressive (step-by-step) planning prompt with state awareness."""
        return PromptTemplate(
            style=PromptStyle.ZERO_SHOT,
            system_prompt=(
                "You are an expert AI planning agent generating one action at a time. "
                "Select the best next action based on the current state.\n\n"
                "DECISION CRITERIA:\n"
                "1. PRECONDITIONS: The action's preconditions MUST ALL be satisfied in the current state\n"
                "2. PROGRESS: Does this action move us closer to any unachieved goal fact?\n"
                "3. NO HARM: This action should not remove facts that are needed for the goal\n"
                "4. DEPENDENCIES: Prefer actions that enable other necessary actions later\n"
                "5. AGENT ASSIGNMENT: Select the correct agent for the action (multi-agent problems)"
            ),
            user_template=(
                "Domain: {domain}\n\n"
                "Problem: {problem}\n\n"
                "Current State (facts that are true):\n{current_state}\n\n"
                "Plan So Far:\n{plan_prefix}\n\n"
                "Unachieved Goals:\n{remaining_goals}\n\n"
                "ANALYZE:\n"
                "- Which operators have preconditions satisfied in the current state?\n"
                "- Which of those operators produce facts needed for unachieved goals?\n"
                "- Avoid operators whose negative effects harm goal achievement\n"
                "- For multi-agent domains, identify the correct agent for the selected action\n\n"
                "Output ONLY the next action in the format: (action_name agent_name arg1 arg2 ...)"
            ),
        )

    @staticmethod
    def val_feedback_backprompt(
        examples: Optional[List[Dict[str, str]]] = None,
    ) -> PromptTemplate:
        """VAL validation feedback backprompt for plan improvement."""
        style = PromptStyle.FEW_SHOT if examples else PromptStyle.ZERO_SHOT
        return PromptTemplate(
            style=style,
            system_prompt=(
                "You are an expert AI planning agent specialized in fixing invalid plans using VAL validation feedback. "
                "Analyze the validation errors from VAL (Validate tool) and generate a corrected plan.\n\n"
                "VALIDATION ERROR ANALYSIS:\n"
                "1. PRECONDITION VIOLATIONS: When VAL reports 'precondition not satisfied' or similar\n"
                "2. TYPE ERRORS: When arguments don't match expected types\n"
                "3. GOAL FAILURES: When 'Goal not satisfied' at the end\n"
                "4. ACTION FAILURES: When specific actions fail to execute\n"
                "5. STATE INCONSISTENCIES: When the plan leads to invalid states\n\n"
                "CORRECTION STRATEGIES:\n"
                "- Add missing prerequisite actions before failed actions\n"
                "- Fix type mismatches in action parameters\n"
                "- Ensure all goal conditions are explicitly achieved\n"
                "- Remove or reorder actions that cause conflicts\n"
                "- For multi-agent problems, verify correct agent assignments"
            ),
            user_template=(
                "Domain: {domain}\n\n"
                "Problem: {problem}\n\n"
                "Previous Plan (INVALID):\n{previous_plan}\n\n"
                "VAL Validation Output:\n{validation_output}\n\n"
                "ANALYSIS: Study the validation output above to understand what went wrong.\n"
                "Look for specific error messages like 'precondition not satisfied', 'Goal not satisfied', "
                "or action execution failures.\n\n"
                "Generate a corrected plan that fixes the validation errors. "
                "Output actions one per line as: (action_name agent_name arg1 arg2 ...)\n"
                "Ensure the new plan is complete and achieves all goals."
            ),
            examples=examples,
        )

    @staticmethod
    def load_few_shot_examples(
        file_path: Optional[str], limit: Optional[int] = None, domain: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Load few-shot examples from a JSON file and optionally filter by domain.

        Returns a list of example dicts containing at least `input` and `output`.
        If examples in the file include `domain`/`problem` metadata it is preserved
        so callers can filter or inspect it.
        """
        if not file_path:
            return []

        path = Path(file_path)
        if not path.is_file():
            return []

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            examples = payload.get("examples", [])
        else:
            examples = payload

        if not isinstance(examples, list):
            return []

        filtered: List[Dict[str, str]] = []
        for item in examples:
            if not isinstance(item, dict):
                continue
            input_text = item.get("input")
            output_text = item.get("output")
            ex_domain = item.get("domain")
            ex_problem = item.get("problem")
            if isinstance(input_text, str) and isinstance(output_text, str):
                # If domain filter provided, only include matching examples
                if domain is None or (isinstance(ex_domain, str) and ex_domain == domain):
                    entry = {"input": input_text, "output": output_text}
                    # preserve metadata if present
                    if ex_domain:
                        entry["domain"] = ex_domain
                    if ex_problem:
                        entry["problem"] = ex_problem
                    filtered.append(entry)

        if limit is not None and limit > 0:
            return filtered[:limit]
        return filtered


class PromptRegistry:
    """Registry for prompt templates."""

    _templates: Dict[str, PromptTemplate] = {}

    @classmethod
    def register(cls, name: str, template: PromptTemplate):
        """Register a new prompt template."""
        cls._templates[name.lower()] = template

    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        """Get a registered template."""
        name = name.lower()
        if name not in cls._templates:
            available = ", ".join(cls._templates.keys())
            raise ValueError(
                f"Unknown prompt template '{name}'. "
                f"Available templates: {available}"
            )
        return cls._templates[name]

    @classmethod
    def list_templates(cls) -> List[str]:
        """List all registered templates."""
        return list(cls._templates.keys())


# Auto-register default templates
def _register_default_templates():
    """Register default planning prompt templates."""
    PromptRegistry.register("zero-shot", PlanningPrompts.zero_shot_planning())
    PromptRegistry.register(
        "chain-of-thought", PlanningPrompts.chain_of_thought_planning()
    )
    PromptRegistry.register("react", PlanningPrompts.react_planning())
    PromptRegistry.register("repair", PlanningPrompts.repair_planning())
    PromptRegistry.register("autoregressive", PlanningPrompts.autoregressive_step())


_register_default_templates()


__all__ = ["PromptStyle", "PromptTemplate", "PlanningPrompts", "PromptRegistry"]
