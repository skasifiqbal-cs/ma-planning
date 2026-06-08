"""Multi-agent planning strategy using decomposition and LLM-based plan merging."""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import subprocess
import re

# Add pyperplan to path if not already available
pyperplan_path = Path(__file__).parent.parent.parent / "pyperplan"
if pyperplan_path.exists() and str(pyperplan_path) not in sys.path:
    sys.path.insert(0, str(pyperplan_path))

from ..converters.agent_decomposer import AgentDecomposer
from ..utils.parsing import extract_parenthesized_actions
from ..validation.evaluator import PlanEvaluator
import json
import time


class LLMMergeStrategy:
    """
    Multi-agent planning using decomposition approach:
    1. Decompose MA-PDDL into single-agent PDDL problems
    2. Solve each agent's problem with pyperplan
    3. Use LLM to merge individual plans into coordinated multi-agent plan
    """

    def __init__(self, llm, config, **kwargs):
        """Initialize strategy.

        Args:
            llm: LLM provider instance
            config: Configuration object
            **kwargs: Additional parameters
        """
        self.llm = llm
        self.config = config
        self.decomposer = AgentDecomposer(python_cmd=config.python_cmd)
        self.debug = getattr(config, "verbose", False)
        self.mode_name = kwargs.get("mode_name", "")
        self.config = config
        self.plan_storage = {
            "agent_plans": {},
            "merged_llm": None,
            "merged_fallback": None,
            "fallback_used": False,
        }

        # Try to import pyperplan
        try:
            from pyperplan import planner

            self.pyperplan = planner
            self.pyperplan_available = True
        except ImportError:
            print("[WARNING] pyperplan not found. Will try command-line execution.")
            self.pyperplan_available = False
            self.pyperplan_cmd = self._find_pyperplan_cmd()

    def _find_pyperplan_cmd(self) -> Optional[str]:
        """Find pyperplan command-line tool."""
        pyperplan_path = Path(__file__).parent.parent.parent / "pyperplan"
        if (pyperplan_path / "pyperplan" / "__main__.py").exists():
            return str(pyperplan_path / "pyperplan" / "__main__.py")
        return None

    def _save_plans_to_disk(
        self,
        domain_name: str,
        problem_stem: str,
        agent_plans: Dict[str, List[str]],
        merged_llm_plan: List[str],
        merged_fallback_plan: List[str],
        fallback_used: bool,
    ) -> Path:
        """Save all plan variants to disk for review.

        Args:
            domain_name: Domain name for organization
            problem_stem: Problem name without extension
            agent_plans: Individual agent plans
            merged_llm_plan: LLM-merged plan
            merged_fallback_plan: Sequential fallback plan
            fallback_used: Whether fallback was used

        Returns:
            Path to plan storage directory
        """
        results_dir = (
            Path(self.config.results_root) / domain_name / problem_stem / "plans"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save individual agent plans
        agents_dir = results_dir / "agents"
        agents_dir.mkdir(exist_ok=True)
        for agent, plan in agent_plans.items():
            agent_file = agents_dir / f"{agent}.plan"
            with agent_file.open("w", encoding="utf-8") as f:
                for action in plan:
                    f.write(action.strip() + "\n")

        # Save LLM-merged plan
        if merged_llm_plan:
            llm_file = results_dir / "merged_llm.plan"
            with llm_file.open("w", encoding="utf-8") as f:
                for action in merged_llm_plan:
                    f.write(action.strip() + "\n")

        # Save fallback plan if used
        if fallback_used and merged_fallback_plan:
            fallback_file = results_dir / "merged_fallback.plan"
            with fallback_file.open("w", encoding="utf-8") as f:
                for action in merged_fallback_plan:
                    f.write(action.strip() + "\n")

        # Save manifest
        manifest = {
            "problem": problem_stem,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agents": list(agent_plans.keys()),
            "agent_plans": {agent: len(plan) for agent, plan in agent_plans.items()},
            "merged_llm_size": len(merged_llm_plan) if merged_llm_plan else 0,
            "merged_fallback_size": (
                len(merged_fallback_plan) if merged_fallback_plan else 0
            ),
            "fallback_used": fallback_used,
            "files": {
                "agents": [f"agents/{agent}.plan" for agent in agent_plans.keys()],
                "merged_llm": "merged_llm.plan" if merged_llm_plan else None,
                "merged_fallback": ("merged_fallback.plan" if fallback_used else None),
            },
        }

        manifest_file = results_dir / "manifest.json"
        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if self.debug:
            print(f"\n[PLANS SAVED] {results_dir}")
            print(f"  Agents: {', '.join(agent_plans.keys())}")
            print(f"  LLM-merged: {len(merged_llm_plan)} actions")
            if fallback_used:
                print(f"  Fallback: {len(merged_fallback_plan)} actions")

        return results_dir

    def generate_plan(
        self,
        ma_domain_file: str = None,
        ma_problem_file: str = None,
        ground_domain_file: str = None,
        ground_problem_file: str = None,
        max_steps: int = 0,
        **kwargs,
    ) -> List[str]:
        """
        Generate multi-agent plan using decomposition approach.

        Args:
            ma_domain_file: Path to unfactored MA-PDDL domain file
            ma_problem_file: Path to unfactored MA-PDDL problem file
            ground_domain_file: Not used (for compatibility)
            ground_problem_file: Not used (for compatibility)
            max_steps: Not used (for compatibility)
            **kwargs: Additional parameters

        Returns:
            List of action strings forming the merged multi-agent plan
        """
        # Use ma_ files (unfactored MA-PDDL) instead of ground_ files
        domain_pddl = ma_domain_file
        problem_pddl = ma_problem_file
        domain_name = Path(domain_pddl).parent.name

        # Enable debug mode for llm-merge to ensure logging
        debug = kwargs.get("debug", self.debug) or (self.mode_name == "llm-merge")

        if debug:
            print("\n" + "=" * 60)
            print("[MULTI-AGENT DECOMPOSITION] Starting decomposition strategy")
            print("=" * 60)

        # Step 1: Decompose into single-agent problems
        if debug:
            print("\n[STEP 1] Decomposing MA-PDDL into single-agent problems...")

        domain_dir = str(Path(domain_pddl).parent)
        domain_file = Path(domain_pddl).name
        problem_file = Path(problem_pddl).name

        output_dir = Path("agent_plans") / Path(problem_pddl).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        agent_files = self.decomposer.decompose_all_agents(
            domain_dir=domain_dir,
            domain_file=domain_file,
            problem_file=problem_file,
            output_base_dir=str(output_dir.parent),
        )

        if not agent_files:
            print("[ERROR] No agents found in MA-PDDL problem")
            return []

        if debug:
            print(f"[STEP 1] Created {len(agent_files)} single-agent problems")

        # Step 2: Solve each agent's problem with pyperplan
        if debug:
            print("\n[STEP 2] Solving individual agent problems with pyperplan...")

        agent_plans = {}
        for agent, (agent_domain, agent_problem) in agent_files.items():
            if debug:
                print(f"\n[PYPERPLAN] Solving for agent: {agent}")

            plan = self._solve_with_pyperplan(agent_domain, agent_problem, agent)

            if plan:
                agent_plans[agent] = plan
                if debug:
                    print(f"[PYPERPLAN] ✓ Found plan for {agent}: {len(plan)} actions")
                    for i, action in enumerate(plan[:5], 1):
                        print(f"  {i}. {action}")
                    if len(plan) > 5:
                        print(f"  ... ({len(plan) - 5} more actions)")
            else:
                if debug:
                    print(f"[PYPERPLAN] ✗ No plan found for {agent}")

        if not agent_plans:
            print("[ERROR] No plans generated for any agent")
            return []

        # Step 3: Merge plans using LLM
        if debug:
            print("\n[STEP 3] Merging individual plans with LLM...")

        merged_llm_plan = self._merge_plans_with_llm(
            agent_plans=agent_plans,
            domain_pddl=domain_pddl,
            problem_pddl=problem_pddl,
            debug=debug,
            ground_domain_file=ground_domain_file,
            ground_problem_file=ground_problem_file,
            max_repair_attempts=3,
        )
        merged_fallback_plan = None
        fallback_used = False

        # For llm-merge mode, check if we need fallback
        # (repairs already happened inside _merge_plans_with_llm)
        if self.mode_name == "llm-merge" and not merged_llm_plan:
            print(
                "[LLM-MERGE] No valid plan generated; falling back to sequential merge."
            )
            merged_fallback_plan = self._fallback_sequential_merge(agent_plans)
            fallback_used = True
            merged_plan = merged_fallback_plan
        else:
            merged_plan = merged_llm_plan

        # Save all plan variants
        problem_stem = Path(problem_pddl).stem
        self._save_plans_to_disk(
            domain_name=domain_name,
            problem_stem=problem_stem,
            agent_plans=agent_plans,
            merged_llm_plan=merged_llm_plan,
            merged_fallback_plan=merged_fallback_plan,
            fallback_used=fallback_used,
        )

        if debug:
            print(f"\n[RESULT] Final plan: {len(merged_plan)} actions")
            print("=" * 60)

        return merged_plan

    def _solve_with_pyperplan(
        self, domain_file: str, problem_file: str, agent_name: str
    ) -> List[str]:
        """Solve single-agent problem with pyperplan.

        Args:
            domain_file: Path to agent's domain file
            problem_file: Path to agent's problem file
            agent_name: Name of the agent

        Returns:
            List of action strings (grounded actions)
        """
        try:
            if self.pyperplan_available:
                # Use pyperplan directly
                plan = self._solve_with_pyperplan_library(domain_file, problem_file)
            else:
                # Use command-line pyperplan
                plan = self._solve_with_pyperplan_cmd(
                    domain_file, problem_file, agent_name
                )

            return plan if plan else []

        except Exception as e:
            print(f"[ERROR] pyperplan failed for {agent_name}: {e}")
            return []

    def _solve_with_pyperplan_library(
        self, domain_file: str, problem_file: str
    ) -> List[str]:
        """Solve using pyperplan as a library."""
        try:
            from pyperplan import planner

            # Use greedy-best-first with FF heuristic for efficiency
            search_alg = planner.SEARCHES.get("gbf", planner.SEARCHES["bfs"])
            heuristic = planner.HEURISTICS.get("hff", None)

            plan = planner.search_plan(
                domain_file=domain_file,
                problem_file=problem_file,
                search=search_alg,
                heuristic_class=heuristic,
                use_preferred_ops=False,
            )

            if plan:
                return [self._to_compact_action(action) for action in plan]
            return []

        except Exception as e:
            print(f"[ERROR] pyperplan library error: {e}")
            return []

    def _to_compact_action(self, action_obj) -> str:
        """Convert pyperplan action/operator object to compact '(action args...)' format."""
        action_name = getattr(action_obj, "name", None)
        if not action_name:
            action_name = str(action_obj).splitlines()[0].strip()

        action_name = action_name.strip()
        if not action_name.startswith("("):
            action_name = f"({action_name})"
        return action_name

    def _solve_with_pyperplan_cmd(
        self, domain_file: str, problem_file: str, agent_name: str
    ) -> List[str]:
        """Solve using pyperplan command-line tool."""
        if not self.pyperplan_cmd:
            print("[ERROR] pyperplan command not found")
            return []

        try:
            # Run pyperplan
            output_file = f"agent_plans/{agent_name}_plan.txt"
            cmd = [
                "python",
                self.pyperplan_cmd,
                domain_file,
                problem_file,
                "-s",
                "bfs",  # Use BFS search
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                # Parse plan from output
                plan = self._parse_pyperplan_output(result.stdout)
                return plan
            else:
                print(f"[ERROR] pyperplan failed: {result.stderr}")
                return []

        except subprocess.TimeoutExpired:
            print(f"[ERROR] pyperplan timeout for {agent_name}")
            return []
        except Exception as e:
            print(f"[ERROR] pyperplan execution error: {e}")
            return []

    def _parse_pyperplan_output(self, output: str) -> List[str]:
        """Parse pyperplan output to extract plan."""
        plan = []
        in_plan = False

        for line in output.split("\n"):
            line = line.strip()

            if "Plan:" in line or "Found plan:" in line:
                in_plan = True
                continue

            if (
                in_plan
                and line
                and not line.startswith("Plan length")
                and not line.startswith("Search")
            ):
                # Clean action format
                action = line.strip("()").strip()
                if action:
                    plan.append(action)

        return plan

    def _merge_plans_with_llm(
        self,
        agent_plans: Dict[str, List[str]],
        domain_pddl: str,
        problem_pddl: str,
        debug: bool = False,
        ground_domain_file: str = None,
        ground_problem_file: str = None,
        max_repair_attempts: int = 2,
    ) -> List[str]:
        """Merge individual agent plans into coordinated multi-agent plan using LLM.

        Args:
            agent_plans: Dictionary mapping agent name to their plan
            domain_pddl: Path to original MA-PDDL domain
            problem_pddl: Path to original MA-PDDL problem
            debug: Enable debug output
            ground_domain_file: Path to grounded domain for validation
            ground_problem_file: Path to grounded problem for validation
            max_repair_attempts: Maximum repair attempts with VAL feedback

        Returns:
            Merged multi-agent plan
        """
        # Read domain and problem
        with open(domain_pddl, "r") as f:
            domain_content = f.read()

        with open(problem_pddl, "r") as f:
            problem_content = f.read()

        # Compress PDDL if enabled
        if getattr(self.config, "compress_pddl", True):
            domain_content = self._compress_pddl(domain_content)
            problem_content = self._compress_pddl(problem_content)

        # Build prompt for LLM
        prompt = self._build_merge_prompt(
            agent_plans=agent_plans,
            domain_content=domain_content,
            problem_content=problem_content,
        )

        action_agent_positions = self._extract_action_agent_positions(
            self._read_text_if_exists(ground_domain_file)
            if ground_domain_file
            else domain_content
        )

        if debug:
            print(f"\n[LLM] Prompt length: {len(prompt)} chars")
            if getattr(self.config, "show_prompts", False):
                print(f"[LLM] Prompt:\n{prompt}\n")

            # Always save prompt to file for inspection
            results_dir = (
                Path(self.config.results_root)
                / Path(domain_pddl).parent.name
                / Path(problem_pddl).stem
                / "plans"
            )
            results_dir.mkdir(parents=True, exist_ok=True)
            prompt_file = results_dir / "llm_merge_prompt.txt"
            with prompt_file.open("w", encoding="utf-8") as f:
                f.write(prompt)
            if debug:
                print(f"[LLM] Prompt saved to: {prompt_file}")

        # Get LLM response
        messages = [
            {
                "role": "system",
                "content": "You are an expert multi-agent planning system. You merge individual agent plans into a coordinated multi-agent plan that achieves the goal while respecting dependencies and concurrency constraints.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm.chat(messages)

            # Save raw LLM response BEFORE any processing
            results_dir = (
                Path(self.config.results_root)
                / Path(domain_pddl).parent.name
                / Path(problem_pddl).stem
                / "plans"
            )
            results_dir.mkdir(parents=True, exist_ok=True)
            raw_response_file = results_dir / "llm_raw_response_initial.txt"
            with raw_response_file.open("w", encoding="utf-8") as f:
                f.write(response)
            if debug:
                print(f"[LLM] Raw response saved to: {raw_response_file}")

            if debug and getattr(self.config, "show_llm_output", True):
                print(f"\n[LLM] Response:\n{response}\n")

            # Extract and normalize actions from response
            merged_actions = extract_parenthesized_actions(response)
            merged_actions = self._normalize_merged_actions(
                merged_actions,
                agent_names=list(agent_plans.keys()),
                action_agent_positions=action_agent_positions,
            )

            if debug:
                print(f"[LLM] Extracted {len(merged_actions)} actions from response")

            # Validate and repair if needed (modulo approach)
            if ground_domain_file and ground_problem_file and max_repair_attempts > 0:
                validate_bin = getattr(self.config, "resolved_val_bin", None)
                if validate_bin:
                    for attempt in range(max_repair_attempts):
                        evaluator = PlanEvaluator(
                            validate_bin=validate_bin,
                            timeout=120,
                            extra_flags=["-v"],
                            print_on_pass=False,
                            print_on_fail=False,
                        )
                        ok, rc, stdout, stderr, _ = evaluator.evaluate_prefix(
                            ground_domain_file,
                            ground_problem_file,
                            merged_actions,
                            keep_temp=False,
                        )

                        if ok:
                            if debug:
                                print(f"[LLM-MERGE] ✓ Validation passed")
                            break
                        else:
                            if debug:
                                print(
                                    f"[LLM-MERGE] ✗ Validation failed (attempt {attempt + 1}/{max_repair_attempts})"
                                )

                            if attempt < max_repair_attempts - 1:
                                # Try repair with VAL feedback
                                repair_prompt = self._build_repair_prompt(
                                    agent_plans=agent_plans,
                                    domain_content=domain_content,
                                    problem_content=problem_content,
                                    failed_plan=merged_actions,
                                    val_output=stdout + "\n" + stderr,
                                )

                                # Save repair prompt
                                repair_prompt_file = (
                                    results_dir
                                    / f"llm_repair_prompt_attempt{attempt + 1}.txt"
                                )
                                with repair_prompt_file.open(
                                    "w", encoding="utf-8"
                                ) as f:
                                    f.write(repair_prompt)
                                if debug:
                                    print(
                                        f"[LLM-REPAIR] Repair prompt saved to: {repair_prompt_file}"
                                    )

                                messages.append(
                                    {"role": "assistant", "content": response}
                                )
                                messages.append(
                                    {"role": "user", "content": repair_prompt}
                                )

                                response = self.llm.chat(messages)

                                # Save raw repair response
                                repair_response_file = (
                                    results_dir
                                    / f"llm_raw_response_repair{attempt + 1}.txt"
                                )
                                with repair_response_file.open(
                                    "w", encoding="utf-8"
                                ) as f:
                                    f.write(response)
                                if debug:
                                    print(
                                        f"[LLM-REPAIR] Raw response saved to: {repair_response_file}"
                                    )

                                if debug and getattr(
                                    self.config, "show_llm_output", True
                                ):
                                    print(f"\n[LLM-REPAIR] Response:\n{response}\n")

                                merged_actions = extract_parenthesized_actions(response)
                                merged_actions = self._normalize_merged_actions(
                                    merged_actions,
                                    agent_names=list(agent_plans.keys()),
                                    action_agent_positions=action_agent_positions,
                                )

            return merged_actions

        except Exception as e:
            print(f"[ERROR] LLM merge failed: {e}")
            # Fallback: concatenate plans sequentially
            return self._fallback_sequential_merge(agent_plans)

    def _normalize_merged_actions(
        self,
        actions: List[str],
        agent_names: List[str],
        action_agent_positions: Optional[Dict[str, Optional[int]]] = None,
    ) -> List[str]:
        """Normalize merged actions and enforce agent position from action schema when available."""
        normalized = []
        agent_set = set(agent_names)
        action_agent_positions = action_agent_positions or {}

        for action in actions:
            text = action.strip()
            if not (text.startswith("(") and text.endswith(")")):
                continue

            body = text[1:-1].strip()
            if not body:
                continue

            tokens = body.split()

            # Remove commas from any token
            tokens = [t.rstrip(",") for t in tokens]

            # Drop lines that are just agent lists like "(a1, a2, a3)"
            if all(t in agent_set for t in tokens):
                continue

            # Drop common non-action lines enclosed in parens
            if tokens and tokens[0].lower() in {"pre:", "add:", "del:", "not"}:
                continue

            action_name = tokens[0]
            args = tokens[1:]
            action_key = action_name.lower()
            expected_agent_idx = action_agent_positions.get(action_key)

            # If schema is available, ignore unknown parenthesized text fragments.
            if action_agent_positions and action_key not in action_agent_positions:
                continue

            # Find agent token if present
            agent_token = None
            agent_idx = None
            for idx, token in enumerate(args):
                if token in agent_set:
                    agent_token = token
                    agent_idx = idx
                    break

            # If schema says there is an agent parameter, force placement accordingly
            if expected_agent_idx is not None and agent_token is not None:
                reordered_args = [
                    token for idx, token in enumerate(args) if idx != agent_idx
                ]
                insert_idx = max(0, min(expected_agent_idx, len(reordered_args)))
                reordered_args.insert(insert_idx, agent_token)
                normalized.append("(" + " ".join([action_name] + reordered_args) + ")")
                continue

            # If no schema available but agent is last, move to second token (action + agent + args)
            if args and args[-1] in agent_set:
                agent = args[-1]
                rest = args[:-1]
                normalized.append("(" + " ".join([action_name, agent] + rest) + ")")
                continue

            # If no schema available and agent already in first arg position, keep
            if args and args[0] in agent_set:
                normalized.append("(" + " ".join([action_name] + args) + ")")
                continue

            # Final guardrail: if an agent appears anywhere else, move it to second token.
            for idx, token in enumerate(args):
                if token in agent_set:
                    rest = [a for j, a in enumerate(args) if j != idx]
                    normalized.append("(" + " ".join([action_name, token] + rest) + ")")
                    break
            else:
                # Keep as-is if no agent can be identified
                normalized.append("(" + " ".join([action_name] + args) + ")")
            continue

        return normalized

    def _read_text_if_exists(self, file_path: Optional[str]) -> str:
        """Read file contents if path exists, otherwise return empty string."""
        if not file_path:
            return ""
        path = Path(file_path)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _extract_action_agent_positions(
        self, domain_text: str
    ) -> Dict[str, Optional[int]]:
        """Extract expected agent argument position for each action from PDDL parameters.

        Returns mapping action_name -> index in action arguments (0-based), or None if no agent parameter.
        """
        positions: Dict[str, Optional[int]] = {}
        if not domain_text:
            return positions

        action_params = re.finditer(
            r"\(:action\s+([^\s\)]+).*?:parameters\s*\((.*?)\)",
            domain_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in action_params:
            action_name = match.group(1).strip().lower()
            params_text = match.group(2)
            tokens = params_text.replace("\n", " ").split()

            typed_params: List[Tuple[str, str]] = []
            pending_vars: List[str] = []
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token.startswith("?"):
                    pending_vars.append(token)
                    i += 1
                    continue
                if token == "-" and i + 1 < len(tokens):
                    param_type = tokens[i + 1].lower()
                    for var in pending_vars:
                        typed_params.append((var, param_type))
                    pending_vars = []
                    i += 2
                    continue
                i += 1

            for var in pending_vars:
                typed_params.append((var, "object"))

            agent_idx = None
            for idx, (_, param_type) in enumerate(typed_params):
                if param_type == "agent":
                    agent_idx = idx
                    break

            positions[action_name] = agent_idx

        return positions

    def _build_merge_prompt(
        self,
        agent_plans: Dict[str, List[str]],
        domain_content: str,
        problem_content: str,
    ) -> str:
        """Build prompt for LLM to merge agent plans.

        Args:
            agent_plans: Individual agent plans
            domain_content: Domain PDDL
            problem_content: Problem PDDL

        Returns:
            Prompt string
        """
        prompt = f"""# Multi-Agent Plan Merging Task

You are given individual plans for multiple agents solving a multi-agent planning problem. Your task is to merge these plans into a single coordinated plan that:

1. Achieves the goal efficiently
2. Respects action dependencies (an action cannot execute until its preconditions are met)
3. Maximizes parallelism where possible (actions by different agents can execute concurrently if they don't conflict)
4. Maintains correct ordering (if agent A's action creates a precondition for agent B's action, A must go first)

## Domain Definition

{domain_content}

## Problem Definition

{problem_content}

## Individual Agent Plans

"""

        for agent, plan in agent_plans.items():
            prompt += f"\n### Agent: {agent}\n"
            prompt += f"Plan length: {len(plan)} actions\n\n"
            for i, action in enumerate(plan, 1):
                prompt += (
                    f"{i}. {self._format_action_with_agent_second(action, agent)}\n"
                )

        prompt += """

## Instructions

Analyze the individual agent plans and create a merged multi-agent plan that:
- Interleaves actions from different agents based on dependencies
- Allows parallel execution where safe (actions with no conflicting effects)
- Ensures all preconditions are satisfied before each action
- Achieves the overall goal

## CRITICAL Output Format Rules

You MUST output actions in this EXACT format:
(action-name agent-name arg1 arg2 ...)

Where:
- action-name: The PDDL action name from the domain definition
- agent-name: The agent performing the action - MUST be the SECOND token
- arg1, arg2, ...: Object arguments for the action

EXAMPLES of CORRECT format:
(action-1 agent-1 obj-a obj-b)
(action-2 agent-2 obj-c obj-d)
(action-3 agent-1 obj-e)
(action-4 agent-3 obj-f obj-g obj-h)

FORBIDDEN:
- Do NOT write: (action-1 obj-a obj-b agent-1) - agent must be second!
- Do NOT write: (agent-1, agent-2, agent-3) - no agent lists!
- Do NOT include PRE:, ADD:, DEL:, or any operator-debug text
- Do NOT add explanations or comments
- Do NOT number the lines (just raw actions)

## Merged Plan:
"""

        return prompt

    def _format_action_with_agent_second(self, action: str, agent: str) -> str:
        """Format action as (action-name agent arg1 arg2 ...), avoiding duplicate agent insertion."""
        text = action.strip()
        if not text:
            return text

        if text.startswith("(") and text.endswith(")"):
            tokens = text[1:-1].strip().split()
        else:
            tokens = text.split()

        if not tokens:
            return text

        action_name = tokens[0]
        args = tokens[1:]

        # Remove existing agent occurrence if present in args, then place it second
        args = [t for t in args if t != agent]
        return "(" + " ".join([action_name, agent] + args) + ")"

    def _build_repair_prompt(
        self,
        agent_plans: Dict[str, List[str]],
        domain_content: str,
        problem_content: str,
        failed_plan: List[str],
        val_output: str,
    ) -> str:
        """Build repair prompt with VAL feedback.

        Args:
            agent_plans: Individual agent plans
            domain_content: Domain PDDL
            problem_content: Problem PDDL
            failed_plan: The plan that failed validation
            val_output: VAL validator output

        Returns:
            Repair prompt string
        """
        relevant_val_feedback = self._extract_relevant_val_feedback(val_output)

        prompt = f"""# Plan Repair Task

The merged plan you provided failed validation. Here is the validation output:

```
    {relevant_val_feedback}
```

## Failed Plan

"""
        for i, action in enumerate(failed_plan, 1):
            prompt += f"{i}. {action}\n"

        prompt += """

## Instructions

Analyze the validation errors and fix the plan. Common issues:
- Actions executed before preconditions are met
- Conflicting actions (e.g., two agents trying to manipulate the same object)
- Missing dependencies between agent actions
- Incorrect action sequencing

## CRITICAL Output Format Rules

You MUST output actions in this EXACT format:
(action-name agent-name arg1 arg2 ...)

Where:
- action-name: The PDDL action name from the domain definition
- agent-name: The agent performing the action - MUST be the SECOND token
- arg1, arg2, ...: Object arguments for the action

EXAMPLES of CORRECT format:
(action-1 agent-1 obj-a obj-b)
(action-2 agent-2 obj-c obj-d)
(action-3 agent-1 obj-e)
(action-4 agent-3 obj-f obj-g obj-h)

FORBIDDEN:
- Do NOT write: (action-1 obj-a obj-b agent-1) - agent must be second!
- Do NOT write: (agent-1, agent-2, agent-3) - no agent lists!
- Do NOT add explanations or comments
- Do NOT number the lines (just raw actions)

## Corrected Plan:
"""
        return prompt

    def _extract_relevant_val_feedback(
        self, val_output: str, max_chars: int = 2500
    ) -> str:
        """Extract the most relevant VAL feedback for plan repair prompts.

        Prioritizes:
        1) Plan Repair Advice block
        2) Goal not satisfied / plan invalid status
        3) Last execution happening block for local context
        """
        if not val_output:
            return ""

        text = val_output.strip()

        # Start from last happening block when available.
        last_happening_idx = text.rfind("Checking next happening")
        tail = text[last_happening_idx:] if last_happening_idx != -1 else text

        # If repair advice exists, keep from there to the end.
        advice_idx = tail.find("Plan Repair Advice:")
        if advice_idx != -1:
            tail = tail[advice_idx:]
            return tail[:max_chars]

        # Otherwise, prefer from goal-status markers.
        for marker in ["Goal not satisfied", "Plan invalid", "Failed plans:"]:
            idx = tail.find(marker)
            if idx != -1:
                tail = tail[idx:]
                break

        return tail[:max_chars]

    def _fallback_sequential_merge(
        self, agent_plans: Dict[str, List[str]]
    ) -> List[str]:
        """Simple fallback: concatenate plans sequentially.

        Args:
            agent_plans: Individual agent plans

        Returns:
            Sequentially merged plan
        """
        print("[FALLBACK] Using sequential merge")
        merged = []
        for agent, plan in agent_plans.items():
            for action in plan:
                action = action.strip()
                if not action.startswith("(") or not action.endswith(")"):
                    merged.append(action)
                    continue
                tokens = action[1:-1].split()
                if not tokens:
                    merged.append(action)
                    continue
                action_name = tokens[0]
                args = tokens[1:]
                merged.append("(" + " ".join([action_name, agent] + args) + ")")
        return merged

    def _compress_pddl(self, content: str) -> str:
        """Compress PDDL by removing comments and formatting actions compactly."""
        import re

        # Remove comments
        lines = []
        for line in content.split("\n"):
            if ";" in line:
                line = line[: line.index(";")]
            line = line.strip()
            if line:
                lines.append(line)
        content = " ".join(lines)

        # Compact action definitions: compress preconditions and effects to single lines
        def compact_action(match):
            action_text = match.group(0)

            # Extract precondition block and compress
            prec_match = re.search(
                r":precondition\s*\(and\s+([^)]+(?:\([^)]*\)[^)]*)*)\)", action_text
            )
            if prec_match:
                prec_content = prec_match.group(1)
                # Extract all atomic predicates
                predicates = re.findall(r"\([^()]+\)", prec_content)
                if predicates:
                    compact_prec = " ; ".join(p.strip() for p in predicates)
                    action_text = action_text.replace(
                        prec_match.group(0), f":precondition {compact_prec}"
                    )

            # Extract effect block and compress
            eff_match = re.search(
                r":effect\s*\(and\s+([^)]+(?:\([^)]*\)[^)]*)*)\)", action_text
            )
            if eff_match:
                eff_content = eff_match.group(1)
                # Separate positive and negative effects
                add_effects = []
                del_effects = []

                # Find all effect literals
                for literal in re.findall(
                    r"\((?:not\s+)?\([^()]+\)|\([^()]+\)", eff_content
                ):
                    literal = literal.strip()
                    if literal.startswith("(not "):
                        # Delete effect
                        inner = re.search(r"\(not\s+(.+)\)", literal)
                        if inner:
                            del_effects.append(inner.group(1).strip())
                    else:
                        # Add effect
                        add_effects.append(literal)

                compact_eff_parts = []
                if add_effects:
                    compact_eff_parts.append("ADD: " + " ".join(add_effects))
                if del_effects:
                    compact_eff_parts.append("DEL: " + " ".join(del_effects))

                if compact_eff_parts:
                    compact_eff = " | ".join(compact_eff_parts)
                    action_text = action_text.replace(
                        eff_match.group(0), f":effect {compact_eff}"
                    )

            return action_text

        # Apply compaction to all actions
        content = re.sub(
            r":action\s+\w+.*?(?=:action|\Z)", compact_action, content, flags=re.DOTALL
        )

        return content
