"""Task decomposition strategy: decompose multi-agent problem into per-agent tasks.

This strategy performs goal-based task decomposition:
1. Parse domain and problem to extract agents and goals
2. Infer which agent should handle which goal based on object involvement
3. Create decomposed problems for each agent (same domain, scoped init/goals)
4. Convert to single-agent PDDL via CoDMAP
5. Solve each with pyperplan
6. Merge plans through LLM coordination
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from copy import deepcopy

# Add pyperplan to path
pyperplan_path = Path(__file__).parent.parent.parent / "pyperplan"
if pyperplan_path.exists() and str(pyperplan_path) not in sys.path:
    sys.path.insert(0, str(pyperplan_path))

from ..converters.agent_decomposer import AgentDecomposer
from ..utils.parsing import extract_parenthesized_actions
from ..validation.evaluator import PlanEvaluator


class TaskDecompositionStrategy:
    """Decompose multi-agent MA-PDDL into per-agent tasks."""

    def __init__(self, llm, config, **kwargs):
        """Initialize strategy.

        Args:
            llm: LLM provider instance
            config: Configuration object
            **kwargs: Additional parameters
        """
        self.llm = llm
        self.config = config
        self.decomposer = AgentDecomposer(
            python_cmd=getattr(config, "python_cmd", "python")
        )
        self.debug = getattr(config, "verbose", False)

    def generate_plan(
        self,
        ma_domain_file: str,
        ma_problem_file: str,
        ground_domain_file: str,
        ground_problem_file: str,
        max_steps: int = 1000,
        **kwargs,
    ) -> Optional[List[str]]:
        """Generate plan through task decomposition + solving + merging.

        Args:
            ma_domain_file: Path to centralized/unfactored MA-PDDL domain
            ma_problem_file: Path to centralized/unfactored MA-PDDL problem
            ground_domain_file: Path to grounded domain (not used in task decomposition)
            ground_problem_file: Path to grounded problem (not used in task decomposition)
            max_steps: Maximum plan steps
            **kwargs: Additional arguments (debug, validate, etc.)

        Returns:
            Merged multi-agent plan
        """
        debug = kwargs.get("debug", self.debug)

        if debug:
            print(f"\n[TASK-DECOMP] Starting task decomposition pipeline")
            print(f"[TASK-DECOMP] Ma-domain: {ma_domain_file}")
            print(f"[TASK-DECOMP] Ma-problem: {ma_problem_file}")

        # Step 1: Parse domain and problem
        domain_content = Path(ma_domain_file).read_text()
        problem_content = Path(ma_problem_file).read_text()

        # Check if this is unfactored MA-PDDL or centralized PDDL
        is_unfactored = (
            ":agent" in domain_content or ":unfactored-privacy" in domain_content
        )

        if not is_unfactored:
            # If we received centralized PDDL, we need the unfactored version for decomposition
            # Try to find unfactored version
            unfactored_dir = (
                Path(ma_domain_file).parent.parent.parent
                / "unfactored"
                / Path(ma_domain_file).parent.name
            )
            if unfactored_dir.exists():
                unfactored_domain = unfactored_dir / "domain.pddl"
                unfactored_problem = unfactored_dir / Path(ma_problem_file).name
                if unfactored_domain.exists() and unfactored_problem.exists():
                    domain_content = unfactored_domain.read_text()
                    problem_content = unfactored_problem.read_text()
                    if debug:
                        print(
                            f"[TASK-DECOMP] Using unfactored MA-PDDL from: {unfactored_dir}"
                        )
                else:
                    print(
                        "[TASK-DECOMP] WARNING: Using centralized PDDL (unfactored not found)"
                    )
            else:
                print(
                    "[TASK-DECOMP] WARNING: Using centralized PDDL (unfactored directory not found)"
                )

        agents = self._extract_agents(problem_content)
        if debug:
            print(f"[TASK-DECOMP] Extracted agents: {agents}")

        # Step 2: Use LLM to decompose tasks
        if debug:
            print(
                f"[TASK-DECOMP] Using LLM to decompose problem into agent-specific sub-problems..."
            )

        decomposed_problems = self._llm_decompose_tasks(
            domain_content=domain_content,
            problem_content=problem_content,
            agents=agents,
            debug=debug,
        )

        if not decomposed_problems:
            print("[TASK-DECOMP] ERROR: LLM failed to decompose problem")
            return None

        if debug:
            print(
                f"[TASK-DECOMP] Created {len(decomposed_problems)} decomposed problems"
            )

        # Step 3: Save decomposed problems as PDDL files
        results_dir = (
            Path(self.config.results_root)
            / Path(ma_domain_file).parent.name
            / Path(ma_problem_file).stem
            / "decomposed"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        problem_name = self._extract_problem_name(problem_content)
        domain_name = self._extract_domain_name(domain_content)
        objects = self._get_objects_section(problem_content)

        problem_files = {}
        for agent, (init_facts, goals) in decomposed_problems.items():
            # Build PDDL problem file
            init_section = "\n    ".join(init_facts) if init_facts else ""
            goals_section = "\n      ".join(goals) if goals else "(and )"

            problem_pddl = f"""(define (problem {problem_name}-{agent})
  (:domain {domain_name})
  (:objects
{objects}
  )
  (:init
    {init_section}
  )
  (:goal
    (and
      {goals_section}
    )
  )
)
"""
            problem_file_path = results_dir / f"problem_{agent}.pddl"
            problem_file_path.write_text(problem_pddl)
            problem_files[agent] = str(problem_file_path)

            # Also save the MA-PDDL domain file for this agent (same domain for all agents)
            domain_file_path = results_dir / f"domain_{agent}.pddl"
            domain_file_path.write_text(domain_content)

            if debug:
                print(
                    f"[TASK-DECOMP] Saved decomposed problem for {agent}: {problem_file_path}"
                )

        # Step 4: Convert to single-agent PDDL using CoDMAP converter
        converted_problems = {}
        converter_script = getattr(self.config, "resolved_converter_script", None)
        python_cmd = getattr(self.config, "python_cmd", "python2")

        if not converter_script:
            print("[TASK-DECOMP] ERROR: CoDMAP converter not found")
            return None

        for agent, problem_file_path in problem_files.items():
            if debug:
                print(f"[TASK-DECOMP] Converting {agent} to single-agent PDDL...")

            # CoDMAP converter expects: <input_dir> <domain_name_no_ext> <problem_name_no_ext> <output_dir>
            # Create output directory for this agent
            agent_output_dir = results_dir / f"converted_{agent}"
            agent_output_dir.mkdir(exist_ok=True)

            # Domain and problem base names (without .pddl)
            domain_base = f"domain_{agent}"
            problem_base = f"problem_{agent}"

            # Run CoDMAP converter
            try:
                import subprocess

                result = subprocess.run(
                    [
                        python_cmd,
                        converter_script,
                        str(results_dir),  # input directory
                        domain_base,  # domain filename without .pddl
                        problem_base,  # problem filename without .pddl
                        str(agent_output_dir),
                    ],  # output directory
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    if debug:
                        print(f"[CODMAP] Conversion error for {agent}:")
                        print(f"  Return code: {result.returncode}")
                        print(f"  Stderr: {result.stderr[:300]}")
                        print(f"  Stdout: {result.stdout[:300]}")
                    continue

                # Check if output files were created
                output_domain = agent_output_dir / f"{domain_base}.pddl"
                output_problem = agent_output_dir / f"{problem_base}.pddl"

                if output_domain.exists() and output_problem.exists():
                    # Post-process: add agent to objects section if not present
                    problem_text = output_problem.read_text()
                    if (
                        agent not in problem_text
                        or f"{agent} - agent" not in problem_text
                    ):
                        # Add agent to objects section after (:objects
                        problem_text = re.sub(
                            r"(\(:objects\s*\n)",
                            f"\\1\t{agent} - agent\n",
                            problem_text,
                            count=1,
                        )
                        output_problem.write_text(problem_text)
                        if debug:
                            print(f"[CODMAP] Added {agent} to objects section")

                    converted_problems[agent] = (
                        str(output_domain),
                        str(output_problem),
                    )
                    if debug:
                        print(f"[CODMAP] Successfully converted {agent}")
                else:
                    if debug:
                        print(f"[CODMAP] Output files not created for {agent}")
                        print(f"  Expected domain: {output_domain}")
                        print(f"  Expected problem: {output_problem}")

            except Exception as e:
                if debug:
                    print(f"[CODMAP] Exception converting {agent}: {e}")
                continue

        if not converted_problems:
            print("[TASK-DECOMP] ERROR: No problems successfully converted")
            return None

        # Step 5: Solve with pyperplan
        agent_plans = {}
        for agent, (converted_domain, converted_problem) in converted_problems.items():
            if debug:
                print(f"[TASK-DECOMP] Solving converted problem for {agent}...")

            plan = self._solve_with_pyperplan(
                converted_domain, converted_problem, agent, debug
            )
            if plan:
                agent_plans[agent] = plan
                if debug:
                    print(f"[TASK-DECOMP] Found {len(plan)} actions for {agent}")
            else:
                if debug:
                    print(f"[TASK-DECOMP] No plan found for {agent}")

        if not agent_plans:
            print("[TASK-DECOMP] ERROR: No agent plans generated")
            return None

        # Step 6: Merge plans (use sequential merge)
        merged_plan = self._merge_plans(
            agent_plans=agent_plans,
            domain_content=domain_content,
            problem_content=problem_content,
            debug=debug,
        )

        if debug:
            print(f"[TASK-DECOMP] Merged plan has {len(merged_plan)} actions")

        return merged_plan

    def _extract_agents(self, problem_content: str) -> List[str]:
        """Extract agent names from problem init state.

        Args:
            problem_content: Problem PDDL content

        Returns:
            List of agent names
        """
        agents = set()

        # Find (handempty agent) or similar predicates
        for match in re.finditer(r"\(handempty\s+(\w+)\)", problem_content):
            agents.add(match.group(1))

        # Also find agents in goal section
        goal_section = re.search(
            r":goal\s*\(([^)]+(?:\([^)]*\))*)\)", problem_content, re.DOTALL
        )
        if goal_section:
            for match in re.finditer(r"\((\w+)\)", goal_section.group(1)):
                # Filter out action names, keep only agent-like names (usually a1, a2, etc.)
                name = match.group(1)
                if name.startswith("a") and name[1:].isdigit():
                    agents.add(name)

        return sorted(agents)

    def _llm_decompose_tasks(
        self,
        domain_content: str,
        problem_content: str,
        agents: List[str],
        debug: bool = False,
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Use LLM to decompose multi-agent problem into per-agent sub-problems.

        Args:
            domain_content: Domain PDDL content
            problem_content: Problem PDDL content
            agents: List of agent names
            debug: Debug flag

        Returns:
            Dictionary mapping agent name to (init_facts, goal_predicates) tuple
        """
        prompt = self._build_decomposition_prompt(
            domain_content=domain_content,
            problem_content=problem_content,
            agents=agents,
        )

        if debug:
            print(f"[LLM-DECOMP] Prompt length: {len(prompt)} chars")

        messages = [
            {
                "role": "system",
                "content": "You are an expert in multi-agent planning and PDDL. You decompose multi-agent planning problems into independent sub-problems for each agent.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm.chat(messages)

            if debug:
                print(f"[LLM-DECOMP] Received response ({len(response)} chars)")

            # Parse LLM response
            decomposed = self._parse_llm_decomposition(response, agents, debug)
            return decomposed

        except Exception as e:
            print(f"[LLM-DECOMP] Error: {e}")
            return {}

    def _build_decomposition_prompt(
        self, domain_content: str, problem_content: str, agents: List[str]
    ) -> str:
        """Build prompt for LLM to decompose problem.

        Args:
            domain_content: Domain PDDL
            problem_content: Problem PDDL
            agents: List of agent names

        Returns:
            Prompt string
        """
        prompt = f"""# Multi-Agent Problem Decomposition Task

You are given a multi-agent planning problem with multiple agents. Your task is to decompose this problem into independent sub-problems for each agent, where each agent has its own initial state facts and goal conditions.

## Domain Definition

{domain_content}

## Problem Definition

{problem_content}

## Agents

{', '.join(agents)}

## Task

Decompose this multi-agent problem into {len(agents)} independent sub-problems, one for each agent. For each agent, determine:

1. **Initial State Facts**: Which predicates from the initial state are relevant for this agent? Include:
   - Facts about the agent itself (e.g., handempty, at-location, etc.)
   - Facts about objects this agent will manipulate to achieve its goals
   - Structural facts needed as preconditions (e.g., block arrangements, connections)

2. **Goal Conditions**: Which goal predicates should this agent achieve? Consider:
   - Distribute goals fairly across agents
   - Goals involving objects an agent is responsible for
   - Ensure all global goals are covered by at least one agent
   - Can assign same goal to multiple agents if coordination is needed

## CRITICAL Output Format

For each agent, output EXACTLY in this format:

```
AGENT: agent-name
INIT:
(predicate1 arg1 arg2)
(predicate2 arg1)
(predicate3 arg1 arg2 arg3)
GOALS:
(goal-predicate1 arg1 arg2)
(goal-predicate2 arg1)
---
```

**Rules:**
- One agent block per agent
- List all init facts for that agent (one per line)
- List all goal predicates for that agent (one per line)  
- Separate agents with `---`
- Use exact predicate and object names from the domain/problem
- Do NOT add explanations or comments
- Do NOT skip any agent

**Example Output:**

```
AGENT: a1
INIT:
(handempty a1)
(at a1 loc1)
(on block1 block2)
GOALS:
(on block1 block3)
---
AGENT: a2
INIT:
(handempty a2)
(at a2 loc2)
(on block4 block5)
GOALS:
(on block4 block6)
---
```

## Decomposition:
"""

        return prompt

    def _parse_llm_decomposition(
        self, response: str, agents: List[str], debug: bool = False
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Parse LLM decomposition response.

        Args:
            response: LLM response text
            agents: Expected agent names
            debug: Debug flag

        Returns:
            Dictionary mapping agent to (init_facts, goals)
        """
        decomposed = {}

        # Split by agent blocks
        agent_blocks = response.split("---")

        for block in agent_blocks:
            block = block.strip()
            if not block or "AGENT:" not in block:
                continue

            # Extract agent name
            agent_match = re.search(r"AGENT:\s*(\w+)", block)
            if not agent_match:
                continue
            agent = agent_match.group(1)

            # Extract init facts
            init_facts = []
            init_match = re.search(r"INIT:\s*(.*?)(?:GOALS:|$)", block, re.DOTALL)
            if init_match:
                init_section = init_match.group(1)
                for line in init_section.split("\n"):
                    line = line.strip()
                    if line.startswith("(") and line.endswith(")"):
                        init_facts.append(line)

            # Extract goals
            goals = []
            goals_match = re.search(r"GOALS:\s*(.*?)(?:---|$)", block, re.DOTALL)
            if goals_match:
                goals_section = goals_match.group(1)
                for line in goals_section.split("\n"):
                    line = line.strip()
                    if line.startswith("(") and line.endswith(")"):
                        goals.append(line)

            if init_facts or goals:
                decomposed[agent] = (init_facts, goals)
                if debug:
                    print(
                        f"[LLM-DECOMP] {agent}: {len(init_facts)} init facts, {len(goals)} goals"
                    )

        return decomposed

    def _extract_goals(self, problem_content: str) -> List[str]:
        """Extract goal predicates from problem.

        Args:
            problem_content: Problem PDDL content

        Returns:
            List of goal predicates
        """
        goals = []

        # Find the :goal section
        goal_match = re.search(r":goal\s*\((.*?)\s*\)\s*\)", problem_content, re.DOTALL)
        if not goal_match:
            return goals

        goal_content = goal_match.group(1)

        # Remove the "and" keyword if present
        goal_content = re.sub(r"\(\s*and\s+", "(", goal_content)

        # Extract all predicates: (predicate arg1 arg2 ...)
        depth = 0
        current = ""
        for char in goal_content:
            if char == "(":
                if depth == 0 and current.strip():
                    # Skip, start of new predicate
                    pass
                current += char
                depth += 1
            elif char == ")":
                depth -= 1
                current += char
                if depth == 0 and current.strip():
                    pred = current.strip()
                    if pred and pred != "()":
                        goals.append(pred)
                    current = ""
            else:
                current += char

        return goals

    def _extract_init_facts(self, problem_content: str) -> List[str]:
        """Extract init facts from problem.

        Args:
            problem_content: Problem PDDL content

        Returns:
            List of init facts
        """
        facts = []

        # Find the :init section
        init_match = re.search(
            r":init\s*\((.*?)\s*\)(?:\s*:goal|\s*:metric|\s*\))",
            problem_content,
            re.DOTALL,
        )
        if not init_match:
            return facts

        init_content = init_match.group(1)

        # Extract all predicates: (predicate arg1 arg2 ...)
        depth = 0
        current = ""
        for char in init_content:
            if char == "(":
                if depth == 0 and current.strip():
                    # Skip, start of new fact
                    pass
                current += char
                depth += 1
            elif char == ")":
                depth -= 1
                current += char
                if depth == 0 and current.strip():
                    fact = current.strip()
                    if fact and fact != "()":
                        facts.append(fact)
                    current = ""
            else:
                current += char

        return facts

    def _infer_goal_to_agent(
        self, goals: List[str], agents: List[str]
    ) -> Dict[str, List[str]]:
        """Infer which agent should handle which goals.

        For now: round-robin assignment of goals to agents.
        Could be enhanced with object-to-agent mapping from init state.

        Args:
            goals: List of goal predicates
            agents: List of agent names

        Returns:
            Mapping of agent to their goals
        """
        agent_goals = {agent: [] for agent in agents}

        # Round-robin assignment
        for i, goal in enumerate(goals):
            agent = agents[i % len(agents)]
            agent_goals[agent].append(goal)

        return agent_goals

    def _filter_facts_for_agent(
        self, facts: List[str], agent_goals: List[str], agent: str
    ) -> Tuple[List[str], Set[str]]:
        """Filter init facts relevant to agent's goals.

        Args:
            facts: All init facts
            agent_goals: Agent's goals
            agent: Agent name

        Returns:
            Tuple of (filtered_facts, involved_objects)
        """
        filtered_facts = []
        involved_objects = set()

        # Extract objects from agent's goals
        goal_objects = set()
        for goal in agent_goals:
            # Extract all objects mentioned in goal
            tokens = goal.replace("(", " ").replace(")", " ").split()
            for token in tokens:
                if (
                    token
                    and not token[0].islower()
                    and token not in ["and", "or", "not"]
                ):
                    goal_objects.add(token)

        # Always include agent
        involved_objects.add(agent)
        involved_objects.update(goal_objects)

        # Filter facts: keep if contains involved objects
        for fact in facts:
            # Always keep agent handempty facts
            if f"(handempty {agent})" in fact:
                filtered_facts.append(fact)
                continue

            # Keep if mentions goal objects
            fact_has_goal_object = any(obj in fact for obj in goal_objects)
            if fact_has_goal_object:
                filtered_facts.append(fact)
                involved_objects.add(agent)  # Agent can manipulate

        return filtered_facts, involved_objects

    def _get_objects_section(self, problem_content: str) -> str:
        """Extract objects section from problem, excluding :private sections and agents.

        Args:
            problem_content: Problem PDDL content

        Returns:
            Objects section text for inclusion in problem, excluding agent objects
        """
        # Extract between :objects and next major keyword
        match = re.search(
            r":objects\s*(.*?)(?:\(:private|\):init|:init|:goal)",
            problem_content,
            re.DOTALL,
        )
        if match:
            obj_section = match.group(1).strip()
            # Clean up and format, exclude agent type objects
            lines = []
            for line in obj_section.split("\n"):
                line = line.strip()
                # Skip empty lines, comments, malformed lines, and agent objects
                if line and not line.startswith(";") and not line.startswith("("):
                    # Skip lines with "- agent" type
                    if "- agent" not in line:
                        lines.append(line)
            return "    " + "\n    ".join(lines)
        return ""

    def _decompose_tasks(
        self,
        domain_content: str,
        problem_content: str,
        agents: List[str],
        debug: bool = False,
    ) -> Dict[str, str]:
        """Decompose multi-agent problem into per-agent tasks.

        Args:
            domain_content: Domain PDDL
            problem_content: Problem PDDL
            agents: Extracted agent names
            debug: Debug flag

        Returns:
            Dictionary mapping agent name to decomposed problem PDDL
        """
        decomposed = {}

        # Extract components
        goals = self._extract_goals(problem_content)
        init_facts = self._extract_init_facts(problem_content)
        objects = self._get_objects_section(problem_content)
        problem_name = self._extract_problem_name(problem_content)
        domain_name = self._extract_domain_name(domain_content)

        # Infer goal-to-agent mapping
        goal_to_agent = self._infer_goal_to_agent(goals, agents)

        if debug:
            print("[TASK-DECOMP] Goal-to-agent mapping:")
            for agent, agent_goals in goal_to_agent.items():
                print(f"  {agent}: {len(agent_goals)} goals")
            print(f"[TASK-DECOMP] Total facts: {len(init_facts)}")
            print(
                f"[TASK-DECOMP] Parse sample fact: {init_facts[0] if init_facts else 'None'}"
            )

        # Create decomposed problem for each agent
        for agent in agents:
            agent_goals = goal_to_agent[agent]
            if not agent_goals:
                # If no goals assigned, give agent empty goal
                agent_goals = ["(and )"]

            # Filter init facts for this agent (include all facts, agent can handle them)
            agent_init_facts = init_facts  # For task decomposition, keep all facts

            # Build decomposed problem PDDL
            domain_ref = domain_name if domain_name else "blocks"
            goal_str = " ".join(agent_goals) if len(agent_goals) > 1 else agent_goals[0]

            init_section = "\n\t".join(agent_init_facts) if agent_init_facts else ""

            problem_str = f"""(define (problem {problem_name}-{agent})
  (:domain {domain_ref})
  (:objects
{objects}
  )
  (:init
    {init_section}
  )
  (:goal
    (and
      {goal_str}
    )
  )
)
"""
            decomposed[agent] = problem_str
            if debug:
                print(
                    f"[TASK-DECOMP] Decomposed problem for {agent}: {len(agent_goals)} goals"
                )

        return decomposed

    def _extract_domain_name(self, domain_content: str) -> str:
        """Extract domain name.

        Args:
            domain_content: Domain PDDL content

        Returns:
            Domain name
        """
        match = re.search(r"\(define\s+\(domain\s+(\w+)\)", domain_content)
        if match:
            return match.group(1)
        return "domain"

    def _extract_problem_name(self, problem_content: str) -> str:
        """Extract problem name.

        Args:
            problem_content: Problem PDDL content

        Returns:
            Problem name
        """
        match = re.search(r"\(define\s+\(problem\s+(\w+)\)", problem_content)
        if match:
            return match.group(1)
        return "problem"

    def _solve_with_pyperplan(
        self, domain_file: str, problem_file: str, agent: str, debug: bool = False
    ) -> Optional[List[str]]:
        """Solve single-agent problem with pyperplan.

        Args:
            domain_file: Path to domain PDDL
            problem_file: Path to problem PDDL
            agent: Agent name
            debug: Debug flag

        Returns:
            Plan as list of actions or None if unsolvable
        """
        import subprocess

        try:
            # Convert to absolute paths
            domain_abs = str(Path(domain_file).resolve())
            problem_abs = str(Path(problem_file).resolve())
            pyperplan_dir = Path(__file__).parent.parent.parent / "pyperplan"

            # Run pyperplan with gbf + hff
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pyperplan",
                    domain_abs,
                    problem_abs,
                    "-s",
                    "gbf",
                    "-H",
                    "hff",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(pyperplan_dir),
            )

            if result.returncode != 0:
                if debug:
                    print(f"[PYPERPLAN] Error for {agent}:")
                    print(f"  Return code: {result.returncode}")
                    print(f"  Stderr: {result.stderr[:500]}")
                return None

            # Parse output plan - look for lines with actions
            plan = []
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line.startswith("(") and line.endswith(")"):
                    plan.append(line)

            if plan and debug:
                print(f"[PYPERPLAN] Found plan for {agent}: {len(plan)} actions")

            return plan if plan else None

        except subprocess.TimeoutExpired:
            if debug:
                print(f"[PYPERPLAN] Timeout for {agent}")
            return None
        except Exception as e:
            if debug:
                print(f"[PYPERPLAN] Exception for {agent}: {e}")
            return None

    def _merge_plans(
        self,
        agent_plans: Dict[str, List[str]],
        domain_content: str,
        problem_content: str,
        debug: bool = False,
    ) -> List[str]:
        """Merge individual agent plans into coordinated plan.

        For now: simple sequential merge (concatenate plans).
        Could be enhanced with LLM-based merge.

        Args:
            agent_plans: Dictionary of agent -> plan
            domain_content: Original domain PDDL
            problem_content: Original problem PDDL
            debug: Debug flag

        Returns:
            Merged plan
        """
        merged = []

        # Sequential merge: concatenate plans
        for agent in sorted(agent_plans.keys()):
            plan = agent_plans[agent]
            for action_str in plan:
                # Ensure agent is second token
                action_str = action_str.strip()
                if action_str.startswith("(") and action_str.endswith(")"):
                    tokens = action_str[1:-1].split()
                    if tokens:
                        action_name = tokens[0]
                        args = tokens[1:]

                        # Insert agent if not present
                        if agent not in args:
                            merged.append(f"({action_name} {agent} {' '.join(args)})")
                        else:
                            merged.append(action_str)
                else:
                    merged.append(action_str)

        if debug:
            print(f"[MERGE] Sequential merge: {len(merged)} actions")

        return merged
