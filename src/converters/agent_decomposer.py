"""Decompose unfactored MA-PDDL into single-agent PDDL problems."""

from pathlib import Path
import subprocess
import re
from typing import List, Dict, Tuple, Optional


class AgentDecomposer:
    """Decompose MA-PDDL into separate single-agent PDDL problems."""

    def __init__(self, python_cmd: str = "python2"):
        """Initialize agent decomposer.

        Args:
            python_cmd: Python 2 command for running converter scripts
        """
        self.python_cmd = python_cmd
        self.centralized_converter = (
            Path(__file__).parent.parent.parent
            / "codmap-2015"
            / "converters"
            / "unfactoredMAPDDL-to-PDDL.py"
        )
        self.factored_converter = (
            Path(__file__).parent.parent.parent
            / "codmap-2015"
            / "converters"
            / "unfactoredMAPDDL-to-factoredMAPDDL.py"
        )

        if not self.centralized_converter.exists():
            raise FileNotFoundError(
                f"Centralized converter not found: {self.centralized_converter}"
            )
        if not self.factored_converter.exists():
            raise FileNotFoundError(
                f"Factored converter not found: {self.factored_converter}"
            )

    def extract_agents(
        self,
        domain_file: str,
        problem_file: str,
        converter_output_dir: Optional[str] = None,
    ) -> List[str]:
        """Extract list of agents from MA-PDDL problem file.

        Args:
            domain_file: Path to MA-PDDL domain file
            problem_file: Path to MA-PDDL problem file

        Returns:
            List of agent names
        """
        agents = []

        # Prefer codmap-2015 converter if available
        converter_agents = self._extract_agents_with_converter(
            domain_file=domain_file,
            problem_file=problem_file,
            output_dir=converter_output_dir,
        )
        if converter_agents:
            return sorted(
                set(a for a in converter_agents if a and a.lower() != "private")
            )

        with open(problem_file, "r") as f:
            content = f.read()

        # Look for :private declarations which indicate agent-specific content
        # Format: (:private <agent> ...)
        private_pattern = r"\(:private\s+(\w+)"
        matches = re.findall(private_pattern, content)
        agents.extend(matches)

        # Also check domain file for agent types and actions
        with open(domain_file, "r") as f:
            domain_content = f.read()

        # Look for :agent parameter in actions
        agent_pattern = r":agent\s+\?(\w+)\s+-\s+(\w+)"
        agent_matches = re.findall(agent_pattern, domain_content)

        # Extract agent types
        agent_types = set([match[1] for match in agent_matches])

        # Now find objects of those types in problem file
        if agent_types:
            for agent_type in agent_types:
                # Look for objects of this type
                obj_pattern = rf"(\w+(?:\s+\w+)*)\s+-\s+{agent_type}"
                obj_matches = re.findall(obj_pattern, content)
                for match in obj_matches:
                    agents.extend(match.split())

        # Remove duplicates and sort, filter invalid tokens
        agents = sorted(set(a for a in agents if a and a.lower() != "private"))

        return agents

    def _extract_agents_with_converter(
        self, domain_file: str, problem_file: str, output_dir: Optional[str] = None
    ) -> List[str]:
        """Use codmap-2015 converter to extract agents list when possible."""
        try:
            domain_path = Path(domain_file)
            problem_path = Path(problem_file)
            domain_dir = domain_path.parent
            domain_stem = domain_path.stem
            problem_stem = problem_path.stem

            if output_dir is None:
                output_dir = str(Path("/tmp/ma-decomp-converted") / problem_stem)

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if not self.centralized_converter.exists():
                return []

            cmd = [
                self.python_cmd,
                str(self.centralized_converter),
                str(domain_dir),
                domain_stem,
                problem_stem,
                str(output_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return []

            agents_file = output_path / f"{problem_stem}.agents"
            if not agents_file.exists():
                return []

            with open(agents_file, "r") as f:
                agents = [line.strip() for line in f.readlines() if line.strip()]

            return agents
        except Exception:
            return []

    def decompose_to_single_agent(
        self,
        domain_dir: str,
        domain_file: str,
        problem_file: str,
        agent: str,
        output_dir: str,
        converter_output_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Create single-agent PDDL problem for a specific agent.

        Prefers converter-generated centralized PDDL when available.
        Falls back to direct MA-PDDL extraction otherwise.

        Args:
            domain_dir: Directory containing MA-PDDL files
            domain_file: Domain file name
            problem_file: Problem file name
            agent: Agent name to extract
            output_dir: Output directory for single-agent PDDL files
            converter_output_dir: Optional converter output directory

        Returns:
            Tuple of (domain_path, problem_path) for single-agent PDDL
        """
        domain_stem = Path(domain_file).stem
        problem_stem = Path(problem_file).stem

        # Try factored converter first for agent-specific domain/problem
        if converter_output_dir is None:
            converter_output_dir = str(Path("/tmp/ma-decomp-factored") / problem_stem)

        factored_path = Path(converter_output_dir)
        factored_domain = factored_path / problem_stem / f"domain-{agent}.pddl"
        factored_problem = factored_path / problem_stem / f"problem-{agent}.pddl"

        if factored_domain.exists() and factored_problem.exists():
            # Convert factored MA-PDDL to single-agent centralized PDDL
            return self._convert_factored_to_centralized(
                factored_domain=str(factored_domain),
                factored_problem=str(factored_problem),
                agent=agent,
                output_dir=output_dir,
            )

        # Fallback: direct extraction
        output_path = Path(output_dir) / agent
        output_path.mkdir(parents=True, exist_ok=True)

        # Read original MA-PDDL domain
        domain_path = Path(domain_dir) / domain_file
        with open(domain_path, "r") as f:
            domain_content = f.read()

        # Read original MA-PDDL problem
        problem_path = Path(domain_dir) / problem_file
        with open(problem_path, "r") as f:
            problem_content = f.read()

        # Extract agent-specific actions
        agent_domain = self._extract_agent_domain(domain_content, agent)

        # Extract agent-specific problem
        agent_problem = self._extract_agent_problem(problem_content, agent, domain_stem)

        # Write single-agent PDDL files
        agent_domain_file = output_path / f"{domain_stem}_{agent}.pddl"
        agent_problem_file = output_path / f"{problem_stem}_{agent}.pddl"

        with open(agent_domain_file, "w") as f:
            f.write(agent_domain)

        with open(agent_problem_file, "w") as f:
            f.write(agent_problem)

        return str(agent_domain_file), str(agent_problem_file)

    def _extract_agent_domain(self, domain_content: str, agent: str) -> str:
        """Extract agent-specific domain definition.

        Args:
            domain_content: Full MA-PDDL domain content
            agent: Agent name

        Returns:
            Single-agent PDDL domain content
        """
        lines = domain_content.split("\n")
        result = []
        in_action = False
        in_agent_action = False
        action_buffer = []
        paren_depth = 0

        for line in lines:
            # Track domain header, requirements, types, predicates
            if any(
                keyword in line
                for keyword in ["(define", ":requirements", ":types", ":constants"]
            ):
                result.append(line)
                continue

            # Handle predicates section
            if ":predicates" in line or ":functions" in line:
                result.append(line)
                continue

            # Track actions
            if "(:action" in line:
                in_action = True
                action_buffer = [line]
                paren_depth = line.count("(") - line.count(")")

                # Check if this action has :agent parameter matching our agent
                # We'll check this when we've read the full action
                continue

            if in_action:
                action_buffer.append(line)
                paren_depth += line.count("(") - line.count(")")

                # Check if action is complete
                if paren_depth == 0 and ")" in line:
                    # Check if this action belongs to our agent
                    action_text = "\n".join(action_buffer)
                    if self._is_agent_action(action_text, agent):
                        # Remove agent parameter and add to result
                        cleaned_action = self._remove_agent_parameter(action_text)
                        result.append(cleaned_action)

                    in_action = False
                    action_buffer = []
                continue

            # Include other lines (closing braces, etc.)
            if not in_action:
                result.append(line)

        return "\n".join(result)

    def _extract_agent_problem(
        self, problem_content: str, agent: str, domain_name: str
    ) -> str:
        """Extract agent-specific problem definition.

        Args:
            problem_content: Full MA-PDDL problem content
            agent: Agent name
            domain_name: Domain name

        Returns:
            Single-agent PDDL problem content
        """
        lines = problem_content.split("\n")
        result = []
        in_private = False
        is_agent_private = False
        skip_depth = 0

        for line in lines:
            # Handle problem header
            if "(define" in line:
                result.append(line)
                continue

            if ":domain" in line:
                result.append(f"  (:domain {domain_name})")
                continue

            # Handle objects - filter out other agents
            if ":objects" in line:
                result.append(line)
                continue

            # Handle init section
            if ":init" in line:
                result.append(line)
                in_private = False
                continue

            # Handle private sections
            if "(:private" in line:
                in_private = True
                is_agent_private = agent in line
                if is_agent_private:
                    # Skip the private wrapper but include contents
                    skip_depth = 1
                else:
                    skip_depth = 1
                continue

            if in_private:
                if "(" in line:
                    skip_depth += line.count("(")
                if ")" in line:
                    skip_depth -= line.count(")")

                if is_agent_private and skip_depth > 0:
                    # Include agent's private init
                    result.append(line)

                if skip_depth == 0:
                    in_private = False
                    is_agent_private = False
                continue

            # Handle goal section
            if ":goal" in line:
                result.append(line)
                continue

            # Include other lines
            if not in_private or is_agent_private:
                result.append(line)

        return "\n".join(result)

    def _is_agent_action(self, action_text: str, agent: str) -> bool:
        """Check if action belongs to specified agent.

        Args:
            action_text: Full action definition
            agent: Agent name to check

        Returns:
            True if action belongs to agent
        """
        # Look for :agent parameter
        # Format: :agent ?<var> - <type>
        agent_pattern = r":agent\s+\?(\w+)\s+-\s+(\w+)"
        match = re.search(agent_pattern, action_text)

        if not match:
            # No agent parameter, this is a shared action - include for all agents
            return True

        # For now, include all actions with agent parameter
        # In more sophisticated version, could filter by agent type
        return True

    def _remove_agent_parameter(self, action_text: str) -> str:
        """Remove :agent parameter from action definition.

        Args:
            action_text: Full action definition

        Returns:
            Action definition without agent parameter
        """
        # Remove :agent ?var - type from parameters
        result = re.sub(r":agent\s+\?\w+\s+-\s+\w+\s*", "", action_text)

        # Also remove agent variable from preconditions/effects if it appears alone
        # This is a simple version - more sophisticated would track variable usage

        return result

    def _convert_factored_to_centralized(
        self, factored_domain: str, factored_problem: str, agent: str, output_dir: str
    ) -> Tuple[str, str]:
        """Convert factored MA-PDDL to centralized single-agent PDDL.

        Manually strips agent parameters and privacy annotations.

        Args:
            factored_domain: Path to factored domain
            factored_problem: Path to factored problem
            agent: Agent name
            output_dir: Output directory base

        Returns:
            Tuple of (domain_path, problem_path)
        """
        output_path = Path(output_dir) / agent
        output_path.mkdir(parents=True, exist_ok=True)

        # Read factored files
        with open(factored_domain) as f:
            domain_content = f.read()
        with open(factored_problem) as f:
            problem_content = f.read()

        # Strip agent parameters and privacy from domain
        centralized_domain = self._strip_agent_params_from_domain(domain_content, agent)

        # Strip privacy from problem
        centralized_problem = self._strip_privacy_from_problem(problem_content, agent)

        # Write centralized files
        agent_domain_file = output_path / f"domain-{agent}.pddl"
        agent_problem_file = output_path / f"problem-{agent}.pddl"

        with open(agent_domain_file, "w") as f:
            f.write(centralized_domain)
        with open(agent_problem_file, "w") as f:
            f.write(centralized_problem)

        print(f"[CONVERTER] Centralized {agent}")
        return str(agent_domain_file), str(agent_problem_file)

    def _copy_factored_output(
        self, factored_domain: str, factored_problem: str, agent: str, output_dir: str
    ) -> Tuple[str, str]:
        """Copy factored converter output to final location.

        The factored converter creates agent-specific domain/problem files.
        Simply copy them to the output location.

        Args:
            factored_domain: Path to factored domain
            factored_problem: Path to factored problem
            agent: Agent name
            output_dir: Output directory base

        Returns:
            Tuple of (domain_path, problem_path)
        """
        output_path = Path(output_dir) / agent
        output_path.mkdir(parents=True, exist_ok=True)

        domain_stem = Path(factored_domain).stem
        problem_stem = Path(factored_problem).stem

        agent_domain_file = output_path / f"{domain_stem}.pddl"
        agent_problem_file = output_path / f"{problem_stem}.pddl"

        # Copy factored files
        with open(factored_domain) as f:
            domain_content = f.read()
        with open(factored_problem) as f:
            problem_content = f.read()

        with open(agent_domain_file, "w") as f:
            f.write(domain_content)
        with open(agent_problem_file, "w") as f:
            f.write(problem_content)

        return str(agent_domain_file), str(agent_problem_file)

    def _extract_agent_from_centralized(
        self, converter_domain: str, converter_problem: str, agent: str, output_dir: str
    ) -> Tuple[str, str]:
        """Extract single-agent PDDL from converter's centralized output.

        Filter out the agent parameter and create single-agent versions of actions.

        Args:
            converter_domain: Path to centralized domain (from converter)
            converter_problem: Path to centralized problem (from converter)
            agent: Agent name
            output_dir: Output directory base

        Returns:
            Tuple of (domain_path, problem_path)
        """
        output_path = Path(output_dir) / agent
        output_path.mkdir(parents=True, exist_ok=True)

        with open(converter_domain) as f:
            domain_content = f.read()
        with open(converter_problem) as f:
            problem_content = f.read()

        # Remove agent parameter from actions
        agent_domain = self._remove_agent_param_from_domain(domain_content)
        # Remove agent grounding from problem init
        agent_problem = self._ground_problem_for_agent(problem_content, agent)

        # Write out
        domain_stem = Path(converter_domain).stem
        problem_stem = Path(converter_problem).stem
        agent_domain_file = output_path / f"{domain_stem}_{agent}.pddl"
        agent_problem_file = output_path / f"{problem_stem}_{agent}.pddl"

        with open(agent_domain_file, "w") as f:
            f.write(agent_domain)
        with open(agent_problem_file, "w") as f:
            f.write(agent_problem)

        return str(agent_domain_file), str(agent_problem_file)

    def _remove_agent_param_from_domain(self, domain_content: str) -> str:
        """Remove first agent parameter from all action definitions.

        The converter generates actions like:
          (:action pick-up :parameters (?a - agent ?x - block) ...)

        We convert to:
          (:action pick-up :parameters (?x - block) ...)
        """
        lines = []
        in_action = False
        in_params = False

        for line in domain_content.split("\n"):
            # Detect action start
            if "(:action" in line:
                in_action = True
                lines.append(line)
                continue

            # Detect end of action
            if in_action and line.strip() == ")":
                lines.append(line)
                in_action = False
                continue

            # Handle parameters section
            if in_action and ":parameters" in line:
                # Remove first agent parameter
                # Pattern: :parameters (?a - agent ?x - block) -> :parameters (?x - block)
                modified = re.sub(r"\(\?\w+\s+-\s+agent\s+", "(", line)
                lines.append(modified)
                continue

            lines.append(line)

        return "\n".join(lines)

    def _ground_problem_for_agent(self, problem_content: str, agent: str) -> str:
        """Replace agent variable references in problem with specific agent.

        The converter generates problems with agent objects. We keep the problem
        as-is since it applies to all agents (the domain actions are parametric).
        """
        return problem_content

    def decompose_all_agents(
        self, domain_dir: str, domain_file: str, problem_file: str, output_base_dir: str
    ) -> Dict[str, Tuple[str, str]]:
        """Decompose MA-PDDL into single-agent problems for all agents.

        Uses factored converter to generate agent-specific PDDL with per-agent goals.

        Args:
            domain_dir: Directory containing MA-PDDL files
            domain_file: Domain file name
            problem_file: Problem file name
            output_base_dir: Base output directory

        Returns:
            Dictionary mapping agent name to (domain_path, problem_path)
        """
        # First, convert to factored MA-PDDL (per-agent)
        problem_stem = Path(problem_file).stem
        output_dir = Path(output_base_dir) / problem_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        factored_output_dir = str(Path("/tmp/ma-decomp-factored") / problem_stem)

        # Run factored converter to generate per-agent PDDL
        domain_path = Path(domain_dir) / domain_file
        problem_path = Path(domain_dir) / problem_file

        self._run_factored_converter(
            domain_dir=domain_dir,
            domain_file=domain_file,
            problem_file=problem_file,
            output_dir=factored_output_dir,
        )

        # Extract agents
        agents = self.extract_agents(
            str(domain_path),
            str(problem_path),
            converter_output_dir=factored_output_dir,
        )

        print(f"[DECOMPOSER] Found {len(agents)} agents: {agents}")

        # For each agent, copy factored output
        agent_files = {}
        for agent in agents:
            print(f"[DECOMPOSER] Creating single-agent problem for: {agent}")
            domain_file_path, problem_file_path = self.decompose_to_single_agent(
                domain_dir=domain_dir,
                domain_file=domain_file,
                problem_file=problem_file,
                agent=agent,
                output_dir=str(output_dir),
                converter_output_dir=factored_output_dir,
            )
            agent_files[agent] = (domain_file_path, problem_file_path)

        return agent_files

    def _run_factored_converter(
        self, domain_dir: str, domain_file: str, problem_file: str, output_dir: str
    ) -> bool:
        """Run factored MA-PDDL converter to generate per-agent problems.

        Args:
            domain_dir: Directory containing MA-PDDL files
            domain_file: Domain file name
            problem_file: Problem file name
            output_dir: Output directory for factored files

        Returns:
            True if successful, False otherwise
        """
        try:
            domain_stem = Path(domain_file).stem
            problem_stem = Path(problem_file).stem
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if not self.factored_converter.exists():
                return False

            cmd = [
                self.python_cmd,
                str(self.factored_converter),
                domain_dir,
                domain_stem,
                problem_stem,
                str(output_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                print(f"[CONVERTER] Factored converter failed: {result.stderr}")
                return False

            print(f"[CONVERTER] Factored conversion successful")
            return True
        except Exception as e:
            print(f"[CONVERTER] Error running factored converter: {e}")
            return False

    def _strip_agent_params_from_domain(self, domain_content: str, agent: str) -> str:
        """Strip agent parameters and move private predicates to public."""
        lines = []
        skip_until_close = 0
        private_predicates = []
        in_predicates = False
        in_action = False

        for line in domain_content.split("\n"):
            # Remove factored-privacy requirement
            if ":factored-privacy" in line:
                line = line.replace(":factored-privacy", "").replace("  ", " ")

            # Track if we're in predicates section
            if "(:predicates" in line:
                in_predicates = True

            # Extract predicates from :private sections
            if "(:private" in line:
                skip_until_close = line.count("(") - line.count(")")
                continue

            if skip_until_close > 0:
                # Capture private predicates
                if "(" in line and not line.strip().startswith(")"):
                    # This is a predicate definition
                    pred_line = line.strip()
                    if pred_line:
                        # Remove agent parameter from predicate completely
                        pred_line = re.sub(r"\?agent\s+-\s+agent\s*", "", pred_line)
                        pred_line = re.sub(
                            r"\s+\)", ")", pred_line
                        )  # Clean trailing space
                        private_predicates.append("\t" + pred_line)

                skip_until_close += line.count("(") - line.count(")")
                if skip_until_close <= 0:
                    skip_until_close = 0
                    # Insert private predicates into public section
                    if in_predicates and private_predicates:
                        lines.extend(private_predicates)
                        private_predicates = []
                continue

            # Check if leaving predicates section
            if in_predicates and line.strip() == ")":
                in_predicates = False

            # Remove agent parameter from action parameters
            if ":parameters" in line and "?a - agent" in line:
                # Remove agent parameter
                line = re.sub(r"\?a\s+-\s+agent\s+", "", line)
                # Clean up empty params or extra spaces
                line = re.sub(r"\(\s+\)", "()", line)

            # Track when we're inside an action
            if "(:action" in line:
                in_action = True

            # Remove agent variable references in action bodies
            if in_action and "?a" in line:
                # Replace (handempty ?a) with (handempty)
                line = re.sub(r"\(handempty\s+\?a\)", "(handempty)", line)
                # Replace (holding ?a ?x) with (holding ?x)
                line = re.sub(r"\(holding\s+\?a\s+(\?\w+)\)", r"(holding \1)", line)

            # Track when action ends
            if in_action and line.strip().endswith(")"):
                paren_count = sum(
                    l.count("(") - l.count(")") for l in lines if in_action
                )
                paren_count += line.count("(") - line.count(")")
                if paren_count == 0:
                    in_action = False

            # Remove agent type from types if it's only for agents
            if "agent" in line and " - object" in line:
                line = re.sub(r"\s*agent\s+", " ", line)

            lines.append(line)

        return "\n".join(lines)

    def _strip_privacy_from_problem(self, problem_content: str, agent: str) -> str:
        """Strip :private sections and preserve agent object typing for single-agent PDDL."""
        lines = []
        skip_until_close = 0
        agent_type = self._infer_agent_type_from_problem(problem_content, agent)
        private_typed_objects: List[Tuple[str, str]] = []

        for line in problem_content.split("\n"):
            # Skip :private sections
            if "(:private" in line:
                private_typed_objects.extend(
                    self._extract_typed_objects_from_private_line(line)
                )
                skip_until_close = line.count("(") - line.count(")")
                continue

            if skip_until_close > 0:
                private_typed_objects.extend(
                    self._extract_typed_objects_from_private_line(line)
                )
                skip_until_close += line.count("(") - line.count(")")
                if skip_until_close <= 0:
                    skip_until_close = 0
                continue

            # Remove agent references from predicates in :init
            # (handempty a1) -> (handempty)
            # (holding a1 ?x) -> (holding ?x)
            if "(" in line and agent in line:
                line = re.sub(r"\(handempty\s+" + agent + r"\)", "(handempty)", line)
                line = re.sub(
                    r"\(holding\s+" + agent + r"\s+(\w+)\)", r"(holding \1)", line
                )

            lines.append(line)

        # Re-add typed objects declared under :private sections (domain-general fix).
        if private_typed_objects:
            existing_typed = set()
            for ln in lines:
                m = re.match(r"\s*([A-Za-z0-9_-]+)\s*-\s*([A-Za-z0-9_-]+)\s*$", ln)
                if m:
                    existing_typed.add((m.group(1), m.group(2)))

            in_objects = False
            insert_idx = None
            for idx, ln in enumerate(lines):
                if "(:objects" in ln:
                    in_objects = True
                    continue
                if in_objects and ln.strip() == ")":
                    insert_idx = idx
                    break

            if insert_idx is not None:
                additions = []
                for obj_name, obj_type in private_typed_objects:
                    if (obj_name, obj_type) not in existing_typed:
                        additions.append(f"\t{obj_name} - {obj_type}")
                        existing_typed.add((obj_name, obj_type))
                if additions:
                    lines[insert_idx:insert_idx] = additions

        # Ensure current agent object is still declared in :objects.
        # Some MA-PDDL inputs store rover objects in :private object blocks, which are removed above.
        if not any(
            re.search(rf"\b{re.escape(agent)}\b\s*-\s*{re.escape(agent_type)}\b", ln)
            for ln in lines
        ):
            in_objects = False
            inserted = False
            for idx, ln in enumerate(lines):
                if "(:objects" in ln:
                    in_objects = True
                    continue
                if in_objects and ln.strip() == ")":
                    lines.insert(idx, f"\t{agent} - {agent_type}")
                    inserted = True
                    break
            if not inserted:
                # Fallback: append a minimal objects block if none exists (defensive)
                lines.append("(:objects")
                lines.append(f"\t{agent} - {agent_type}")
                lines.append(")")

        return "\n".join(lines)

    def _infer_agent_type_from_problem(self, problem_content: str, agent: str) -> str:
        """Infer the concrete type of the given agent from MA-PDDL problem text.

        Looks for patterns in private object declarations, e.g.:
        (:private plane1 plane1 - aircraft)
        """
        if not problem_content or not agent:
            return "object"

        # Case 1: inline private declaration on one line
        inline_pattern = rf"\(:private\s+{re.escape(agent)}\s+{re.escape(agent)}\s*-\s*([A-Za-z0-9_-]+)"
        match = re.search(
            inline_pattern, problem_content, flags=re.IGNORECASE | re.DOTALL
        )
        if match:
            inferred = match.group(1).strip()
            return "object" if inferred.lower() == "agent" else inferred

        # Case 2: multiline private block
        # (:private plane1
        #    plane1 - aircraft
        # )
        block_pattern = rf"\(:private\s+{re.escape(agent)}\b(.*?)\)"
        block_match = re.search(
            block_pattern, problem_content, flags=re.IGNORECASE | re.DOTALL
        )
        if block_match:
            block = block_match.group(1)
            typed_match = re.search(
                rf"\b{re.escape(agent)}\s*-\s*([A-Za-z0-9_-]+)",
                block,
                flags=re.IGNORECASE,
            )
            if typed_match:
                inferred = typed_match.group(1).strip()
                return "object" if inferred.lower() == "agent" else inferred

        # Case 3: generic typed object mention (common in factored outputs)
        generic_match = re.search(
            rf"\b{re.escape(agent)}\s*-\s*([A-Za-z0-9_-]+)",
            problem_content,
            flags=re.IGNORECASE,
        )
        if generic_match:
            inferred = generic_match.group(1).strip()
            return "object" if inferred.lower() == "agent" else inferred

        # Fallback if no explicit private declaration is found.
        return "object"

    def _extract_typed_objects_from_private_line(
        self, line: str
    ) -> List[Tuple[str, str]]:
        """Extract typed objects from lines inside a :private objects block.

        Supports forms like:
            plane1 - aircraft
            cit2 - city
            a b - type
        """
        text = line.strip()
        if not text or text.startswith("(:private") or text in {")", "("}:
            return []

        # Remove trailing/leading parentheses noise
        text = text.strip("()").strip()
        if "-" not in text:
            return []

        match = re.match(r"^([A-Za-z0-9_\-\s]+?)\s*-\s*([A-Za-z0-9_-]+)$", text)
        if not match:
            return []

        names_part = match.group(1).strip()
        obj_type = match.group(2).strip()
        if obj_type.lower() == "agent":
            obj_type = "object"
        names = [n for n in names_part.split() if n and n != ":private"]
        return [(name, obj_type) for name in names]
