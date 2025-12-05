"""Base protocol for planning strategies."""

from typing import Protocol, List


class PlanGenerationStrategy(Protocol):
    """Protocol for plan generation strategies."""

    def generate_plan(
        self,
        ma_domain_file: str,
        ma_problem_file: str,
        ground_domain_file: str,
        ground_problem_file: str,
        max_steps: int,
    ) -> List[str]:
        """
        Generate a plan.

        Args:
            ma_domain_file: Path to MA-PDDL domain file
            ma_problem_file: Path to MA-PDDL problem file
            ground_domain_file: Path to grounded (centralized) domain file
            ground_problem_file: Path to grounded (centralized) problem file
            max_steps: Maximum number of plan steps

        Returns:
            List of action strings forming the plan
        """
        ...
