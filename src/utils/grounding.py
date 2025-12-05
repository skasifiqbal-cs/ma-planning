"""Action grounding using pyperplan."""

from pyperplan.pddl.parser import Parser
from pyperplan.grounding import ground


class ActionGrounder:
    """Grounds PDDL actions and maintains state."""

    def __init__(self, domain_pddl: str, problem_pddl: str):
        """
        Initialize action grounder.

        Args:
            domain_pddl: Path to domain PDDL file
            problem_pddl: Path to problem PDDL file
        """
        parser = Parser(domain_pddl, problem_pddl)
        domain = parser.parse_domain()
        problem = parser.parse_problem(domain)
        self.task = ground(problem, domain)
        self.operators = list(self.task.operators)
        self.state = self.task.initial_state

    def applicable_actions(self):
        """Get all applicable actions in the current state."""
        return [op for op in self.operators if op.applicable(self.state)]

    def apply_action(self, op):
        """Apply an action and update state."""
        self.state = op.apply(self.state)
