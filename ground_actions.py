from pyperplan.pddl.parser import Parser
from pyperplan.grounding import ground

class ActionGrounder:
    def __init__(self, domain_pddl, problem_pddl):
        parser = Parser(domain_pddl, problem_pddl)
        domain = parser.parse_domain()
        problem = parser.parse_problem(domain)
        self.task = ground(problem, domain)
        self.operators = list(self.task.operators)
        self.state = self.task.initial_state

    def applicable_actions(self):
        return [op for op in self.operators if op.applicable(self.state)]

    def apply_action(self, op):
        self.state = op.apply(self.state)