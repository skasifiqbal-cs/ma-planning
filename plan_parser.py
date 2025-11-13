import re
from typing import List, Set

# Simple action pattern: (op arg1 arg2 ...)
ACTION_REGEX = re.compile(r"\([^\(\)]+\)")


def extract_parenthesized_actions(text: str) -> List[str]:
    """Extract all parenthesized tokens that look like actions, in order."""
    if not text:
        return []
    return [m.group(0).strip() for m in ACTION_REGEX.finditer(text)]


def parse_actions_no_validation(
    text: str,
    valid_ground_ops: Set[str],
    disable_name_checks: bool = False,
) -> List[str]:
    """
    No Validation parsing:
    - Scan entire LLM response for parenthesized action strings.
    - If disable_name_checks is False, keep only those that match a known grounded operator name.
      (No applicability checks.)
    - If True, accept all parenthesized strings as actions.
    """
    actions = extract_parenthesized_actions(text)
    if disable_name_checks:
        return actions
    return [a for a in actions if a in valid_ground_ops]
