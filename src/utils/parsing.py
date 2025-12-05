"""Plan parsing utilities."""

import re
from typing import List, Set

# Simple action pattern: (op arg1 arg2 ...)
ACTION_REGEX = re.compile(r"\([^\(\)]+\)")


def extract_parenthesized_actions(text: str) -> List[str]:
    """
    Extract all parenthesized tokens that look like actions.

    Args:
        text: Raw text to parse

    Returns:
        List of action strings
    """
    if not text:
        return []
    return [m.group(0).strip() for m in ACTION_REGEX.finditer(text)]


def parse_actions_no_validation(
    text: str,
    valid_ground_ops: Set[str],
    disable_name_checks: bool = False,
) -> List[str]:
    """
    Parse actions without validation.

    Args:
        text: Raw text from LLM
        valid_ground_ops: Set of valid grounded operator names
        disable_name_checks: If True, skip operator name validation

    Returns:
        List of action strings
    """
    actions = extract_parenthesized_actions(text)
    if disable_name_checks:
        return actions
    return [a for a in actions if a in valid_ground_ops]
