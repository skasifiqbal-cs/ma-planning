"""Plan parsing utilities."""

import re
from typing import List, Set

# Simple action pattern: (op arg1 arg2 ...)
ACTION_REGEX = re.compile(r"\([^\(\)]+\)")


def compress_pddl(
    text: str, strip_comments: bool = True, compact_whitespace: bool = True
) -> str:
    """
    Compress PDDL text to reduce token count.

    Args:
        text: PDDL text to compress
        strip_comments: Remove semicolon comments
        compact_whitespace: Reduce multiple spaces/newlines to single spaces

    Returns:
        Compressed PDDL text
    """
    if strip_comments:
        # Remove semicolon comments (but preserve the content before them)
        lines = []
        for line in text.split("\n"):
            # Find semicolon not inside strings/parentheses
            comment_pos = line.find(";")
            if comment_pos >= 0:
                line = line[:comment_pos]
            if line.strip():
                lines.append(line)
        text = "\n".join(lines)

    if compact_whitespace:
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Replace multiple newlines with single newline
        text = re.sub(r"\n\s*\n+", "\n", text)
        # Remove spaces around parentheses to save tokens
        text = re.sub(r"\s*\(\s*", "(", text)
        text = re.sub(r"\s*\)\s*", ")", text)
        text = text.strip()

    return text


def extract_parenthesized_actions(text: str) -> List[str]:
    """
    Extract all parenthesized tokens that look like actions, with fallback to line-based parsing.

    Args:
        text: Raw text to parse

    Returns:
        List of action strings
    """
    if not text:
        return []

    # First try parenthesized format: (action_name arg1 arg2 ...)
    actions = [m.group(0).strip() for m in ACTION_REGEX.finditer(text)]

    # If no parenthesized actions found, try line-based format (one action per line)
    # This handles LLM output that omits parentheses
    if not actions:
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines that don't look like actions
            if (
                not line
                or line.startswith(";")
                or line.startswith("WARNING")
                or "WARNING" in line
            ):
                continue
            # If line starts with a word followed by arguments, wrap it in parens
            # This is a simple heuristic: action names are typically lowercase with hyphens
            tokens = line.split()
            if tokens and ("-" in tokens[0] or tokens[0].islower()):
                # Looks like an action line; wrap it
                actions.append(f"({line})")

    return actions


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
