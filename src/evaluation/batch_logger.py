"""Batch execution logger for tracking LLM output and final plans."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class BatchLogger:
    """
    Simple logger for batch planning execution.

    Tracks:
    - Raw LLM output
    - Action modifications (original → modified)
    - Final plan after all modifications
    - Validation results
    """

    def __init__(self, domain_name: str, model_name: str, output_dir: str = "results"):
        """
        Initialize batch logger.

        Args:
            domain_name: Name of the planning domain
            model_name: LLM model name
            output_dir: Base directory for logs
        """
        self.domain_name = domain_name
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.model_slug = model_name.replace("/", "-").replace(":", "-")
        self.logs: Dict[str, Dict[str, Any]] = {}

    def start_problem(self, problem_name: str):
        """Initialize log entry for a problem."""
        self.logs[problem_name] = {
            "problem": problem_name,
            "start_time": datetime.now().isoformat(),
            "llm_output": None,
            "llm_actions": [],
            "modifications": [],
            "final_plan": [],
            "validation": {"passed": False, "message": "", "details": ""},
            "timing": {"start": datetime.now().timestamp()},
        }

    def log_llm_output(
        self, problem_name: str, raw_output: str, extracted_actions: List[str]
    ):
        """Log the raw LLM output and extracted actions."""
        if problem_name not in self.logs:
            self.start_problem(problem_name)

        self.logs[problem_name]["llm_output"] = raw_output
        self.logs[problem_name]["llm_actions"] = extracted_actions.copy()

    def log_prompt(self, problem_name: str, prompt_messages: List[Dict[str, str]]):
        """Log the formatted LLM prompt (list of messages) for a problem."""
        if problem_name not in self.logs:
            self.start_problem(problem_name)

        # Store the prompt messages and a compact text version
        self.logs[problem_name]["llm_prompt_messages"] = prompt_messages
        # Create a readable prompt text for quick inspection
        prompt_text_parts = []
        for m in prompt_messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            prompt_text_parts.append(f"[{role}] {content}")
        self.logs[problem_name]["llm_prompt"] = "\n\n".join(prompt_text_parts)

    def log_action_modification(
        self,
        problem_name: str,
        action_index: int,
        original_action: str,
        modified_action: str,
        reason: str,
    ):
        """Log an action modification."""
        if problem_name not in self.logs:
            self.start_problem(problem_name)

        modification = {
            "action_index": action_index,
            "original": original_action,
            "modified": modified_action,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        self.logs[problem_name]["modifications"].append(modification)

    def log_final_plan(
        self, problem_name: str, final_actions: List[str], num_modifications: int
    ):
        """Log the final plan."""
        if problem_name not in self.logs:
            self.start_problem(problem_name)

        self.logs[problem_name]["final_plan"] = final_actions.copy()

    def log_validation_result(
        self, problem_name: str, passed: bool, message: str = "", details: str = ""
    ):
        """Log validation result."""
        if problem_name not in self.logs:
            self.start_problem(problem_name)

        self.logs[problem_name]["validation"]["passed"] = passed
        self.logs[problem_name]["validation"]["message"] = message
        self.logs[problem_name]["validation"]["details"] = details

    def log_validation_attempts(self, problem_name: str, attempts: list):
        """Log per-attempt plan and VAL output for llm-modulo strategy."""
        if problem_name not in self.logs:
            self.start_problem(problem_name)
        self.logs[problem_name]["validation_attempts"] = [
            {"attempt": n, "plan": p, "val_output": v} for n, p, v in attempts
        ]

    def log_error(self, problem_name: str, error: str, traceback_str: str = ""):
        """Log an error during planning."""
        if problem_name not in self.logs:
            self.start_problem(problem_name)

        # Store error info in validation field
        self.logs[problem_name]["validation"]["passed"] = False
        self.logs[problem_name]["validation"]["message"] = f"Error: {error}"
        if traceback_str:
            self.logs[problem_name]["validation"]["details"] = traceback_str

    def end_problem(self, problem_name: str):
        """Mark end of problem and calculate duration."""
        if problem_name in self.logs:
            self.logs[problem_name]["timing"]["end"] = datetime.now().timestamp()
            duration = (
                self.logs[problem_name]["timing"]["end"]
                - self.logs[problem_name]["timing"]["start"]
            )
            self.logs[problem_name]["timing"]["duration"] = duration

    def generate_log_report(self) -> Path:
        """
        Generate JSON log file with all details.

        Returns:
            Path to the generated log file
        """
        domain_dir = self.output_dir / self.domain_name
        domain_dir.mkdir(parents=True, exist_ok=True)

        log_file = domain_dir / "log.json"

        log_data = {
            "metadata": {
                "domain": self.domain_name,
                "model": self.model_name,
                "generated_at": datetime.now().isoformat(),
            },
            "problems": self.logs,
        }

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

        return log_file

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_problems = len(self.logs)
        total_modifications = sum(
            len(log.get("modifications", [])) for log in self.logs.values()
        )
        total_actions = sum(
            len(log.get("final_plan", [])) for log in self.logs.values()
        )
        total_llm_actions = sum(
            len(log.get("llm_actions", [])) for log in self.logs.values()
        )

        validation_stats = {
            "passed": sum(
                1
                for log in self.logs.values()
                if log.get("validation", {}).get("passed", False)
            ),
            "failed": sum(
                1
                for log in self.logs.values()
                if not log.get("validation", {}).get("passed", False)
            ),
        }

        total_time = sum(
            log.get("timing", {}).get("duration", 0) for log in self.logs.values()
        )

        return {
            "total_problems": total_problems,
            "total_llm_actions": total_llm_actions,
            "total_modifications": total_modifications,
            "total_actions_in_plans": total_actions,
            "avg_modifications_per_problem": (
                total_modifications / total_problems if total_problems > 0 else 0
            ),
            "avg_actions_per_plan": (
                total_actions / total_problems if total_problems > 0 else 0
            ),
            "validation": validation_stats,
            "total_time": total_time,
            "avg_time_per_problem": (
                total_time / total_problems if total_problems > 0 else 0
            ),
        }
