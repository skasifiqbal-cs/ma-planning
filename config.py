"""
Configuration module for the ma-planning pipeline.

Features:
- Environment variable overrides (highest precedence before runtime CLI overrides).
- Automatic tool path resolution (Validate, python interpreter).
- Auto-detection / fallback for a misspelled converter folder ('centalized' vs 'centralized').
- Safe creation of centralized/output directories.
- Simple serialization with as_dict().
- One-step resolve() that must be called before using external tools.

Environment Variables:
  MAP_PLANNING_CONVERTER_SCRIPT
  MAP_PLANNING_PYTHON_CMD
  MAP_PLANNING_LLM_MODEL
  MAP_PLANNING_LLM_URL
  MAP_PLANNING_VALIDATE_BIN
  MAP_PLANNING_EMBED_MODEL
  MAP_PLANNING_VALIDATION_MODE
  MAP_PLANNING_UNFACTORED_ROOT
  MAP_PLANNING_CENTRALIZED_ROOT
  MAP_PLANNING_RESULTS_ROOT
  MAP_PLANNING_MAX_STEPS
  MAP_PLANNING_PLAN_TIMEOUT
  MAP_PLANNING_TEMPERATURE
  MAP_PLANNING_MAX_TOKENS
  MAP_PLANNING_DEBUG        (use "0"/"1")
  MAP_PLANNING_DEBUG_PRINT_PROMPT     # "0"/"1", default "0"
  MAP_PLANNING_DEBUG_PRINT_RESPONSE   # "0"/"1", default "1"

Typical usage:
    from config import Config
    cfg = Config()
    cfg.resolve()
    print(cfg.resolved_val_bin)
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional


# ---------- Helper functions ----------


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return non-empty environment variable value or default."""
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def _to_bool(val: str) -> bool:
    return val.lower() in {"1", "true", "yes", "on"}


# ---------- Config dataclass ----------


@dataclass
class Config:
    # Converter settings
    converter: Dict[str, str] = field(
        default_factory=lambda: {
            "converter_script": _get_env(
                "MAP_PLANNING_CONVERTER_SCRIPT",
                "/home/rr/ma-planning/codmap-2015/competition/centalized/ma-to-pddl.py",
            ),
            "python_cmd": _get_env("MAP_PLANNING_PYTHON_CMD", "python2"),
        }
    )

    # LLM settings (base URL only; LLMPrompt will append /api/chat if needed)
    llm_model: str = _get_env("MAP_PLANNING_LLM_MODEL", "llama3:8b")
    llm_url: str = _get_env("MAP_PLANNING_LLM_URL", "http://localhost:11434")

    # Validator
    val_bin: str = _get_env("MAP_PLANNING_VALIDATE_BIN", "Validate")

    # Future embedding model (unused now)
    embed_model: str = _get_env("MAP_PLANNING_EMBED_MODEL", "paraphrase-MiniLM-L6-v2")

    # Validation mode (pipeline-level)
    validation_mode: str = _get_env("MAP_PLANNING_VALIDATION_MODE", "no-val")

    # Data roots
    unfactored_root: str = _get_env(
        "MAP_PLANNING_UNFACTORED_ROOT",
        "/home/rr/Downloads/pddl-data-master/codmap-2015/unfactored/",
    )
    centralized_root: str = _get_env(
        "MAP_PLANNING_CENTRALIZED_ROOT", "/home/rr/ma-planning/centralized"
    )
    results_root: str = _get_env(
        "MAP_PLANNING_RESULTS_ROOT", "/home/rr/ma-planning/results"
    )

    # Planning parameters
    max_steps: int = int(_get_env("MAP_PLANNING_MAX_STEPS", "30"))
    plan_timeout: int = int(_get_env("MAP_PLANNING_PLAN_TIMEOUT", "300"))

    # LLM inference parameters
    temperature: float = float(_get_env("MAP_PLANNING_TEMPERATURE", "0.7"))
    max_tokens: int = int(_get_env("MAP_PLANNING_MAX_TOKENS", "128"))

    # Debug flag (controls full prompt printing elsewhere)
    debug: bool = _to_bool(_get_env("MAP_PLANNING_DEBUG", "1"))

    # Fine-grained debug printing controls
    debug_print_prompt: bool = _to_bool(
        _get_env("MAP_PLANNING_DEBUG_PRINT_PROMPT", "0")
    )
    debug_print_response: bool = _to_bool(
        _get_env("MAP_PLANNING_DEBUG_PRINT_RESPONSE", "1")
    )

    # Resolved fields (populated after resolve())
    resolved_converter_script: Optional[str] = None
    resolved_python_cmd: Optional[str] = None
    resolved_val_bin: Optional[str] = None

    # ---------- Public Methods ----------

    def resolve(self) -> None:
        """
        Resolve external paths:
          - Ensure centralized_root & results_root exist.
          - Resolve converter script (with typo correction & fallback search).
          - Resolve python interpreter.
          - Resolve Validate binary.
        """
        self._ensure_dirs()
        self._resolve_converter_script()
        self.resolved_python_cmd = self._resolve_executable(
            self.converter["python_cmd"], required=True, label="python_cmd"
        )
        self.resolved_val_bin = self._resolve_executable(
            self.val_bin, required=True, label="Validate"
        )

        if self.debug:
            print("[CONFIG] Resolution complete:")
            print(f"  converter_script: {self.resolved_converter_script}")
            print(f"  python_cmd:       {self.resolved_python_cmd}")
            print(f"  validate_bin:     {self.resolved_val_bin}")
            print(f"  centralized_root: {self.centralized_root}")
            print(f"  results_root:     {self.results_root}")
            print(f"  llm_url (base):   {self.llm_url}")
            print(f"  llm_model:        {self.llm_model}")
            print(f"  max_tokens:       {self.max_tokens}")
            print(f"  temperature:      {self.temperature}")
            print(f"  debug_print_prompt:   {self.debug_print_prompt}")
            print(f"  debug_print_response: {self.debug_print_response}")

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary snapshot of all current configuration values + resolved fields."""
        base = {
            "converter": self.converter,
            "llm_model": self.llm_model,
            "llm_url": self.llm_url,
            "val_bin": self.val_bin,
            "embed_model": self.embed_model,
            "validation_mode": self.validation_mode,
            "unfactored_root": self.unfactored_root,
            "centralized_root": self.centralized_root,
            "results_root": self.results_root,
            "max_steps": self.max_steps,
            "plan_timeout": self.plan_timeout,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "debug": self.debug,
            "debug_print_prompt": self.debug_print_prompt,
            "debug_print_response": self.debug_print_response,
        }
        base["resolved"] = {
            "converter_script": self.resolved_converter_script,
            "python_cmd": self.resolved_python_cmd,
            "validate_bin": self.resolved_val_bin,
        }
        return base

    # ---------- Internal Helpers ----------

    def _ensure_dirs(self) -> None:
        """Create centralized_root and results_root if absent."""
        for d in [self.centralized_root, self.results_root]:
            path = Path(d)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                if self.debug:
                    print(f"[CONFIG] Created directory: {path}")

    def _resolve_converter_script(self) -> None:
        """
        Resolve converter script:
          - Use provided path if exists.
          - Attempt 'centalized' -> 'centralized' correction.
          - Fallback search under /home/rr/ma-planning for 'ma-to-pddl.py'.
        """
        raw = self.converter["converter_script"]
        candidate = Path(raw)
        if candidate.is_file():
            self.resolved_converter_script = str(candidate)
            return

        if "centalized" in raw:
            alt = raw.replace("centalized", "centralized")
            if Path(alt).is_file():
                if self.debug:
                    print(f"[CONFIG] Auto-corrected converter_script path: {alt}")
                self.converter["converter_script"] = alt
                self.resolved_converter_script = alt
                return

        search_root = Path("/home/rr/ma-planning")
        matches = list(search_root.rglob("ma-to-pddl.py"))
        if matches:
            picked = matches[0]
            if self.debug:
                print(f"[CONFIG] Auto-detected converter script: {picked}")
            self.converter["converter_script"] = str(picked)
            self.resolved_converter_script = str(picked)
            return

        raise FileNotFoundError(
            f"[CONFIG ERROR] Converter script not found: {raw}\n"
            "Tried typo correction and fallback search.\n"
            "Set MAP_PLANNING_CONVERTER_SCRIPT or update Config.converter['converter_script']."
        )

    def _resolve_executable(
        self, spec: str, required: bool, label: str
    ) -> Optional[str]:
        """
        Resolve an executable:
          - If spec is a path (absolute or contains separator), check executable bit.
          - Else search PATH via shutil.which().
        """
        if os.path.isabs(spec) or os.path.sep in spec:
            if os.access(spec, os.X_OK):
                return spec
            if required:
                raise RuntimeError(
                    f"[CONFIG ERROR] {label} not executable or not found: {spec}"
                )
            return None

        found = shutil.which(spec)
        if found:
            return found
        if required:
            raise RuntimeError(f"[CONFIG ERROR] {label} '{spec}' not found in PATH.")
        return None


# ---------- Standalone usage ----------

if __name__ == "__main__":
    cfg = Config()
    try:
        cfg.resolve()
    except Exception as e:
        print(f"FAILED TO RESOLVE CONFIG: {e}")
    else:
        import pprint

        pprint.pprint(cfg.as_dict())
