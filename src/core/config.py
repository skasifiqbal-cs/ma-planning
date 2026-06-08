"""Configuration for MA-PDDL planning pipeline.

Environment variable overrides (all prefixed MAP_PLANNING_):
  LLM_PROVIDER, LLM_MODEL, LLM_URL, LLM_API_KEY
  VALIDATE_BIN, CONVERTER_SCRIPT, PYTHON_CMD
  EMBED_MODEL, CENTRALIZED_ROOT, RESULTS_ROOT
  MAX_STEPS, TEMPERATURE, MAX_TOKENS, STRATEGY
  BACKPROMPT_MAX_RETRIES, DEBUG (0/1)
"""

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def _load_env_file():
    env_file = Path(__file__).parent.parent.parent / ".env"
    if not env_file.exists():
        return
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key and not os.getenv(key):
                    os.environ[key] = value.strip()
    except Exception:
        pass


_load_env_file()


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name, "").strip()
    return val if val else default


def _bool(val: str) -> bool:
    return val.lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    """Runtime configuration for the planning pipeline."""

    # LLM
    llm_provider: str = _env("MAP_PLANNING_LLM_PROVIDER", "ollama")
    llm_model: str = _env("MAP_PLANNING_LLM_MODEL", "llama3:8b")
    llm_url: str = _env("MAP_PLANNING_LLM_URL", "http://localhost:11434")
    llm_api_key: Optional[str] = _env("MAP_PLANNING_LLM_API_KEY")
    temperature: float = float(_env("MAP_PLANNING_TEMPERATURE", "0.0"))
    max_tokens: int = int(_env("MAP_PLANNING_MAX_TOKENS", "8000"))

    # Planning — universal
    strategy: str = _env("MAP_PLANNING_STRATEGY", "llm-modulo")
    max_steps: int = int(_env("MAP_PLANNING_MAX_STEPS", "1000"))
    compress_pddl: bool = True
    skip_pyperplan: bool = _bool(_env("MAP_PLANNING_SKIP_PYPERPLAN", "0"))
    show_applicable_operators: bool = False

    # llm-modulo specific
    backprompt_max_retries: int = int(_env("MAP_PLANNING_BACKPROMPT_MAX_RETRIES", "2"))
    condense_val_errors: bool = True
    stop_on_empty_plan: bool = True

    # llm-repair specific
    action_validation: str = _env("MAP_PLANNING_ACTION_VALIDATION", "repair")
    embed_model: str = _env("MAP_PLANNING_EMBED_MODEL", "paraphrase-MiniLM-L6-v2")

    # Few-shot
    use_few_shot: bool = False
    few_shot_example_count: int = 0
    few_shot_examples_file: Optional[str] = "results/few_shot_examples.json"

    # Paths
    val_bin: str = _env("MAP_PLANNING_VALIDATE_BIN", "Validate")
    converter_script: str = _env(
        "MAP_PLANNING_CONVERTER_SCRIPT",
        "/home/rr/ma-planning/codmap-2015/competition/centalized/ma-to-pddl.py",
    )
    python_cmd: str = _env("MAP_PLANNING_PYTHON_CMD", "python3")
    centralized_root: str = _env("MAP_PLANNING_CENTRALIZED_ROOT", "/home/rr/ma-planning/centralized")
    results_root: str = _env("MAP_PLANNING_RESULTS_ROOT", "/home/rr/ma-planning/results")

    # Debug (all default off; set via config.yaml debug: section or env MAP_PLANNING_DEBUG=1)
    debug: bool = _bool(_env("MAP_PLANNING_DEBUG", "0"))
    debug_prompts: bool = False
    debug_val_feedback: bool = False
    show_prompts: bool = False
    show_llm_output: bool = False

    # Resolved at runtime by resolve()
    resolved_converter_script: Optional[str] = None
    resolved_python_cmd: Optional[str] = None
    resolved_val_bin: Optional[str] = None

    # ── Loading ────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, config_file: str = "config.yaml") -> "Config":
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required: pip install pyyaml")

        config_path = Path(config_file)
        cfg = cls()

        if not config_path.exists():
            return cfg

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        def _set(section: dict, yaml_key: str, attr: str, cast=None):
            val = section.get(yaml_key)
            if val is not None:
                setattr(cfg, attr, cast(val) if cast else val)

        llm = data.get("llm", {})
        _set(llm, "provider",    "llm_provider")
        _set(llm, "model",       "llm_model")
        _set(llm, "url",         "llm_url")
        _set(llm, "api_key",     "llm_api_key")
        _set(llm, "temperature", "temperature",  float)
        _set(llm, "max_tokens",  "max_tokens",   int)

        planning = data.get("planning", {})
        _set(planning, "strategy",    "strategy",    str)
        _set(planning, "max_steps",   "max_steps",   int)
        _set(planning, "compress_pddl", "compress_pddl", bool)
        _set(planning, "skip_pyperplan", "skip_pyperplan", bool)
        _set(planning, "show_applicable_operators", "show_applicable_operators", bool)

        # Read strategy-specific section (e.g., llm-modulo:, llm-repair:, etc.)
        strategy_section = data.get(cfg.strategy, {}) or {}
        _set(strategy_section, "backprompt_max_retries", "backprompt_max_retries", int)
        _set(strategy_section, "condense_val_errors",    "condense_val_errors",    bool)
        _set(strategy_section, "stop_on_empty_plan",     "stop_on_empty_plan",     bool)
        _set(strategy_section, "action_validation",      "action_validation",      str)
        _set(strategy_section, "embed_model",            "embed_model",            str)

        fs = data.get("few_shot", {})
        _set(fs, "enabled",       "use_few_shot",           bool)
        _set(fs, "count",         "few_shot_example_count", int)
        _set(fs, "examples_file", "few_shot_examples_file", str)

        paths = data.get("paths", {})
        _set(paths, "val_bin",           "val_bin")
        _set(paths, "converter_script",  "converter_script")
        _set(paths, "python2_cmd",       "python_cmd")

        dbg = data.get("debug", {})
        _set(dbg, "enabled",        "debug",           bool)
        _set(dbg, "show_prompts",   "show_prompts",    bool)
        _set(dbg, "show_llm_output","show_llm_output", bool)
        _set(dbg, "val_feedback",   "debug_val_feedback", bool)
        if cfg.debug:
            cfg.debug_prompts = True

        return cfg

    # ── Resolution ─────────────────────────────────────────────────────────────

    def resolve(self) -> None:
        Path(self.results_root).mkdir(parents=True, exist_ok=True)
        self._resolve_converter_script()
        self._resolve_python_cmd()
        self._resolve_val_bin()

    def _resolve_converter_script(self) -> None:
        script = Path(self.converter_script)
        if script.is_file():
            self.resolved_converter_script = str(script.resolve())
            return
        alt = script.parent.parent / "centralized" / script.name
        if alt.is_file():
            self.resolved_converter_script = str(alt.resolve())
            return
        raise FileNotFoundError(f"Converter script not found: {script}")

    def _resolve_python_cmd(self) -> None:
        cmd = shutil.which(self.python_cmd)
        if cmd:
            self.resolved_python_cmd = cmd
        else:
            raise FileNotFoundError(f"Python command not found: {self.python_cmd}")

    def _resolve_val_bin(self) -> None:
        val = Path(self.val_bin)
        if val.is_file():
            self.resolved_val_bin = str(val.resolve())
            return
        cmd = shutil.which(self.val_bin)
        if cmd:
            self.resolved_val_bin = cmd
            return
        repo_root = Path(__file__).parent.parent.parent
        for candidate in [
            repo_root / "VAL" / "build" / "bin" / "Validate",
            repo_root / "VAL" / "Validate",
            repo_root / "VAL" / "build" / "linux64" / "Release" / "bin" / "Validate",
        ]:
            if candidate.is_file():
                self.resolved_val_bin = str(candidate.resolve())
                return
        raise FileNotFoundError(f"Validate binary not found: {self.val_bin}")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
