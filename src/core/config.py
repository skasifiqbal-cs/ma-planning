"""
Configuration management for MA-PDDL planning pipeline.

Environment Variables:
  MAP_PLANNING_CONVERTER_SCRIPT
  MAP_PLANNING_PYTHON_CMD
  MAP_PLANNING_LLM_MODEL
  MAP_PLANNING_LLM_URL
  MAP_PLANNING_VALIDATE_BIN
  MAP_PLANNING_EMBED_MODEL
  MAP_PLANNING_CENTRALIZED_ROOT
  MAP_PLANNING_RESULTS_ROOT
  MAP_PLANNING_MAX_STEPS
  MAP_PLANNING_TEMPERATURE
  MAP_PLANNING_MAX_TOKENS
  MAP_PLANNING_DEBUG (0/1)
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Auto-load .env file if it exists
def _load_env_file():
    """Load environment variables from .env file in project root."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        # Only set if not already in environment (environment takes precedence)
                        if key and not os.getenv(key):
                            os.environ[key] = value
        except Exception:
            pass  # Silently fail if .env can't be read


_load_env_file()


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return non-empty environment variable value or default."""
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def _to_bool(val: str) -> bool:
    """Convert string to boolean."""
    return val.lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    """Configuration for MA-PDDL planning pipeline."""

    # Converter settings
    converter_script: str = _get_env(
        "MAP_PLANNING_CONVERTER_SCRIPT",
        "/home/rr/ma-planning/codmap-2015/competition/centalized/ma-to-pddl.py",
    )
    python_cmd: str = _get_env("MAP_PLANNING_PYTHON_CMD", "python2")

    # LLM settings
    llm_provider: str = _get_env("MAP_PLANNING_LLM_PROVIDER", "ollama")
    llm_model: str = _get_env("MAP_PLANNING_LLM_MODEL", "llama3:8b")
    llm_url: str = _get_env("MAP_PLANNING_LLM_URL", "http://localhost:11434")
    llm_api_key: Optional[str] = _get_env(
        "MAP_PLANNING_LLM_API_KEY"
    )  # For OpenAI, Groq, etc.

    # Validation
    val_bin: str = _get_env("MAP_PLANNING_VALIDATE_BIN", "Validate")

    # Embedding model
    embed_model: str = _get_env("MAP_PLANNING_EMBED_MODEL", "paraphrase-MiniLM-L6-v2")

    # Data roots
    centralized_root: str = _get_env(
        "MAP_PLANNING_CENTRALIZED_ROOT", "/home/rr/ma-planning/centralized"
    )
    results_root: str = _get_env(
        "MAP_PLANNING_RESULTS_ROOT", "/home/rr/ma-planning/results"
    )

    # Planning parameters
    max_steps: int = int(_get_env("MAP_PLANNING_MAX_STEPS", "30"))
    temperature: float = float(_get_env("MAP_PLANNING_TEMPERATURE", "0.7"))
    max_tokens: int = int(_get_env("MAP_PLANNING_MAX_TOKENS", "2048"))
    state_based_validation: bool = _to_bool(
        _get_env("MAP_PLANNING_STATE_VALIDATION", "1")
    )

    # Debug flags
    debug: bool = _to_bool(_get_env("MAP_PLANNING_DEBUG", "1"))
    debug_prompts: bool = False  # Set by from_yaml

    # Resolved paths (populated after resolve())
    resolved_converter_script: Optional[str] = None
    resolved_python_cmd: Optional[str] = None
    resolved_val_bin: Optional[str] = None

    def resolve(self) -> None:
        """Resolve and validate all paths and tools."""
        self._ensure_dirs()
        self._resolve_converter_script()
        self._resolve_python_cmd()
        self._resolve_val_bin()

        if self.debug:
            print("[CONFIG] All paths resolved successfully:")
            print(f"  Converter: {self.resolved_converter_script}")
            print(f"  Python: {self.resolved_python_cmd}")
            print(f"  Validate: {self.resolved_val_bin}")

    @classmethod
    def from_yaml(cls, config_file: str = "config.yaml") -> "Config":
        """Load configuration from YAML file.

        YAML structure:
            llm:
              provider: ollama
              model: llama3:8b
              temperature: 0.0
              url: http://localhost:11434
            planning:
              strategy: no-val
              max_steps: 50
              validate: true
            paths:
              val_bin: VAL/build/bin/Validate
              converter_script: codmap-2015/converters/multi-agent.py
              python2_cmd: python2
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

        config_path = Path(config_file)
        config = cls()

        if not config_path.exists():
            return config

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        # LLM settings
        llm = data.get("llm", {})
        if "provider" in llm:
            config.llm_provider = llm["provider"]
        if "model" in llm:
            config.llm_model = llm["model"]
        if "url" in llm:
            config.llm_url = llm["url"]
        if "api_key" in llm and llm["api_key"] is not None:
            config.llm_api_key = llm["api_key"]
        if "temperature" in llm and llm["temperature"] is not None:
            config.temperature = float(llm["temperature"])
        if "max_tokens" in llm and llm["max_tokens"] is not None:
            config.max_tokens = int(llm["max_tokens"])

        # Planning settings
        planning = data.get("planning", {})
        if "max_steps" in planning:
            config.max_steps = int(planning["max_steps"])
        if (
            "state_based_validation" in planning
            and planning["state_based_validation"] is not None
        ):
            config.state_based_validation = bool(planning["state_based_validation"])

        # Paths (only override if not None in YAML)
        paths = data.get("paths", {})
        if "converter_script" in paths and paths["converter_script"] is not None:
            config.converter_script = paths["converter_script"]
        if "python2_cmd" in paths and paths["python2_cmd"] is not None:
            config.python_cmd = paths["python2_cmd"]
        if "val_bin" in paths and paths["val_bin"] is not None:
            config.val_bin = paths["val_bin"]

        # Output settings
        output = data.get("output", {})
        if "debug_prompts" in output and output["debug_prompts"] is not None:
            config.debug_prompts = bool(output["debug_prompts"])

        return config

    def _ensure_dirs(self) -> None:
        """Create necessary directories."""
        for d in [self.centralized_root, self.results_root]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _resolve_converter_script(self) -> None:
        """Resolve converter script path."""
        script = Path(self.converter_script)
        if script.is_file():
            self.resolved_converter_script = str(script.resolve())
            return

        # Try alternate spelling
        alt = script.parent.parent / "centralized" / script.name
        if alt.is_file():
            self.resolved_converter_script = str(alt.resolve())
            return

        raise FileNotFoundError(f"Converter script not found: {script}")

    def _resolve_python_cmd(self) -> None:
        """Resolve Python command."""
        cmd = shutil.which(self.python_cmd)
        if cmd:
            self.resolved_python_cmd = cmd
        else:
            raise FileNotFoundError(f"Python command not found: {self.python_cmd}")

    def _resolve_val_bin(self) -> None:
        """Resolve VAL binary."""
        val = Path(self.val_bin)
        if val.is_file():
            self.resolved_val_bin = str(val.resolve())
            return

        cmd = shutil.which(self.val_bin)
        if cmd:
            self.resolved_val_bin = cmd
            return

        # Check VAL directory in project
        val_dir = Path(__file__).parent.parent.parent / "VAL"
        validate = val_dir / "Validate"
        if validate.is_file():
            self.resolved_val_bin = str(validate.resolve())
            return

        raise FileNotFoundError(f"Validate binary not found: {self.val_bin}")

    def as_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return asdict(self)


if __name__ == "__main__":
    cfg = Config()
    try:
        cfg.resolve()
        print("[SUCCESS] Configuration resolved")
        print(f"Config: {cfg.as_dict()}")
    except Exception as e:
        print(f"[ERROR] {e}")
