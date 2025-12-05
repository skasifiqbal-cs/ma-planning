"""Configuration management with YAML support."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import itertools


class ConfigManager:
    """Manage experiment configurations."""

    @staticmethod
    def load_config(path: Path) -> Dict[str, Any]:
        """
        Load config from YAML or JSON file.

        Args:
            path: Path to config file

        Returns:
            Configuration dictionary
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            if path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif path.suffix == ".json":
                return json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config format: {path.suffix}. "
                    "Use .yaml, .yml, or .json"
                )

    @staticmethod
    def save_config(config: Dict[str, Any], path: Path):
        """
        Save config to YAML or JSON file.

        Args:
            config: Configuration dictionary
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            if path.suffix in [".yaml", ".yml"]:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(config, f, indent=2)

    @staticmethod
    def generate_sweep(
        base_config: Dict[str, Any], sweep_params: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter sweep configurations.

        Example:
            base = {"model": "gpt-4", "temperature": 0.0}
            sweep = {"strategy": ["no-val", "repair"], "temperature": [0.0, 0.5, 1.0]}
            configs = ConfigManager.generate_sweep(base, sweep)
            # Returns 6 configs (2 strategies × 3 temperatures)

        Args:
            base_config: Base configuration
            sweep_params: Parameters to sweep (name -> list of values)

        Returns:
            List of configuration dictionaries
        """
        # Get all parameter combinations
        keys = list(sweep_params.keys())
        values = [sweep_params[k] for k in keys]

        configs = []
        for combination in itertools.product(*values):
            config = base_config.copy()
            for key, value in zip(keys, combination):
                config[key] = value
            configs.append(config)

        return configs

    @staticmethod
    def create_experiment_config(
        name: str,
        strategy: str,
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
        max_steps: int = 50,
        validate: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a standard experiment configuration.

        Args:
            name: Experiment name
            strategy: Planning strategy
            model: LLM model
            provider: LLM provider
            temperature: Sampling temperature
            max_steps: Maximum planning steps
            validate: Whether to validate plans
            **kwargs: Additional parameters

        Returns:
            Configuration dictionary
        """
        config = {
            "experiment_name": name,
            "strategy": strategy,
            "llm": {
                "provider": provider,
                "model": model,
                "temperature": temperature,
            },
            "planning": {
                "max_steps": max_steps,
                "validate": validate,
            },
            **kwargs,
        }
        return config


# Example config templates
EXAMPLE_CONFIGS = {
    "zero_shot_gpt4": {
        "experiment_name": "zero-shot-gpt4",
        "strategy": "no-val",
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.0,
        },
        "planning": {
            "max_steps": 50,
            "validate": True,
        },
    },
    "repair_claude": {
        "experiment_name": "repair-claude",
        "strategy": "repair",
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.0,
        },
        "planning": {
            "max_steps": 50,
            "validate": True,
            "max_repair_attempts": 3,
        },
    },
    "randomized_ollama": {
        "experiment_name": "randomized-ollama",
        "strategy": "randomized",
        "llm": {
            "provider": "ollama",
            "model": "llama3:8b",
            "temperature": 0.7,
            "url": "http://localhost:11434",
        },
        "planning": {
            "max_steps": 50,
            "validate": True,
            "num_candidates": 5,
        },
    },
}


def save_example_configs(output_dir: Path = Path("configs")):
    """Save example configurations to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, config in EXAMPLE_CONFIGS.items():
        output_file = output_dir / f"{name}.yaml"
        ConfigManager.save_config(config, output_file)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    # Generate example configs when run directly
    save_example_configs()
