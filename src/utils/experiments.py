"""Experiment tracking for research."""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class ExperimentTracker:
    """Track planning experiments for research analysis."""

    def __init__(self, output_dir: Path = Path("experiments")):
        """
        Initialize experiment tracker.

        Args:
            output_dir: Directory for experiment logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_file = self.output_dir / "experiments.jsonl"

    def log_run(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a single experimental run.

        Args:
            experiment_id: Unique ID for this experiment
            config: Configuration used (strategy, model, etc.)
            results: Results (plan length, validity, metrics)
            metadata: Additional metadata (domain, problem, etc.)
        """
        entry = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": results,
            "metadata": metadata or {},
        }

        with open(self.experiment_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_dataframe(self) -> pd.DataFrame:
        """
        Load all experiments as a pandas DataFrame.

        Returns:
            DataFrame with all experiment data
        """
        if not self.experiment_file.exists():
            return pd.DataFrame()

        experiments = []
        with open(self.experiment_file) as f:
            for line in f:
                if line.strip():
                    experiments.append(json.loads(line))

        if not experiments:
            return pd.DataFrame()

        # Flatten nested dicts for easier analysis
        flat_data = []
        for exp in experiments:
            flat = {
                "experiment_id": exp["experiment_id"],
                "timestamp": exp["timestamp"],
            }

            # Flatten config
            for k, v in exp.get("config", {}).items():
                flat[f"config_{k}"] = v

            # Flatten results
            for k, v in exp.get("results", {}).items():
                flat[f"result_{k}"] = v

            # Flatten metadata
            for k, v in exp.get("metadata", {}).items():
                flat[f"meta_{k}"] = v

            flat_data.append(flat)

        return pd.DataFrame(flat_data)

    def summarize(self) -> Dict[str, Any]:
        """
        Generate summary statistics.

        Returns:
            Dictionary of summary statistics
        """
        df = self.get_dataframe()

        if df.empty:
            return {"message": "No experiments logged yet"}

        summary = {
            "total_runs": len(df),
            "unique_experiments": (
                df["experiment_id"].nunique() if "experiment_id" in df else 0
            ),
        }

        # Strategy breakdown
        if "config_strategy" in df:
            summary["strategies"] = df["config_strategy"].value_counts().to_dict()

        # Model breakdown
        if "config_model" in df:
            summary["models"] = df["config_model"].value_counts().to_dict()

        # Performance metrics
        if "result_plan_length" in df:
            summary["avg_plan_length"] = float(df["result_plan_length"].mean())
            summary["min_plan_length"] = int(df["result_plan_length"].min())
            summary["max_plan_length"] = int(df["result_plan_length"].max())

        if "result_valid" in df:
            summary["success_rate"] = float(df["result_valid"].sum() / len(df))
            summary["valid_plans"] = int(df["result_valid"].sum())
            summary["invalid_plans"] = int((~df["result_valid"]).sum())

        if "result_time_seconds" in df:
            summary["avg_time_seconds"] = float(df["result_time_seconds"].mean())

        return summary

    def export_csv(self, output_file: Path = None):
        """
        Export experiments to CSV.

        Args:
            output_file: Output CSV file path
        """
        if output_file is None:
            output_file = self.output_dir / "experiments.csv"

        df = self.get_dataframe()
        df.to_csv(output_file, index=False)
        return output_file

    def filter_experiments(
        self,
        strategy: Optional[str] = None,
        model: Optional[str] = None,
        valid_only: bool = False,
    ) -> pd.DataFrame:
        """
        Filter experiments by criteria.

        Args:
            strategy: Filter by strategy name
            model: Filter by model name
            valid_only: Only include valid plans

        Returns:
            Filtered DataFrame
        """
        df = self.get_dataframe()

        if df.empty:
            return df

        if strategy and "config_strategy" in df:
            df = df[df["config_strategy"] == strategy]

        if model and "config_model" in df:
            df = df[df["config_model"] == model]

        if valid_only and "result_valid" in df:
            df = df[df["result_valid"] == True]

        return df
