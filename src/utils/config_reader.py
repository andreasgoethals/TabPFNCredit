import yaml
from pathlib import Path
from typing import Dict, Any
import sys

def get_config_dir(experiment_name: str) -> Path:
    """
    Get the configuration directory for a specific experiment.
    Path: project_root/scripts/{experiment_name}/config
    """
    # Find project root (up 3 levels from src/utils/config_reader.py)
    current = Path(__file__).resolve()
    project_root = current.parent.parent.parent
    
    config_dir = project_root / "scripts" / experiment_name / "config"
    
    if not config_dir.exists():
        raise FileNotFoundError(
            f"Config directory not found for experiment '{experiment_name}'.\n"
            f"Expected location: {config_dir}"
        )
        
    return config_dir


def _load_standard_config(config_dir: Path) -> Dict[str, Any]:
    """
    Standard loading logic for experiments using the same 3-file structure.
    Works for: Experiment0, Experiment1
    Expects: CONFIG_DATA.yaml, CONFIG_METHOD.yaml, CONFIG_EXPERIMENT.yaml
    """
    try:
        with open(config_dir / "CONFIG_DATA.yaml") as f:
            data_config = yaml.safe_load(f)

        with open(config_dir / "CONFIG_METHOD.yaml") as f:
            method_config = yaml.safe_load(f)

        with open(config_dir / "CONFIG_EXPERIMENT.yaml") as f:
            experiment_config = yaml.safe_load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Missing required YAML config file in {config_dir}.\n"
            f"Error: {e}"
        )

    # Standard merging logic
    config = {
        "split": data_config["split"],
        "paths": data_config["paths"],
        "datasets": {
            "pd": {k: v for k, v in data_config["dataset_pd"].items() if v},
            "lgd": {k: v for k, v in data_config["dataset_lgd"].items() if v}
        },
        "methods": {
            "pd": {k: v for k, v in method_config["methods"]["pd"].items() if v},
            "lgd": {k: v for k, v in method_config["methods"]["lgd"].items() if v}
        },
        "training": {
            "max_epochs": experiment_config["max_epochs"],
            "batch_size": experiment_config["batch_size"],
            "early_stopping": experiment_config["early_stopping"],
            "early_stopping_patience": experiment_config["early_stopping_patience"]
        },
        "tuning": {
            "n_trials": experiment_config["n_trials"]
        }
    }
    return config


def _load_experiment2_config(config_dir: Path) -> Dict[str, Any]:
    """
    Loading logic for Experiment2 (Learning Curve Analysis).
    Uses standard 3-file structure with additional learning curve parameters.
    Expects: CONFIG_DATA.yaml, CONFIG_METHOD.yaml, CONFIG_EXPERIMENT.yaml

    Additional parameters in CONFIG_EXPERIMENT.yaml:
        - row_max: Upper bound for row limit
        - row_min: Lower bound for row limit
        - row_step: Step size for decreasing rows
    """
    try:
        with open(config_dir / "CONFIG_DATA.yaml") as f:
            data_config = yaml.safe_load(f)

        with open(config_dir / "CONFIG_METHOD.yaml") as f:
            method_config = yaml.safe_load(f)

        with open(config_dir / "CONFIG_EXPERIMENT.yaml") as f:
            experiment_config = yaml.safe_load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Missing required YAML config file in {config_dir}.\n"
            f"Error: {e}"
        )

    # Validate learning curve parameters
    required_lc_params = ['row_max', 'row_min', 'row_step']
    for param in required_lc_params:
        if param not in experiment_config:
            raise ValueError(
                f"Missing required learning curve parameter '{param}' in CONFIG_EXPERIMENT.yaml"
            )

    # Validate parameter relationships
    if experiment_config['row_min'] > experiment_config['row_max']:
        raise ValueError(
            f"row_min ({experiment_config['row_min']}) cannot be greater than "
            f"row_max ({experiment_config['row_max']})"
        )
    if experiment_config['row_step'] <= 0:
        raise ValueError(f"row_step must be positive, got {experiment_config['row_step']}")

    # Build config with learning curve parameters
    config = {
        "split": data_config["split"],
        "paths": data_config["paths"],
        "datasets": {
            "pd": {k: v for k, v in data_config["dataset_pd"].items() if v},
            "lgd": {k: v for k, v in data_config["dataset_lgd"].items() if v}
        },
        "methods": {
            "pd": {k: v for k, v in method_config["methods"]["pd"].items() if v},
            "lgd": {k: v for k, v in method_config["methods"]["lgd"].items() if v}
        },
        "training": {
            "max_epochs": experiment_config["max_epochs"],
            "batch_size": experiment_config["batch_size"],
            "early_stopping": experiment_config["early_stopping"],
            "early_stopping_patience": experiment_config["early_stopping_patience"]
        },
        "tuning": {
            "n_trials": experiment_config["n_trials"]
        },
        # NEW: Learning curve control parameters
        "learning_curve": {
            "row_max": experiment_config["row_max"],
            "row_min": experiment_config["row_min"],
            "row_step": experiment_config["row_step"]
        }
    }
    return config


def _load_experiment3_config(config_dir: Path) -> Dict[str, Any]:
    """
    Loading logic for Experiment3 (Class Imbalance Analysis).
    Uses standard 3-file structure with additional imbalance parameters.
    Expects: CONFIG_DATA.yaml, CONFIG_METHOD.yaml, CONFIG_EXPERIMENT.yaml

    Additional parameters in CONFIG_EXPERIMENT.yaml:
        - minority_proportion_max: Upper bound for minority class proportion
        - minority_proportion_min: Lower bound for minority class proportion
        - minority_proportion_step: Step size for decreasing proportion
    """
    try:
        with open(config_dir / "CONFIG_DATA.yaml") as f:
            data_config = yaml.safe_load(f)

        with open(config_dir / "CONFIG_METHOD.yaml") as f:
            method_config = yaml.safe_load(f)

        with open(config_dir / "CONFIG_EXPERIMENT.yaml") as f:
            experiment_config = yaml.safe_load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Missing required YAML config file in {config_dir}.\n"
            f"Error: {e}"
        )

    # Validate imbalance parameters
    required_imb_params = ['minority_proportion_max', 'minority_proportion_min', 'minority_proportion_step']
    for param in required_imb_params:
        if param not in experiment_config:
            raise ValueError(
                f"Missing required imbalance parameter '{param}' in CONFIG_EXPERIMENT.yaml"
            )

    # Validate parameter relationships
    if experiment_config['minority_proportion_min'] > experiment_config['minority_proportion_max']:
        raise ValueError(
            f"minority_proportion_min ({experiment_config['minority_proportion_min']}) cannot be greater than "
            f"minority_proportion_max ({experiment_config['minority_proportion_max']})"
        )
    if experiment_config['minority_proportion_step'] <= 0:
        raise ValueError(f"minority_proportion_step must be positive, got {experiment_config['minority_proportion_step']}")
    if experiment_config['minority_proportion_min'] < 0 or experiment_config['minority_proportion_max'] > 0.5:
        raise ValueError(
            f"minority_proportion must be in range [0, 0.5]. "
            f"Got min={experiment_config['minority_proportion_min']}, max={experiment_config['minority_proportion_max']}"
        )

    # Build config with imbalance parameters
    config = {
        "split": data_config["split"],
        "paths": data_config["paths"],
        "datasets": {
            "pd": {k: v for k, v in data_config["dataset_pd"].items() if v},
            "lgd": {k: v for k, v in data_config["dataset_lgd"].items() if v}
        },
        "methods": {
            "pd": {k: v for k, v in method_config["methods"]["pd"].items() if v},
            "lgd": {k: v for k, v in method_config["methods"]["lgd"].items() if v}
        },
        "training": {
            "max_epochs": experiment_config["max_epochs"],
            "batch_size": experiment_config["batch_size"],
            "early_stopping": experiment_config["early_stopping"],
            "early_stopping_patience": experiment_config["early_stopping_patience"]
        },
        "tuning": {
            "n_trials": experiment_config["n_trials"]
        },
        # NEW: Class imbalance control parameters
        "imbalance": {
            "minority_proportion_max": experiment_config["minority_proportion_max"],
            "minority_proportion_min": experiment_config["minority_proportion_min"],
            "minority_proportion_step": experiment_config["minority_proportion_step"]
        }
    }
    return config


def load_config(experiment_name: str) -> Dict[str, Any]:
    """
    Dispatcher: Load configuration based on the experiment name.
    Different experiments can have different loading logic.
    """
    if not experiment_name:
        raise ValueError("experiment_name must be provided (e.g., 'Experiment0', 'Experiment1')")

    config_dir = get_config_dir(experiment_name)

    # =========================================================
    # DISPATCHER LOGIC PER EXPERIMENT
    # =========================================================

    if experiment_name in ["Experiment0", "Experiment1"]:
        # Both use standard 3-file structure
        return _load_standard_config(config_dir)

    elif experiment_name == "Experiment2":
        # Learning Curve Analysis - uses extended config with row_max/min/step
        return _load_experiment2_config(config_dir)

    elif experiment_name == "Experiment3":
        # Class Imbalance Analysis - uses extended config with minority_proportion params
        return _load_experiment3_config(config_dir)

    else:
        # Fallback or error for unknown experiments
        raise NotImplementedError(f"Unknown experiment '{experiment_name}'. No loading logic defined.")


if __name__ == "__main__":
    # Test
    for exp in ["Experiment0", "Experiment1", "Experiment2", "Experiment3"]:
        try:
            cfg = load_config(exp)
            print(f"[OK] {exp} config loaded successfully.")
            if exp == "Experiment2":
                lc = cfg.get("learning_curve", {})
                print(f"  Learning curve: row_max={lc.get('row_max')}, "
                      f"row_min={lc.get('row_min')}, row_step={lc.get('row_step')}")
            if exp == "Experiment3":
                imb = cfg.get("imbalance", {})
                print(f"  Imbalance: prop_max={imb.get('minority_proportion_max')}, "
                      f"prop_min={imb.get('minority_proportion_min')}, "
                      f"prop_step={imb.get('minority_proportion_step')}")
        except FileNotFoundError:
            print(f"[--] {exp} config not found (directory doesn't exist yet)")
        except Exception as e:
            print(f"[ERR] {exp} error: {e}")