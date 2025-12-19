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
        # Placeholder for future logic
        raise NotImplementedError(f"Config loading logic for '{experiment_name}' is not implemented yet.")
        
    else:
        # Fallback or error for unknown experiments
        raise NotImplementedError(f"Unknown experiment '{experiment_name}'. No loading logic defined.")


if __name__ == "__main__":
    # Test
    for exp in ["Experiment0", "Experiment1"]:
        try:
            cfg = load_config(exp)
            print(f"✓ {exp} config loaded successfully.")
        except FileNotFoundError:
            print(f"✗ {exp} config not found (directory doesn't exist yet)")
        except Exception as e:
            print(f"✗ {exp} error: {e}")