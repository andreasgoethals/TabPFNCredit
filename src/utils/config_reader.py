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

def _load_experiment1_config(config_dir: Path) -> Dict[str, Any]:
    """
    Specific loading logic for Experiment1.
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
            f"Missing required YAML config file for Experiment1 in {config_dir}.\n"
            f"Error: {e}"
        )

    # Experiment1 specific merging logic
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
        raise ValueError("experiment_name must be provided (e.g., 'Experiment1')")

    config_dir = get_config_dir(experiment_name)
    
    # =========================================================
    # DISPATCHER LOGIC PER EXPERIMENT
    # =========================================================
    
    if experiment_name == "Experiment1":
        return _load_experiment1_config(config_dir)
        
    elif experiment_name == "Experiment0":
        # Placeholder for future logic
        pass 
        
    elif experiment_name == "Experiment2":
        # Placeholder for future logic
        pass
        
    else:
        # Fallback or error for unknown experiments
        raise NotImplementedError(f"Config loading logic for '{experiment_name}' is not implemented yet.")

if __name__ == "__main__":
    # Test
    try:
        cfg = load_config("Experiment1")
        print("Experiment1 config loaded successfully.")
    except Exception as e:
        print(f"Error: {e}")