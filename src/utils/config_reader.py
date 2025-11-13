# config_reader.py
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_dir: str = "config") -> Dict[str, Any]:
    """Load all configuration files and return as a unified dictionary."""
    config_dir = Path(config_dir)
    
    # Load individual configs
    with open(config_dir / "CONFIG_DATA.yaml") as f:
        data_config = yaml.safe_load(f)
    
    with open(config_dir / "CONFIG_METHOD.yaml") as f:
        method_config = yaml.safe_load(f)
    
    with open(config_dir / "CONFIG_EXPERIMENT.yaml") as f:
        experiment_config = yaml.safe_load(f)
    
    # Combine into unified structure
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


if __name__ == "__main__":
    config = load_config()
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))