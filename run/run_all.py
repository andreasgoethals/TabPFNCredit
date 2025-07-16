import yaml
import subprocess
from pathlib import Path
import os

# SETUP CONFIGURATION
base_config_dir = Path("config")
temp_config_dir = Path("tmp_configs")
temp_config_dir.mkdir(exist_ok=True)

# Base paths
data_config_path = base_config_dir / "CONFIG_DATA.yaml"
experiment_config_path = base_config_dir / "CONFIG_EXPERIMENT.yaml"
method_config_path = base_config_dir / "CONFIG_METHOD.yaml"
evaluation_config_path = base_config_dir / "CONFIG_EVALUATION.yaml"

# Load base configs
with open(data_config_path) as f:
    base_data_config = yaml.safe_load(f)

with open(experiment_config_path) as f:
    base_experiment_config = yaml.safe_load(f)

# Loop over tasks
for task in ["pd"]: # can add lgd if needed later on
    dataset_section = f"dataset_{task}"
    if dataset_section not in base_data_config:
        print(f"No datasets found for task '{task}'. Skipping...")
        continue

    # Loop over datasets
    for dataset_name in base_data_config[dataset_section]:
        if not base_data_config[dataset_section][dataset_name]:
            continue

        print(f"\n Running experiment: task='{task}', dataset='{dataset_name}'")

        # Create copies of the base configs
        data_config = yaml.safe_load(data_config_path.read_text())
        experiment_config = yaml.safe_load(experiment_config_path.read_text())

        # Set the task in the experiment config
        experiment_config["task"] = task

        # Set all datasets to false, then activate the one we want
        for key in data_config[dataset_section]:
            data_config[dataset_section][key] = (key == dataset_name)

        # Save temp files
        tag = f"{task}_{dataset_name}"
        tmp_data_path = temp_config_dir / f"CONFIG_DATA_{tag}.yaml"
        tmp_experiment_path = temp_config_dir / f"CONFIG_EXPERIMENT_{tag}.yaml"

        with open(tmp_data_path, "w") as f:
            yaml.dump(data_config, f)

        with open(tmp_experiment_path, "w") as f:
            yaml.dump(experiment_config, f)

        # Run
        cmd = [
            "python", "run/main.py",
            "--data", str(tmp_data_path),
            "--experiment", str(tmp_experiment_path),
            "--method", str(method_config_path),
            "--evaluation", str(evaluation_config_path),
            "--workdir", str(Path.cwd())
        ]

        # subprocess.run(cmd)
        env = dict(**os.environ)
        env["PYTHONPATH"] = str(Path.cwd())

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Check 
        success = result.returncode == 0
        message = "Success" if success else result.stderr.strip()

        # Cleanup
        if tmp_data_path.exists():
            os.remove(tmp_data_path)
        if tmp_experiment_path.exists():
            os.remove(tmp_experiment_path)

        print(f"Temporary config files deleted: {tmp_data_path}, {tmp_experiment_path}")