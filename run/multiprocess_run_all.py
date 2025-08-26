import os
import subprocess
import yaml
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

os.chdir(project_root)

def run_experiment(task, dataset_section, dataset_name, data_config_path,
                   experiment_config_path, method_config_path, evaluation_config_path,
                   temp_config_dir,
                   gpu_id=None):

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

    # Load and update configs
    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)
    with open(experiment_config_path) as f:
        experiment_config = yaml.safe_load(f)
    dataset_section = f"dataset_{task}"
    experiment_config["task"] = task
    for key in data_config[dataset_section]:
        data_config[dataset_section][key] = (key == dataset_name)
    tag = f"{task}_{dataset_name}"
    tmp_data_path = temp_config_dir / f"CONFIG_DATA_{tag}.yaml"
    tmp_experiment_path = temp_config_dir / f"CONFIG_EXPERIMENT_{tag}.yaml"
    with open(tmp_data_path, "w") as f:
        yaml.dump(data_config, f)
    with open(tmp_experiment_path, "w") as f:
        yaml.dump(experiment_config, f)


    cmd = [
        "python", "run/main.py",
        "--data", str(tmp_data_path),
        "--experiment", str(tmp_experiment_path),
        "--method", str(method_config_path),
        "--evaluation", str(evaluation_config_path),
        "--workdir", str(Path.cwd())
    ]
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(Path.cwd())
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Run the experiment
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Cleanup
    if tmp_data_path.exists():
        os.remove(tmp_data_path)
    if tmp_experiment_path.exists():
        os.remove(tmp_experiment_path)
    return dataset_name, result.returncode, result.stdout, result.stderr

# ----- MAIN -----
base_config_dir = Path("config")
temp_config_dir = Path("tmp_configs")
temp_config_dir.mkdir(exist_ok=True)
data_config_path = base_config_dir / "CONFIG_DATA.yaml"
experiment_config_path = base_config_dir / "CONFIG_EXPERIMENT.yaml"
method_config_path = base_config_dir / "CONFIG_METHOD.yaml"
evaluation_config_path = base_config_dir / "CONFIG_EVALUATION.yaml"

# List datasets to run
datasets = []
for task in ["pd"]:
    with open(data_config_path) as f:
        base_data_config = yaml.safe_load(f)
    dataset_section = f"dataset_{task}"
    if dataset_section not in base_data_config:
        continue
    for dataset_name in base_data_config[dataset_section]:
        if base_data_config[dataset_section][dataset_name]:
            datasets.append((task, dataset_section, dataset_name))


# Assign one GPU per process, as available
gpu_ids = list(range(3))  # Change if more/less GPUs

with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
    future_to_dataset = {}
    for i, (task, dataset_section, dataset_name) in enumerate(datasets):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        print(f"\n[STARTED] {dataset_name}")
        future = executor.submit(
            run_experiment,
            task, dataset_section, dataset_name, data_config_path,
            experiment_config_path, method_config_path, evaluation_config_path,
            temp_config_dir, gpu_id
        )
        future_to_dataset[future] = dataset_name


    for future in as_completed(future_to_dataset):
        ds = future_to_dataset[future]
        try:
            dataset_name, code, out, err = future.result()
            print(f"\n[RESULT] {dataset_name}: {'SUCCESS' if code == 0 else 'FAIL'}")
            if code != 0:
                print(err)
        except Exception as exc:
            print(f'\n[ERROR] {ds} generated an exception: {exc}')