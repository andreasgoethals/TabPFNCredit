import os
import sys
import subprocess
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support, set_start_method

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

os.chdir(project_root)

def detect_gpu_ids():
    """
    Return a list of integer GPU IDs available to this process.
    Order of precedence:
      1) CUDA_VISIBLE_DEVICES (if set)
      2) torch.cuda.device_count()
      3) nvidia-smi --list-gpus parsing
    If nothing found, return [] to indicate CPU-only.
    """
    # 1) Respect CUDA_VISIBLE_DEVICES if present and not empty
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and cvd.lower() not in {"none", "-1"}:
        # CUDA_VISIBLE_DEVICES may be like "0,1", or UUIDs; keep indices we can parse.
        ids = []
        for tok in cvd.split(","):
            tok = tok.strip()
            # If it's a number, keep it
            if tok.isdigit():
                ids.append(int(tok))
            # If it's a UUID, map to a positional index (by order)
            # e.g., CUDA_VISIBLE_DEVICES="GPU-xxx,GPU-yyy" -> map to [0,1]
            else:
                # Treat any non-integer token as a slot; position is index
                ids.append(len(ids))
        # De-dup just in case
        ids = list(dict.fromkeys(ids))
        return ids

    # 2) Try torch
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n and n > 0:
                return list(range(n))
    except Exception:
        pass

    # 3) Try nvidia-smi
    try:
        out = subprocess.check_output(["nvidia-smi", "--list-gpus"], text=True, stderr=subprocess.STDOUT)
        # Lines look like: "GPU 0: GeForce RTX 3090 (UUID: GPU-xxxx)"
        ids = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("GPU "):
                # Extract integer after "GPU "
                try:
                    num = int(line.split()[1].rstrip(":"))
                    ids.append(num)
                except Exception:
                    # ignore parse errors
                    pass
        if ids:
            return ids
    except Exception:
        pass

    # No GPUs detected
    return []


def run_experiment(task, dataset_section, dataset_name, data_config_path,
                   experiment_config_path, method_config_path, evaluation_config_path,
                   temp_config_dir, gpu_id=None):

    # --- Load base configs from disk (paths are passed in) ---
    data_config = yaml.safe_load(Path(data_config_path).read_text())
    experiment_config = yaml.safe_load(Path(experiment_config_path).read_text())

    # Set the task in the experiment config
    experiment_config["task"] = task

    # Turn all datasets off, enable only the requested dataset
    for key in data_config[dataset_section]:
        data_config[dataset_section][key] = (key == dataset_name)

    # Save temp config files unique per (task, dataset)
    tag = f"{task}_{dataset_name}"
    temp_config_dir = Path(temp_config_dir)
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    tmp_data_path = temp_config_dir / f"CONFIG_DATA_{tag}.yaml"
    tmp_experiment_path = temp_config_dir / f"CONFIG_EXPERIMENT_{tag}.yaml"
    tmp_data_path.write_text(yaml.safe_dump(data_config))
    tmp_experiment_path.write_text(yaml.safe_dump(experiment_config))

    # Build subprocess command; use current Python executable for portability
    cmd = [
        sys.executable, "run/main.py",
        "--data", str(tmp_data_path),
        "--experiment", str(tmp_experiment_path),
        "--method", str(method_config_path),
        "--evaluation", str(evaluation_config_path),
        "--workdir", str(Path.cwd())
    ]

    # Environment (optional CUDA device pinning)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    if gpu_id is not None:
        # Pin this child process to a single GPU
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Run the experiment
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Cleanup temp files
    try:
        if tmp_data_path.exists():
            tmp_data_path.unlink()
        if tmp_experiment_path.exists():
            tmp_experiment_path.unlink()
    except OSError:
        # Best-effort cleanup; ignore race conditions on Windows
        pass

    return dataset_name, result.returncode, result.stdout, result.stderr


def main():
    base_config_dir = Path("config")
    temp_config_dir = Path("tmp_configs")
    temp_config_dir.mkdir(exist_ok=True)

    data_config_path = base_config_dir / "CONFIG_DATA.yaml"
    experiment_config_path = base_config_dir / "CONFIG_EXPERIMENT.yaml"
    method_config_path = base_config_dir / "CONFIG_METHOD.yaml"
    evaluation_config_path = base_config_dir / "CONFIG_EVALUATION.yaml"

    # Build list of datasets to run
    datasets = []
    for task in ["pd"]:  # extend to ["pd", "lgd"] if needed
        base_data_config = yaml.safe_load(data_config_path.read_text())
        dataset_section = f"dataset_{task}"
        if dataset_section not in base_data_config:
            continue
        for dataset_name, enabled in base_data_config[dataset_section].items():
            # be tolerant to strings like "true"/"false"
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() in ("1", "true", "yes", "y")
            if enabled:
                datasets.append((task, dataset_section, dataset_name))

    if not datasets:
        print("[ERROR] No datasets enabled in CONFIG_DATA.yaml")
        return

    # --- Detect GPUs dynamically ---
    gpu_ids = detect_gpu_ids()  # e.g., [0,1,2] or []
    if gpu_ids:
        print(f"[INFO] Detected GPUs: {gpu_ids}")
        max_workers = min(len(gpu_ids), len(datasets))
    else:
        print("[INFO] No GPUs detected. Running on CPU.")
        max_workers = min(os.cpu_count() or 1, len(datasets))

    if max_workers < 1:
        max_workers = 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_dataset = {}
        for i, (task, dataset_section, dataset_name) in enumerate(datasets):
            gpu_id = (gpu_ids[i % len(gpu_ids)] if gpu_ids else None)
            fut = executor.submit(
                run_experiment,
                task, dataset_section, dataset_name,
                str(data_config_path),
                str(experiment_config_path),
                str(method_config_path),
                str(evaluation_config_path),
                str(temp_config_dir),
                gpu_id
            )
            future_to_dataset[fut] = dataset_name

        for future in as_completed(future_to_dataset):
            ds = future_to_dataset[future]
            try:
                dataset_name, code, out, err = future.result()
                print(f"\n[RESULT] {dataset_name}: {'SUCCESS' if code == 0 else 'FAIL'}")
                if code != 0:
                    print(err)
            except Exception as exc:
                print(f'\n[ERROR] {ds} generated an exception: {exc}')


if __name__ == "__main__":
    # Windows/Spawn-safe entry point
    freeze_support()  # harmless on POSIX; required for frozen executables on Windows
    try:
        # On Unix this is 'fork' by default; on Windows it's 'spawn'
        set_start_method("spawn")
    except RuntimeError:
        # Start method already set in this interpreter session
        pass

    main()