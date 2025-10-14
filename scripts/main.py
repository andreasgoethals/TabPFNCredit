import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.classes.experiment import Experiment
from utils.utils import set_random_seed, load_config



# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='config/CONFIG_DATA.yaml')
parser.add_argument('--experiment', type=str, default='config/CONFIG_EXPERIMENT.yaml')
parser.add_argument('--method', type=str, default='config/CONFIG_METHOD.yaml')
parser.add_argument('--evaluation', type=str, default='config/CONFIG_EVALUATION.yaml')
parser.add_argument('--tuning', type=str, default='config/CONFIG_TUNING.yaml')
parser.add_argument('--workdir', type=str, default=str(Path(__file__).resolve().parent.parent))
args = parser.parse_args()

# Set working directory
os.chdir(args.workdir)

# Reproducibility
set_random_seed(0)

# Load config files
dataconfig = load_config(Path(args.data))
experimentconfig = load_config(Path(args.experiment))
methodconfig = load_config(Path(args.method))
evaluationconfig = load_config(Path(args.evaluation))
tuningconfig = load_config(Path(args.tuning))

# Get task type (pd or lgd)
task = experimentconfig.get('task')

# Select dataset based on the task
dataset_name = ""
if task == 'pd':
    # Select dataset where 'true' is marked under dataset_pd
    dataset_name = next((key for key, value in dataconfig['dataset_pd'].items() if value), 'dataset_pd_unknown')
elif task == 'lgd':
    # Select dataset where 'true' is marked under dataset_lgd
    dataset_name = next((key for key, value in dataconfig['dataset_lgd'].items() if value), 'dataset_lgd_unknown')

# Format timestamp
timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')

# Format names
base_folder = f"{dataset_name}"
experiment_folder = f"{dataset_name}_tune-{tuningconfig['tune_hyperparameters']}_imbal-{experimentconfig['imbalance']}_{timestamp}"

# Create output and config backup
output_dir = Path("outputs") / task / base_folder
output_dir.mkdir(parents=True, exist_ok=True)
config_backup_dir = Path('outputs') / task / base_folder / 'config' / experiment_folder
config_backup_dir.mkdir(parents=True, exist_ok=True)
logs_dir = Path('outputs') / 'logs' / experiment_folder
config_backup_dir.mkdir(parents=True, exist_ok=True)

# Run experiment
experiment = Experiment(dataconfig, experimentconfig, methodconfig, evaluationconfig, tuningconfig, logs_dir)
experiment.run()

# Prepare config info for the results table
imbalance = experimentconfig.get('imbalance', False)
imbalance_ratio = experimentconfig.get('imbalance_ratio', None)
tuning = tuningconfig.get('tune_hyperparameters', False)
dataset_col = dataset_name

# Compose imbalance_setting string
if not imbalance:
    imbalance_setting = 'off'
else:
    imbalance_setting = f'on (ratio={imbalance_ratio})'

# Get the results
rows = []
for model_name, splits_dict in experiment.results.items():
    for split_idx, metrics_dict in splits_dict.items():
        row = {
            'dataset': dataset_col,
            'imbalance_setting': imbalance_setting,
            'tuning': tuning,
            'model': model_name,
            'split': split_idx,
        }
        row.update(metrics_dict)
        rows.append(row)

df = pd.DataFrame(rows)

# Print the res
print(df)

now2 = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
print('\nExperiment ended at: ', now2)

# Save results
output_file = output_dir / f"{dataset_name}_tune-{tuningconfig['tune_hyperparameters']}_imbal-{experimentconfig['imbalance']}_{timestamp}.csv"
df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

print("Logs saved to: ", output_dir / "experiment.log")

# Copy config files
shutil.copy(args.data, config_backup_dir / "CONFIG_DATA.yaml")
shutil.copy(args.experiment, config_backup_dir / "CONFIG_EXPERIMENT.yaml")
shutil.copy(args.method, config_backup_dir / "CONFIG_METHOD.yaml")
shutil.copy(args.evaluation, config_backup_dir / "CONFIG_EVALUATION.yaml")
shutil.copy(args.tuning, config_backup_dir / "CONFIG_TUNING.yaml")
print(f"Configs backed up to: {config_backup_dir}")
