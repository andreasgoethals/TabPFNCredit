import os
import argparse

from src.utils import set_random_seed, load_config
from src.classes.experiment import Experiment
import pandas as pd
import shutil
import datetime

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

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

# Run experiment
experiment = Experiment(dataconfig, experimentconfig, methodconfig, evaluationconfig, tuningconfig)
experiment.run()


# Get the results
rows = []
for model_name, splits_dict in experiment.results.items():
    for split_idx, metrics_dict in splits_dict.items():
        row = {'model': model_name, 'split': split_idx}
        row.update(metrics_dict)
        rows.append(row)

df = pd.DataFrame(rows)

# Print the res
print(df)

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

now2 = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
print('\nExperiment ended at: ', now2)

# Format names
base_folder = f"{task}_{dataset_name}"
filename_prefix = f"{task}_{dataset_name}_{timestamp}"

# Create output and config backup
output_dir = Path("outputs") / base_folder
output_dir.mkdir(parents=True, exist_ok=True)
config_backup_dir = output_dir / f"configs_{filename_prefix}"
config_backup_dir.mkdir(parents=True, exist_ok=True)

# Save results
output_file = output_dir / f"{filename_prefix}.csv"
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

# Copy config files
shutil.copy(args.data, config_backup_dir / "CONFIG_DATA.yaml")
shutil.copy(args.experiment, config_backup_dir / "CONFIG_EXPERIMENT.yaml")
shutil.copy(args.method, config_backup_dir / "CONFIG_METHOD.yaml")
shutil.copy(args.evaluation, config_backup_dir / "CONFIG_EVALUATION.yaml")
shutil.copy(args.tuning, config_backup_dir / "CONFIG_TUNING.yaml")
print(f"Configs backed up to: {config_backup_dir}")
# rows = []
# for model_name, splits_dict in experiment.results.items():
#     for split_idx, metrics_dict in splits_dict.items():
#         row = {'model': model_name, 'split': split_idx}
#         row.update(metrics_dict)
#         rows.append(row)
#
# df = pd.DataFrame(rows)
#
# # Print the res
# print(df)
#
# # Get task type (pd or lgd)
# task = experimentconfig.get('task')
#
# # Select dataset based on the task
# dataset_name = ""
# if task == 'pd':
#     # Select dataset where 'true' is marked under dataset_pd
#     dataset_name = next((key for key, value in dataconfig['dataset_pd'].items() if value), 'dataset_pd_unknown')
# elif task == 'lgd':
#     # Select dataset where 'true' is marked under dataset_lgd
#     dataset_name = next((key for key, value in dataconfig['dataset_lgd'].items() if value), 'dataset_lgd_unknown')
#
# # Format timestamp
# timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
#
# now2 = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
# print('\nExperiment ended at: ', now2)
#
# # Format names
# base_folder = f"{task}_{dataset_name}"
# filename_prefix = f"{task}_{dataset_name}_{timestamp}"
#
# # Create output and config backup
# output_dir = Path("outputs") / base_folder
# output_dir.mkdir(parents=True, exist_ok=True)
# config_backup_dir = output_dir / f"configs_{filename_prefix}"
# config_backup_dir.mkdir(parents=True, exist_ok=True)
#
# # Save results
# output_file = output_dir / f"{filename_prefix}.csv"
# df.to_csv(output_file, index=False)
# print(f"Results saved to {output_file}")
#
# # Copy config files
# shutil.copy(args.data, config_backup_dir / "CONFIG_DATA.yaml")
# shutil.copy(args.experiment, config_backup_dir / "CONFIG_EXPERIMENT.yaml")
# shutil.copy(args.method, config_backup_dir / "CONFIG_METHOD.yaml")
# shutil.copy(args.evaluation, config_backup_dir / "CONFIG_EVALUATION.yaml")
# shutil.copy(args.tuning, config_backup_dir / "CONFIG_TUNING.yaml")
# print(f"Configs backed up to: {config_backup_dir}")




