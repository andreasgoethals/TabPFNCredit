#!/usr/bin/env python3
"""
Scan an experiment's result pickles for missing ``(dataset, method, hpo_mode)``
cells and emit a focused SLURM array that runs only those.

This is the rerun companion to ``src/utils/remove_results.py``:

* delete some results -> ``python scripts/retry_failed.py --experiment experiment1``
* it inspects ``results/{experiment}/{pd,lgd}/*.pkl`` and the enabled
  config to compute the "expected" cells minus the "present" cells.
* it then writes ``scripts/Experiment{N}/Experiment{N}_Retry.slurm`` whose
  array body calls the orchestrator's single-cell ``--dataset/--method/...``
  mode for each missing cell.

It also handles the "new method just added" case: if the config enables a
method that has zero results in any pickle, every (dataset, that_method)
cell is reported as missing and queued.

Usage
-----
    python scripts/retry_failed.py --experiment experiment1
    python scripts/retry_failed.py --experiment experiment1 --dry-run
    python scripts/retry_failed.py --experiment experiment1 --tasks gpu  # gpu | cpu | both
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.methods.method_config import GPU_METHODS, CPU_METHODS, NO_HPO_METHODS


# --------------------------------------------------------------------------- #
# Missing-cell detection
# --------------------------------------------------------------------------- #

def _load_pickle_safely(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return None


def _expected_hpo_modes(method: str) -> list[str]:
    return ["NO_HPO"] if method in NO_HPO_METHODS else ["NO_HPO", "HPO"]


def find_missing_cells(experiment: str, config: dict) -> list[tuple[str, str, str, str]]:
    """Return ``[(dataset, method, task_type, hpo_mode), ...]`` that are missing.

    A cell is missing if, for some enabled (dataset, method) and an expected
    HPO mode, the pickle either doesn't exist, is empty, or lacks that
    (hpo_mode, method) entry.
    """
    results_root = PROJECT_ROOT / "results" / experiment
    missing: list[tuple[str, str, str, str]] = []

    for task_type in ("pd", "lgd"):
        datasets = list(config["datasets"][task_type].keys())
        methods = list(config["methods"][task_type].keys())
        task_dir = results_root / task_type

        for dataset in datasets:
            pkl_path = task_dir / f"{dataset}.pkl"
            data = _load_pickle_safely(pkl_path)

            for method in methods:
                for hpo_mode in _expected_hpo_modes(method):
                    if (data is None or
                            hpo_mode not in data or
                            method not in data[hpo_mode]):
                        missing.append((dataset, method, task_type, hpo_mode))

    return missing


# --------------------------------------------------------------------------- #
# Hardware classification
# --------------------------------------------------------------------------- #

def _classify(cell):
    _ds, method, _task, _hpo = cell
    if method in GPU_METHODS:
        return "gpu"
    if method in CPU_METHODS:
        return "cpu"
    return "unknown"


# --------------------------------------------------------------------------- #
# SLURM emission (one focused script per hardware class)
# --------------------------------------------------------------------------- #

RETRY_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=exp{exp_n}_retry_{hw}
#SBATCH --cluster={cluster}
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
{gpus_line}#SBATCH --mem={memory}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --output=${{VSC_DATA}}/TabPFNCredit/results/experiment{exp_n}/logs/slurm/retry_{hw}_%A_%a.out
#SBATCH --error=${{VSC_DATA}}/TabPFNCredit/results/experiment{exp_n}/logs/slurm/retry_{hw}_%A_%a.err
#SBATCH --requeue
#SBATCH --array=0-{last_idx}%{max_concurrent}

set -euo pipefail
sleep $((${{SLURM_ARRAY_TASK_ID:-0}} % 30))
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export PATH="${{VSC_DATA}}/miniconda3/bin:${{PATH}}"
source activate TabPFNCredit
export LD_LIBRARY_PATH="${{VSC_DATA}}/miniconda3/envs/TabPFNCredit/lib:${{LD_LIBRARY_PATH:-}}"

cd "${{VSC_DATA}}/TabPFNCredit"
mkdir -p "${{VSC_DATA}}/TabPFNCredit/results/experiment{exp_n}/logs/slurm"

# Per-array-index lookup table (Bash arrays for cell coords)
DATASETS=({datasets_arr})
METHODS=({methods_arr})
TASK_TYPES=({tasks_arr})
HPO_MODES=({hpos_arr})

DS="${{DATASETS[${{SLURM_ARRAY_TASK_ID}}]}}"
M="${{METHODS[${{SLURM_ARRAY_TASK_ID}}]}}"
T="${{TASK_TYPES[${{SLURM_ARRAY_TASK_ID}}]}}"
H="${{HPO_MODES[${{SLURM_ARRAY_TASK_ID}}]}}"

echo "==> retrying ${{DS}}/${{M}}/${{T}}/${{H}}"

python -u scripts/Experiment{exp_n}/Experiment{exp_n}_{orch_suffix}.py \\
    --dataset="${{DS}}" --method="${{M}}" --task_type="${{T}}" --hpo_mode="${{H}}" --verbose
"""


def _bash_array(items):
    return " ".join(f'"{x}"' for x in items)


def emit_retry_script(experiment: str, cells: list[tuple[str, str, str, str]],
                      hw: str, out_path: Path) -> None:
    if not cells:
        return

    exp_n = experiment.replace("experiment", "")
    if hw == "gpu":
        resources = dict(cluster="genius", partition="gpu_p100",
                         cpus=4, memory="45G", time="70:00:00",
                         gpus_line="#SBATCH --gpus-per-node=1\n")
        orch_suffix = "GPU"
        max_conc = min(16, len(cells))
    else:  # cpu
        resources = dict(cluster="genius", partition="batch",
                         cpus=8, memory="40G", time="48:00:00",
                         gpus_line="")
        orch_suffix = "CPU"
        max_conc = min(32, len(cells))

    script = RETRY_TEMPLATE.format(
        exp_n=exp_n, hw=hw,
        last_idx=len(cells) - 1, max_concurrent=max_conc,
        datasets_arr=_bash_array(c[0] for c in cells),
        methods_arr=_bash_array(c[1] for c in cells),
        tasks_arr=_bash_array(c[2] for c in cells),
        hpos_arr=_bash_array(c[3] for c in cells),
        orch_suffix=orch_suffix,
        **resources,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script, encoding="utf-8", newline="\n")
    os.chmod(out_path, 0o755)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate a retry SLURM array for missing cells")
    parser.add_argument("--experiment", default="experiment1",
                        choices=["experiment0", "experiment1", "experiment2", "experiment3"])
    parser.add_argument("--tasks", default="both", choices=["gpu", "cpu", "both"])
    parser.add_argument("--dry-run", action="store_true",
                        help="List missing cells without writing SLURM files")
    args = parser.parse_args()

    exp_capitalised = args.experiment.replace("experiment", "Experiment")
    config = load_config(exp_capitalised)

    missing = find_missing_cells(args.experiment, config)
    if not missing:
        print(f"[retry_failed] No missing cells for {args.experiment} -- nothing to do.")
        return

    by_hw = {"gpu": [], "cpu": [], "unknown": []}
    for cell in missing:
        by_hw[_classify(cell)].append(cell)

    print(f"\n{'='*70}\nRETRY MANIFEST -- {args.experiment}\n{'='*70}")
    print(f"Missing cells: {len(missing)}")
    print(f"  GPU: {len(by_hw['gpu'])}")
    print(f"  CPU: {len(by_hw['cpu'])}")
    if by_hw["unknown"]:
        print(f"  WARNING: {len(by_hw['unknown'])} cells reference methods that are neither GPU nor CPU; skipping.")

    if args.dry_run:
        print("\n--- missing cells ---")
        for ds, m, t, h in missing:
            print(f"  {t}/{ds}/{m}/{h}")
        return

    scripts_dir = PROJECT_ROOT / "scripts" / exp_capitalised

    wrote_any = False
    if args.tasks in ("gpu", "both") and by_hw["gpu"]:
        out = scripts_dir / f"{exp_capitalised}_Retry_GPU.slurm"
        emit_retry_script(args.experiment, by_hw["gpu"], "gpu", out)
        print(f"\n  wrote {out.relative_to(PROJECT_ROOT)} ({len(by_hw['gpu'])} cells)")
        wrote_any = True

    if args.tasks in ("cpu", "both") and by_hw["cpu"]:
        out = scripts_dir / f"{exp_capitalised}_Retry_CPU.slurm"
        emit_retry_script(args.experiment, by_hw["cpu"], "cpu", out)
        print(f"  wrote {out.relative_to(PROJECT_ROOT)} ({len(by_hw['cpu'])} cells)")
        wrote_any = True

    if wrote_any:
        print(f"\nSubmit with:")
        if by_hw["gpu"] and args.tasks in ("gpu", "both"):
            print(f"  sbatch scripts/{exp_capitalised}/{exp_capitalised}_Retry_GPU.slurm")
        if by_hw["cpu"] and args.tasks in ("cpu", "both"):
            print(f"  sbatch scripts/{exp_capitalised}/{exp_capitalised}_Retry_CPU.slurm")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
