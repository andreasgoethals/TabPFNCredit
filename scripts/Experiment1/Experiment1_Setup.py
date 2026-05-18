#!/usr/bin/env python3
"""
Experiment 1 setup: generate batched SLURM scripts for the full benchmark.

Task model
----------
* GPU task = one ``(dataset, method, task_type)`` -- runs both NO_HPO and HPO
  inside the slot (folds cache means HPO reuses the data prep).
* CPU task = one ``(dataset, task_type)``         -- runs ALL enabled CPU
  methods (NO_HPO + HPO each) inside the slot, sharing the cached folds.

This cuts the GPU array length by ~2x and the CPU array length by
``|cpu_methods| x 2``, with the same total compute (now amortised against
shared data loading).

CLI filters
-----------
``--methods-only A,B,C``  -- only emit tasks for the listed methods
``--datasets-only X,Y``   -- only emit tasks for the listed datasets
These let you regenerate a focused batch for a partial rerun without
editing the YAML config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.methods.method_config import (
    GPU_METHODS,
    CPU_METHODS,
    NO_HPO_METHODS,
    FOUNDATION_METHODS,
)
from scripts._slurm_templates import (
    SlurmResources,
    _array_range,
    assemble_array_script,
    slurm_header,
    write_batched_scripts,
)

MAX_TASKS_PER_SLURM = 400


# --------------------------------------------------------------------------- #
# Task counting (matches the bundled orchestrators)
# --------------------------------------------------------------------------- #

def count_gpu_tasks(methods, datasets):
    """One task per (dataset, method); HPO mode is iterated inside the slot."""
    return len(methods) * len(datasets)


def count_cpu_tasks(methods, datasets):
    """One task per dataset (all CPU methods bundle into it)."""
    if not methods:
        return 0
    return len(datasets)


# --------------------------------------------------------------------------- #
# Script builders
# --------------------------------------------------------------------------- #

def _build_exp1_script(*, resources, job_type, log_subdir, orchestrator_relpath,
                       soft_isolation=False, extra_args=""):
    def _build(batch_id, start, end, max_concurrent):
        n_tasks = end - start
        header = slurm_header(
            job_name=f"exp1_{job_type}{batch_id}",
            resources=resources,
            log_subdir=log_subdir,
            array_range=_array_range(n_tasks, max_concurrent),
            soft_isolation=soft_isolation,
        )
        banner = f"EXPERIMENT 1 - {job_type.upper()} - BATCH {batch_id}"
        python_cmd = (
            f'python -u scripts/Experiment1/{orchestrator_relpath} '
            f'--array_id=${{GLOBAL_TASK_ID}} --verbose{extra_args}'
        )
        return assemble_array_script(
            header=header, banner_title=banner, log_subdir=log_subdir,
            python_command=python_cmd, task_offset=start,
        )
    return _build


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _split_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _apply_filters(items, allow):
    if allow is None:
        return list(items)
    return [x for x in items if x in allow]


def print_method_summary(gpu_methods, cpu_methods, pd_datasets, lgd_datasets):
    print(f"\n{'='*70}\nCONFIGURATION SUMMARY\n{'='*70}")
    print(f"\nDatasets:")
    print(f"  PD  ({len(pd_datasets)}):  {', '.join(sorted(pd_datasets))}")
    print(f"  LGD ({len(lgd_datasets)}): {', '.join(sorted(lgd_datasets))}")
    print(f"\nGPU methods ({len(gpu_methods)}):")
    for i, m in enumerate(sorted(gpu_methods), 1):
        tag = "NO_HPO only" if m in NO_HPO_METHODS else "NO_HPO + HPO"
        print(f"  {i:2d}. {m:<20} ({tag})")
    print(f"\nCPU methods ({len(cpu_methods)}):")
    for i, m in enumerate(sorted(cpu_methods), 1):
        tag = "NO_HPO only" if m in NO_HPO_METHODS else "NO_HPO + HPO"
        print(f"  {i:2d}. {m:<20} ({tag})")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 1 SLURM scripts")
    parser.add_argument(
        "--foundation-on-wice", action="store_true",
        help="Emit a separate wICE H100 batch for foundation methods (recommended).",
    )
    parser.add_argument(
        "--methods-only", type=str, default=None,
        help="Comma-separated method allow-list (overrides config flags)."
    )
    parser.add_argument(
        "--datasets-only", type=str, default=None,
        help="Comma-separated dataset allow-list (overrides config flags)."
    )
    args = parser.parse_args()

    method_allow = _split_csv(args.methods_only)
    dataset_allow = _split_csv(args.datasets_only)

    print(f"\n{'='*70}\nEXPERIMENT 1 SETUP\n{'='*70}\n")
    config = load_config("Experiment1")

    pd_datasets = _apply_filters(config["datasets"]["pd"].keys(), dataset_allow)
    lgd_datasets = _apply_filters(config["datasets"]["lgd"].keys(), dataset_allow)
    if not (pd_datasets or lgd_datasets):
        print("ERROR: No datasets selected (config + filters yield empty set)")
        sys.exit(1)

    pd_methods = _apply_filters(config["methods"]["pd"].keys(), method_allow)
    lgd_methods = _apply_filters(config["methods"]["lgd"].keys(), method_allow)

    gpu_pd = [m for m in pd_methods if m in GPU_METHODS]
    gpu_lgd = [m for m in lgd_methods if m in GPU_METHODS]
    cpu_pd = [m for m in pd_methods if m in CPU_METHODS]
    cpu_lgd = [m for m in lgd_methods if m in CPU_METHODS]

    all_gpu_methods = set(gpu_pd + gpu_lgd)
    all_cpu_methods = set(cpu_pd + cpu_lgd)
    if not (all_gpu_methods or all_cpu_methods):
        print("ERROR: No methods selected (config + filters yield empty set)")
        sys.exit(1)

    print_method_summary(all_gpu_methods, all_cpu_methods, pd_datasets, lgd_datasets)

    if args.foundation_on_wice:
        std_gpu_pd  = [m for m in gpu_pd  if m not in FOUNDATION_METHODS]
        std_gpu_lgd = [m for m in gpu_lgd if m not in FOUNDATION_METHODS]
        fnd_gpu_pd  = [m for m in gpu_pd  if m in FOUNDATION_METHODS]
        fnd_gpu_lgd = [m for m in gpu_lgd if m in FOUNDATION_METHODS]
    else:
        std_gpu_pd, std_gpu_lgd = gpu_pd, gpu_lgd
        fnd_gpu_pd, fnd_gpu_lgd = [], []

    n_std_gpu = count_gpu_tasks(std_gpu_pd, pd_datasets) + count_gpu_tasks(std_gpu_lgd, lgd_datasets)
    n_fnd_gpu = count_gpu_tasks(fnd_gpu_pd, pd_datasets) + count_gpu_tasks(fnd_gpu_lgd, lgd_datasets)
    n_cpu = count_cpu_tasks(cpu_pd, pd_datasets) + count_cpu_tasks(cpu_lgd, lgd_datasets)

    print(f"{'='*70}\nTASK COUNTS (bundled)\n{'='*70}")
    print(f"Standard GPU tasks:   {n_std_gpu:4d}  (each runs NO_HPO + HPO)")
    print(f"Foundation GPU tasks: {n_fnd_gpu:4d}  (each runs NO_HPO + HPO)")
    print(f"CPU tasks:            {n_cpu:4d}  (each runs all {len(cpu_pd) or len(cpu_lgd)} CPU methods x 2)")
    print(f"Total:                {n_std_gpu + n_fnd_gpu + n_cpu:4d}")
    print(f"Batch size:           {MAX_TASKS_PER_SLURM} tasks per SLURM file")
    print(f"{'='*70}\n")

    scripts_dir = PROJECT_ROOT / "scripts" / "Experiment1"
    log_subdir = "experiment1/logs/slurm"

    # Thread filters into the orchestrator command so the array-runtime task
    # list matches the Setup-time task count.
    extra_args = ""
    if method_allow:
        extra_args += f" --methods-only={','.join(method_allow)}"
    if dataset_allow:
        extra_args += f" --datasets-only={','.join(dataset_allow)}"

    std_gpu_files = write_batched_scripts(
        total_tasks=n_std_gpu, max_tasks_per_batch=MAX_TASKS_PER_SLURM,
        max_concurrent=min(16, max(1, n_std_gpu)),
        prefix="Experiment1_GPU", scripts_dir=scripts_dir,
        build_script=_build_exp1_script(
            resources=SlurmResources(cluster="genius", partition="gpu_p100",
                                     cpus=4, gpus=1, memory="45G", time="70:00:00"),
            job_type="gpu", log_subdir=log_subdir,
            orchestrator_relpath="Experiment1_GPU.py",
            extra_args=extra_args,
        ),
    )

    fnd_gpu_files = write_batched_scripts(
        total_tasks=n_fnd_gpu, max_tasks_per_batch=MAX_TASKS_PER_SLURM,
        max_concurrent=min(8, max(1, n_fnd_gpu)),
        prefix="Experiment1_GPU_Foundation", scripts_dir=scripts_dir,
        build_script=_build_exp1_script(
            resources=SlurmResources(cluster="wice", partition="gpu_h100",
                                     cpus=16, gpus=1, memory="100G", time="70:00:00"),
            job_type="foundation", log_subdir=log_subdir,
            orchestrator_relpath="Experiment1_GPU.py", soft_isolation=True,
            extra_args=extra_args,
        ),
    )

    cpu_files = write_batched_scripts(
        total_tasks=n_cpu, max_tasks_per_batch=MAX_TASKS_PER_SLURM,
        max_concurrent=min(32, max(1, n_cpu)),
        prefix="Experiment1_CPU", scripts_dir=scripts_dir,
        build_script=_build_exp1_script(
            resources=SlurmResources(cluster="genius", partition="batch",
                                     cpus=8, gpus=0, memory="40G", time="48:00:00"),
            job_type="cpu", log_subdir=log_subdir,
            orchestrator_relpath="Experiment1_CPU.py",
            extra_args=extra_args,
        ),
    )

    print(f"{'='*70}\nSETUP COMPLETE\n{'='*70}\n")

    def _print_batch(label, files):
        if files:
            print(f"  {label} ({len(files)} file(s)):")
            for fn, s, e in files:
                print(f"    - {fn} (tasks {s}-{e-1})")
            print()

    _print_batch("Standard GPU (genius/gpu_p100)", std_gpu_files)
    _print_batch("Foundation GPU (wICE/gpu_h100, soft isolation)", fnd_gpu_files)
    _print_batch("CPU (genius/batch)", cpu_files)

    print("Submit with e.g.:")
    for fn, _, _ in std_gpu_files + fnd_gpu_files + cpu_files:
        print(f"  sbatch scripts/Experiment1/{fn}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
