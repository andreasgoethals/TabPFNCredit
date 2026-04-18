#!/usr/bin/env python3
"""
Experiment 1 setup: generate batched SLURM scripts for the full benchmark.

Splits enabled methods into GPU vs CPU sets via ``src.methods.method_config``,
counts ``(dataset, method, hpo_mode)`` tasks, and emits batched array scripts
bounded at 400 tasks each (below VSC's 500-element ``--array`` cap).

Improvements over the original:

* Shared SLURM templates (``scripts/_slurm_templates.py``) deduplicate the
  three experiments' header/prologue logic and standardise:
  absolute ``${VSC_DATA}`` log paths, ``#SBATCH --requeue``, ``set -euo pipefail``,
  ``mkdir -p`` of the log dir, and optional failure email via
  ``TABPFN_SLURM_NOTIFY_EMAIL``.
* Foundation methods on Experiment 1 can be promoted to wICE H100 via the
  ``--foundation-on-wice`` flag, mirroring Experiment 2's split (historically
  P100 was OOM'ing on ``home_credit``).
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
# Task counting
# --------------------------------------------------------------------------- #

def count_tasks(methods, datasets):
    """Count (dataset, method, hpo_mode) tasks. NO_HPO-only methods count once."""
    count = 0
    for _ in datasets:
        for method in methods:
            count += 1 if method in NO_HPO_METHODS else 2
    return count


# --------------------------------------------------------------------------- #
# Script builders
# --------------------------------------------------------------------------- #

def _build_exp1_script(
    *,
    resources: SlurmResources,
    job_type: str,
    log_subdir: str,
    orchestrator_relpath: str,
    soft_isolation: bool = False,
):
    """Return a closure suitable for ``write_batched_scripts``."""

    def _build(batch_id: int, start: int, end: int, max_concurrent: int) -> str:
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
            f'--array_id=${{GLOBAL_TASK_ID}} --verbose'
        )
        return assemble_array_script(
            header=header,
            banner_title=banner,
            log_subdir=log_subdir,
            python_command=python_cmd,
            task_offset=start,
        )

    return _build


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Experiment 1 SLURM scripts")
    parser.add_argument(
        "--foundation-on-wice",
        action="store_true",
        help="Emit a separate wICE H100 batch for foundation methods (recommended).",
    )
    args = parser.parse_args()

    print(f"\n{'='*70}\nEXPERIMENT 1 SETUP\n{'='*70}\n")
    config = load_config("Experiment1")

    pd_datasets = list(config["datasets"]["pd"].keys())
    lgd_datasets = list(config["datasets"]["lgd"].keys())
    if not (pd_datasets or lgd_datasets):
        print("ERROR: No datasets enabled in config")
        sys.exit(1)

    pd_methods = list(config["methods"]["pd"].keys())
    lgd_methods = list(config["methods"]["lgd"].keys())

    gpu_pd = [m for m in pd_methods if m in GPU_METHODS]
    gpu_lgd = [m for m in lgd_methods if m in GPU_METHODS]
    cpu_pd = [m for m in pd_methods if m in CPU_METHODS]
    cpu_lgd = [m for m in lgd_methods if m in CPU_METHODS]

    all_gpu_methods = set(gpu_pd + gpu_lgd)
    all_cpu_methods = set(cpu_pd + cpu_lgd)

    if not all_gpu_methods and not all_cpu_methods:
        print("ERROR: No methods enabled in config")
        sys.exit(1)

    print_method_summary(all_gpu_methods, all_cpu_methods, pd_datasets, lgd_datasets)

    # Split GPU tasks into Standard + Foundation if requested
    if args.foundation_on_wice:
        std_gpu_pd = [m for m in gpu_pd if m not in FOUNDATION_METHODS]
        std_gpu_lgd = [m for m in gpu_lgd if m not in FOUNDATION_METHODS]
        fnd_gpu_pd = [m for m in gpu_pd if m in FOUNDATION_METHODS]
        fnd_gpu_lgd = [m for m in gpu_lgd if m in FOUNDATION_METHODS]
    else:
        std_gpu_pd, std_gpu_lgd = gpu_pd, gpu_lgd
        fnd_gpu_pd, fnd_gpu_lgd = [], []

    n_std_gpu = count_tasks(std_gpu_pd, pd_datasets) + count_tasks(std_gpu_lgd, lgd_datasets)
    n_fnd_gpu = count_tasks(fnd_gpu_pd, pd_datasets) + count_tasks(fnd_gpu_lgd, lgd_datasets)
    n_cpu = count_tasks(cpu_pd, pd_datasets) + count_tasks(cpu_lgd, lgd_datasets)

    print(f"{'='*70}\nTASK COUNTS\n{'='*70}")
    print(f"Standard GPU tasks:   {n_std_gpu:4d}")
    print(f"Foundation GPU tasks: {n_fnd_gpu:4d}")
    print(f"CPU tasks:            {n_cpu:4d}")
    print(f"Total:                {n_std_gpu + n_fnd_gpu + n_cpu:4d}")
    print(f"Batch size:           {MAX_TASKS_PER_SLURM} tasks per SLURM file")
    print(f"{'='*70}\n")

    scripts_dir = PROJECT_ROOT / "scripts" / "Experiment1"
    log_subdir = "experiment1/logs/slurm"

    # Standard GPU
    std_gpu_files = write_batched_scripts(
        total_tasks=n_std_gpu,
        max_tasks_per_batch=MAX_TASKS_PER_SLURM,
        max_concurrent=min(16, max(1, n_std_gpu)),
        prefix="Experiment1_GPU",
        scripts_dir=scripts_dir,
        build_script=_build_exp1_script(
            resources=SlurmResources(
                cluster="genius", partition="gpu_p100",
                cpus=4, gpus=1, memory="45G", time="70:00:00",
            ),
            job_type="gpu",
            log_subdir=log_subdir,
            orchestrator_relpath="Experiment1_GPU.py",
        ),
    )

    # Foundation GPU (wICE H100, soft isolation)
    fnd_gpu_files = write_batched_scripts(
        total_tasks=n_fnd_gpu,
        max_tasks_per_batch=MAX_TASKS_PER_SLURM,
        max_concurrent=min(8, max(1, n_fnd_gpu)),
        prefix="Experiment1_GPU_Foundation",
        scripts_dir=scripts_dir,
        build_script=_build_exp1_script(
            resources=SlurmResources(
                cluster="wice", partition="gpu_h100",
                cpus=16, gpus=1, memory="100G", time="70:00:00",
            ),
            job_type="foundation",
            log_subdir=log_subdir,
            orchestrator_relpath="Experiment1_GPU.py",
            soft_isolation=True,
        ),
    )

    # CPU
    cpu_files = write_batched_scripts(
        total_tasks=n_cpu,
        max_tasks_per_batch=MAX_TASKS_PER_SLURM,
        max_concurrent=min(32, max(1, n_cpu)),
        prefix="Experiment1_CPU",
        scripts_dir=scripts_dir,
        build_script=_build_exp1_script(
            resources=SlurmResources(
                cluster="genius", partition="batch",
                cpus=8, gpus=0, memory="40G", time="48:00:00",
            ),
            job_type="cpu",
            log_subdir=log_subdir,
            orchestrator_relpath="Experiment1_CPU.py",
        ),
    )

    # Report
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
