#!/usr/bin/env python3
"""
GPU Orchestrator: Manages GPU method execution on GPU nodes.

Task model
----------
Each SLURM array slot processes ONE ``(dataset, method, task_type)`` pair.
Inside the slot we run both ``NO_HPO`` and ``HPO`` consecutively (HPO is
skipped for methods listed in ``NO_HPO_METHODS``). The data preparation
performed by ``DataFeeder`` is cached process-locally by
``method_runner._get_or_prepare_folds``, so the HPO call reuses the folds
prepared during the NO_HPO call -- a ~2x wallclock saving per dataset on
methods that support both modes.

Backwards-compatible CLI modes:
* ``--array_id K``                              -> array-driven execution.
* ``--dataset/--method/--task_type/--hpo_mode`` -> single-task retry mode
  (run exactly one cell; used by ``scripts/retry_failed.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.methods.method_config import GPU_METHODS, NO_HPO_METHODS
from scripts.Experiment1.Experiment1 import run_single_method


def _apply_allow(items, allow):
    return list(items) if allow is None else [x for x in items if x in allow]


def build_gpu_task_list(config, methods_allow=None, datasets_allow=None):
    """Return a list of ``(dataset, method, task_type)`` tuples.

    Each tuple is one SLURM array slot. Inside the slot both NO_HPO and
    (where supported) HPO are run sequentially.

    The optional allow-lists let the orchestrator honour the same
    ``--methods-only`` / ``--datasets-only`` filters that ``Experiment1_Setup.py``
    used when it emitted the SLURM scripts -- without this, the Setup-time
    task count and the array-runtime task list disagree and array indices
    point to the wrong cells.
    """
    tasks = []

    pd_datasets = _apply_allow(config['datasets']['pd'].keys(), datasets_allow)
    pd_methods = [m for m in _apply_allow(config['methods']['pd'].keys(), methods_allow)
                  if m in GPU_METHODS]
    for dataset in pd_datasets:
        for method in pd_methods:
            tasks.append((dataset, method, 'pd'))

    lgd_datasets = _apply_allow(config['datasets']['lgd'].keys(), datasets_allow)
    lgd_methods = [m for m in _apply_allow(config['methods']['lgd'].keys(), methods_allow)
                   if m in GPU_METHODS]
    for dataset in lgd_datasets:
        for method in lgd_methods:
            tasks.append((dataset, method, 'lgd'))

    return tasks


def _run_both_hpo_modes(dataset, method, task_type, config, experiment_name, verbose):
    """Run NO_HPO followed by HPO for one (dataset, method) cell.

    NO_HPO_METHODS only execute NO_HPO; the existing run_single_method
    behaviour (duplicating into the HPO bucket) preserves downstream
    aggregation semantics.
    """
    hpo_modes = ['NO_HPO'] if method in NO_HPO_METHODS else ['NO_HPO', 'HPO']
    for hpo_mode in hpo_modes:
        run_single_method(
            dataset=dataset,
            method=method,
            task_type=task_type,
            hpo_mode=hpo_mode,
            config=config,
            experiment_name=experiment_name,
            verbose=verbose,
        )


def main():
    parser = argparse.ArgumentParser(description='Run GPU methods')
    parser.add_argument('--array_id', type=int, help='SLURM array task ID')
    parser.add_argument('--dataset', type=str, help='Specific dataset to run')
    parser.add_argument('--method', type=str, help='Specific method to run')
    parser.add_argument('--task_type', type=str, help='Task type (pd or lgd)')
    parser.add_argument('--hpo_mode', type=str, help='Single HPO mode (NO_HPO or HPO) -- retry mode only')
    parser.add_argument('--methods-only', type=str, default=None,
                        help='CSV allow-list; must match the value passed to Setup')
    parser.add_argument('--datasets-only', type=str, default=None,
                        help='CSV allow-list; must match the value passed to Setup')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--experiment', type=str, default='experiment1')
    args = parser.parse_args()

    def _csv(s):
        return [x.strip() for x in s.split(",") if x.strip()] if s else None
    methods_allow = _csv(getattr(args, "methods_only", None))
    datasets_allow = _csv(getattr(args, "datasets_only", None))

    config = load_config("Experiment1")

    print(f"\n{'='*70}")
    print(f"GPU ORCHESTRATOR")
    print(f"{'='*70}")

    # --- Retry / single-cell mode ---------------------------------------
    if args.dataset and args.method and args.task_type:
        if args.hpo_mode:
            print(f"Mode: single cell (retry)")
            print(f"Dataset:  {args.dataset}")
            print(f"Method:   {args.method}")
            print(f"Task:     {args.task_type}")
            print(f"HPO mode: {args.hpo_mode}")
            print(f"{'='*70}\n")
            run_single_method(
                dataset=args.dataset, method=args.method,
                task_type=args.task_type, hpo_mode=args.hpo_mode,
                config=config, experiment_name=args.experiment,
                verbose=args.verbose,
            )
            return
        # No hpo_mode -> run both
        print(f"Mode: single cell (both HPO modes)")
        print(f"Dataset: {args.dataset}  Method: {args.method}  Task: {args.task_type}")
        print(f"{'='*70}\n")
        _run_both_hpo_modes(args.dataset, args.method, args.task_type,
                            config, args.experiment, args.verbose)
        return

    # --- Array-driven mode ----------------------------------------------
    if args.array_id is None:
        print("ERROR: provide either --array_id or (--dataset --method --task_type)")
        sys.exit(1)

    gpu_tasks = build_gpu_task_list(config, methods_allow, datasets_allow)
    print(f"Mode: array-based")
    print(f"Total tasks:  {len(gpu_tasks)}")
    print(f"Array ID:     {args.array_id}")
    print(f"{'='*70}\n")

    if args.array_id < 0 or args.array_id >= len(gpu_tasks):
        print(f"ERROR: Array ID {args.array_id} out of range [0, {len(gpu_tasks)-1}]")
        sys.exit(1)

    dataset, method, task_type = gpu_tasks[args.array_id]
    print(f"Running task {args.array_id}: {dataset}/{method}/{task_type} -> NO_HPO + HPO\n")
    _run_both_hpo_modes(dataset, method, task_type, config, args.experiment, args.verbose)


if __name__ == "__main__":
    main()
