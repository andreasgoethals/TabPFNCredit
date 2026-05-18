#!/usr/bin/env python3
"""
CPU Orchestrator: Manages CPU method execution on CPU nodes.

Task model
----------
A single SLURM array slot runs ALL enabled CPU methods on ONE
``(dataset, task_type)`` pair, across both NO_HPO and HPO modes. The folds
prepared by ``DataFeeder`` are cached process-locally inside
``method_runner._get_or_prepare_folds``, so the per-dataset data load
happens exactly once per slot regardless of how many methods run -- a large
speedup for tasks where individual classical methods finish in milliseconds.

Backwards-compatible single-cell mode is preserved for ``scripts/retry_failed.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.methods.method_config import CPU_METHODS, NO_HPO_METHODS
from src.methods.method_runner import clear_folds_cache
from scripts.Experiment1.Experiment1 import run_single_method


def _apply_allow(items, allow):
    return list(items) if allow is None else [x for x in items if x in allow]


def build_cpu_task_list(config, methods_allow=None, datasets_allow=None):
    """Return a list of ``(dataset, task_type)`` tuples.

    Each tuple is one SLURM array slot. Inside the slot every enabled CPU
    method runs (NO_HPO and HPO) sharing one cached folds dict. Allow-lists
    must match those passed to Experiment1_Setup so the Setup-time count
    and the runtime task list agree.
    """
    tasks = []
    pd_datasets = _apply_allow(config['datasets']['pd'].keys(), datasets_allow)
    pd_methods = _apply_allow(config['methods']['pd'].keys(), methods_allow)
    if any(m in CPU_METHODS for m in pd_methods):
        for dataset in pd_datasets:
            tasks.append((dataset, 'pd'))
    lgd_datasets = _apply_allow(config['datasets']['lgd'].keys(), datasets_allow)
    lgd_methods = _apply_allow(config['methods']['lgd'].keys(), methods_allow)
    if any(m in CPU_METHODS for m in lgd_methods):
        for dataset in lgd_datasets:
            tasks.append((dataset, 'lgd'))
    return tasks


def _cpu_methods_for(task_type, config, methods_allow=None):
    methods = _apply_allow(config['methods'][task_type].keys(), methods_allow)
    return [m for m in methods if m in CPU_METHODS]


def _run_cpu_bundle(dataset, task_type, config, experiment_name, verbose,
                    methods_allow=None):
    """Run every CPU method (NO_HPO + HPO) for one (dataset, task_type)."""
    methods = _cpu_methods_for(task_type, config, methods_allow)
    print(f"[bundle] {dataset}/{task_type}: {len(methods)} CPU methods")
    for method in methods:
        hpo_modes = ['NO_HPO'] if method in NO_HPO_METHODS else ['NO_HPO', 'HPO']
        for hpo_mode in hpo_modes:
            try:
                run_single_method(
                    dataset=dataset, method=method, task_type=task_type,
                    hpo_mode=hpo_mode, config=config,
                    experiment_name=experiment_name, verbose=verbose,
                )
            except Exception as exc:  # noqa: BLE001 -- isolate per-method failures
                print(f"[bundle] FAIL {dataset}/{method}/{hpo_mode}: {exc}")
                # Continue to the next method; the failure is already logged
                # to results/{experiment}/logs/errors.log by run_single_method.
    # Free dataset memory before the next array slot
    clear_folds_cache()


def main():
    parser = argparse.ArgumentParser(description='Run CPU methods')
    parser.add_argument('--array_id', type=int, help='SLURM array task ID')
    parser.add_argument('--dataset', type=str, help='Specific dataset to run')
    parser.add_argument('--method', type=str, help='Specific method to run (retry mode)')
    parser.add_argument('--task_type', type=str, help='Task type (pd or lgd)')
    parser.add_argument('--hpo_mode', type=str, help='Single HPO mode -- retry mode only')
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
    print(f"CPU ORCHESTRATOR")
    print(f"{'='*70}")

    # --- Retry / single-cell mode ---------------------------------------
    if args.dataset and args.method and args.task_type and args.hpo_mode:
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

    # --- Array-driven bundled mode --------------------------------------
    if args.array_id is None:
        print("ERROR: provide either --array_id or (--dataset --method --task_type --hpo_mode)")
        sys.exit(1)

    cpu_tasks = build_cpu_task_list(config, methods_allow, datasets_allow)
    print(f"Mode: array-based (bundled)")
    print(f"Total tasks (datasets):  {len(cpu_tasks)}")
    print(f"Array ID:                {args.array_id}")
    print(f"{'='*70}\n")

    if args.array_id < 0 or args.array_id >= len(cpu_tasks):
        print(f"ERROR: Array ID {args.array_id} out of range [0, {len(cpu_tasks)-1}]")
        sys.exit(1)

    dataset, task_type = cpu_tasks[args.array_id]
    _run_cpu_bundle(dataset, task_type, config, args.experiment, args.verbose,
                    methods_allow=methods_allow)


if __name__ == "__main__":
    main()
