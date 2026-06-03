#!/usr/bin/env python3
"""Experiment 2: learning-curve analysis (metric vs training rows).

For each (dataset, method) the script walks ``row_limit`` DOWN from
``row_max`` to ``row_min`` in steps of ``row_step`` (both configurable
per task in ``CONFIG_EXPERIMENT.yaml`` under ``learning_curve.<task>``).
Each sweep point becomes its own result file via the suffix convention
documented in :mod:`src.utils.result_io`::

    results/experiment2/<task>/<dataset>/<method>__row<N>.json
    results/experiment2/<task>/<dataset>/<method>__row<N>.npz

Resumable: completed sweep points are skipped automatically (the
existence of the suffixed JSON is enough).
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import preprocess_dataset
from src.methods.method_runner import run_talent_method
from src.utils.config_reader import load_config
from src.utils.logging_setup import configure_task_logging, task_timer
from src.utils.result_io import build_method_name, has_complete_result, save_method
from src.utils.storage_handler import StorageHandler

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _cleanup_gpu() -> None:
    """Free GPU memory between sweep iterations."""
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_dataset_size(task: str, dataset: str) -> int:
    """Number of rows in the processed dataset."""
    _, _, y, _ = preprocess_dataset(task, dataset)
    return len(y)


def generate_row_limit_sequence(
    dataset_size: int, row_max: int, row_min: int, row_step: int,
) -> List[int]:
    """Build the descending sequence of row_limits to evaluate.

    Starts at ``min(dataset_size, row_max)``, optionally aligns to a
    multiple of ``row_step``, then steps down by ``row_step`` until
    crossing below ``row_min``.
    """
    current = min(dataset_size, row_max)
    if current < row_min:
        return [current]
    sequence = [current]
    aligned = (current // row_step) * row_step
    if aligned < current and aligned >= row_min:
        current = aligned
        sequence.append(current)
    while current - row_step >= row_min:
        current -= row_step
        sequence.append(current)
    return sequence


def run_learning_curve(
    *,
    dataset: str,
    method: str,
    task_type: str,
    config: dict,
    experiment_name: str = "experiment2",
    verbose: bool = False,
) -> None:
    """Run a learning-curve sweep for ONE (dataset, method, task)."""

    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "pd").mkdir(exist_ok=True)
    (experiment_path / "lgd").mkdir(exist_ok=True)

    log_file = configure_task_logging(
        experiment=experiment_name,
        dataset=dataset,
        method=method,
        task=task_type,
        results_root=experiment_path.parent,
        verbose=verbose,
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "node=%s job=%s array=%s log=%s",
        os.environ.get("SLURMD_NODENAME", "LOCAL"),
        os.environ.get("SLURM_JOB_ID", "-"),
        os.environ.get("SLURM_ARRAY_TASK_ID", "-"),
        log_file,
    )

    # --- read per-task sweep parameters ---
    lc = config["learning_curve"][task_type]
    row_max: int = lc["row_max"]
    row_min: int = lc["row_min"]
    row_step: int = lc["row_step"]
    min_dataset_size: int = lc.get("min_dataset_size", 0)

    dataset_size = get_dataset_size(task_type, dataset)
    if dataset_size < min_dataset_size:
        logger.info(
            "Skipping %s (size %d < min_dataset_size %d for task %s)",
            dataset, dataset_size, min_dataset_size, task_type,
        )
        return

    row_limits = generate_row_limit_sequence(dataset_size, row_max, row_min, row_step)
    logger.info("Sweep: %d points from %d down to %d (step %d)",
                len(row_limits), row_limits[0], row_limits[-1], row_step)

    cv_splits = config["split"]["cv_splits"]

    for i, row_limit in enumerate(row_limits, start=1):
        sweep_method_name = build_method_name(method, {"row": row_limit})
        if has_complete_result(
            base=experiment_path.parent,
            experiment=experiment_name,
            task=task_type,
            dataset=dataset,
            method=sweep_method_name,
            expected_folds=cv_splits,
        ):
            logger.debug("[%d/%d] SKIP row_limit=%d (already done)", i, len(row_limits), row_limit)
            continue

        with task_timer(f"row_limit={row_limit} ({i}/{len(row_limits)})", logger):
            fold_results = run_talent_method(
                task=task_type,
                dataset=dataset,
                method=method,
                test_size=config["split"]["test_size"],
                val_size=config["split"]["val_size"],
                cv_splits=cv_splits,
                seed=config["split"]["seed"],
                row_limit=row_limit,
                sampling=config["split"].get("sampling"),
                max_epoch=config["training"]["max_epochs"],
                batch_size=config["training"]["batch_size"],
                tune=False,
                n_trials=1,
                early_stopping=config["training"]["early_stopping"],
                early_stopping_patience=config["training"]["early_stopping_patience"],
                verbose=verbose,
            )

        if fold_results:
            save_method(
                fold_results,
                base=experiment_path.parent,
                experiment=experiment_name,
                task=task_type,
                dataset=dataset,
                method=sweep_method_name,
            )

        _cleanup_gpu()


if __name__ == "__main__":  # pragma: no cover
    print("Run via Experiment2_CPU.py / Experiment2_GPU_*.py "
          "or `tabpfncredit run --experiment Experiment2 ... --row-limit N`")
