#!/usr/bin/env python3
"""Experiment 3: class-imbalance analysis (PD only).

For each (dataset, method) the script walks ``minority_proportion`` DOWN
from ``minority_proportion_max`` to ``minority_proportion_min`` in steps
of ``minority_proportion_step``. Each sweep point gets its own file via
the suffix convention in :mod:`src.utils.result_io`::

    results/experiment3/pd/<dataset>/<method>__min<X>.json
    results/experiment3/pd/<dataset>/<method>__min<X>.npz

Resumable: completed sweep points are skipped automatically.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import List

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_minority_sequence(
    p_max: float, p_min: float, p_step: float,
) -> List[float]:
    """Descending sequence of minority-class proportions."""
    seq: List[float] = []
    p = p_max
    while p >= p_min - 1e-12:
        seq.append(round(p, 6))
        p -= p_step
    return seq


def run_imbalance_sweep(
    *,
    dataset: str,
    method: str,
    task_type: str,
    config: dict,
    experiment_name: str = "experiment3",
    verbose: bool = False,
) -> None:
    """Run a class-imbalance sweep for ONE (dataset, method). PD only."""
    if task_type.lower() != "pd":
        # Experiment 3 only makes sense for binary classification.
        return

    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "pd").mkdir(exist_ok=True)

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

    p_max = config["imbalance"]["minority_proportion_max"]
    p_min = config["imbalance"]["minority_proportion_min"]
    p_step = config["imbalance"]["minority_proportion_step"]
    proportions = generate_minority_sequence(p_max, p_min, p_step)
    logger.info("Sweep: %d minority-proportion points from %g to %g (step %g)",
                len(proportions), p_max, p_min, p_step)

    cv_splits = config["split"]["cv_splits"]

    for i, p in enumerate(proportions, start=1):
        sweep_method_name = build_method_name(method, {"min": p})
        if has_complete_result(
            base=experiment_path.parent,
            experiment=experiment_name,
            task=task_type,
            dataset=dataset,
            method=sweep_method_name,
            expected_folds=cv_splits,
        ):
            logger.debug("[%d/%d] SKIP minority=%g (already done)", i, len(proportions), p)
            continue

        with task_timer(f"minority={p:g} ({i}/{len(proportions)})", logger):
            # DataFeeder's `sampling` parameter undersamples the majority class
            # to hit the requested minority proportion.
            fold_results = run_talent_method(
                task=task_type,
                dataset=dataset,
                method=method,
                test_size=config["split"]["test_size"],
                val_size=config["split"]["val_size"],
                cv_splits=cv_splits,
                seed=config["split"]["seed"],
                row_limit=config["split"].get("row_limit"),
                sampling=p,
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
    print("Run via Experiment3_CPU.py / Experiment3_GPU_*.py "
          "or `tabpfncredit run --experiment Experiment3 ... --minority-proportion P`")
