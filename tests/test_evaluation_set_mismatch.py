"""Tests for the evaluation-set mismatch detector.

The benchmark's comparability claim is that every method is scored on identical
observations. That broke once in practice: a preprocessing change shrank one LGD
dataset, only some methods were re-run, and the stale files kept predicting on
the older, larger version -- so that dataset's *observed* target mean differed by
method and every pooled mean, rank and test mixing them was invalid.

``evaluation_set_mismatches`` exists to catch that automatically. These tests pin
both halves of its contract: it must flag methods that disagree on the test
folds, and it must stay silent when sizes differ for a legitimate reason -- sweep
points are *meant* to have different row counts, and ``__HPO`` shares its twin's
folds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.utils.results_checking import evaluation_set_mismatches


def _write(root: Path, task: str, dataset: str, method: str,
           fold_sizes: tuple[int, ...]) -> None:
    """Write one npz holding per-fold y_true of the given lengths."""
    out = root / "experiment1" / task / dataset
    out.mkdir(parents=True, exist_ok=True)
    arrays = {f"fold_{i}_y_true": np.zeros(n)
              for i, n in enumerate(fold_sizes, start=1)}
    arrays.update({f"fold_{i}_y_pred": np.zeros(n)
                   for i, n in enumerate(fold_sizes, start=1)})
    np.savez_compressed(out / f"{method}.npz", **arrays)


def test_agreeing_methods_are_not_flagged(tmp_path: Path):
    for method in ("catboost", "tabpfn_v3", "LogReg"):
        _write(tmp_path, "pd", "0001.demo", method, (100, 100, 99))
    assert evaluation_set_mismatches("Experiment1", results_root=tmp_path).empty


def test_disagreeing_methods_are_flagged(tmp_path: Path):
    _write(tmp_path, "lgd", "0001.demo", "catboost", (100, 100, 99))
    _write(tmp_path, "lgd", "0001.demo", "LinearRegression", (100, 100, 99))
    _write(tmp_path, "lgd", "0001.demo", "tabpfn_v3", (110, 110, 109))   # stale

    found = evaluation_set_mismatches("Experiment1", results_root=tmp_path)
    assert not found.empty, "a method on a different row count must be reported"

    flagged = found[found["verdict"] == "MISMATCH"]
    assert len(flagged) == 1
    assert flagged.iloc[0]["methods"] == "tabpfn_v3"
    assert flagged.iloc[0]["total_rows"] == 329
    # the two agreeing methods are the majority, not the offender
    majority = found[found["verdict"] == "majority"].iloc[0]
    assert majority["n_methods"] == 2
    assert majority["total_rows"] == 299


def test_hpo_variant_shares_its_twins_folds(tmp_path: Path):
    """``__HPO`` is not a data axis -- it must be compared against the twin."""
    _write(tmp_path, "pd", "0001.demo", "xgboost", (100, 100))
    _write(tmp_path, "pd", "0001.demo", "xgboost__HPO", (110, 110))

    flagged = evaluation_set_mismatches("Experiment1", results_root=tmp_path)
    assert not flagged.empty, "HPO must not excuse a different evaluation set"


def test_sweep_points_of_different_sizes_are_not_flagged(tmp_path: Path):
    """Experiment 2/3 sweep points are *supposed* to differ in size."""
    _write(tmp_path, "pd", "0001.demo", "xgboost__row1000", (200, 200))
    _write(tmp_path, "pd", "0001.demo", "xgboost__row20000", (4000, 4000))
    _write(tmp_path, "pd", "0001.demo", "catboost__row1000", (200, 200))
    _write(tmp_path, "pd", "0001.demo", "catboost__row20000", (4000, 4000))

    assert evaluation_set_mismatches("Experiment1", results_root=tmp_path).empty


def test_mismatch_within_one_sweep_point_is_flagged(tmp_path: Path):
    """Grouping by sweep value must not hide a disagreement inside a value."""
    _write(tmp_path, "pd", "0001.demo", "xgboost__row1000", (200, 200))
    _write(tmp_path, "pd", "0001.demo", "catboost__row1000", (250, 250))

    flagged = evaluation_set_mismatches("Experiment1", results_root=tmp_path)
    assert not flagged.empty
    assert set(flagged["sweep"]) == {"row=1000"}


def test_missing_experiment_directory_is_empty_not_an_error(tmp_path: Path):
    assert evaluation_set_mismatches("Experiment1", results_root=tmp_path).empty


def test_shard_files_are_skipped(tmp_path: Path):
    """A shard holds only part of a cell, so its fold count means nothing."""
    _write(tmp_path, "pd", "0001.demo", "xgboost", (100, 100))
    _write(tmp_path, "pd", "0001.demo", "xgboost__shard_123_4", (7,))

    assert evaluation_set_mismatches("Experiment1", results_root=tmp_path).empty
