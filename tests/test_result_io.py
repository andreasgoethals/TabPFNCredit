"""Tests for src.utils.result_io.save_method / load_method.

Pins the JSON+npz round-trip contract: metrics + aggregates land in the
JSON, prediction arrays land in the npz, and load_method reassembles the
original per-fold dict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestSaveMethod:
    """Tests for the per-(dataset, method) JSON+npz layout (result_io.save_method)."""

    def test_save_method_round_trip(self, tmp_path: Path):
        import numpy as np
        from src.utils.result_io import save_method, load_method

        fold_results = {
            0: {
                "fold_id": 0,
                "metrics": {"AUC": 0.85, "F1": 0.7},
                "train_time": 1.2,
                "predict_time": 0.3,
                "threshold": 0.4,
                "used_hpo": False,
                "method": "catboost",
                "dataset": "ds",
                "task": "pd",
                "y_true": np.array([0, 1, 0, 1, 0]),
                "y_prob": np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0.6, 0.4]]),
                "y_pred": np.array([0, 1, 0, 1, 0]),
                "info": {"task_type": "binclass"},
            },
            1: {
                "fold_id": 1,
                "metrics": {"AUC": 0.87, "F1": 0.72},
                "train_time": 1.4,
                "predict_time": 0.35,
                "threshold": 0.42,
                "used_hpo": False,
                "method": "catboost",
                "dataset": "ds",
                "task": "pd",
                "y_true": np.array([1, 0, 1, 0, 1]),
                "y_prob": np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9]]),
                "y_pred": np.array([1, 0, 1, 0, 1]),
                "info": {"task_type": "binclass"},
            },
        }
        save_method(
            fold_results,
            base=tmp_path, experiment="exp_test",
            task="pd", dataset="ds", method="catboost",
        )
        json_path = tmp_path / "exp_test" / "pd" / "ds" / "catboost.json"
        npz_path = tmp_path / "exp_test" / "pd" / "ds" / "catboost.npz"
        assert json_path.exists()
        assert npz_path.exists()

        data = json.loads(json_path.read_text())
        assert data["n_folds"] == 2
        assert "AUC" in data["aggregates"]
        assert data["aggregates"]["AUC"]["mean"] == pytest.approx(0.86, abs=1e-9)

        # Round-trip via load_method
        loaded = load_method(
            base=tmp_path, experiment="exp_test",
            task="pd", dataset="ds", method="catboost",
        )
        assert set(loaded.keys()) == {0, 1}
        assert loaded[0]["metrics"]["AUC"] == 0.85
        np.testing.assert_array_equal(loaded[0]["y_true"], fold_results[0]["y_true"])
        np.testing.assert_array_almost_equal(loaded[1]["y_prob"], fold_results[1]["y_prob"])
