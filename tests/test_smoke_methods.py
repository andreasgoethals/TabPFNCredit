"""Smoke tests: minimal end-to-end runs on synthetic data.

Replaces the old ``method_debugger.py`` script -- pytest infrastructure
gives us collection / filtering / parallelism for free.

Invoke just the smoke tests with::

    pytest -m smoke
"""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.smoke]


# Only the cheapest CPU methods -- the GPU / foundation ones need real
# packages installed (tabpfn, tabicl, ...). CI runs this list on a
# matrix of methods to verify that ``TALENT.run()`` integration still
# works end-to-end through the wrapper.
_SMOKE_METHODS_PD = ["LogReg", "RandomForest", "knn", "xgboost"]
_SMOKE_METHODS_LGD = ["LinearRegression", "RandomForest", "knn", "xgboost"]


def _make_tiny_dataset_dir(tmp_path, task: str, N, C, y):
    """Write a TALENT-format dataset directory under tmp_path."""
    import json

    dataset_dir = tmp_path / "data" / "synthetic"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # 60/20/20 split
    n = len(y)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    for split, sl in (("train", slice(0, i1)), ("val", slice(i1, i2)), ("test", slice(i2, n))):
        if N is not None:
            np.save(dataset_dir / f"N_{split}.npy", N[sl])
        if C is not None:
            np.save(dataset_dir / f"C_{split}.npy", C[sl])
        np.save(dataset_dir / f"y_{split}.npy", y[sl])
    info = {
        "task_type": "binclass" if task == "pd" else "regression",
        "n_num_features": int(N.shape[1]) if N is not None else 0,
        "n_cat_features": int(C.shape[1]) if C is not None else 0,
    }
    (dataset_dir / "info.json").write_text(json.dumps(info))
    return dataset_dir.parent


@pytest.mark.parametrize("method", _SMOKE_METHODS_PD)
def test_smoke_pd_method(method, tiny_pd_dataset, tmp_path):
    """Every cheap PD method must complete one fold without crashing."""
    pytest.importorskip("TALENT")
    N, C, y, _idx = tiny_pd_dataset

    # Build the train/val/test data the way TALENT expects.
    n = len(y)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    N_data = {"train": N[:i1].astype(float), "val": N[i1:i2].astype(float)}
    N_test = {"test": N[i2:].astype(float)}
    C_data = {"train": C[:i1].astype(str), "val": C[i1:i2].astype(str)}
    C_test = {"test": C[i2:].astype(str)}
    y_data = {"train": y[:i1], "val": y[i1:i2]}
    y_test = {"test": y[i2:]}
    info = {"task_type": "binclass", "n_num_features": 4, "n_cat_features": 2}

    import TALENT
    result = TALENT.run(
        method,
        (N_data, C_data, y_data),
        (N_test, C_test, y_test),
        info,
        save_path=str(tmp_path / "ckpt"),
    )
    assert result.predictions is not None
    assert result.predict_proba is not None
    assert result.predict_proba.shape[1] == 2
    assert result.predict_proba.shape[0] == len(y[i2:])
    # Threshold tuning fires for binclass
    assert result.threshold is not None
    assert 0.0 < result.threshold < 1.0


@pytest.mark.parametrize("method", _SMOKE_METHODS_LGD)
def test_smoke_lgd_method(method, tiny_lgd_dataset, tmp_path):
    """Every cheap LGD method must complete one fold without crashing."""
    pytest.importorskip("TALENT")
    N, C, y = tiny_lgd_dataset

    n = len(y)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    N_data = {"train": N[:i1].astype(float), "val": N[i1:i2].astype(float)}
    N_test = {"test": N[i2:].astype(float)}
    C_data = {"train": C[:i1].astype(str), "val": C[i1:i2].astype(str)}
    C_test = {"test": C[i2:].astype(str)}
    y_data = {"train": y[:i1], "val": y[i1:i2]}
    y_test = {"test": y[i2:]}
    info = {"task_type": "regression", "n_num_features": 4, "n_cat_features": 2}

    import TALENT
    result = TALENT.run(
        method,
        (N_data, C_data, y_data),
        (N_test, C_test, y_test),
        info,
        save_path=str(tmp_path / "ckpt"),
    )
    assert result.predictions is not None
    # No predict_proba for regression
    assert result.predict_proba is None
    assert result.threshold is None
