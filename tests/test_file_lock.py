"""Tests for src.utils.file_lock + the read-modify-write race in remove_results.

Specifically covers the bug fixed in remove_results.py: concurrent
writers should not lose updates when modifying the same pickle file.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import pytest

from src.utils.file_lock import FileLock, _HAS_FCNTL, _HAS_PORTALOCKER


# Concurrent-writer tests are only meaningful when a real locking backend
# is available; on Windows dev machines without portalocker they would
# always fail (the FileLock degrades to a warning + no-op by design).
HAS_LOCK_BACKEND = _HAS_FCNTL or _HAS_PORTALOCKER


def _writer_proc(path: str, key: str, value: int) -> None:
    """Read pickle, mutate, write -- all under one exclusive lock."""
    with FileLock(path, exclusive=True, binary=True) as f:
        f.seek(0)
        content = f.read()
        try:
            data = pickle.loads(content) if content else {}
        except Exception:
            data = {}
        data[key] = value
        f.seek(0)
        f.truncate()
        pickle.dump(data, f)
        f.flush()
        time.sleep(0.05)  # increase the chance of contention


class TestFileLock:

    @pytest.mark.skipif(
        not HAS_LOCK_BACKEND,
        reason="No locking backend (install portalocker on Windows / fcntl on POSIX).",
    )
    def test_concurrent_writers_no_lost_updates(self, tmp_path: Path):
        """Spawn 8 writers, each adding a unique key. All 8 keys must survive."""
        path = tmp_path / "shared.pkl"
        # Initialise the file (FileLock opens in a+b mode, so empty start is fine).
        with FileLock(path, exclusive=True, binary=True) as f:
            pickle.dump({}, f)

        procs = []
        for i in range(8):
            p = mp.Process(target=_writer_proc, args=(str(path), f"k{i}", i))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=10)
            assert p.exitcode == 0, f"writer crashed (exit {p.exitcode})"

        with open(path, "rb") as f:
            final = pickle.load(f)
        assert len(final) == 8, f"lost updates: got {sorted(final.keys())}"


class TestSummaryUpdate:
    """Tests for the JSON+npz summary writer (B1)."""

    def test_update_method_summary_round_trip(self, tmp_path: Path):
        from src.utils.result_io import update_method_summary
        import numpy as np

        fold_results = {
            0: {
                "metrics": {"AUC": 0.85, "F1": 0.7},
                "train_time": 1.2,
                "predict_time": 0.3,
                "threshold": 0.4,
                "used_hpo": False,
            },
            1: {
                "metrics": {"AUC": 0.87, "F1": 0.72},
                "train_time": 1.4,
                "predict_time": 0.35,
                "threshold": 0.42,
                "used_hpo": False,
            },
        }
        update_method_summary(
            fold_results,
            base=tmp_path, experiment="exp_test",
            task="pd", dataset="ds", method="catboost",
            hpo_mode="NO_HPO",
        )
        summary_path = (
            tmp_path / "exp_test" / "pd" / "ds" / "catboost" / "NO_HPO" / "summary.json"
        )
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["n_folds"] == 2
        assert "AUC" in data["aggregates"]
        assert data["aggregates"]["AUC"]["mean"] == pytest.approx(0.86, abs=1e-9)
