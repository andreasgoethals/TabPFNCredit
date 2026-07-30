"""Invariants every processed dataset must satisfy before it is benchmarked.

Two real bugs motivated this, both in one dataset and both invisible in the
results tables:

* German Credit's raw file has no header row, so the default ``header=0`` ate its
  first record and the benchmark scored 999 of 1000 observations.
* That raw file had itself been written out from such a mis-read, so the recovered
  record carried pandas' duplicate-column suffixes ("4" -> "4.1", "1" -> "1.1").
  Reading with ``header=None`` restored the row but put a THIRD value, 1.1, into
  a binary target -- a worse failure than the missing row, and one that no
  row-count check would notice.

These checks run against ``data/processed/`` rather than against the private
preprocessing module, so they hold for every dataset and stay meaningful in a
clone that has data but not that module. They skip when no processed data is
present (a fresh clone, or CI).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


def _datasets() -> list[tuple[str, Path]]:
    if not PROCESSED.is_dir():
        return []
    return [(f"{task}/{d.name}", d)
            for task in ("pd", "lgd")
            for d in sorted((PROCESSED / task).glob("*"))
            if (d / "info.json").exists() and (d / "y.npy").exists()]


DATASETS = _datasets()

pytestmark = pytest.mark.skipif(
    not DATASETS, reason="no processed datasets present (fresh clone / CI)")


def _load(path: Path):
    info = json.loads((path / "info.json").read_text(encoding="utf-8"))
    y = np.load(path / "y.npy", allow_pickle=True)
    return info, y


@pytest.mark.parametrize("name,path", DATASETS, ids=[n for n, _ in DATASETS])
def test_row_counts_agree(name: str, path: Path):
    """info.json, y and the feature matrices must describe the same rows."""
    info, y = _load(path)
    assert info["n_samples"] == len(y), (
        f"{name}: info.json says {info['n_samples']} samples, y has {len(y)}")
    for matrix in ("N.npy", "C.npy"):
        file = path / matrix
        if not file.exists():
            continue
        arr = np.load(file, allow_pickle=True)
        assert arr.shape[0] == len(y), (
            f"{name}: {matrix} has {arr.shape[0]} rows, y has {len(y)}")


@pytest.mark.parametrize("name,path", DATASETS, ids=[n for n, _ in DATASETS])
def test_pd_targets_are_binary(name: str, path: Path):
    """A PD target may only be 0 or 1.

    This is the check that catches a mis-parsed record: a stray 1.1 sails through
    every row-count and range test but makes AUC, the base rate and every
    calibration number wrong.
    """
    if not name.startswith("pd/"):
        pytest.skip("classification check")
    _info, y = _load(path)
    values = set(np.unique(y[np.isfinite(y)]).tolist())
    assert values <= {0.0, 1.0}, f"{name}: non-binary PD target values {sorted(values)}"
    assert len(values) == 2, f"{name}: PD target has only {values} -- no variation"


@pytest.mark.parametrize("name,path", DATASETS, ids=[n for n, _ in DATASETS])
def test_lgd_targets_are_within_the_unit_interval(name: str, path: Path):
    """LGD is a loss fraction; preprocessing clips it to [0, 1]."""
    if not name.startswith("lgd/"):
        pytest.skip("regression check")
    _info, y = _load(path)
    finite = y[np.isfinite(y)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0, (
        f"{name}: LGD target outside [0, 1] -> [{finite.min()}, {finite.max()}]")


@pytest.mark.parametrize("name,path", DATASETS, ids=[n for n, _ in DATASETS])
def test_feature_counts_match_info(name: str, path: Path):
    """The declared numerical/categorical widths must match the arrays."""
    info, _y = _load(path)
    for key, matrix in (("n_num_features", "N.npy"), ("n_cat_features", "C.npy")):
        declared = int(info.get(key) or 0)
        file = path / matrix
        actual = int(np.load(file, allow_pickle=True).shape[1]) if file.exists() else 0
        assert declared == actual, (
            f"{name}: info.json {key}={declared} but {matrix} has {actual} columns")


def test_no_dataset_lost_a_row_to_a_header():
    """A headerless raw CSV must yield one processed row per non-empty line.

    Guards the specific failure that cost German Credit its first record. Only
    datasets whose raw CSV is still present can be checked; the rest skip.
    """
    raw_root = PROJECT_ROOT / "data" / "raw"
    if not raw_root.is_dir():
        pytest.skip("no raw data present")

    def first_line_is_data(csv_path: Path) -> bool:
        """A header is all-text above numeric data; a data row is not."""
        import csv as _csv
        with csv_path.open(encoding="utf-8", errors="ignore", newline="") as fh:
            reader = _csv.reader(fh)
            rows = [next(reader, None) for _ in range(3)]
        if any(r is None for r in rows) or len({len(r) for r in rows}) != 1:
            return False

        def mask(row):
            out = []
            for field in row:
                try:
                    float(field.strip())
                    out.append(True)
                except ValueError:
                    out.append(False)
            return out

        masks = [mask(r) for r in rows]
        return masks[0] == masks[1] == masks[2] and any(masks[1])

    checked = 0
    for name, path in DATASETS:
        task, slug = name.split("/", 1)
        csv_path = raw_root / task / f"{slug}.csv"
        if not csv_path.exists() or not first_line_is_data(csv_path):
            continue
        checked += 1
        with csv_path.open(encoding="utf-8", errors="ignore") as fh:
            n_lines = sum(1 for line in fh if line.strip())
        _info, y = _load(path)
        assert len(y) == n_lines, (
            f"{name}: headerless raw file has {n_lines} records but only "
            f"{len(y)} were processed -- the first row was probably read as a header")
    if not checked:
        pytest.skip("no headerless raw CSVs present")
