"""Tests for DataFeeder's nested minority subsampling (Experiment 3).

The Experiment-3 imbalance sweep walks the minority proportion DOWN in tiny
steps. The scientific requirement is that lowering the target only ever
*deletes more* of the same minority rows -- it must never re-draw a fresh
random subset -- so the performance trend reflects fewer minority cases, not
lucky/unlucky draws. ``DataFeeder._nested_minority_keep_mask`` enforces this by
keeping a PREFIX of a single fixed-seed permutation of the minority indices.

These tests pin that contract: nesting, monotonicity, correct achieved
proportion, determinism, and the no-op edge cases. They import only the static
mask helper's behaviour via the class, so they do not need TALENT installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.data_feeder import DataFeeder


def _make_y(n_minority: int, n_majority: int, seed: int = 0) -> np.ndarray:
    y = np.array([1] * n_minority + [0] * n_majority)
    np.random.default_rng(seed).shuffle(y)
    return y


def test_nested_and_monotone():
    y = _make_y(200, 800)
    n_maj = int((y == 0).sum())
    targets = [0.15, 0.10, 0.05, 0.025, 0.01, 0.0025]

    kept_sets = []
    for t in targets:
        mask = DataFeeder._nested_minority_keep_mask(y, t, seed=42)
        # majority is always kept in full -- only minority rows are removed
        assert mask[y == 0].all()
        kept_min = set(np.where(mask & (y == 1))[0].tolist())
        kept_sets.append(kept_min)
        # achieved proportion is within a row of the target
        achieved = len(kept_min) / (len(kept_min) + n_maj)
        assert abs(achieved - t) < 0.01

    # each lower target keeps a STRICT SUBSET of the next-higher target
    for hi, lo in zip(kept_sets, kept_sets[1:]):
        assert lo.issubset(hi)
        assert len(lo) < len(hi)


def test_determinism():
    y = _make_y(150, 850)
    a = DataFeeder._nested_minority_keep_mask(y, 0.05, seed=42)
    b = DataFeeder._nested_minority_keep_mask(y, 0.05, seed=42)
    assert np.array_equal(a, b)


def test_above_prevalence_is_noop():
    # prevalence is 0.20; you cannot raise the minority share by deleting
    # minority rows, so a target above prevalence keeps everything.
    y = _make_y(200, 800)
    assert DataFeeder._nested_minority_keep_mask(y, 0.50, seed=1).all()


@pytest.mark.parametrize("y", [np.array([0, 0, 0]), np.array([1, 1]), np.array([])])
def test_degenerate_inputs_are_noop(y):
    # single-class or empty -> nothing to subsample
    assert DataFeeder._nested_minority_keep_mask(y, 0.1, seed=1).all()


def test_nesting_holds_across_seeds_internally():
    # For a FIXED seed the chain is nested; different seeds give different (but
    # internally still nested) chains. This guards the prefix-of-permutation
    # implementation against accidentally re-drawing per target.
    y = _make_y(300, 700)
    for seed in (0, 7, 123):
        big = set(np.where(DataFeeder._nested_minority_keep_mask(y, 0.10, seed=seed) & (y == 1))[0])
        small = set(np.where(DataFeeder._nested_minority_keep_mask(y, 0.02, seed=seed) & (y == 1))[0])
        assert small.issubset(big)


# ---------------------------------------------------------------------------
# Experiment 2: nested row-limit subsampling (the learning curve)
# ---------------------------------------------------------------------------

def test_rowlimit_nested_and_stratified_pd():
    # Lowering the row cap must keep a STRICT SUBSET of the larger cap's rows,
    # preserve the class ratio, and never drop the minority class entirely.
    y = _make_y(200, 800)  # prevalence 0.20
    caps = [800, 500, 300, 100, 50]
    sets = []
    for c in caps:
        idx = DataFeeder._nested_subsample_indices(y, c, seed=42, stratify=True)
        s = set(idx.tolist())
        sets.append(s)
        # class ratio preserved and minority present
        assert (y[idx] == 1).sum() > 0
        assert abs((y[idx] == 1).mean() - 0.20) < 0.02
    for hi, lo in zip(sets, sets[1:]):
        assert lo.issubset(hi)
        assert len(lo) < len(hi)


def test_rowlimit_nested_regression_and_determinism():
    y = np.arange(1000).astype(float)  # regression target
    sets = []
    for c in [800, 500, 100, 20]:
        idx = DataFeeder._nested_subsample_indices(y, c, seed=7, stratify=False)
        sets.append(set(idx.tolist()))
    for hi, lo in zip(sets, sets[1:]):
        assert lo.issubset(hi)
    a = DataFeeder._nested_subsample_indices(y, 500, seed=7, stratify=False)
    b = DataFeeder._nested_subsample_indices(y, 500, seed=7, stratify=False)
    assert np.array_equal(a, b)


def test_rowlimit_noop_when_cap_exceeds_size():
    y = _make_y(20, 80)
    idx = DataFeeder._nested_subsample_indices(y, 1000, seed=1, stratify=True)
    assert np.array_equal(idx, np.arange(100))
