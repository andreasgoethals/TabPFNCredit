"""Shared pytest fixtures for TabPFNCredit.

Synthesises tiny PD / LGD datasets so tests run in < 5 seconds and never
depend on the real credit-risk data being checked into the repo.

Auto-skips GPU-required methods when CUDA is unavailable -- ``pytest -m
"not gpu"`` is the canonical CI invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

# Ensure src.* imports work without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ============================================================================
#  GPU detection
# ============================================================================

def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


HAS_CUDA = _has_cuda()


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU-marked tests if CUDA is missing."""
    if HAS_CUDA:
        return
    skip_marker = pytest.mark.skip(reason="No CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_marker)


# ============================================================================
#  Synthetic data fixtures
# ============================================================================

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def tiny_pd_dataset(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """50 rows, 4 num features + 2 cat features, binary target with ~30% positives."""
    n = 50
    N = rng.normal(size=(n, 4))
    C = rng.integers(0, 5, size=(n, 2))
    # Plant a signal so AUC > 0.5
    y = (N[:, 0] + N[:, 1] + 0.3 * C[:, 0] > 0).astype(int)
    return N, C, y, np.arange(n)


@pytest.fixture
def tiny_lgd_dataset(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """50 rows, 4 num features + 2 cat features, continuous target in [0, 1]."""
    n = 50
    N = rng.normal(size=(n, 4))
    C = rng.integers(0, 5, size=(n, 2))
    raw = 0.5 + 0.3 * N[:, 0] - 0.2 * C[:, 0] + 0.1 * rng.normal(size=n)
    y = np.clip(raw, 0.0, 1.0)
    return N, C, y


@pytest.fixture
def synthetic_probas(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Realistic (y_true, y_proba) for binary classification."""
    n = 200
    y_true = (rng.random(n) < 0.3).astype(int)
    # Predicted prob is correct on average but noisy.
    base = np.where(y_true == 1, 0.7, 0.2)
    y_proba_pos = np.clip(base + 0.2 * rng.normal(size=n), 0.01, 0.99)
    y_proba = np.stack([1.0 - y_proba_pos, y_proba_pos], axis=1)
    return y_true, y_proba
