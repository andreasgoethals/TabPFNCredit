"""Central filesystem-path resolution for TabPFNCredit.

Lookup policy (matches the project's storage layout on the HPC cluster):

* **Datasets & checkpoints** are read from the repo-local folder FIRST,
  then from the shared *project storage* (the "staging" area). A workstation
  uses its local ``data/`` / ``checkpoints/``; on the cluster, where those
  folders may be absent, every lookup transparently falls back to the large,
  non-purged project storage.
* **Results** default to the project storage (large, non-purged), overridable
  with ``$TABPFN_RESULTS_ROOT``.
* **Logs** stay on the repo root (general data storage), never on project
  storage. See :func:`logs_root`.
* **Regenerable caches / scratch** go to the project storage so they cannot
  fill the (small) general data storage. Overridable with ``$TABPFN_CACHE_ROOT``.

The project-storage root is ``$TABPFN_STAGING_ROOT`` (default
``/staging/leuven/stg_00211``). When that directory does not exist (e.g. a
laptop), every project-storage lookup is skipped and the repo-local folders
are used, so this module never changes local behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# Repo root: <repo>/src/utils/paths.py -> parents[2] == <repo>.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default shared project-storage root on the cluster. Env-overridable so a
# different project number / mount point needs no code change.
_DEFAULT_STAGING_ROOT = "/staging/leuven/stg_00211"


# ============================================================================
#  Project-storage ("staging") root
# ============================================================================

def staging_root() -> Optional[Path]:
    """Return the shared project-storage root, or ``None`` if unavailable.

    Honours ``$TABPFN_STAGING_ROOT`` (default ``/staging/leuven/stg_00211``).
    Returns ``None`` when the directory does not exist -- e.g. on a laptop --
    so callers fall back to the repo-local folders automatically.
    """
    raw = os.environ.get("TABPFN_STAGING_ROOT", _DEFAULT_STAGING_ROOT).strip()
    if not raw:
        return None
    p = Path(raw)
    try:
        return p if p.is_dir() else None
    except OSError:
        return None


def _ordered_roots(subdir: str) -> List[Path]:
    """``<repo>/<subdir>`` first, then ``<staging>/<subdir>`` (if it exists)."""
    roots = [PROJECT_ROOT / subdir]
    st = staging_root()
    if st is not None:
        cand = st / subdir
        if cand not in roots:
            roots.append(cand)
    return roots


def data_roots() -> List[Path]:
    """All ``data/`` roots in lookup order: repo-local first, then project storage."""
    return _ordered_roots("data")


def checkpoint_roots() -> List[Path]:
    """All ``checkpoints/`` roots in lookup order: repo-local first, then project storage."""
    return _ordered_roots("checkpoints")


# ============================================================================
#  Dataset resolution (datasets: repo first, then project storage)
# ============================================================================

def find_processed_dir(task: str, dataset: str) -> Optional[Path]:
    """Return the processed-dataset directory, repo first then project storage.

    A directory counts as "processed" once its ``y.npy`` exists (written last
    by the preprocessing step). Returns ``None`` if no root has it yet.
    """
    for root in data_roots():
        d = root / "processed" / task.lower() / dataset
        if (d / "y.npy").exists():
            return d
    return None


def find_raw_path(task: str, dataset: str) -> Optional[Path]:
    """Return the raw dataset path *stem* (no extension) that exists, else ``None``.

    Checks repo-local ``data/raw`` first, then project storage, for
    ``<dataset>.csv`` or ``<dataset>.parquet``. The returned stem is what the
    loader appends ``.csv`` / ``.parquet`` to.
    """
    for root in data_roots():
        stem = root / "raw" / task.lower() / dataset
        if stem.with_suffix(".csv").exists() or stem.with_suffix(".parquet").exists():
            return stem
    return None


def processed_write_dir(task: str, dataset: str) -> Path:
    """Directory to WRITE freshly preprocessed arrays into.

    Writes next to wherever the raw file lives, so data staged on project
    storage produces its processed cache on project storage too (never filling
    the general data storage). Falls back to the repo-local ``data/processed``
    when the raw file's root can't be determined.
    """
    raw = find_raw_path(task, dataset)
    if raw is not None:
        root = raw.parents[2]  # <root>/raw/<task>/<ds> -> <root>
        return root / "processed" / task.lower() / dataset
    return PROJECT_ROOT / "data" / "processed" / task.lower() / dataset


def raw_task_dirs(task: str) -> List[Path]:
    """All existing ``data/raw/<task>`` dirs (repo first, then project storage)."""
    return [r / "raw" / task.lower() for r in data_roots()]


def processed_task_dirs(task: str) -> List[Path]:
    """All existing ``data/processed/<task>`` dirs (repo first, then project storage)."""
    return [r / "processed" / task.lower() for r in data_roots()]


# ============================================================================
#  Output roots: results (project storage) / logs (general data storage)
# ============================================================================

def results_root() -> Path:
    """Where result JSON/npz + summary CSVs are written.

    Honours ``$TABPFN_RESULTS_ROOT`` (the SLURM scripts point it at project
    storage); otherwise defaults to ``<staging>/results`` when project storage
    exists, else the repo-local ``results/`` for laptop runs.
    """
    env = os.environ.get("TABPFN_RESULTS_ROOT")
    if env:
        return Path(env)
    st = staging_root()
    if st is not None:
        return st / "results"
    return PROJECT_ROOT / "results"


def logs_root() -> Path:
    """Where logs live -- ALWAYS the repo root (general data storage), never project storage."""
    return PROJECT_ROOT / "logs"


def cache_root() -> Path:
    """Root for regenerable caches (joblib folds cache, TALENT scratch).

    Honours ``$TABPFN_CACHE_ROOT``. Otherwise prefers ``$VSC_SCRATCH`` -- the
    cluster's purge-after-28-days parallel filesystem meant for regenerable job
    I/O. We deliberately do NOT cache on the project storage: joblib writes
    thousands of tiny files and that filesystem is inode-limited
    (~150k inodes/TB), so a cache there exhausts the file quota. Falls back to
    the repo-local ``.cache`` off-cluster.
    """
    env = os.environ.get("TABPFN_CACHE_ROOT")
    if env:
        return Path(env)
    scratch = os.environ.get("VSC_SCRATCH")
    if scratch and Path(scratch).is_dir():
        return Path(scratch) / "tabpfncredit" / "cache"
    return PROJECT_ROOT / ".cache"


def describe() -> dict:
    """Human-readable snapshot of resolved roots (used by ``tabpfncredit doctor``)."""
    st = staging_root()
    return {
        "project_root": str(PROJECT_ROOT),
        "staging_root": str(st) if st else "(not found)",
        "data_roots": [str(p) for p in data_roots()],
        "checkpoint_roots": [str(p) for p in checkpoint_roots()],
        "results_root": str(results_root()),
        "logs_root": str(logs_root()),
        "cache_root": str(cache_root()),
    }


__all__ = [
    "PROJECT_ROOT",
    "staging_root",
    "data_roots",
    "checkpoint_roots",
    "find_processed_dir",
    "find_raw_path",
    "processed_write_dir",
    "raw_task_dirs",
    "processed_task_dirs",
    "results_root",
    "logs_root",
    "cache_root",
    "describe",
]
