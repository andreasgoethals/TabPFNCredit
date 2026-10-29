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
* **Regenerable caches / scratch** use ``$VSC_SCRATCH`` when available, or
  the local ``.cache`` directory. Override with ``$TABPFN_CACHE_ROOT``.

``$TABPFN_STAGING_ROOT`` keeps its legacy meaning: the exact project directory.
Alternatively, ``$TABPFN_PROJECT_ROOT`` or ``$VSC_PROJECT_ROOT`` selects a shared
allocation; ``TabPFNCredit`` is appended. Explicit new roots must exist. Without
these settings the historical Lustre default and local fallback are unchanged.

The opt-in ``TABPFN_VSC_PROFILE=container`` validates selected datasets and
persistent writes under explicit PROJECT storage, and caches under scratch.
Read precedence remains repo first: an invalid selected dataset raises instead
of silently switching to another copy. Repo links into PROJECT remain valid.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# Repo root: <repo>/src/utils/paths.py -> parents[2] == <repo>.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default shared project-storage root on the cluster: this project's own
# subfolder of the shared ``stg_00211`` allocation (everything lives under
# ``.../stg_00211/TabPFNCredit``, never directly in the allocation root).
# Env-overridable so a different project number / mount point needs no code
# change.
_DEFAULT_STAGING_ROOT = "/lustre1/project/stg_00211/TabPFNCredit"


# ============================================================================
#  Project-storage ("staging") root
# ============================================================================

def _container_profile() -> bool:
    return os.environ.get("TABPFN_VSC_PROFILE", "").strip().lower() == "container"


def _require_directory(path: Path, setting: str) -> Path:
    """Validate an explicit absolute directory without creating or changing it."""
    if not path.is_absolute():
        raise ValueError(f"{setting} must name an absolute directory: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"{setting} is not a directory: {path}")
    return resolved


def _validated_write_path(path: Path, root: Path, setting: str) -> Path:
    """Check containment and the existing writable ancestor, including symlinks."""
    resolved_root = _require_directory(root, setting + " storage root")
    if not path.is_absolute():
        raise ValueError(f"{setting} must be absolute: {path}")
    resolved = path.resolve()
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{setting} must be beneath {resolved_root}: {path}")
    ancestor = resolved
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise NotADirectoryError(f"{setting} has a non-directory ancestor: {ancestor}")
    if not os.access(ancestor, os.W_OK | os.X_OK):
        raise PermissionError(f"{setting} is not writable beneath {ancestor}")
    return path


def _validate_project_read(path: Path, description: str) -> None:
    """Validate the already selected input without changing lookup precedence."""
    root = _require_directory(staging_root(), "PROJECT storage")
    resolved = path.resolve(strict=True)
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"{description} must be beneath PROJECT {root}: {path}. "
                         "The selected dataset was not changed; review its storage migration.")


def staging_root() -> Optional[Path]:
    """Return the project directory, preserving the exact legacy root override.

    Precedence: TABPFN_STAGING_ROOT (exact directory), TABPFN_PROJECT_ROOT
    (shared allocation), VSC_PROJECT_ROOT (shared allocation), historical
    Lustre default. Only legacy mode permits an unavailable root/local fallback.
    """
    strict = _container_profile()
    setting = "TABPFN_STAGING_ROOT"
    if setting in os.environ:
        raw = os.environ[setting].strip()
    else:
        setting = next((name for name in ("TABPFN_PROJECT_ROOT", "VSC_PROJECT_ROOT")
                        if name in os.environ), "")
        if setting:
            raw = os.environ[setting].strip()
            if not raw:
                raise ValueError(f"{setting} must name an existing shared PROJECT root")
            raw = str(Path(raw) / "TabPFNCredit")
            strict = True
        elif strict:
            raise ValueError("Container profile requires explicit PROJECT storage: "
                             "TABPFN_STAGING_ROOT, TABPFN_PROJECT_ROOT or VSC_PROJECT_ROOT")
        else:
            raw = _DEFAULT_STAGING_ROOT
    if not raw:
        if strict:
            raise ValueError(f"{setting} must name an existing PROJECT directory")
        return None
    p = Path(raw)
    if strict:
        _require_directory(p, setting)
        return p
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
    In the container profile, validate this selection and its files on PROJECT.
    """
    for root in data_roots():
        d = root / "processed" / task.lower() / dataset
        if (d / "y.npy").exists():
            if _container_profile():
                _validate_project_read(d, "Processed dataset")
                for array in d.glob("*.npy"):
                    _validate_project_read(array, "Processed dataset array")
                info = d / "info.json"
                if info.exists() or info.is_symlink():
                    _validate_project_read(info, "Processed dataset metadata")
            return d
    return None


def raw_file_for(stem: Path, ext: str) -> Path:
    """``stem`` + ``ext``, APPENDED -- never ``Path.with_suffix``.

    Every dataset slug has the form ``NNNN.name`` ("0001.gmsc"), so pathlib reads
    ``.gmsc`` as the suffix and ``with_suffix(".csv")`` REPLACES it, yielding
    "0001.csv". That silently made :func:`find_raw_path` return ``None`` for every
    dataset in the benchmark: preprocessing always fell back to the repo-local
    path (so raw data on project storage was never seen) and the raw row-count
    fallback never fired, which let datasets drop out of experiment selection
    whenever their processed cache was absent.
    """
    return stem.parent / (stem.name + ext)


def find_raw_path(task: str, dataset: str) -> Optional[Path]:
    """Return the raw dataset path *stem* (no extension) that exists, else ``None``.

    Checks repo-local ``data/raw`` first, then project storage, for
    ``<dataset>.csv`` or ``<dataset>.parquet``. The returned stem is what the
    loader appends ``.csv`` / ``.parquet`` to. The container profile validates
    the selected physical file, preserving the loader's CSV-first precedence.
    """
    for root in data_roots():
        stem = root / "raw" / task.lower() / dataset
        for ext in (".csv", ".parquet"):
            candidate = raw_file_for(stem, ext)
            if candidate.exists():
                if _container_profile():
                    _validate_project_read(candidate, "Raw dataset")
                return stem
    return None


def processed_write_dir(task: str, dataset: str) -> Path:
    """Directory to WRITE freshly preprocessed arrays into.

    Writes next to wherever the raw file lives, so data staged on project
    storage produces its processed cache on project storage too (never filling
    the general data storage). Falls back to the repo-local ``data/processed``
    when the raw file's root can't be determined. The container profile refuses
    a destination outside PROJECT rather than changing input selection.
    """
    raw = find_raw_path(task, dataset)
    if raw is not None:
        root = raw.parents[2]  # <root>/raw/<task>/<ds> -> <root>
        destination = root / "processed" / task.lower() / dataset
    else:
        destination = PROJECT_ROOT / "data" / "processed" / task.lower() / dataset
    if _container_profile():
        return _validated_write_path(destination, staging_root(), "Processed data")
    return destination


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
    exists, else the repo-local ``results/`` for laptop runs. The container
    profile requires a destination beneath its explicit PROJECT directory.
    """
    env = os.environ.get("TABPFN_RESULTS_ROOT")
    if _container_profile():
        st = staging_root()
        return _validated_write_path(Path(env) if env else st / "results", st,
                                     "TABPFN_RESULTS_ROOT")
    # Resolve explicitly selected new aliases even when a results override exists.
    if "TABPFN_STAGING_ROOT" not in os.environ and any(
        name in os.environ for name in ("TABPFN_PROJECT_ROOT", "VSC_PROJECT_ROOT")
    ):
        staging_root()
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
    cluster's temporary filesystem. Small-file caches can exhaust PROJECT's
    file quota. Off-cluster, the fallback remains the repo-local ``.cache``.
    The container profile requires an explicit job cache beneath VSC_SCRATCH.
    """
    env = os.environ.get("TABPFN_CACHE_ROOT")
    if _container_profile():
        staging_root()
        scratch = os.environ.get("VSC_SCRATCH", "").strip()
        if not scratch or not env:
            raise ValueError("Container profile requires VSC_SCRATCH and an explicit "
                             "TABPFN_CACHE_ROOT beneath it")
        return _validated_write_path(Path(env), Path(scratch), "TABPFN_CACHE_ROOT")
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
    "raw_file_for",
    "processed_write_dir",
    "raw_task_dirs",
    "processed_task_dirs",
    "results_root",
    "logs_root",
    "cache_root",
    "describe",
]
