"""Result storage: one JSON + one .npz per (experiment, task, dataset, method[, sweep_point]).

Layout
------
Base layout::

    results/<experiment>/<task>/<dataset>/<method>.json
    results/<experiment>/<task>/<dataset>/<method>.npz

For experiments that sweep an extra axis (Experiment 1 sweeps HPO mode,
Experiment 2 sweeps ``row_limit``, Experiment 3 sweeps
``minority_proportion``), the sweep point is appended to the method
name with ``__`` so every sweep point still gets its own pair::

    results/experiment1/pd/0001.gmsc/xgboost.json            # NO_HPO
    results/experiment1/pd/0001.gmsc/xgboost__HPO.json       # HPO
    results/experiment2/pd/<dataset>/tabpfn_v3__row20000.json
    results/experiment3/pd/<dataset>/tabicl_v2__min0p1500.json

Use :func:`build_method_name` to construct the right suffix.

Rationale
---------
Each SLURM array task writes ONE pair of files for its slot. No two
array slots ever touch the same file, so file locks are not required
for the result writes -- a major simplification over the old monolithic
per-dataset pickle.

Public API
----------
* :func:`save_method` -- write the JSON + npz for one (dataset, method[, sweep_point]).
* :func:`load_method` -- inverse: returns ``{fold_id: fold_dict}``.
* :func:`has_complete_result` -- skip helper for resumable re-runs.
* :func:`scan_results` -- yield ``(experiment, task, dataset, method, payload)``
  tuples for every result file under a results root.
* :func:`build_method_name` -- ``(method, sweep_axis, sweep_value)`` -> filename stem.
* :func:`parse_method_name` -- inverse: split ``"xgboost__row20000"`` into parts.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


_ARRAY_KEYS = ("y_true", "y_prob", "y_pred", "val_y_true", "val_y_prob")
_SCALAR_KEYS_PER_FOLD = (
    "metrics", "train_time", "predict_time", "threshold",
    "used_hpo", "hpo_n_trials",
    "n_clipped_below", "n_clipped_above",
)


def _result_paths(
    base: Path, experiment: str, task: str, dataset: str, method: str
) -> Tuple[Path, Path]:
    out_dir = base / experiment.lower() / task.lower() / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{method}.json", out_dir / f"{method}.npz"


def save_method(
    method_results: Mapping[int, Mapping[str, Any]],
    *,
    base: Path,
    experiment: str,
    task: str,
    dataset: str,
    method: str,
) -> Tuple[Path, Path]:
    """Persist all folds of one ``(dataset, method)`` in one JSON + one npz.

    ``method_results`` is the dict returned by
    :func:`src.methods.method_runner.run_talent_method`
    (``{fold_id: fold_dict}``). Existing files are overwritten atomically.
    """
    if not method_results:
        raise ValueError(f"save_method called with empty results for {method}")

    json_path, npz_path = _result_paths(base, experiment, task, dataset, method)

    # ------ JSON (scalars only) -----------------------------------------
    payload: Dict[str, Any] = {
        "experiment": experiment.lower(),
        "task": task.lower(),
        "dataset": dataset,
        "method": method,
        "folds": {},
    }
    for fold_id, fold in sorted(method_results.items()):
        payload["folds"][str(fold_id)] = {
            k: _to_jsonable(fold.get(k)) for k in _SCALAR_KEYS_PER_FOLD
        }
    payload["info"] = _to_jsonable(next(iter(method_results.values())).get("info", {}))
    payload["aggregates"] = _aggregate(payload["folds"])
    payload["n_folds"] = len(payload["folds"])

    # Write the npz (arrays) FIRST, then the JSON (scalars) LAST -- each via a
    # temp file + os.replace so a crash, kill, or "Disk quota exceeded" error
    # mid-write can never leave a half-written file. Because has_complete_result()
    # keys on the JSON, writing it last guarantees the invariant: a complete,
    # parseable <method>.json implies its <method>.npz is already fully on disk.
    # So an interrupted save is never mistaken for a finished (skippable) result
    # -- a re-run cleanly redoes that point.
    # Remove any stale temp files left by a previously killed save of this
    # point (e.g. after `scancel`) so interrupted runs don't leak inodes on
    # the file-count-limited project storage.
    out_dir = json_path.parent
    for _stale in list(out_dir.glob(f".{json_path.name}.*.tmp")) + \
                  list(out_dir.glob(f".{npz_path.name}.*.tmp")):
        try:
            _stale.unlink()
        except OSError:
            pass

    arrays: Dict[str, np.ndarray] = {}
    for fold_id, fold in method_results.items():
        for k in _ARRAY_KEYS:
            v = fold.get(k)
            if v is None:
                continue
            arrays[f"fold_{fold_id}_{k}"] = np.asarray(v)
    if arrays:
        tmp_npz = npz_path.parent / f".{npz_path.name}.{os.getpid()}.tmp"
        with open(tmp_npz, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp_npz, npz_path)
    elif npz_path.exists():
        npz_path.unlink()

    tmp_json = json_path.parent / f".{json_path.name}.{os.getpid()}.tmp"
    tmp_json.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_json, json_path)

    return json_path, npz_path


def load_method(
    *,
    base: Path,
    experiment: str,
    task: str,
    dataset: str,
    method: str,
) -> Dict[int, Dict[str, Any]]:
    """Inverse of :func:`save_method` -- returns ``{fold_id: fold_dict}``."""
    json_path, npz_path = _result_paths(base, experiment, task, dataset, method)
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    payload = json.loads(json_path.read_text())
    arrays: Dict[str, np.ndarray] = {}
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}

    out: Dict[int, Dict[str, Any]] = {}
    for fold_id_str, fold in payload["folds"].items():
        fold_id = int(fold_id_str)
        fold_dict = dict(fold)
        fold_dict["fold_id"] = fold_id
        fold_dict["info"] = payload.get("info", {})
        fold_dict["method"] = method
        fold_dict["dataset"] = dataset
        fold_dict["task"] = task
        for k in _ARRAY_KEYS:
            key = f"fold_{fold_id}_{k}"
            if key in arrays:
                fold_dict[k] = arrays[key]
        out[fold_id] = fold_dict
    return out


def scan_results(base: Path) -> Iterator[Tuple[str, str, str, str, Dict[str, Any]]]:
    """Walk every JSON result file and yield ``(experiment, task, dataset, method, payload)``.

    Packed files (Experiment 2/3 -- one ``<method>.json`` holding many sweep
    points under ``payload["points"]``) are expanded: each point is yielded as
    its own logical result, with ``method`` set to the point name and a payload
    carrying that point's ``folds`` / ``aggregates``.
    """
    if not base.exists():
        return
    for json_path in base.rglob("*.json"):
        # Path: results/<experiment>/<task>/<dataset>/<method>.json
        parts = json_path.relative_to(base).parts
        if len(parts) != 4:
            continue  # not our layout (e.g. summaries/)
        experiment, task, dataset, method_file = parts
        method = method_file[:-5]  # strip .json
        try:
            payload = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Skipping malformed JSON: {json_path}")
            continue
        if isinstance(payload, dict) and "points" in payload:
            info = payload.get("info", {})
            for point_name, entry in (payload.get("points") or {}).items():
                point_payload = dict(entry)
                point_payload.setdefault("info", info)
                yield experiment, task, dataset, point_name, point_payload
        else:
            yield experiment, task, dataset, method, payload


# ============================================================================
#  Helpers
# ============================================================================

def _aggregate(folds: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Mean / std for every numeric metric across folds."""
    metric_names: set = set()
    for fold in folds.values():
        m = fold.get("metrics") or {}
        metric_names.update(m.keys())
    out: Dict[str, Dict[str, float]] = {}
    for name in sorted(metric_names):
        values = []
        for fold in folds.values():
            v = (fold.get("metrics") or {}).get(name)
            if v is None:
                continue
            try:
                fv = float(v)
                if fv != fv:  # NaN
                    continue
                values.append(fv)
            except (TypeError, ValueError):
                continue
        if values:
            out[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
    return out


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def has_complete_result(
    *,
    base: Path,
    experiment: str,
    task: str,
    dataset: str,
    method: str,
    expected_folds: int,
) -> bool:
    """Return True iff a ``<method>.json`` with ``expected_folds`` folds exists.

    Used by every experiment driver to **skip** (dataset, method) cells that
    already have complete results -- so re-running an experiment over a
    partial sweep is safe.
    """
    json_path, _ = _result_paths(base, experiment, task, dataset, method)
    if not json_path.exists():
        return False
    try:
        payload = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return False
    folds = payload.get("folds") or {}
    return len(folds) >= int(expected_folds)


# ============================================================================
#  Packed results (Experiment 2 / 3) -- many sweep points in ONE file
# ============================================================================
#
# Experiments 2 & 3 sweep over many points per (dataset, method) cell. Writing
# one ``<method>__<suffix>.{json,npz}`` per point creates tens of thousands of
# tiny files and exhausts the project-storage inode quota (~150k inodes/TB).
# Instead we PACK every point of a cell into a single ``<method>.json``:
#
#     {"experiment","task","dataset","method","packed":true,
#      "points": {"<method>__rowN": {"folds":{...},"aggregates":{...},"n_folds":k}, ...},
#      "info": {...}}
#
# No npz is written (predictions are dropped for these experiments). The SLURM
# generator keeps all of a cell's points in ONE array task, so the
# read-modify-write below is a safe single writer -- no file lock needed.

def _scalars_payload(method_results: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    """Build the metrics-only ``{folds, aggregates, n_folds}`` block for a result."""
    folds = {
        str(fid): {k: _to_jsonable(fold.get(k)) for k in _SCALAR_KEYS_PER_FOLD}
        for fid, fold in sorted(method_results.items())
    }
    return {"folds": folds, "aggregates": _aggregate(folds), "n_folds": len(folds)}


def save_packed_point(
    method_results: Mapping[int, Mapping[str, Any]],
    *,
    base: Path,
    experiment: str,
    task: str,
    dataset: str,
    method_base: str,
    point_name: str,
) -> Path:
    """Add/overwrite ONE sweep point inside the cell's packed ``<method_base>.json``.

    Metrics only (no npz). Safe because a cell's points are all run by a single
    array task (see the SLURM generator's cell-grouped packing), so this
    read-modify-write never races another writer. Written atomically.
    """
    if not method_results:
        raise ValueError(f"save_packed_point called with empty results for {point_name}")
    out_dir = base / experiment.lower() / task.lower() / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{method_base}.json"

    payload: Optional[Dict[str, Any]] = None
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            payload = None  # truncated by an earlier kill -> start fresh
    if not isinstance(payload, dict) or "points" not in payload:
        payload = {
            "experiment": experiment.lower(), "task": task.lower(),
            "dataset": dataset, "method": method_base, "packed": True, "points": {},
        }

    payload["points"][point_name] = _scalars_payload(method_results)
    payload["n_points"] = len(payload["points"])
    if "info" not in payload:
        payload["info"] = _to_jsonable(next(iter(method_results.values())).get("info", {}))

    # stale-temp cleanup + atomic write (see save_method)
    for _stale in out_dir.glob(f".{json_path.name}.*.tmp"):
        try:
            _stale.unlink()
        except OSError:
            pass
    tmp = out_dir / f".{json_path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, json_path)
    return json_path


def has_complete_packed_point(
    *,
    base: Path,
    experiment: str,
    task: str,
    dataset: str,
    method_base: str,
    point_name: str,
    expected_folds: int,
) -> bool:
    """True iff the packed ``<method_base>.json`` holds ``point_name`` with all folds."""
    json_path, _ = _result_paths(base, experiment, task, dataset, method_base)
    if not json_path.exists():
        return False
    try:
        payload = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return False
    point = (payload.get("points") or {}).get(point_name)
    if not point:
        return False
    return len(point.get("folds") or {}) >= int(expected_folds)


# ============================================================================
#  Sweep-suffix helpers (Experiment 1 HPO mode / Exp 2 row_limit / Exp 3 minority_proportion)
# ============================================================================
#
# We keep the "one file per (dataset, method) cell" invariant by encoding any
# sweep axis as a suffix on the method name. Two suffixes can be combined
# (e.g. `xgboost__HPO__row20000`) when an experiment varies BOTH HPO and a
# sweep axis, but typically only one applies at a time.

_SWEEP_SEPARATOR = "__"


def _format_sweep_value(axis: str, value: Any) -> str:
    """Encode a sweep-point value into a filename-safe suffix.

    Rules:
      * Integers and bools render as themselves (`row20000`, `HPO`).
      * Floats render with the decimal replaced by ``p`` and four significant
        digits (`min0p1500`, `min0p0025`) so ordering is intuitive.
      * Strings render as-is.
    """
    if isinstance(value, bool):
        return f"{axis}{int(value)}"
    if isinstance(value, int):
        return f"{axis}{value}"
    if isinstance(value, float):
        # 4 sig figs, decimal -> 'p' (filesystem-safe)
        return f"{axis}{value:.4f}".replace(".", "p").rstrip("0").rstrip("p") + (
            "" if value != int(value) else "p0"
        )
    return f"{axis}{value}"


def build_method_name(method: str, sweep: Optional[Mapping[str, Any]] = None) -> str:
    """Compose a filename stem from ``method`` and an optional sweep dict.

    Example::

        >>> build_method_name("xgboost", {"row": 20000})
        'xgboost__row20000'
        >>> build_method_name("tabicl_v2", {"min": 0.0025})
        'tabicl_v2__min0p0025'
        >>> build_method_name("xgboost", {"HPO": True})
        'xgboost__HPO1'
    """
    if not sweep:
        return method
    suffixes = []
    for axis in sorted(sweep):
        value = sweep[axis]
        if axis.upper() == "HPO" and value is True:
            suffixes.append("HPO")
        elif axis.upper() == "HPO" and value is False:
            continue  # NO_HPO is the default; no suffix
        else:
            suffixes.append(_format_sweep_value(axis, value))
    if not suffixes:
        return method
    return method + _SWEEP_SEPARATOR + _SWEEP_SEPARATOR.join(suffixes)


def parse_method_name(filename_stem: str) -> Dict[str, Any]:
    """Inverse of :func:`build_method_name`.

    Returns ``{"method": <base>, "sweep": {axis: value, ...}}``. Unknown
    suffixes are returned verbatim as strings (the summariser keeps them).
    """
    parts = filename_stem.split(_SWEEP_SEPARATOR)
    base = parts[0]
    sweep: Dict[str, Any] = {}
    for piece in parts[1:]:
        if piece == "HPO":
            sweep["HPO"] = True
            continue
        # axisNNN or axisNpMMM -- split on first digit
        for i, ch in enumerate(piece):
            if ch.isdigit():
                axis = piece[:i]
                rest = piece[i:]
                try:
                    if "p" in rest:
                        sweep[axis] = float(rest.replace("p", "."))
                    else:
                        sweep[axis] = int(rest)
                except ValueError:
                    sweep[axis] = rest
                break
        else:
            sweep[piece] = True
    return {"method": base, "sweep": sweep}


__all__ = [
    "save_method", "load_method", "scan_results", "has_complete_result",
    "save_packed_point", "has_complete_packed_point",
    "build_method_name", "parse_method_name",
]
