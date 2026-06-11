"""Gap scan for resumable submissions: find every sweep point NOT yet done.

Backs the ``tabpfncredit resubmit`` CLI command. Given an experiment, this
module compares what the experiment's YAML config says SHOULD exist against
the result files actually on disk (local downloaded copy or the cluster's
project storage -- whatever :func:`src.utils.paths.results_root` resolves to)
and returns work items for ONLY the missing points.

Why not just re-run ``tabpfncredit experiment``? That re-shards *all* points
(done + missing) across array slots; slots whose points are all done still
get submitted, queue, start, and exit -- wasted queue positions and scheduler
churn. The resubmit path packs only the missing points into the smallest
possible number of dense array slots.

The returned work items have exactly the shape
:func:`src.utils.slurm_generator.generate_scripts_for_experiment` expects
(including ``copy_from`` for non-tunable HPO copy points, which cost seconds,
not GPU hours).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.methods.runtime_profile import estimate_point_seconds
from src.utils.config_reader import load_config
from src.utils.paths import results_root as _default_results_root
from src.utils.result_io import has_complete_packed_point, has_complete_result

# Experiments whose sweep points are PACKED into one <method>.json per cell.
_PACKED_EXPERIMENTS = ("experiment2", "experiment3")


def find_missing_work_items(
    experiment: str,
    results_root: Optional[Path | str] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Return ``(work_items, summary)`` for every sweep point not yet complete.

    ``work_items`` is ready for the SLURM generator; ``summary`` carries the
    counts (``expected`` / ``done`` / ``missing``) plus a per-method breakdown
    of what is missing.
    """
    # Imported lazily: src.cli pulls in Typer + the full method registry.
    from src.cli import _build_task_list, _sweep_points

    config = load_config(experiment)
    root = Path(results_root) if results_root else _default_results_root()
    exp = experiment.lower()
    cv = int(config["split"]["cv_splits"])
    n_trials = (config.get("tuning") or {}).get("n_trials", 1)
    packed = exp in _PACKED_EXPERIMENTS

    missing: List[dict] = []
    by_method: Dict[str, int] = {}
    n_done = 0

    # For PACKED experiments the per-point checker would re-read and re-parse
    # the cell's whole packed JSON for EVERY point (thousands of parses per
    # cell -- this froze `resubmit` on Experiment 2). Instead, parse each
    # cell's packed file ONCE into the set of complete point names.
    def _complete_points_of_cell(task: str, dataset: str, method: str) -> set:
        import json
        path = root / exp / task.lower() / dataset / f"{method}.json"
        if not path.exists():
            return set()
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return set()
        return {
            pname for pname, entry in (payload.get("points") or {}).items()
            if len(entry.get("folds") or {}) >= cv
        }

    for cell in _build_task_list(config):
        if packed:
            cell_done = _complete_points_of_cell(
                cell["task"], cell["dataset"], cell["method"]
            )
        for point in _sweep_points(exp, config, cell):
            name = point["name"]
            if packed:
                complete = name in cell_done
            else:
                complete = has_complete_result(
                    base=root, experiment=exp, task=cell["task"],
                    dataset=cell["dataset"], method=name, expected_folds=cv,
                )
            if complete:
                n_done += 1
                continue

            copy_from = point.get("copy_from")
            missing.append({
                "dataset": cell["dataset"],
                "method": cell["method"],
                "task": cell["task"],
                "name": name,
                "tune": point.get("tune", False),
                "row_limit": point.get("row_limit"),
                "sampling": point.get("sampling"),
                "copy_from": copy_from,
                "est_seconds": 5 if copy_from else estimate_point_seconds(
                    cell["method"], n_folds=cv,
                    row_limit=point.get("row_limit"),
                    tune=point.get("tune", False), n_trials=n_trials,
                ),
            })
            by_method[cell["method"]] = by_method.get(cell["method"], 0) + 1

    summary = {
        "experiment": experiment,
        "results_root": str(root),
        "expected": n_done + len(missing),
        "done": n_done,
        "missing": len(missing),
        "missing_by_method": dict(sorted(by_method.items(), key=lambda kv: -kv[1])),
    }
    return missing, summary


__all__ = ["find_missing_work_items"]
