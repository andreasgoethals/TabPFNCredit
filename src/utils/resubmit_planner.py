"""Build work items for Slurm plans and check existing results before resuming."""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.paths import experiment_results_dir
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.methods.runtime_profile import estimate_point_seconds
from src.utils.config_reader import load_config
from src.utils.paths import results_root as _default_results_root
from src.utils.result_io import complete_packed_points, has_complete_result

# Experiments whose sweep points are PACKED into one <method>.json per cell.
_PACKED_EXPERIMENTS = ("experiment2", "experiment3")


def find_missing_work_items(
    experiment: str,
    results_root: Optional[Path | str] = None,
    *, cells: Optional[Sequence[dict]] = None, config: Optional[dict] = None,
    skip_completed: bool = True,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Return ``(work_items, summary)`` for every sweep point not yet complete.

    ``work_items`` is ready for the SLURM generator; ``summary`` carries the
    counts (``expected`` / ``done`` / ``missing``) plus a per-method breakdown
    of what is missing.
    """
    # Imported lazily: the CLI module pulls in Typer + the full method registry.
    from src.utils.cli import _build_task_list, _sweep_points

    if config is None:
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
    # cell -- this froze `resubmit` on Experiment 2). Instead, parse each cell's
    # packed file(s) ONCE into the set of complete point names. The cell may be
    # spread over MANY per-task shard files (intra-cell parallelism), so this
    # unions across ``<method>.json`` + every ``<method>__shard_*.json``.
    def _complete_points_of_cell(task: str, dataset: str, method: str) -> set:
        return complete_packed_points(
            base=root, experiment=exp, task=task, dataset=dataset,
            method_base=method, expected_folds=cv,
        )

    revision = os.environ.get("TABPFN_REVISION") == "1"
    checked = set()
    for cell in (_build_task_list(config) if cells is None else cells):
        key = (cell["task"], cell["dataset"])
        if skip_completed and revision and key not in checked:
            from src.utils.revision_results import verify_revision_inputs
            if (experiment_results_dir(root, exp) / key[0] / key[1]).is_dir():
                verify_revision_inputs(root, exp, *key, config, record=False)
            checked.add(key)
        if skip_completed and packed:
            cell_done = _complete_points_of_cell(
                cell["task"], cell["dataset"], cell["method"]
            )
        for point in _sweep_points(exp, config, cell):
            name = point["name"]
            if not skip_completed:
                complete = False
            elif packed:
                complete = name in cell_done
            else:
                complete = has_complete_result(
                    base=root, experiment=exp, task=cell["task"],
                    dataset=cell["dataset"], method=name, expected_folds=cv, require_predictions=revision,
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
