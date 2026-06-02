"""Migrate legacy ``.pkl`` results into the new per-(dataset, method) JSON+npz layout.

Old layout::

    results/<experiment>/<task>/<dataset>.pkl
       =  {HPO_mode: {method: {fold_id: {y_true, y_prob, y_pred, metrics, ...}}}}

New layout::

    results/<experiment>/<task>/<dataset>/<method>.json   (scalar metrics, times, info)
    results/<experiment>/<task>/<dataset>/<method>.npz    (per-fold arrays)

Usage::

    # Migrate Experiment 1 (the only one with results we want to keep)
    python scripts/migrate_pkl_to_json.py --experiment experiment1

    # Dry-run -- just list what would be written
    python scripts/migrate_pkl_to_json.py --experiment experiment1 --dry-run

    # Migrate everything and remove the old pickles
    python scripts/migrate_pkl_to_json.py --experiment experiment1 --delete-old

NO_HPO and HPO results are written into separate "method" filenames so
both modes survive the migration without colliding:

    <method>.json         <- NO_HPO results (default mode)
    <method>__HPO.json    <- HPO results

If you only ever have one mode for a given method this still works:
there's just one file. The downstream readers in
``src/utils/result_io.py`` and the polars summariser handle both names.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.result_io import save_method  # noqa: E402

logger = logging.getLogger(__name__)


def _suffix(hpo_mode: str) -> str:
    """Decorate the method filename with the HPO mode for non-NO_HPO results."""
    return "" if hpo_mode.upper() == "NO_HPO" else f"__{hpo_mode.upper()}"


def _stamp_method_meta(
    fold_results: Mapping[int, Mapping[str, Any]],
    *,
    method: str,
    dataset: str,
    task: str,
) -> Dict[int, Dict[str, Any]]:
    """Ensure each fold carries the metadata that ``save_method`` expects."""
    out: Dict[int, Dict[str, Any]] = {}
    for fold_id, fold in fold_results.items():
        f = dict(fold)
        f.setdefault("method", method)
        f.setdefault("dataset", dataset)
        f.setdefault("task", task)
        f.setdefault("fold_id", fold_id)
        out[fold_id] = f
    return out


def migrate_one_pkl(
    pkl_path: Path,
    *,
    base: Path,
    experiment: str,
    task: str,
    dataset: str,
    dry_run: bool,
) -> int:
    """Convert one ``<dataset>.pkl`` into the new per-method layout.

    Returns the number of (method, hpo_mode) tuples written.
    """
    with open(pkl_path, "rb") as fp:
        data = pickle.load(fp)

    if not isinstance(data, dict):
        logger.warning("Unexpected pickle structure at %s -- skipping", pkl_path)
        return 0

    n_written = 0

    # Old format: {HPO_mode: {method: {fold_id: fold_dict}}}
    # Newer "flat" pickles (Experiment 0 saved a single mode at the top
    # level via results[method] = method_results) look like:
    #     {method: {fold_id: fold_dict}}
    sample_value = next(iter(data.values()))
    is_hpo_keyed = (
        isinstance(sample_value, dict)
        and next(iter(sample_value.values()), {})
        and isinstance(next(iter(sample_value.values())), dict)
        and "fold_id" not in next(iter(sample_value.values()))
        # heuristic: if the inner-inner values are also dicts, treat as HPO-keyed
    )

    iteration = (
        data.items() if is_hpo_keyed else (("NO_HPO", data),)
    )

    for hpo_mode, methods_dict in iteration:
        if not isinstance(methods_dict, dict):
            continue
        for method, fold_results in methods_dict.items():
            if not isinstance(fold_results, dict):
                continue
            target_method_name = method + _suffix(hpo_mode)
            stamped = _stamp_method_meta(
                fold_results, method=target_method_name, dataset=dataset, task=task
            )
            target_json = base / experiment / task / dataset / f"{target_method_name}.json"
            if dry_run:
                logger.info("[DRY] would write %s (%d folds)", target_json, len(stamped))
            else:
                save_method(
                    stamped,
                    base=base,
                    experiment=experiment,
                    task=task,
                    dataset=dataset,
                    method=target_method_name,
                )
                logger.info("Wrote %s (%d folds)", target_json, len(stamped))
            n_written += 1

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True, help="e.g. experiment1")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_PROJECT_ROOT / "results",
        help="Path to the results directory (default: ./results).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write anything; just print what would be migrated.",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="After successful migration, delete the old <dataset>.pkl files.",
    )
    parser.add_argument(
        "--task",
        choices=("pd", "lgd", "both"),
        default="both",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    exp_dir = args.results_root / args.experiment
    if not exp_dir.exists():
        parser.error(f"Experiment directory not found: {exp_dir}")

    tasks = ("pd", "lgd") if args.task == "both" else (args.task,)
    total_pkls = 0
    total_methods = 0
    for task in tasks:
        task_dir = exp_dir / task
        if not task_dir.exists():
            logger.info("Skipping %s (no %s task directory)", task, task_dir)
            continue
        for pkl_path in sorted(task_dir.glob("*.pkl")):
            dataset = pkl_path.stem
            logger.info("--- %s / %s / %s ---", args.experiment, task, dataset)
            n = migrate_one_pkl(
                pkl_path,
                base=args.results_root,
                experiment=args.experiment,
                task=task,
                dataset=dataset,
                dry_run=args.dry_run,
            )
            total_pkls += 1
            total_methods += n
            if args.delete_old and not args.dry_run and n > 0:
                pkl_path.unlink()
                logger.info("Deleted old pickle: %s", pkl_path)

    logger.info(
        "Migration done: %d pickles processed, %d (method, mode) files written.",
        total_pkls, total_methods,
    )


if __name__ == "__main__":
    main()
