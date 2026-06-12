"""Backfill the Average-Precision metric family into OLD result files.

Newer runs store ``AP``, ``AP_baseline``, ``AP_minus_baseline`` and
``AP_normalized`` for every PD fold; results produced before that change lack
them. This script recomputes the four metrics from the predictions that every
Experiment 0/1 result already carries in its ``.npz`` (``fold_<i>_y_true`` +
``fold_<i>_y_prob``) and writes them into the JSON's fold metrics, then
refreshes the per-file aggregates. Writes are atomic (temp file + replace).

Packed results (Experiment 2/3) store metrics only -- no predictions -- so a
missing AP there cannot be recomputed; those files are counted and reported
(they regenerate with the metrics on a re-run, which Exp 2/3 already had).

Usage (repo root, venv active -- works on the VSC against project storage and
locally against a downloaded copy)::

    python -m src.utils.backfill_ap_metrics --dry-run
    python -m src.utils.backfill_ap_metrics
    python -m src.utils.backfill_ap_metrics --results-root /staging/leuven/stg_00211/results

Re-run ``tabpfncredit summarize --experiment <name>`` afterwards so the CSVs
pick up the new columns.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.methods.method_metrics import average_precision_deviation
from src.utils.paths import results_root as _default_results_root
from src.utils.result_io import _aggregate  # canonical aggregates builder

_AP_KEYS = ("AP", "AP_baseline", "AP_minus_baseline", "AP_normalized")


def _pos_proba(y_prob: np.ndarray) -> np.ndarray | None:
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        return y_prob[:, 1]
    if y_prob.ndim == 1:
        return y_prob
    return None


def backfill(results_root: Path, dry_run: bool = False) -> Dict[str, List[str]]:
    report = {"updated": [], "already_ok": [], "no_predictions": [], "errors": []}

    for json_path in sorted(results_root.rglob("*.json")):
        parts = json_path.relative_to(results_root).parts
        if len(parts) != 4 or parts[1].lower() != "pd":
            continue  # AP is a classification metric; PD results only
        label = "/".join(parts)[:-5]
        try:
            payload = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            report["errors"].append(f"{label} (unreadable JSON)")
            continue

        if "points" in payload:  # packed (Exp 2/3): no stored predictions
            missing = any(
                "AP_normalized" not in (f.get("metrics") or {})
                for pt in (payload.get("points") or {}).values()
                for f in (pt.get("folds") or {}).values()
            )
            if missing:
                report["no_predictions"].append(label)
            else:
                report["already_ok"].append(label)
            continue

        folds = payload.get("folds") or {}
        if all("AP_normalized" in (f.get("metrics") or {}) for f in folds.values()):
            report["already_ok"].append(label)
            continue

        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            report["no_predictions"].append(label)
            continue
        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                arrays = {k: npz[k] for k in npz.files}
        except Exception as exc:
            report["errors"].append(f"{label} (npz: {exc})")
            continue

        changed = False
        for fid, fold in folds.items():
            metrics = fold.setdefault("metrics", {})
            if "AP_normalized" in metrics:
                continue
            yt = arrays.get(f"fold_{fid}_y_true")
            yp = arrays.get(f"fold_{fid}_y_prob")
            pos = _pos_proba(yp) if yp is not None else None
            if yt is None or pos is None:
                continue
            ap = average_precision_deviation(np.asarray(yt).ravel(), pos)
            metrics.update({k: float(ap[k]) for k in _AP_KEYS if k in ap})
            changed = True

        if not changed:
            report["no_predictions"].append(label)
            continue

        payload["aggregates"] = _aggregate(folds)
        report["updated"].append(label)
        if not dry_run:
            tmp = json_path.parent / f".{json_path.name}.{os.getpid()}.tmp"
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, json_path)

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = args.results_root or _default_results_root()
    print(f"Results root: {root}\nMode        : "
          f"{'DRY RUN' if args.dry_run else 'APPLY'}\n")
    rep = backfill(root, dry_run=args.dry_run)
    verb = "would be" if args.dry_run else "were"
    print(f"-- {len(rep['updated'])} file(s) {verb} UPDATED with AP metrics")
    for x in rep["updated"][:40]:
        print(f"   {x}")
    if len(rep["updated"]) > 40:
        print(f"   ... and {len(rep['updated']) - 40} more")
    print(f"-- {len(rep['already_ok'])} already had AP metrics (untouched)")
    print(f"-- {len(rep['no_predictions'])} without stored predictions "
          f"(packed Exp 2/3 or missing npz) -- cannot recompute:")
    for x in rep["no_predictions"][:15]:
        print(f"   {x}")
    if rep["errors"]:
        print(f"-- {len(rep['errors'])} ERRORS: {rep['errors'][:10]}")
    print("\nNow refresh the CSVs:  tabpfncredit summarize --experiment <name> [...]")


if __name__ == "__main__":
    main()
