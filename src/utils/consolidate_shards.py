"""Merge Experiment 2/3 per-array-task shard files back into one file per cell.

What the shards are
-------------------
Experiment 2/3 parallelise a single ``(dataset, method)`` cell's sweep points
across MANY SLURM array tasks (so e.g. HackerEarth's class-imbalance sweep is
not run serially). Each task writes its OWN
``<method>__shard_<jobid>_<task>.json`` -- it is the sole writer of that file,
so no locks are needed. The summariser and the resubmit gap-scan read the
**union** across a cell's shards, so the results are already complete and
correct. The only downside is cosmetic: the shards accumulate as many small
files (one per task per resubmission).

This tool consolidates them: for every cell it unions all shard points into the
canonical ``<method>.json`` (last writer wins on a duplicated point) and deletes
the shard files, leaving one tidy file per cell. Safe to run repeatedly; a later
resubmit simply creates fresh shards you can consolidate again.

    python -m src.utils.consolidate_shards --dry-run
    python -m src.utils.consolidate_shards
    python -m src.utils.consolidate_shards --experiment experiment3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import results_root as _default_results_root  # noqa: E402


def consolidate_shards(
    results_root: Optional[Path | str] = None,
    experiments: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Merge ``<method>__shard_*.json`` into ``<method>.json`` per cell, deleting
    the shards. Returns ``{cells, shards_merged, shards_deleted}``."""
    root = Path(results_root or _default_results_root())
    exps = {e.lower() for e in experiments} if experiments else None
    report = {"cells": 0, "shards_merged": 0, "shards_deleted": 0}
    if not root.exists():
        return report

    cells: Dict[tuple, List[Path]] = {}
    for sp in root.rglob("*__shard_*.json"):
        parts = sp.relative_to(root).parts
        if len(parts) != 4:
            continue
        if exps and parts[0].lower() not in exps:
            continue
        base = parts[3][:-5].split("__shard_")[0]
        cells.setdefault((sp.parent, base, parts[0], parts[1], parts[2]), []).append(sp)

    for (cell_dir, base, exp, task, dataset), shards in sorted(cells.items(), key=lambda kv: str(kv[0])):
        canon = cell_dir / f"{base}.json"
        merged: Optional[dict] = None
        if canon.exists():
            try:
                merged = json.loads(canon.read_text())
            except (OSError, json.JSONDecodeError):
                merged = None
        if not isinstance(merged, dict) or "points" not in merged:
            merged = {"experiment": exp, "task": task, "dataset": dataset,
                      "method": base, "packed": True, "points": {}}
        for sp in sorted(shards):
            try:
                payload = json.loads(sp.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            merged["points"].update(payload.get("points") or {})
            if "info" not in merged and isinstance(payload.get("info"), dict):
                merged["info"] = payload["info"]
            report["shards_merged"] += 1
        merged["n_points"] = len(merged["points"])
        report["cells"] += 1
        if dry_run:
            continue
        tmp = cell_dir / f".{base}.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(merged, indent=2))
        os.replace(tmp, canon)
        for sp in shards:
            sp.unlink(missing_ok=True)
            report["shards_deleted"] += 1
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", nargs="*", help="limit to these experiments (default: all)")
    ap.add_argument("--results-root", type=Path, default=None,
                    help="results root (default: auto-resolved; project storage on the VSC)")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    args = ap.parse_args(argv)
    root = Path(args.results_root or _default_results_root())
    print(f"Results root : {root}")
    print(f"Mode         : {'DRY RUN' if args.dry_run else 'APPLY'}")
    rep = consolidate_shards(results_root=root, experiments=args.experiment, dry_run=args.dry_run)
    verb = "would merge" if args.dry_run else "merged"
    print(f"\n{verb} {rep['shards_merged']} shard file(s) into {rep['cells']} cell file(s); "
          f"{'would delete' if args.dry_run else 'deleted'} "
          f"{rep['shards_merged'] if args.dry_run else rep['shards_deleted']} shard(s).")
    if rep["cells"] == 0:
        print("   (no shard files found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
