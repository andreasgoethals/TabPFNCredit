"""Delete ONE (experiment, task, dataset, method) result cell.

Use this to drop a single bugged result so the next ``tabpfncredit resubmit``
recomputes only that cell -- e.g. a broken ``tabicl_v2`` run on the
``vehicle_loan`` dataset in Experiment 2. It removes that cell's
``<method>.json`` / ``.npz`` AND every per-task packed shard
(``<method>__shard_*.json``) and HPO variant (``<method>__HPO.*``), matching
the method name EXACTLY so removing ``tabicl_v2`` never touches ``tabicl`` or
``tabicl_v2_5``. Other methods / datasets / experiments are untouched.

Usage (repo root, venv active -- works locally and on the VSC):

    # see what would go, then do it:
    python scripts/remove_result.py --experiment experiment2 --task pd \\
        --dataset 0003.vehicle_loan --method tabicl_v2 --dry-run
    python scripts/remove_result.py --experiment experiment2 --task pd \\
        --dataset 0003.vehicle_loan --method tabicl_v2

    # then recompute only the missing cell across all clusters:
    TABPFN_ALL_CLUSTERS=1 tabpfncredit resubmit Experiment2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import results_root as _default_results_root  # noqa: E402


def _targets(cell_dir: Path, method: str):
    """Every file in ``cell_dir`` belonging to ``method`` (exact stem + __ variants)."""
    hits = []
    if not cell_dir.is_dir():
        return hits
    for path in sorted(cell_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".json", ".npz"):
            continue
        stem = path.stem  # filename without extension
        if stem == method or stem.startswith(f"{method}__"):
            hits.append(path)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", required=True, help="e.g. experiment2")
    ap.add_argument("--task", required=True, help="pd or lgd")
    ap.add_argument("--dataset", required=True, help="dataset dir name, e.g. 0003.vehicle_loan")
    ap.add_argument("--method", required=True, help="method name, e.g. tabicl_v2")
    ap.add_argument("--results-root", type=Path, default=None,
                    help="Results root (default: auto-resolved; project storage on the VSC).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be removed without deleting anything.")
    args = ap.parse_args()

    root = Path(args.results_root or _default_results_root())
    cell_dir = root / args.experiment.lower() / args.task.lower() / args.dataset
    hits = _targets(cell_dir, args.method)

    print(f"Results root : {root}")
    print(f"Cell         : {args.experiment.lower()}/{args.task.lower()}/{args.dataset}/{args.method}")
    print(f"Mode         : {'DRY RUN' if args.dry_run else 'DELETE'}")
    if not cell_dir.is_dir():
        print(f"  (cell directory does not exist: {cell_dir})")
        return
    print(f"Matched {len(hits)} file(s):")
    for h in hits:
        print(f"   {h.relative_to(root)}")
        if not args.dry_run:
            h.unlink(missing_ok=True)
    if hits and not args.dry_run:
        print(f"\nDeleted {len(hits)} file(s). Re-run "
              f"`tabpfncredit resubmit {args.experiment.capitalize()}` to recompute this cell.")
    elif not hits:
        print("   (nothing matched -- check the dataset dir name and method spelling)")


if __name__ == "__main__":
    main()
