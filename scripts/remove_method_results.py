"""Delete every result file for one method across the results tree.

Use this to drop a method you no longer want in the benchmark (e.g. the
``dummy`` baseline) without re-running anything. It removes the method's
``<method>.json`` / ``.npz`` and its ``<method>__HPO.*`` copies, in every
experiment / task / dataset, matching exactly (so removing ``tabpfn`` never
touches ``tabpfn_v2``).

Usage (repo root, venv active -- works locally and on the VSC):

    python scripts/remove_method_results.py --method dummy --dry-run
    python scripts/remove_method_results.py --method dummy
    python scripts/remove_method_results.py --method dummy \\
        --results-root /staging/leuven/stg_00211/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import results_root as _default_results_root  # noqa: E402


def _targets(root: Path, method: str):
    """Every file belonging to ``method`` (exact base name + __HPO variant)."""
    hits = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".json", ".npz"):
            continue
        stem = path.stem  # filename without extension
        if stem == method or stem == f"{method}__HPO":
            hits.append(path)
    return sorted(hits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", default="dummy",
                    help="Method whose results to remove (default: dummy).")
    ap.add_argument("--results-root", type=Path, default=None,
                    help="Results root (default: auto-resolved; project storage on the VSC).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be removed without deleting anything.")
    args = ap.parse_args()

    root = args.results_root or _default_results_root()
    hits = _targets(Path(root), args.method)
    print(f"Results root : {root}")
    print(f"Method       : {args.method!r}")
    print(f"Mode         : {'DRY RUN' if args.dry_run else 'DELETE'}")
    print(f"Matched {len(hits)} file(s):")
    for h in hits:
        print(f"   {h.relative_to(root)}")
        if not args.dry_run:
            h.unlink(missing_ok=True)
    if hits and not args.dry_run:
        print(f"\nDeleted {len(hits)} file(s). Re-run "
              f"`tabpfncredit summarize --experiment <name>` to refresh the CSVs.")
    elif not hits:
        print("   (nothing to remove)")


if __name__ == "__main__":
    main()
