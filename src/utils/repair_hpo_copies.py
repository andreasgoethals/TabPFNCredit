"""One-shot repair: make ``<method>__HPO`` identical to ``<method>`` for
non-tunable methods.

Why this exists
---------------
For methods that cannot be tuned (``supports_hpo=False`` in TALENT's method
registry -- the tabular foundation models plus LinearRegression), the HPO
result is BY DEFINITION the NO_HPO result: Experiment 1's ``__HPO`` point for
them is supposed to be a pure file copy. Two historical bugs broke that
invariant on disk:

1. Old ``__HPO`` files predate the copy mechanism (they were genuine re-runs
   from earlier regimes) and were never overwritten because skip-if-done saw
   them as complete.
2. The SLURM work-item builder dropped ``copy_from``, so on the cluster the
   ``__HPO`` points re-RAN the model instead of copying -- GPU nondeterminism
   then made the pair differ slightly.

This script walks the results tree and, for every ``<m>__HPO.json`` whose base
method is non-tunable:

* if the NO_HPO sibling exists and the two differ -> re-copies NO_HPO over
  ``__HPO`` (JSON + npz, exact);
* if the NO_HPO sibling is missing -> deletes the orphan ``__HPO`` (it cannot
  be validated; the copy point regenerates it on the next submit, after the
  NO_HPO source has run);
* if they already match -> leaves it alone.

Which methods count as "non-tunable" comes from ONE place: TALENT's method
registry (``MethodSpec.supports_hpo``), surfaced in the wrapper as
``NO_HPO_METHODS`` / ``HPO_METHODS`` in :mod:`src.methods.method_config`.

Usage (from the repo root, venv active; works locally and on the VSC)::

    python -m src.utils.repair_hpo_copies                 # apply
    python -m src.utils.repair_hpo_copies --dry-run       # report only
    python -m src.utils.repair_hpo_copies --results-root /staging/leuven/stg_00211/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from src.methods.method_config import NO_HPO_METHODS
from src.utils.paths import results_root as _default_results_root
from src.utils.result_io import load_method, save_method

_HPO_SUFFIX = "__HPO"


def _aggregates(path: Path) -> dict | None:
    """Parse a result JSON and return its ``aggregates`` block (None if unreadable)."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if "points" in payload:  # packed file (Exp 2/3) -- no HPO axis there
        return None
    return payload.get("aggregates") or {}


def repair(results_root: Path, dry_run: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """Scan + repair. Returns (recopied, deleted_orphans, already_consistent)."""
    recopied: List[str] = []
    deleted: List[str] = []
    consistent: List[str] = []

    for hpo_json in sorted(results_root.rglob(f"*{_HPO_SUFFIX}.json")):
        parts = hpo_json.relative_to(results_root).parts
        if len(parts) != 4:
            continue  # not the <exp>/<task>/<dataset>/<file>.json layout
        experiment, task, dataset, fname = parts
        name = fname[:-5]
        base = name[: -len(_HPO_SUFFIX)]
        if base not in NO_HPO_METHODS:
            continue  # tunable method -- its __HPO is a genuine tuned run

        label = f"{experiment}/{task}/{dataset}/{name}"
        nohpo_json = hpo_json.parent / f"{base}.json"

        if not nohpo_json.exists():
            # Orphan: no NO_HPO source to validate against. Delete so the copy
            # point regenerates it after the NO_HPO run lands.
            deleted.append(label)
            if not dry_run:
                hpo_json.unlink(missing_ok=True)
                (hpo_json.parent / f"{name}.npz").unlink(missing_ok=True)
            continue

        if _aggregates(hpo_json) == _aggregates(nohpo_json):
            consistent.append(label)
            continue

        recopied.append(label)
        if not dry_run:
            src = load_method(
                base=results_root, experiment=experiment,
                task=task, dataset=dataset, method=base,
            )
            save_method(
                src, base=results_root, experiment=experiment,
                task=task, dataset=dataset, method=name,
            )

    return recopied, deleted, consistent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", type=Path, default=None,
                    help="Results root (default: auto-resolved, see src.utils.paths).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without touching any file.")
    args = ap.parse_args()

    root = args.results_root or _default_results_root()
    print(f"Results root : {root}")
    print(f"Non-tunable  : {sorted(NO_HPO_METHODS)}")
    print(f"Mode         : {'DRY RUN (no changes)' if args.dry_run else 'APPLY'}\n")

    recopied, deleted, consistent = repair(root, dry_run=args.dry_run)

    verb = "would be" if args.dry_run else "were"
    print(f"-- {len(recopied)} __HPO file(s) {verb} RE-COPIED from NO_HPO (differed):")
    for x in recopied:
        print(f"   {x}")
    print(f"-- {len(deleted)} orphan __HPO file(s) {verb} DELETED (no NO_HPO source):")
    for x in deleted:
        print(f"   {x}")
    print(f"-- {len(consistent)} already consistent (untouched).")


if __name__ == "__main__":
    main()
