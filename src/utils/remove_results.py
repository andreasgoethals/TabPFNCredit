"""Delete benchmark results by any combination of filters -- the one tool for
pruning the results tree (replaces the old per-method / per-cell scripts).

Works **locally and on the VSC** (results location is auto-resolved by
:func:`src.utils.paths.results_root`, or pass ``--results-root``).

Select what to delete with any combination of:

* ``--experiment`` / ``--task`` / ``--dataset`` / ``--method`` (repeatable;
  omit a filter to match everything on that axis),
* ``--hpo`` / ``--no-hpo`` (only the tuned ``__HPO`` results, or only the
  untuned ones; omit for both), and
* ``--folds`` -- instead of deleting whole result files, drop **only** those
  fold ids from every matched result (the file is rewritten with the remaining
  folds and refreshed aggregates).

Method matching is exact on the *base* name, so ``--method tabicl`` never
touches ``tabicl_v2``; it does match that method's ``__HPO`` copy and its
packed per-task ``__shard_*`` files (Experiment 2/3).

Whenever results change, the affected experiments' **summary CSVs are removed**
(they are now stale); pass ``--resummarize`` to regenerate them immediately.

Examples (repo root, venv active)::

    # drop the dummy baseline everywhere, refresh the summaries
    python -m src.utils.remove_results --method dummy --resummarize

    # one bugged cell
    python -m src.utils.remove_results --experiment experiment2 --task pd \\
        --dataset 0003.vehicle_loan --method tabicl_v2

    # only the tuned copies of two methods in Experiment 1
    python -m src.utils.remove_results --experiment experiment1 --method xgboost catboost --hpo

    # drop just folds 3 and 4 of every catboost result (keep the rest)
    python -m src.utils.remove_results --method catboost --folds 3 4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

# Make ``src`` importable when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import results_root as _default_results_root  # noqa: E402
from src.utils.result_io import _aggregate  # noqa: E402  (canonical aggregate builder)

_ARRAY_PREFIXES = ("y_true", "y_prob", "y_pred", "val_y_true", "val_y_prob")


def _norm(values: Optional[Iterable[str]]) -> Optional[Set[str]]:
    """Lower-cased set of filter values, or ``None`` to mean 'match all'."""
    if not values:
        return None
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _parse_stem(stem: str):
    """``(base_method, is_hpo)`` from a result-file stem.

    ``tabicl_v2`` -> ("tabicl_v2", False); ``tabicl_v2__HPO`` -> (..., True);
    ``tabicl_v2__shard_123_0`` -> ("tabicl_v2", False)."""
    parts = stem.split("__")
    return parts[0], ("HPO" in parts[1:])


def _drop_folds_in_place(json_path: Path, fold_set: Set[int], dry_run: bool) -> bool:
    """Remove ``fold_set`` from one result file (packed or plain). Returns True
    if anything changed. Rewrites the JSON (+ trims the npz) atomically."""
    try:
        payload = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    drop = {str(f) for f in fold_set}
    changed = False

    def _prune(folds: dict) -> dict:
        nonlocal changed
        kept = {k: v for k, v in folds.items() if k not in drop}
        if len(kept) != len(folds):
            changed = True
        return kept

    if isinstance(payload, dict) and "points" in payload:           # packed (Exp 2/3)
        for pt in (payload.get("points") or {}).values():
            pt["folds"] = _prune(pt.get("folds") or {})
            pt["aggregates"] = _aggregate(pt["folds"])
            pt["n_folds"] = len(pt["folds"])
    else:                                                           # plain (Exp 0/1)
        payload["folds"] = _prune(payload.get("folds") or {})
        payload["aggregates"] = _aggregate(payload["folds"])
        payload["n_folds"] = len(payload["folds"])

    if not changed or dry_run:
        return changed

    tmp = json_path.parent / f".{json_path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, json_path)

    # Trim the matching arrays out of the npz (plain results only).
    npz_path = json_path.with_suffix(".npz")
    if npz_path.exists():
        try:
            import numpy as np
            with np.load(npz_path, allow_pickle=False) as npz:
                keep = {k: npz[k] for k in npz.files
                        if not any(k == f"fold_{f}_{p}" for f in fold_set for p in _ARRAY_PREFIXES)}
            if keep:
                tmpz = npz_path.parent / f".{npz_path.name}.{os.getpid()}.tmp"
                with open(tmpz, "wb") as fh:
                    np.savez_compressed(fh, **keep)
                os.replace(tmpz, npz_path)
            else:
                npz_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 -- npz trim is best-effort
            pass
    return True


def remove_results(
    *,
    results_root: Optional[Path | str] = None,
    experiments: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    hpo: Optional[bool] = None,
    folds: Optional[Sequence[int]] = None,
    dry_run: bool = False,
    drop_summaries: bool = True,
    resummarize: bool = False,
) -> Dict[str, list]:
    """Delete (or fold-trim) every result matching the filters.

    Returns a report dict with the lists of touched files. See the module
    docstring for the filter semantics.
    """
    root = Path(results_root or _default_results_root())
    f_exp, f_task, f_ds, f_meth = (_norm(experiments), _norm(tasks),
                                   _norm(datasets), _norm(methods))
    fold_set = {int(x) for x in folds} if folds else None

    report: Dict[str, list] = {"removed": [], "fold_trimmed": [], "summaries": []}
    affected_exps: Set[str] = set()

    if root.exists():
        for json_path in sorted(root.rglob("*.json")):
            parts = json_path.relative_to(root).parts
            if len(parts) != 4 or not parts[3].endswith(".json"):
                continue
            exp, task, dataset, fname = parts
            stem = fname[:-5]
            base, is_hpo = _parse_stem(stem)
            if f_exp and exp.lower() not in f_exp:       continue
            if f_task and task.lower() not in f_task:    continue
            if f_ds and dataset.lower() not in f_ds:     continue
            if f_meth and base.lower() not in f_meth:    continue
            if hpo is True and not is_hpo:               continue
            if hpo is False and is_hpo:                  continue

            rel = "/".join(parts)
            if fold_set is None:
                report["removed"].append(rel)
                affected_exps.add(exp.lower())
                if not dry_run:
                    json_path.unlink(missing_ok=True)
                    json_path.with_suffix(".npz").unlink(missing_ok=True)
            elif _drop_folds_in_place(json_path, fold_set, dry_run):
                report["fold_trimmed"].append(rel)
                affected_exps.add(exp.lower())

    # Summaries are stale the moment any result changes -> drop them.
    if drop_summaries and affected_exps:
        sumdir = root / "summaries"
        for exp in sorted(affected_exps):
            for suffix in ("_per_fold.csv", "_per_method.csv"):
                p = sumdir / f"{exp}{suffix}"
                if p.exists():
                    report["summaries"].append(f"summaries/{p.name}")
                    if not dry_run:
                        p.unlink(missing_ok=True)

    if resummarize and not dry_run and affected_exps:
        from src.utils.result_summary import summarize_to_csv
        for exp in sorted(affected_exps):
            try:
                summarize_to_csv(root, exp, root / "summaries")
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] could not re-summarize {exp}: {exc}")

    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", nargs="*", help="e.g. experiment1 experiment2 (default: all)")
    ap.add_argument("--task", nargs="*", help="pd / lgd (default: both)")
    ap.add_argument("--dataset", nargs="*", help="dataset dir name(s), e.g. 0003.vehicle_loan")
    ap.add_argument("--method", nargs="*", help="method base name(s), e.g. dummy tabicl_v2")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--hpo", action="store_true", help="only the tuned (__HPO) results")
    grp.add_argument("--no-hpo", action="store_true", help="only the untuned results")
    ap.add_argument("--folds", nargs="*", type=int,
                    help="drop ONLY these fold ids from each matched result (keeps the rest)")
    ap.add_argument("--results-root", type=Path, default=None,
                    help="results root (default: auto-resolved; project storage on the VSC)")
    ap.add_argument("--keep-summaries", action="store_true",
                    help="do NOT delete the affected experiments' summary CSVs")
    ap.add_argument("--resummarize", action="store_true",
                    help="regenerate the affected summaries immediately after pruning")
    ap.add_argument("--dry-run", action="store_true", help="list what would change, do nothing")
    args = ap.parse_args(argv)

    hpo = True if args.hpo else (False if args.no_hpo else None)
    root = Path(args.results_root or _default_results_root())
    print(f"Results root : {root}")
    print(f"Filters      : experiment={args.experiment or 'ALL'} task={args.task or 'ALL'} "
          f"dataset={args.dataset or 'ALL'} method={args.method or 'ALL'} "
          f"hpo={hpo} folds={args.folds or '(whole files)'}")
    print(f"Mode         : {'DRY RUN' if args.dry_run else 'APPLY'}")

    rep = remove_results(
        results_root=root, experiments=args.experiment, tasks=args.task,
        datasets=args.dataset, methods=args.method, hpo=hpo, folds=args.folds,
        dry_run=args.dry_run, drop_summaries=not args.keep_summaries,
        resummarize=args.resummarize,
    )
    verb = "would remove" if args.dry_run else "removed"
    if args.folds:
        print(f"\n{('would trim' if args.dry_run else 'trimmed')} folds {args.folds} in "
              f"{len(rep['fold_trimmed'])} result(s):")
        for x in rep["fold_trimmed"][:50]:
            print(f"   {x}")
    else:
        print(f"\n{verb} {len(rep['removed'])} result file(s):")
        for x in rep["removed"][:50]:
            print(f"   {x}")
        if len(rep["removed"]) > 50:
            print(f"   ... and {len(rep['removed']) - 50} more")
    if rep["summaries"]:
        print(f"\n{('would remove' if args.dry_run else 'removed')} {len(rep['summaries'])} stale "
              f"summary CSV(s): {', '.join(rep['summaries'])}")
    if not rep["removed"] and not rep["fold_trimmed"]:
        print("   (nothing matched the filters)")
    elif not args.dry_run and not args.resummarize and rep["summaries"]:
        print("\nRe-run `tabpfncredit summarize --experiment <name>` (or pass --resummarize) to "
              "rebuild the CSVs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
