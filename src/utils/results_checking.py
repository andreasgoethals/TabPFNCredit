"""Results auditing for TabPFNCredit — completeness, integrity & sanity checks.

This is the backing module for the **Results_Checking** notebook: all logic
lives here, the notebook is a thin viewer (``from src.utils.results_checking
import *``).

For one experiment it answers:

* **Coverage** — which ``(task, dataset, method[, sweep point])`` does the
  *config* say should exist, and which of those are present on disk?
* **Completeness** — which result files have fewer folds than ``cv_splits``,
  or are malformed / unreadable?
* **Sanity** — which results look anomalous (NaN/inf, out-of-range metrics,
  worse-than-random AUC, absurd R², zero-variance folds, …)?
* **Stale** — which files exist on disk but are *not* in the current config
  (a disabled method, a renamed dataset, an old sweep)?
* **Visualisation** — dataset×method coverage heatmaps, per-method fold
  completeness, sweep coverage (Exp 2/3), and metric distributions / curves.

It works against any results root: pass ``results_root=`` explicitly, else it
uses ``$TABPFN_RESULTS_ROOT`` / the resolved project-storage ``results/``
(see :mod:`src.utils.paths`). Run it wherever the results live — locally, or
on the cluster via a VSCode-on-OnDemand session (which can see the Lustre
project storage that the OOD file browser cannot).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.paths import results_root as _default_results_root

Combo = Tuple[str, str, str]  # (task, dataset, name)

# Methods whose predictions are constant by design, so a zero across-fold
# variance (std=0) is EXPECTED, not anomalous (e.g. the Dummy baseline always
# predicts the prior, giving an identical AUC=0.5 every fold).
_CONSTANT_BASELINES = {"dummy"}


# ============================================================================
#  Expected combinations (the source of truth: the experiment's YAML config)
# ============================================================================

def expected_points(experiment: str) -> Tuple[set, int]:
    """Return ``(expected_combos, cv_splits)`` from the experiment's config.

    ``expected_combos`` is a set of ``(task, dataset, name)`` where ``name`` is
    the result-file stem (method + any sweep suffix, e.g. ``xgboost__row20000``).
    Reuses the CLI's own cell-builder and sweep-expander, so it matches exactly
    what a run would attempt. Requires the datasets to be visible (repo or
    project storage), since Exp 2/3 select datasets by row count / imbalance.
    """
    # Imported lazily: the CLI module pulls in Typer + the method registry.
    from src.utils.cli import _build_task_list, _sweep_points
    from src.utils.config_reader import load_config

    config = load_config(experiment)
    cv = int((config.get("split") or {}).get("cv_splits", 5))
    exp = experiment.lower()
    expected: set = set()
    for cell in _build_task_list(config):
        for pt in _sweep_points(exp, config, cell):
            expected.add((cell["task"], cell["dataset"], pt["name"]))
    return expected, cv


def base_method(name: str) -> str:
    """Strip the sweep/HPO suffix: ``xgboost__row20000`` -> ``xgboost``."""
    return name.split("__", 1)[0]


def sweep_axis(name: str) -> Optional[Tuple[str, float]]:
    """Decode a sweep suffix into ``(axis, value)`` or ``None``.

    ``xgboost__row20000`` -> ``("row", 20000.0)``;
    ``tabicl_v2__min0p0025`` -> ``("min", 0.0025)``; ``xgboost__HPO`` -> ``("HPO", 1)``.
    """
    if "__" not in name:
        return None
    suffix = name.split("__", 1)[1]
    for axis in ("row", "min"):
        if suffix.startswith(axis):
            raw = suffix[len(axis):].replace("p", ".")
            try:
                return axis, float(raw)
            except ValueError:
                return axis, math.nan
    if suffix == "HPO":
        return "HPO", 1.0
    return suffix, math.nan


# ============================================================================
#  Metric sanity rules
# ============================================================================

# Metrics that must live in [0, 1] (per task). AP_normalized & R2 may be
# negative (worse than baseline), so they are checked separately.
_UNIT_INTERVAL_PD = ("AUC", "Brier", "ECE", "AP", "F1", "KS", "Accuracy")
_NONNEG_LGD = ("RMSE", "MAE")


def metric_anomalies(task: str, metrics: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable anomaly strings for one metric dict."""
    issues: List[str] = []

    def fval(k: str) -> Optional[float]:
        v = metrics.get(k)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # NaN / inf anywhere
    for k, v in metrics.items():
        fv = fval(k)
        if fv is not None and (math.isnan(fv) or math.isinf(fv)):
            issues.append(f"{k}=NaN/inf")

    if task.lower() == "pd":
        auc = fval("AUC")
        if auc is not None and not (math.isnan(auc)):
            if not (-1e-9 <= auc <= 1 + 1e-9):
                issues.append(f"AUC out of [0,1] ({auc:.3f})")
            elif auc < 0.5 - 1e-6:
                issues.append(f"AUC<0.5 worse-than-random ({auc:.3f})")
        for k in _UNIT_INTERVAL_PD:
            v = fval(k)
            if v is not None and not math.isnan(v) and not (-1e-9 <= v <= 1 + 1e-9):
                issues.append(f"{k} out of [0,1] ({v:.3f})")
        apn = fval("AP_normalized")
        if apn is not None and not math.isnan(apn) and apn > 1 + 1e-9:
            issues.append(f"AP_normalized>1 ({apn:.3f})")
    else:  # lgd regression
        r2 = fval("R2")
        if r2 is not None and not math.isnan(r2) and r2 < -1.0:
            issues.append(f"R2 very negative ({r2:.3f})")
        for k in _NONNEG_LGD:
            v = fval(k)
            if v is not None and not math.isnan(v) and v < -1e-9:
                issues.append(f"{k}<0 ({v:.3f})")
    return issues


# ============================================================================
#  Audit
# ============================================================================

@dataclass
class Audit:
    """Structured result of auditing one experiment."""
    experiment: str
    results_root: Path
    cv_splits: int
    expected: set = field(default_factory=set)
    present: set = field(default_factory=set)
    missing: set = field(default_factory=set)
    unexpected: set = field(default_factory=set)
    incomplete: List[Tuple[Combo, int]] = field(default_factory=list)
    malformed: List[Combo] = field(default_factory=list)
    anomalies: List[Tuple[Combo, List[str]]] = field(default_factory=list)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # -- convenience summaries -------------------------------------------
    def summary(self) -> pd.DataFrame:
        n_complete = sum(
            1 for _ in self.present
            if _ not in {c for c, _n in self.incomplete}
        )
        rows = [
            ("expected (from config)", len(self.expected)),
            ("present on disk", len(self.present)),
            ("complete (>= cv folds)", n_complete),
            ("MISSING (expected, absent)", len(self.missing)),
            ("INCOMPLETE (< cv folds)", len(self.incomplete)),
            ("MALFORMED / unreadable", len(self.malformed)),
            ("ANOMALOUS metrics", len(self.anomalies)),
            ("STALE (on disk, not in config)", len(self.unexpected)),
        ]
        return pd.DataFrame(rows, columns=["check", "count"]).set_index("check")

    def missing_frame(self) -> pd.DataFrame:
        return _combo_frame(sorted(self.missing))

    def incomplete_frame(self) -> pd.DataFrame:
        df = _combo_frame([c for c, _n in self.incomplete])
        if not df.empty:
            df["n_folds"] = [n for _c, n in self.incomplete]
            df["need"] = self.cv_splits
        return df

    def stale_frame(self) -> pd.DataFrame:
        return _combo_frame(sorted(self.unexpected))

    def anomalies_frame(self) -> pd.DataFrame:
        df = _combo_frame([c for c, _ in self.anomalies])
        if not df.empty:
            df["issues"] = ["; ".join(i) for _c, i in self.anomalies]
        return df


def _ds_display_name(slug: str) -> str:
    """Paper display name for ``slug`` (raw slug if the registry is absent)."""
    try:
        from src.data.dataset_names import display_name
        return display_name(str(slug))
    except Exception:  # pragma: no cover -- never break an audit over a label
        return str(slug)


def _combo_frame(combos: Sequence[Combo]) -> pd.DataFrame:
    """Tabulate ``(task, dataset, method)`` combos for PRINTING.

    Every caller feeds the result straight to ``print(... .to_string())``, so the
    ``dataset`` column carries the PAPER DISPLAY NAME rather than the on-disk
    slug, so the proprietary datasets are anonymised here exactly as they are in
    every figure and table. Printing the slug leaked the real dataset identities
    into the audit notebook's committed outputs -- the one tracked place they
    appeared.
    """
    if not combos:
        return pd.DataFrame(columns=["task", "dataset", "name", "base_method"])
    df = pd.DataFrame(combos, columns=["task", "dataset", "name"])
    df["dataset"] = df["dataset"].map(_ds_display_name)
    df["base_method"] = df["name"].map(base_method)
    return df


def audit_experiment(experiment: str, results_root: Optional[Path | str] = None) -> Audit:
    """Audit one experiment end-to-end. See :class:`Audit`."""
    base = Path(results_root) if results_root else _default_results_root()
    exp = experiment.lower()

    try:
        expected, cv = expected_points(experiment)
    except Exception as exc:  # config missing or datasets not visible
        print(f"[warn] could not compute expected combos for {experiment}: {exc}")
        expected, cv = set(), 5

    existing: Dict[Combo, dict] = {}
    malformed: List[Combo] = []
    # scan_results already skips JSONs it can't parse (logged); to surface
    # malformed files we re-walk and try to parse, catching failures.
    exp_dir = base / exp
    if exp_dir.exists():
        for jp in exp_dir.rglob("*.json"):
            parts = jp.relative_to(base).parts
            if len(parts) != 4:
                continue  # summaries/ etc.
            _e, task, dataset, method_file = parts
            name = method_file[:-5]
            import json
            try:
                payload = json.loads(jp.read_text())
            except Exception:
                malformed.append((task, dataset, name))
                continue
            # Packed file (Exp 2/3): one <method>.json holds many sweep points
            # under "points"; expand each into its own logical result.
            if isinstance(payload, dict) and "points" in payload:
                for point_name, entry in (payload.get("points") or {}).items():
                    existing[(task, dataset, point_name)] = entry
            else:
                existing[(task, dataset, name)] = payload

    present = set(existing)
    missing = expected - present
    unexpected = present - expected if expected else set()

    incomplete: List[Tuple[Combo, int]] = []
    anomalies: List[Tuple[Combo, List[str]]] = []
    rows: List[dict] = []

    for key, payload in existing.items():
        task, dataset, name = key
        folds = payload.get("folds") or {}
        n_folds = len(folds)
        complete = n_folds >= cv
        if not complete:
            incomplete.append((key, n_folds))

        aggregates = payload.get("aggregates") or {}
        # aggregates: {metric: {"mean": .., "std": ..}}; flatten to means.
        means = {
            m: (a.get("mean") if isinstance(a, dict) else a)
            for m, a in aggregates.items()
        }
        issues = metric_anomalies(task, means)
        # zero-variance across folds (suspicious) -- but EXPECTED for constant
        # baselines like the Dummy classifier, so skip those.
        if n_folds > 1 and base_method(name) not in _CONSTANT_BASELINES:
            for m, a in aggregates.items():
                if isinstance(a, dict) and a.get("std") == 0 and m in ("AUC", "R2"):
                    issues.append(f"{m} identical across folds (std=0)")
        if issues:
            anomalies.append((key, issues))

        row = {
            "task": task, "dataset": dataset, "name": name,
            "base_method": base_method(name), "n_folds": n_folds,
            "complete": complete,
        }
        ax = sweep_axis(name)
        if ax:
            row["sweep_axis"], row["sweep_value"] = ax
        for m, mv in means.items():
            row[m] = mv
        rows.append(row)

    table = pd.DataFrame(rows)
    return Audit(
        experiment=experiment, results_root=base, cv_splits=cv,
        expected=expected, present=present, missing=missing,
        unexpected=unexpected, incomplete=incomplete, malformed=malformed,
        anomalies=anomalies, table=table,
    )


# ============================================================================
#  Coverage matrices
# ============================================================================

def coverage_matrix(audit: Audit, task: str) -> pd.DataFrame:
    """dataset × base_method matrix of present/expected sweep-point fraction.

    1.0 = every expected point present & complete; 0.0 = none; NaN = not
    expected for that cell. Useful as a heatmap.
    """
    exp_by_cell: Dict[Tuple[str, str], int] = {}
    got_by_cell: Dict[Tuple[str, str], int] = {}
    incomplete_keys = {c for c, _n in audit.incomplete}

    for (t, d, n) in audit.expected:
        if t != task:
            continue
        exp_by_cell[(d, base_method(n))] = exp_by_cell.get((d, base_method(n)), 0) + 1
    for (t, d, n) in audit.present:
        if t != task:
            continue
        if (t, d, n) in incomplete_keys:
            continue
        got_by_cell[(d, base_method(n))] = got_by_cell.get((d, base_method(n)), 0) + 1

    from src.data.dataset_names import sort_datasets as _ds_sorted
    datasets = _ds_sorted({d for (d, _m) in exp_by_cell} | {d for (d, _m) in got_by_cell})
    methods = sorted({m for (_d, m) in exp_by_cell} | {m for (_d, m) in got_by_cell})
    mat = pd.DataFrame(index=datasets, columns=methods, dtype=float)
    for d in datasets:
        for m in methods:
            exp_n = exp_by_cell.get((d, m))
            got_n = got_by_cell.get((d, m), 0)
            if exp_n:
                mat.loc[d, m] = got_n / exp_n
            elif got_n:
                mat.loc[d, m] = -1.0  # present but not expected (stale)
            else:
                mat.loc[d, m] = np.nan
    return mat


# ============================================================================
#  Plotting (matplotlib / seaborn). Each returns a Figure; safe to skip if the
#  plotting stack is unavailable.
# ============================================================================

def _import_plt():
    import matplotlib.pyplot as plt  # noqa: F401
    try:
        import seaborn as sns  # noqa: F401
    except Exception:
        sns = None
    return plt, sns


def plot_coverage_heatmap(audit: Audit, task: str, ax=None):
    """Heatmap of dataset×method completion fraction (1=done, 0=missing, red=stale)."""
    plt, sns = _import_plt()
    mat = coverage_matrix(audit, task)
    if mat.empty:
        return None
    if ax is None:
        import matplotlib.pyplot as _p
        _, ax = _p.subplots(figsize=(max(6, 0.7 * mat.shape[1] + 3),
                                     max(4, 0.4 * mat.shape[0] + 2)))
    data = mat.astype(float)
    if sns is not None:
        sns.heatmap(data, annot=True, fmt=".2f", vmin=0, vmax=1,
                    cmap="RdYlGn", cbar_kws={"label": "fraction complete"},
                    linewidths=0.5, linecolor="white", ax=ax)
    else:
        im = ax.imshow(data.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(data.columns)), data.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(data.index)), data.index)
        ax.figure.colorbar(im, ax=ax, label="fraction complete")
    ax.set_title(f"{audit.experiment} — {task.upper()} coverage "
                 f"(complete / expected sweep points)")
    return ax.figure


def plot_fold_completeness(audit: Audit, ax=None):
    """Bar chart of how many cells are complete vs incomplete vs missing per method."""
    plt, _ = _import_plt()
    if audit.table.empty and not audit.missing:
        return None
    import matplotlib.pyplot as _p
    methods = sorted({base_method(n) for (_t, _d, n) in (audit.expected or audit.present)})
    complete = []
    incomplete = []
    missing = []
    inc_keys = {c for c, _n in audit.incomplete}
    for m in methods:
        exp_m = {(t, d, n) for (t, d, n) in audit.expected if base_method(n) == m}
        pres_m = {(t, d, n) for (t, d, n) in audit.present if base_method(n) == m}
        inc_m = pres_m & inc_keys
        complete.append(len(pres_m - inc_m))
        incomplete.append(len(inc_m))
        missing.append(len(exp_m - pres_m))
    if ax is None:
        _, ax = _p.subplots(figsize=(max(6, 1.2 * len(methods) + 2), 4))
    x = range(len(methods))
    ax.bar(x, complete, label="complete", color="#2ca02c")
    ax.bar(x, incomplete, bottom=complete, label="incomplete", color="#ff7f0e")
    ax.bar(x, missing, bottom=[c + i for c, i in zip(complete, incomplete)],
           label="missing", color="#d62728")
    ax.set_xticks(list(x), methods, rotation=45, ha="right")
    ax.set_ylabel("sweep points")
    ax.set_title(f"{audit.experiment} — completeness by method")
    ax.legend()
    return ax.figure


def plot_metric_distributions(audit: Audit, metric: str = "AUC", ax=None):
    """Box/strip of a metric across datasets, grouped by method (sanity spread)."""
    plt, sns = _import_plt()
    df = audit.table
    if df.empty or metric not in df.columns:
        return None
    import matplotlib.pyplot as _p
    if ax is None:
        _, ax = _p.subplots(figsize=(max(6, 1.2 * df["base_method"].nunique() + 2), 4))
    sub = df.dropna(subset=[metric])
    if sub.empty:
        return None
    if sns is not None:
        sns.boxplot(data=sub, x="base_method", y=metric, ax=ax, color="#cfe8ff")
        sns.stripplot(data=sub, x="base_method", y=metric, ax=ax,
                      color="#1f77b4", size=3, alpha=0.5)
    else:
        groups = [g[metric].values for _k, g in sub.groupby("base_method")]
        ax.boxplot(groups, labels=list(sub["base_method"].unique()))
    ax.set_title(f"{audit.experiment} — {metric} distribution by method")
    ax.tick_params(axis="x", rotation=45)
    return ax.figure


def plot_sweep_curves(audit: Audit, task: str, metric: str = "AUC", ax=None):
    """For Exp 2/3: metric vs sweep value -- ONE line per METHOD, the metric
    averaged over every included dataset at each sweep value (+-1 std band).

    One call per task gives the two headline graphs (PD and LGD). Holes in a
    line still reveal missing sweep points.
    """
    plt, _ = _import_plt()
    df = audit.table
    if df.empty or "sweep_value" not in df.columns or metric not in df.columns:
        return None
    import matplotlib.pyplot as _p
    sub = df[(df["task"] == task)].dropna(subset=["sweep_value", metric])
    if sub.empty:
        return None
    if ax is None:
        _, ax = _p.subplots(figsize=(10, 6))
    grp = (sub.groupby(["base_method", "sweep_value"])[metric]
           .agg(["mean", "std"]).reset_index())
    for meth, g in grp.groupby("base_method"):
        g = g.sort_values("sweep_value")
        line, = ax.plot(g["sweep_value"], g["mean"], marker="o", ms=3, lw=1.8,
                        label=meth)
        ax.fill_between(g["sweep_value"], g["mean"] - g["std"].fillna(0),
                        g["mean"] + g["std"].fillna(0),
                        alpha=0.12, color=line.get_color())
    axis = sub["sweep_axis"].iloc[0] if "sweep_axis" in sub.columns else "sweep"
    ax.set_xlabel(axis)
    ax.set_ylabel(metric)
    ax.set_title(f"{audit.experiment} — {task.upper()} {metric} vs {axis} "
                 f"(mean over datasets)")
    ax.legend(fontsize=9)
    return ax.figure


# ============================================================================
#  One-call driver (used by the notebook)
# ============================================================================

def evaluation_set_mismatches(
    experiment: str,
    results_root: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Datasets whose methods were NOT scored on the same observations.

    The benchmark's central comparability claim is that every method sees
    byte-identical folds, so a dataset's *observed* target vector -- and hence
    its test-fold row counts -- must be the same for every method. When a
    preprocessing change lands and only some results are re-run, the stale files
    keep the old dataset version and that claim quietly breaks: pooled means,
    ranks and significance tests then mix two different evaluation sets.

    Detection compares the per-fold ``fold_<k>_y_true`` lengths stored in each
    npz. Sweep points legitimately differ in size (Experiment 2 caps rows,
    Experiment 3 subsamples the minority class), so methods are grouped by
    ``(task, dataset, sweep axis+value)``; ``__HPO`` shares the sweep group with
    its untuned twin because tuning does not change the folds.

    Returns one row per (group, distinct shape), empty when everything agrees.
    """
    root = Path(results_root) if results_root else _default_results_root()
    exp_dir = root / experiment.lower()
    groups: Dict[Tuple[str, str, str], Dict[Tuple[int, ...], List[str]]] = {}

    for npz_path in sorted(exp_dir.glob("*/*/*.npz")):
        task, dataset = npz_path.parts[-3], npz_path.parts[-2]
        name = npz_path.stem
        if "shard" in name:                      # a shard holds part of a cell
            continue
        axis = sweep_axis(name)
        # __HPO is not a data axis -- it must land in the same group as its twin
        key_axis = "" if axis is None or axis[0] == "HPO" else f"{axis[0]}={axis[1]:g}"
        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                folds = sorted(
                    (int(m.group(1)), k)
                    for k in npz.files
                    if (m := re.fullmatch(r"fold_(\d+)_y_true", k))
                )
                if not folds:
                    continue
                shape = tuple(len(npz[k]) for _i, k in folds)
        except Exception:                        # unreadable npz is `malformed`'s job
            continue
        groups.setdefault((task, dataset, key_axis), {}).setdefault(shape, []).append(name)

    rows: List[Dict[str, Any]] = []
    for (task, dataset, key_axis), shapes in sorted(groups.items()):
        if len(shapes) < 2:
            continue
        ranked = sorted(shapes.items(), key=lambda kv: -len(kv[1]))
        for rank, (shape, methods) in enumerate(ranked):
            rows.append({
                "task": task,
                "dataset": _ds_display_name(dataset),
                "sweep": key_axis or "-",
                "total_rows": sum(shape),
                "n_methods": len(methods),
                "verdict": "majority" if rank == 0 else "MISMATCH",
                "methods": ", ".join(sorted(methods)[:6])
                           + (" ..." if len(methods) > 6 else ""),
            })
    return pd.DataFrame(rows)


def run_full_audit(
    experiments: Sequence[str] = ("Experiment0", "Experiment1", "Experiment2", "Experiment3"),
    results_root: Optional[Path | str] = None,
    show_plots: bool = True,
) -> Dict[str, Audit]:
    """Audit several experiments, print the headline tables, and draw plots.

    Returns ``{experiment: Audit}`` so the notebook can drill in further.
    """
    out: Dict[str, Audit] = {}
    for exp in experiments:
        print("=" * 78)
        print(f"  {exp}")
        print("=" * 78)
        audit = audit_experiment(exp, results_root=results_root)
        out[exp] = audit
        print(f"results root: {audit.results_root}\n")
        try:
            from IPython.display import display
            display(audit.summary())
        except Exception:
            print(audit.summary())

        if audit.missing:
            print(f"\n-- {len(audit.missing)} MISSING (showing up to 30) --")
            print(audit.missing_frame().head(30).to_string(index=False))
        if audit.incomplete:
            print(f"\n-- {len(audit.incomplete)} INCOMPLETE (showing up to 30) --")
            print(audit.incomplete_frame().head(30).to_string(index=False))
        if audit.malformed:
            print(f"\n-- {len(audit.malformed)} MALFORMED --")
            print(_combo_frame(audit.malformed).to_string(index=False))
        if audit.anomalies:
            print(f"\n-- {len(audit.anomalies)} ANOMALOUS (showing up to 30) --")
            print(audit.anomalies_frame().head(30).to_string(index=False))
        if audit.unexpected:
            print(f"\n-- {len(audit.unexpected)} STALE / not-in-config (showing up to 30) --")
            print(audit.stale_frame().head(30).to_string(index=False))

        mismatch = evaluation_set_mismatches(exp, results_root=results_root)
        if len(mismatch):
            bad = int((mismatch["verdict"] == "MISMATCH").sum())
            print(f"\n-- EVALUATION-SET MISMATCH: {bad} group(s) not scored on "
                  f"the same observations --")
            print("   Methods below disagree on the test folds, so pooled means, "
                  "ranks and tests")
            print("   for these datasets mix two dataset versions. Delete the "
                  "minority rows'")
            print("   results (src.utils.remove_results) and re-run them.")
            print(mismatch.to_string(index=False))

        if show_plots:
            try:
                import matplotlib.pyplot as plt
                for task in ("pd", "lgd"):
                    fig = plot_coverage_heatmap(audit, task)
                    if fig is not None:
                        plt.show()
                fig = plot_fold_completeness(audit)
                if fig is not None:
                    plt.show()
                metric = "AUC" if any(t == "pd" for (t, _d, _n) in audit.present) else "R2"
                fig = plot_metric_distributions(audit, metric)
                if fig is not None:
                    plt.show()
                if exp.lower() in ("experiment2", "experiment3"):
                    for task in ("pd", "lgd"):
                        fig = plot_sweep_curves(audit, task, metric="AUC" if task == "pd" else "R2")
                        if fig is not None:
                            plt.show()
            except Exception as exc:
                print(f"[warn] plotting skipped: {exc}")
        print()
    return out


__all__ = [
    "Audit", "audit_experiment", "expected_points", "base_method", "sweep_axis",
    "metric_anomalies", "coverage_matrix",
    "plot_coverage_heatmap", "plot_fold_completeness",
    "plot_metric_distributions", "plot_sweep_curves",
    "evaluation_set_mismatches", "run_full_audit",
]
