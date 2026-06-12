"""Statistical comparison of methods over multiple datasets.

Implements the full recommended methodology of:

* Demšar (2006), "Statistical Comparisons of Classifiers over Multiple Data
  Sets", JMLR 7 -- average ranks, Friedman test, Iman-Davenport correction,
  Nemenyi post-hoc + critical-difference (CD) diagrams, Bonferroni-Dunn
  comparison with a control, Wilcoxon signed-ranks test, sign test.
* García & Herrera (2008), "An Extension on 'Statistical Comparisons of
  Classifiers over Multiple Data Sets' for all Pairwise Comparisons", JMLR 9
  -- adjusted p-values (APVs) for all-pairwise comparisons: Nemenyi, Holm,
  Shaffer's static procedure, Bergmann-Hommel's dynamic procedure.
* Plus the standard one-vs-control step procedures often used alongside them:
  Holm, Hochberg, Hommel, Holland, Finner, Li.
* Benchmark summary statistics: win/loss/tie counts and PAMA (percentage of
  the maximum metric achieved per dataset, averaged).

Input convention
----------------
Everything operates on a **metric matrix**: a pandas DataFrame with datasets
as rows and methods as columns, one scalar per cell (the fold-mean of a
metric). Build it from the benchmark's per-method summary CSV with
:func:`metric_matrix`. ``N`` = number of datasets, ``k`` = number of methods.

All tests treat datasets as the independent samples (one measurement per
dataset per method), exactly as prescribed by Demšar.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as ss

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt


# ============================================================================
#  Matrix construction + ranks
# ============================================================================

def metric_matrix(
    df: pd.DataFrame,
    metric: str,
    *,
    hpo_mode: Optional[str] = None,
    min_datasets: int = 1,
) -> pd.DataFrame:
    """datasets x methods matrix of ``metric`` means from a summary DataFrame.

    ``df`` is the per-method summary CSV (columns ``dataset``, ``method``,
    ``hpo_mode``, ``{metric}_mean`` or ``metric.{metric}_mean``). Methods with
    missing values on any dataset are DROPPED (rank tests need a complete
    matrix), with a printed note so holes are never silent.
    """
    col = next(
        (c for c in (f"metric.{metric}_mean", f"{metric}_mean", metric) if c in df.columns),
        None,
    )
    if col is None:
        raise KeyError(f"No column for metric {metric!r} in {list(df.columns)[:15]}")
    sub = df
    if hpo_mode is not None and "hpo_mode" in df.columns:
        sub = df[df["hpo_mode"] == hpo_mode]
    mat = sub.pivot_table(index="dataset", columns="method", values=col, aggfunc="mean")
    incomplete = mat.columns[mat.isna().any()].tolist()
    if incomplete:
        print(f"[stat] dropping methods with missing datasets: {incomplete}")
        mat = mat.drop(columns=incomplete)
    mat = mat.dropna(axis=0, how="any")
    if mat.shape[0] < min_datasets:
        raise ValueError(f"Only {mat.shape[0]} complete dataset rows for {metric}")
    return mat


def average_ranks(matrix: pd.DataFrame, *, higher_is_better: bool = True) -> pd.Series:
    """Average rank per method (rank 1 = best; ties get average ranks)."""
    ranks = matrix.rank(axis=1, ascending=not higher_is_better)
    return ranks.mean(axis=0).sort_values()


# ============================================================================
#  Demšar (2006): omnibus tests
# ============================================================================

def friedman_test(matrix: pd.DataFrame, *, higher_is_better: bool = True) -> Dict[str, float]:
    """Friedman test + Iman-Davenport correction (Demšar 2006, §3.2.2).

    chi2_F = 12N / (k(k+1)) * [sum_j R_j^2 - k(k+1)^2 / 4]
    F_F    = (N-1) chi2_F / (N(k-1) - chi2_F)   ~  F((k-1), (k-1)(N-1))
    """
    N, k = matrix.shape
    R = average_ranks(matrix, higher_is_better=higher_is_better)
    chi2 = 12.0 * N / (k * (k + 1)) * (float((R ** 2).sum()) - k * (k + 1) ** 2 / 4.0)
    p_chi2 = float(ss.chi2.sf(chi2, k - 1))
    denom = N * (k - 1) - chi2
    if denom <= 0:  # extreme case: chi2 at its maximum
        ff, p_ff = float("inf"), 0.0
    else:
        ff = (N - 1) * chi2 / denom
        p_ff = float(ss.f.sf(ff, k - 1, (k - 1) * (N - 1)))
    return {
        "N": N, "k": k,
        "chi2_F": float(chi2), "p_chi2": p_chi2,
        "iman_davenport_F": float(ff), "p_iman_davenport": p_ff,
    }


def nemenyi_cd(k: int, N: int, alpha: float = 0.05) -> float:
    """Critical difference CD = q_alpha * sqrt(k(k+1)/(6N)) (Demšar Eq. after Tab.5).

    ``q_alpha`` is the Studentized-range quantile divided by sqrt(2), computed
    exactly via scipy (matches Demšar's Table 5(a) for any k).
    """
    q = ss.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    return float(q * math.sqrt(k * (k + 1) / (6.0 * N)))


def bonferroni_dunn_cd(k: int, N: int, alpha: float = 0.05) -> float:
    """CD for comparisons against ONE control (Demšar §3.2.2, Table 5(b))."""
    z = ss.norm.ppf(1 - alpha / (2 * (k - 1)))
    return float(z * math.sqrt(k * (k + 1) / (6.0 * N)))


# ============================================================================
#  Demšar (2006): two-classifier tests
# ============================================================================

def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-ranks test over datasets (Demšar §3.1.3).

    Zero differences are split between positive/negative ranks
    (``zero_method='zsplit'``), as the paper prescribes.
    """
    res = ss.wilcoxon(np.asarray(a, float), np.asarray(b, float),
                      zero_method="zsplit", alternative="two-sided")
    return {"statistic": float(res.statistic), "p": float(res.pvalue)}


def sign_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Sign test on win counts (Demšar §3.1.4); ties split evenly."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    wins = float((a > b).sum()) + 0.5 * float((a == b).sum())
    n = len(a)
    p = float(ss.binomtest(int(round(wins)), n, 0.5, alternative="two-sided").pvalue)
    return {"wins": wins, "n": n, "p": p}


# ============================================================================
#  Pairwise rank z-statistics (basis of every post-hoc below)
# ============================================================================

def _pairwise_rank_pvalues(
    matrix: pd.DataFrame, *, higher_is_better: bool = True
) -> pd.DataFrame:
    """Unadjusted p-values for all method pairs from the rank statistic
    z = (R_i - R_j) / sqrt(k(k+1)/(6N))  (Demšar §3.2.2; García & Herrera §2).
    """
    N, k = matrix.shape
    R = average_ranks(matrix, higher_is_better=higher_is_better)
    se = math.sqrt(k * (k + 1) / (6.0 * N))
    rows = []
    for m1, m2 in itertools.combinations(R.index, 2):
        z = (R[m1] - R[m2]) / se
        rows.append({"method_1": m1, "method_2": m2, "z": float(z),
                     "p_unadjusted": float(2 * ss.norm.sf(abs(z)))})
    return pd.DataFrame(rows).sort_values("p_unadjusted").reset_index(drop=True)


# ============================================================================
#  Step procedures: one-vs-control APVs
#  (Bonferroni-Dunn per Demšar; Holm/Hochberg/Hommel/Holland/Finner/Rom/Li
#   as in the post-hoc literature extending it)
# ============================================================================

def _control_pvalues(matrix, control, *, higher_is_better=True) -> pd.DataFrame:
    N, k = matrix.shape
    R = average_ranks(matrix, higher_is_better=higher_is_better)
    se = math.sqrt(k * (k + 1) / (6.0 * N))
    rows = []
    for m in R.index:
        if m == control:
            continue
        z = (R[m] - R[control]) / se
        rows.append({"method": m, "z": float(z),
                     "p_unadjusted": float(2 * ss.norm.sf(abs(z)))})
    return pd.DataFrame(rows).sort_values("p_unadjusted").reset_index(drop=True)


def control_apv_table(
    matrix: pd.DataFrame,
    control: Optional[str] = None,
    *,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """All one-vs-control adjusted p-values. Control defaults to the best-ranked.

    Columns: Bonferroni-Dunn, Holm, Hochberg, Hommel, Holland, Finner, Li.
    """
    R = average_ranks(matrix, higher_is_better=higher_is_better)
    control = control or R.index[0]
    tab = _control_pvalues(matrix, control, higher_is_better=higher_is_better)
    p = tab["p_unadjusted"].to_numpy()
    m = len(p)
    idx = np.arange(1, m + 1)

    tab["bonferroni_dunn"] = np.minimum(p * m, 1.0)
    tab["holm"] = np.minimum.reduce([np.maximum.accumulate((m - idx + 1) * p), np.ones(m)])
    # Hochberg (step-up): APV_i = min_{j>=i} ((m-j+1) p_j), monotone from the back
    hoch = (m - idx + 1) * p
    tab["hochberg"] = np.minimum(np.minimum.accumulate(hoch[::-1])[::-1], 1.0)
    tab["hommel"] = _hommel_apv(p)
    tab["holland"] = np.minimum(np.maximum.accumulate(1 - (1 - p) ** (m - idx + 1)), 1.0)
    tab["finner"] = np.minimum(np.maximum.accumulate(1 - (1 - p) ** (m / idx)), 1.0)
    tab["li"] = p / (p + 1 - p[-1]) if p[-1] < 1 else np.ones(m)
    tab.insert(0, "control", control)
    return tab


def _hommel_apv(p: np.ndarray) -> np.ndarray:
    """Hommel adjusted p-values (Wright 1992 algorithm)."""
    m = len(p)
    q = p.astype(float).copy()
    pa = p.astype(float).copy()
    for mm in range(m, 1, -1):
        i1 = np.arange(m - mm + 1)            # indices 1..m-mm+1 (0-based)
        i2 = np.arange(m - mm + 1, m)         # indices m-mm+2..m (0-based)
        q1 = np.min(mm * p[i2] / np.arange(2, mm + 1)) if len(i2) else np.inf
        q[i1] = np.minimum(mm * p[i1], q1)
        q[i2] = q[m - mm]
        pa = np.maximum(pa, q)
    return np.minimum(pa, 1.0)


# ============================================================================
#  García & Herrera (2008): all-pairwise APVs
# ============================================================================

@lru_cache(maxsize=None)
def _shaffer_true_counts(k: int) -> frozenset:
    """S(k): possible numbers of simultaneously-true hypotheses among k(k-1)/2
    pairwise hypotheses (García & Herrera 2008, after Shaffer 1986):
    S(k) = U_{j=1..k} { C(j,2) + x : x in S(k-j) },  S(0) = S(1) = {0}.
    """
    if k <= 1:
        return frozenset({0})
    out = set()
    for j in range(1, k + 1):
        cj2 = j * (j - 1) // 2
        for x in _shaffer_true_counts(k - j):
            out.add(cj2 + x)
    return frozenset(out)


def _exhaustive_sets(methods: Tuple[str, ...]) -> List[frozenset]:
    """All exhaustive sets of pairwise hypotheses over ``methods``
    (García & Herrera 2008, Appendix: a set of hypotheses is exhaustive iff it
    is exactly the within-group pairs of some partition of the methods).
    """
    @lru_cache(maxsize=None)
    def parts(items: Tuple[str, ...]) -> Tuple[frozenset, ...]:
        if len(items) <= 1:
            return (frozenset(),)
        first, rest = items[0], items[1:]
        out = set()
        # choose the block containing `first`
        for r in range(len(rest) + 1):
            for combo in itertools.combinations(rest, r):
                block = (first,) + combo
                block_pairs = frozenset(itertools.combinations(sorted(block), 2))
                remaining = tuple(x for x in rest if x not in combo)
                for sub in parts(remaining):
                    out.add(block_pairs | sub)
        return tuple(out)

    return [e for e in parts(tuple(sorted(methods))) if e]


def pairwise_apv_table(
    matrix: pd.DataFrame,
    *,
    higher_is_better: bool = True,
    bergmann_max_k: int = 8,
) -> pd.DataFrame:
    """All-pairwise comparisons with the García & Herrera (2008) APVs.

    Columns: ``nemenyi`` (= unadjusted * m), ``holm``, ``shaffer`` (static),
    ``bergmann_hommel`` (dynamic; only computed when k <= ``bergmann_max_k``
    because it enumerates exhaustive hypothesis sets, which grows
    super-exponentially in k).
    """
    tab = _pairwise_rank_pvalues(matrix, higher_is_better=higher_is_better)
    p = tab["p_unadjusted"].to_numpy()
    m = len(p)
    k = matrix.shape[1]
    idx = np.arange(1, m + 1)

    tab["nemenyi"] = np.minimum(p * m, 1.0)
    tab["holm"] = np.minimum(np.maximum.accumulate((m - idx + 1) * p), 1.0)

    # Shaffer static: t_i = max #hypotheses that can be simultaneously true
    # given i-1 rejections = max{ s in S(k) : s <= m - i + 1 }.
    S = sorted(_shaffer_true_counts(k))
    t = np.array([max(s for s in S if s <= m - i + 1) for i in idx], dtype=float)
    t = np.maximum(t, 1.0)
    tab["shaffer"] = np.minimum(np.maximum.accumulate(t * p), 1.0)

    if k <= bergmann_max_k:
        # APV_h = max{ |I| * min_{j in I} p_j : I exhaustive, h in I }
        # (Garcia & Herrera 2008, Sec. 2.3 + Appendix). Hypotheses are stored
        # as sorted 2-tuples inside each exhaustive set.
        hyp_p = {tuple(sorted((r.method_1, r.method_2))): r.p_unadjusted
                 for r in tab.itertuples()}
        exhaustive = _exhaustive_sets(tuple(matrix.columns))
        ex_stats = [(I, len(I) * min(hyp_p[h] for h in I)) for I in exhaustive]
        bh = []
        for r in tab.itertuples():
            h = tuple(sorted((r.method_1, r.method_2)))
            v = max((val for I, val in ex_stats if h in I), default=r.p_unadjusted)
            bh.append(min(v, 1.0))
        tab["bergmann_hommel"] = np.minimum(np.maximum.accumulate(np.array(bh)), 1.0)
    else:
        tab["bergmann_hommel"] = np.nan
        print(f"[stat] Bergmann-Hommel skipped (k={k} > {bergmann_max_k}; "
              f"exhaustive-set enumeration infeasible) -- use Shaffer instead.")
    return tab


# ============================================================================
#  Win / loss / tie + PAMA
# ============================================================================

def win_loss_tie(matrix: pd.DataFrame, *, higher_is_better: bool = True) -> pd.DataFrame:
    """Pairwise matrix: entry (i, j) = #datasets where method i beats j."""
    cols = list(matrix.columns)
    sign = 1 if higher_is_better else -1
    out = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
    for m1, m2 in itertools.permutations(cols, 2):
        out.loc[m1, m2] = int((sign * (matrix[m1] - matrix[m2]) > 0).sum())
    return out


def wlt_summary(matrix: pd.DataFrame, *, higher_is_better: bool = True) -> pd.DataFrame:
    """Per-method total wins / losses / ties across all pairwise duels."""
    W = win_loss_tie(matrix, higher_is_better=higher_is_better)
    n_duels = (matrix.shape[1] - 1) * matrix.shape[0]
    wins = W.sum(axis=1)
    losses = W.sum(axis=0)
    ties = n_duels - wins - losses
    return (pd.DataFrame({"wins": wins, "losses": losses, "ties": ties})
            .sort_values("wins", ascending=False))


def percent_of_max(matrix: pd.DataFrame, *, higher_is_better: bool = True,
                   p_threshold: float = 0.95) -> pd.DataFrame:
    """Percentage of maximum performance (NOT PAMA -- see :func:`pama_fold_level`).

    Per dataset, the % of the best method's metric a method achieves, averaged
    over datasets; plus the share of datasets where the method reaches
    ``p_threshold`` (e.g. 95%) of the best ("P95").
    """
    if higher_is_better:
        frac = matrix.div(matrix.max(axis=1), axis=0)
    else:
        frac = pd.DataFrame(
            matrix.min(axis=1).to_numpy()[:, None] / matrix.to_numpy(),
            index=matrix.index, columns=matrix.columns,
        )
    return (pd.DataFrame({
        "PctOfMax_%": 100 * frac.mean(axis=0),
        f"P{int(p_threshold*100)}_%": 100 * (frac >= p_threshold).mean(axis=0),
    }).sort_values("PctOfMax_%", ascending=False))


def pama_fold_level(
    per_fold_df: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """PAMA -- Probability of Achieving MAximal accuracy (Fernandez-Delgado
    et al., 2014): the relative frequency with which a learner achieves the
    TOP score across all fold-level observations (every (dataset, fold) pair).

    ``per_fold_df`` is the per-fold summary CSV (columns ``dataset``,
    ``method``, ``fold_id``, ``metric.<metric>``). Only (dataset, fold) cells
    where EVERY method has a value are counted, so no method is advantaged by
    missing competitors. Ties at the top credit each tied method.

    Returns a DataFrame with ``wins``, ``n_folds`` and ``PAMA_%`` per method.
    """
    col = next((c for c in (f"metric.{metric}", metric) if c in per_fold_df.columns), None)
    if col is None:
        raise KeyError(f"No fold-level column for {metric!r}")
    piv = per_fold_df.pivot_table(index=["dataset", "fold_id"], columns="method",
                                  values=col, aggfunc="mean")
    piv = piv.dropna(axis=0, how="any")  # complete fold observations only
    if piv.empty:
        raise ValueError("No complete (dataset, fold) observations across all methods")
    best = piv.max(axis=1) if higher_is_better else piv.min(axis=1)
    is_top = piv.eq(best, axis=0)
    wins = is_top.sum(axis=0)
    out = pd.DataFrame({
        "wins": wins.astype(int),
        "n_folds": len(piv),
        "PAMA_%": 100.0 * wins / len(piv),
    }).sort_values("PAMA_%", ascending=False)
    return out


# ============================================================================
#  Pairwise Wilcoxon signed-rank tests with Holm correction (benchmark-paper
#  methodology: Wilcoxon over datasets per pair, Holm step-down over ALL pairs,
#  reported as a Win/Loss matrix with significance asterisks)
# ============================================================================

def wilcoxon_holm_pairwise(
    matrix: pd.DataFrame,
    *,
    higher_is_better: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """All-pairs Wilcoxon signed-rank tests + Holm step-down adjustment.

    For every method pair the null is a zero median performance difference
    across datasets (Wilcoxon, zsplit zero handling per Demsar). Holm's
    correction is applied over ALL k(k-1)/2 pairs. Returns one row per pair:
    wins/losses/ties (from ``higher_is_better`` direction), ``p_unadjusted``,
    ``p_holm`` and ``significant`` (p_holm <= alpha).
    """
    sign = 1.0 if higher_is_better else -1.0
    rows = []
    for m1, m2 in itertools.combinations(matrix.columns, 2):
        a, b = matrix[m1].to_numpy(float), matrix[m2].to_numpy(float)
        d = sign * (a - b)
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = float(ss.wilcoxon(a, b, zero_method="zsplit",
                                  alternative="two-sided").pvalue)
        rows.append({"method_1": m1, "method_2": m2,
                     "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                     "ties": int((d == 0).sum()), "p_unadjusted": p})
    tab = pd.DataFrame(rows).sort_values("p_unadjusted").reset_index(drop=True)
    p = tab["p_unadjusted"].to_numpy()
    m = len(p)
    idx = np.arange(1, m + 1)
    tab["p_holm"] = np.minimum(np.maximum.accumulate((m - idx + 1) * p), 1.0)
    tab["significant"] = tab["p_holm"] <= alpha
    return tab


# ============================================================================
#  Plots
# ============================================================================

def plot_cd_diagram(
    ranks: pd.Series,
    cd: float,
    *,
    title: str = "Critical difference diagram",
    out_path: Optional[Path] = None,
):
    """Demšar-style critical difference diagram.

    Layout (top to bottom): CD ruler, the rank axis (rank 1 = best, on the
    LEFT), a dedicated band with one row per clique bar (methods whose rank
    difference is below ``cd``), then the method labels -- best half on the
    left margin, worst half on the right, each label on its OWN row so the
    connector lines never overlap.
    """
    ranks = ranks.sort_values()
    k = len(ranks)
    vals = ranks.to_numpy()
    lo, hi = math.floor(vals.min()), math.ceil(vals.max())
    span = max(hi - lo, 1)

    # ---- cliques: maximal intervals of methods within CD of each other ----
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i:
            cliques.append((i, j))
    cliques = [c for c in cliques
               if not any(o[0] <= c[0] and c[1] <= o[1] and o != c for o in cliques)]
    n_cl = len(cliques)

    # ---- geometry ----
    half = math.ceil(k / 2)                  # labels per side
    row_h = 0.55                             # vertical gap between label rows
    cl_h = 0.30                              # vertical gap between clique bars
    axis_y = 0.0
    clique_top = axis_y - 0.35
    label_top = clique_top - n_cl * cl_h - 0.45
    y_bottom = label_top - half * row_h
    margin = 0.34 * span                     # x room for the labels

    fig_w = max(11.0, 1.6 * span + 7)
    fig_h = max(3.2, 0.95 + n_cl * 0.26 + half * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(y_bottom - 0.3, axis_y + 1.25)
    ax.axis("off")

    # ---- rank axis + ticks ----
    ax.plot([lo, hi], [axis_y, axis_y], c="black", lw=1.4)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [axis_y, axis_y + 0.12], c="black", lw=1.2)
        ax.text(t, axis_y + 0.18, str(t), ha="center", va="bottom", fontsize=10)
        if t < hi:  # minor half-ticks
            ax.plot([t + 0.5, t + 0.5], [axis_y, axis_y + 0.06], c="black", lw=0.8)

    # ---- CD ruler (above the axis, anchored at the left end) ----
    ruler_y = axis_y + 0.78
    ax.plot([lo, lo + cd], [ruler_y, ruler_y], lw=2.0, c="black")
    for xx in (lo, lo + cd):
        ax.plot([xx, xx], [ruler_y - 0.06, ruler_y + 0.06], lw=2.0, c="black")
    ax.text(lo + cd / 2, ruler_y + 0.10, f"CD = {cd:.3f}",
            ha="center", va="bottom", fontsize=10)

    # ---- clique bars: one row each, just below the axis ----
    for h, (i, j) in enumerate(cliques):
        y = clique_top - h * cl_h
        ax.plot([vals[i] - 0.03 * span, vals[j] + 0.03 * span], [y, y],
                c="black", lw=3.2, solid_capstyle="round")

    # ---- labels: best half left, worst half right; one row per label ----
    for i, (name, r) in enumerate(ranks.items()):
        left_side = i < half
        row = i if left_side else (k - 1 - i)        # best/worst closest to top
        y = label_top - row * row_h
        x_text = lo - margin * 0.97 if left_side else hi + margin * 0.97
        x_elbow = lo - margin * 0.92 if left_side else hi + margin * 0.92
        ax.plot([r, r], [axis_y, y], c="0.25", lw=0.8)                 # stem
        ax.plot([r, x_elbow], [y, y], c="0.25", lw=0.8)                # connector
        ax.text(x_text, y, f"{name}  ({r:.2f})" if left_side else f"({r:.2f})  {name}",
                ha="left" if left_side else "right", va="center", fontsize=9)

    ax.set_title(title, fontweight="bold", pad=14)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_significance_matrix(
    apv_table: pd.DataFrame,
    methods: Sequence[str],
    *,
    procedure: str = "shaffer",
    alpha: float = 0.05,
    title: Optional[str] = None,
    out_path: Optional[Path] = None,
):
    """k x k heatmap of adjusted p-values; cells < alpha are highlighted."""
    import seaborn as sns
    mat = pd.DataFrame(np.nan, index=list(methods), columns=list(methods), dtype=float)
    for r in apv_table.itertuples():
        v = getattr(r, procedure)
        mat.loc[r.method_1, r.method_2] = v
        mat.loc[r.method_2, r.method_1] = v
    fig, ax = plt.subplots(figsize=(1.0 * len(methods) + 3, 0.8 * len(methods) + 2))
    annot = mat.map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    sns.heatmap(mat, annot=annot, fmt="", cmap="RdYlGn", vmin=0, vmax=2 * alpha,
                center=alpha, linewidths=0.5, cbar_kws={"label": f"APV ({procedure})"},
                ax=ax)
    ax.set_title(title or f"Pairwise APVs ({procedure}); green < {alpha}",
                 fontweight="bold")
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_win_loss_matrix(
    matrix: pd.DataFrame,
    *,
    higher_is_better: bool = True,
    title: str = "Pairwise wins (row beats column)",
    out_path: Optional[Path] = None,
):
    import seaborn as sns
    W = win_loss_tie(matrix, higher_is_better=higher_is_better)
    order = wlt_summary(matrix, higher_is_better=higher_is_better).index
    W = W.loc[order, order]
    fig, ax = plt.subplots(figsize=(1.0 * len(order) + 3, 0.8 * len(order) + 2))
    sns.heatmap(W, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                cbar_kws={"label": "# datasets won"}, ax=ax)
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_percent_of_max_bars(
    matrix: pd.DataFrame,
    *,
    higher_is_better: bool = True,
    metric_name: str = "",
    out_path: Optional[Path] = None,
):
    """Bars of the percentage-of-maximum-performance summary (NOT PAMA)."""
    t = percent_of_max(matrix, higher_is_better=higher_is_better)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(t) + 4), 6))
    ax.barh(t.index[::-1], t["PctOfMax_%"][::-1], color="#4878CF")
    ax.set_xlabel(f"Mean % of best-per-dataset {metric_name}".strip())
    ax.set_xlim(max(0.0, t["PctOfMax_%"].min() - 5), 100.5)
    ax.set_title(f"Percentage of maximum performance ({metric_name})"
                 if metric_name else "Percentage of maximum performance",
                 fontweight="bold")
    for y, v in enumerate(t["PctOfMax_%"][::-1]):
        ax.text(v + 0.15, y, f"{v:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_pama_bars(
    per_fold_df: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool = True,
    metric_name: str = "",
    foundation_methods: Optional[Sequence[str]] = None,
    out_path: Optional[Path] = None,
):
    """PAMA bars (Fernandez-Delgado et al., 2014): share of fold-level
    observations where each method achieves the top score. Optionally
    highlights foundation models and prints their collective share."""
    t = pama_fold_level(per_fold_df, metric, higher_is_better=higher_is_better)
    t = t[t["PAMA_%"] > 0]
    fm = set(foundation_methods or [])
    colors = ["#d62728" if mth in fm else "#4878CF" for mth in t.index]
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(t) + 4), max(4, 0.35 * len(t) + 2)))
    ax.barh(t.index[::-1], t["PAMA_%"][::-1], color=colors[::-1])
    nm = metric_name or metric
    ax.set_xlabel(f"PAMA: % of fold-level observations with the top {nm}")
    ax.set_title(f"Probability of Achieving MAximal accuracy ({nm}, "
                 f"n = {int(t['n_folds'].iloc[0])} folds)", fontweight="bold")
    for y, (v, w) in enumerate(zip(t["PAMA_%"][::-1], t["wins"][::-1])):
        ax.text(v + 0.2, y, f"{v:.1f}%  ({w})", va="center", fontsize=9)
    if fm:
        share = t.loc[t.index.isin(fm), "PAMA_%"].sum()
        ax.text(0.98, 0.02, f"foundation models collectively: {share:.1f}%",
                transform=ax.transAxes, ha="right", fontsize=10,
                bbox=dict(boxstyle="round", fc="#ffe9e9", ec="#d62728"))
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_wilcoxon_wl_matrix(
    matrix: pd.DataFrame,
    *,
    higher_is_better: bool = True,
    alpha: float = 0.05,
    metric_name: str = "",
    out_path: Optional[Path] = None,
):
    """k x k matrix of pairwise Wilcoxon results: cell text = "W/L" of the ROW
    method vs the COLUMN method, with a trailing ``*`` when the Holm-corrected
    p-value is <= alpha. Cell colour = win-loss margin. Methods ordered by
    average rank (best first)."""
    import seaborn as sns
    tab = wilcoxon_holm_pairwise(matrix, higher_is_better=higher_is_better, alpha=alpha)
    order = list(average_ranks(matrix, higher_is_better=higher_is_better).index)
    k = len(order)
    margin = pd.DataFrame(np.nan, index=order, columns=order, dtype=float)
    annot = pd.DataFrame("", index=order, columns=order, dtype=object)
    n_sig = 0
    for r in tab.itertuples():
        star = "*" if r.significant else ""
        n_sig += int(r.significant)
        annot.loc[r.method_1, r.method_2] = f"{r.wins}/{r.losses}{star}"
        annot.loc[r.method_2, r.method_1] = f"{r.losses}/{r.wins}{star}"
        margin.loc[r.method_1, r.method_2] = r.wins - r.losses
        margin.loc[r.method_2, r.method_1] = r.losses - r.wins
    vmax = float(np.nanmax(np.abs(margin.values))) or 1.0
    fig, ax = plt.subplots(figsize=(0.62 * k + 4, 0.5 * k + 3))
    sns.heatmap(margin, annot=annot, fmt="", cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, linewidths=0.4, linecolor="white",
                cbar_kws={"label": "win - loss margin (row vs column)"},
                annot_kws={"fontsize": 7}, ax=ax)
    ax.set_title(
        f"Pairwise Wilcoxon signed-rank ({metric_name}): W/L of row vs column; "
        f"* = significant at alpha={alpha} after Holm "
        f"({n_sig}/{len(tab)} pairs significant)",
        fontweight="bold", fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout()
    return _finish(fig, out_path)


def _finish(fig, out_path: Optional[Path]):
    saved = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        saved = out_path
    try:
        from src.visualizations.data_exploration import _display_inline
        _display_inline(fig)
    except Exception:
        pass
    plt.close(fig)
    return saved


__all__ = [
    "metric_matrix", "average_ranks",
    "friedman_test", "nemenyi_cd", "bonferroni_dunn_cd",
    "wilcoxon_signed_rank", "sign_test",
    "control_apv_table", "pairwise_apv_table",
    "wilcoxon_holm_pairwise",
    "win_loss_tie", "wlt_summary",
    "percent_of_max", "pama_fold_level",
    "plot_cd_diagram", "plot_significance_matrix",
    "plot_win_loss_matrix", "plot_pama_bars",
    "plot_percent_of_max_bars", "plot_wilcoxon_wl_matrix",
]
