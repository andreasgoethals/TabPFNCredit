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


def pama(matrix: pd.DataFrame, *, higher_is_better: bool = True,
         p_threshold: float = 0.95) -> pd.DataFrame:
    """PAMA: per dataset, the % of the best method's metric a method achieves,
    averaged over datasets. Also the share of datasets where the method reaches
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
        "PAMA_%": 100 * frac.mean(axis=0),
        f"P{int(p_threshold*100)}_%": 100 * (frac >= p_threshold).mean(axis=0),
    }).sort_values("PAMA_%", ascending=False))


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
    """Classic Demšar CD diagram: methods on a rank axis, bold bars join
    groups whose rank difference is below ``cd``."""
    ranks = ranks.sort_values()
    k = len(ranks)
    lo, hi = math.floor(ranks.min()), math.ceil(ranks.max())
    fig, ax = plt.subplots(figsize=(max(8, k * 1.1), 0.55 * k + 2.4))
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0, k / 2 + 2.6)
    ax.invert_xaxis()  # rank 1 (best) on the right, like the paper
    ax.spines[["left", "right", "bottom"]].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.set_xticks(range(lo, hi + 1))
    axis_y = k / 2 + 2.0

    # CD ruler
    ax.plot([hi, hi - cd], [axis_y + 0.45, axis_y + 0.45], lw=2.5, c="black")
    ax.text(hi - cd / 2, axis_y + 0.55, f"CD = {cd:.3f}", ha="center", fontsize=10)
    ax.axhline(axis_y, c="black", lw=1)

    # method stems: left half labels on the left, right half on the right
    half = math.ceil(k / 2)
    for i, (name, r) in enumerate(ranks.items()):
        side_left = i >= half          # worse half -> labels left
        row = (i - half) if side_left else (half - 1 - i)
        y_label = axis_y - 0.9 - 0.5 * row
        x_label = lo - 0.25 if side_left else hi + 0.25
        ha = "right" if not side_left else "left"
        ax.plot([r, r], [axis_y, y_label], c="black", lw=0.9)
        ax.plot([r, x_label], [y_label, y_label], c="black", lw=0.9)
        ax.text(x_label, y_label + 0.05, f"{name} ({r:.2f})",
                ha=ha, va="bottom", fontsize=10)

    # cliques: maximal groups of methods within CD of each other
    vals = ranks.to_numpy()
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i:
            cliques.append((i, j))
    cliques = [c for c in cliques
               if not any(o[0] <= c[0] and c[1] <= o[1] and o != c for o in cliques)]
    for h, (i, j) in enumerate(cliques):
        y = axis_y - 0.22 - 0.18 * h
        ax.plot([vals[i] - 0.04, vals[j] + 0.04], [y, y], c="black", lw=3.5,
                solid_capstyle="round")

    ax.set_title(title, pad=28, fontweight="bold")
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


def plot_pama_bars(
    matrix: pd.DataFrame,
    *,
    higher_is_better: bool = True,
    metric_name: str = "",
    out_path: Optional[Path] = None,
):
    t = pama(matrix, higher_is_better=higher_is_better)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(t) + 4), 6))
    ax.barh(t.index[::-1], t["PAMA_%"][::-1], color="#4878CF")
    ax.set_xlabel(f"PAMA: mean % of best-per-dataset {metric_name}".strip())
    ax.set_xlim(max(0.0, t["PAMA_%"].min() - 5), 100.5)
    ax.set_title(f"PAMA ({metric_name})" if metric_name else "PAMA", fontweight="bold")
    for y, v in enumerate(t["PAMA_%"][::-1]):
        ax.text(v + 0.15, y, f"{v:.1f}", va="center", fontsize=9)
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
    "win_loss_tie", "wlt_summary", "pama",
    "plot_cd_diagram", "plot_significance_matrix",
    "plot_win_loss_matrix", "plot_pama_bars",
]
