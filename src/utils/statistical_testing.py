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

# Shared styling so the stats figures match the experiment notebooks exactly
# (same fonts, the crimson foundation-model highlight, R2 -> R²).
from src.visualizations.experiment_plots import (  # noqa: E402
    TICK_FS, LABEL_FS, TITLE_FS,
    _pretty_metric, _color_foundation_ticks, _foundation_methods, _best_to_worst_colors,
)
from src.methods.method_names import display_name as _display_name  # noqa: E402


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


def friedman_aligned_ranks_test(matrix: pd.DataFrame, *, higher_is_better: bool = True) -> Dict[str, float]:
    """Friedman Aligned Ranks test (Hodges & Lehmann 1962; García, Fernández,
    Luengo & Herrera 2010, *Information Sciences*).

    A more powerful omnibus alternative to Friedman when the number of methods
    ``k`` is small-to-moderate: instead of ranking within each dataset only,
    each observation is *aligned* by subtracting its dataset's mean across
    methods, then ALL ``N*k`` aligned values are ranked together. ~chi² with
    ``k-1`` df.
    """
    N, k = matrix.shape
    vals = matrix.to_numpy(dtype=float)
    aligned = vals - vals.mean(axis=1, keepdims=True)          # subtract per-dataset mean
    flat = aligned.ravel()
    # rank descending for higher-is-better so rank 1 = best (direction does not
    # affect the (squared) statistic, but keeps semantics consistent).
    ranks = pd.Series(flat).rank(ascending=not higher_is_better).to_numpy().reshape(N, k)
    Rj = ranks.sum(axis=0)                                     # per-method rank sums
    Ri = ranks.sum(axis=1)                                     # per-dataset rank sums
    kN = k * N
    # numerator: (k-1) [ Σ R_j² − (k N² / 4)(kN+1)² ]
    num = (k - 1) * (float((Rj ** 2).sum()) - (k * N ** 2 / 4.0) * (kN + 1) ** 2)
    den = (kN * (kN + 1) * (2 * kN + 1)) / 6.0 - (1.0 / k) * float((Ri ** 2).sum())
    T = num / den if den != 0 else float("inf")
    return {"N": N, "k": k, "aligned_ranks_chi2": float(T),
            "p": float(ss.chi2.sf(T, k - 1))}


def quade_test(matrix: pd.DataFrame, *, higher_is_better: bool = True) -> Dict[str, float]:
    """Quade test (Quade 1979; Conover 1999; Demšar 2006).

    Weights each dataset by its performance *range* (datasets that discriminate
    more between methods count more), then tests for a treatment effect. Often
    more powerful than Friedman for a small number of datasets. ~F with
    ``(k-1, (N-1)(k-1))`` df.
    """
    N, k = matrix.shape
    vals = matrix.to_numpy(dtype=float)
    # within-dataset ranks (1 = best), dataset weights from the range rank
    within = pd.DataFrame(vals).rank(axis=1, ascending=not higher_is_better).to_numpy()
    ranges = vals.max(axis=1) - vals.min(axis=1)
    Q = pd.Series(ranges).rank().to_numpy()                    # rank of the ranges, 1..N
    S = Q[:, None] * (within - (k + 1) / 2.0)
    Sj = S.sum(axis=0)
    A = float((S ** 2).sum())
    B = float((Sj ** 2).sum()) / N
    if A == B:                                                 # degenerate (all blocks identical)
        return {"N": N, "k": k, "quade_F": float("inf"), "p": 0.0}
    F = (N - 1) * B / (A - B)
    return {"N": N, "k": k, "quade_F": float(F),
            "p": float(ss.f.sf(F, k - 1, (N - 1) * (k - 1)))}


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


def bayesian_sign_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    rope: float = 0.01,
    prior_strength: float = 1.0,
    n_samples: int = 50_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Bayesian sign test with a region of practical equivalence (ROPE)
    (Benavoli, Corani, Demšar & Zaffalon 2017, *JMLR* 18 — "Time for a change").

    A modern complement to null-hypothesis testing that sidesteps the
    multiple-comparison / family-wise-error explosion entirely: instead of a
    p-value it returns the posterior probability that method ``a`` is
    practically better (``P_right``), that the two are practically equivalent
    (``P_rope``, both within ``rope`` of each other), or that ``b`` is better
    (``P_left``). Per-dataset differences ``a-b`` are classified into the three
    regions and a symmetric Dirichlet prior (total mass ``prior_strength``) is
    updated to a Dirichlet posterior, summarised in closed form and by
    Monte-Carlo (which region is most probable).
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n_left = int((d < -rope).sum())          # b practically better
    n_rope = int((np.abs(d) <= rope).sum())  # practically equivalent
    n_right = int((d > rope).sum())          # a practically better
    alpha = np.array([prior_strength / 3.0] * 3) + np.array([n_left, n_rope, n_right])
    total = float(alpha.sum())
    samp = np.random.default_rng(seed).dirichlet(alpha, size=n_samples)
    region = np.argmax(samp, axis=1)
    return {
        "rope": rope, "n_left": n_left, "n_rope": n_rope, "n_right": n_right,
        "P_left": float(alpha[0] / total),    # posterior mean P(b better)
        "P_rope": float(alpha[1] / total),    # posterior mean P(equivalent)
        "P_right": float(alpha[2] / total),   # posterior mean P(a better)
        "P_left_is_max": float((region == 0).mean()),
        "P_rope_is_max": float((region == 1).mean()),
        "P_right_is_max": float((region == 2).mean()),
    }


def bayesian_signed_rank_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    rope: float = 0.01,
    s: float = 0.5,
    n_samples: int = 50_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Bayesian signed-rank test with a ROPE (Benavoli, Mangili, Corani,
    Zaffalon & Ruggeri 2014; Benavoli, Corani, Demšar & Zaffalon 2017).

    The Bayesian analogue of the Wilcoxon *signed-rank* test, and the procedure
    used by **Gunnarsson et al. (2021)** for credit scoring. Unlike the Bayesian
    SIGN test (:func:`bayesian_sign_test`), it uses the *magnitudes* of the
    per-dataset differences via their Walsh averages ``(z_i + z_j)/2``, under a
    Dirichlet-process prior (one pseudo-observation at 0 with concentration
    ``s``). Returns the posterior probability that method ``a`` is practically
    better (``P_right``), that the two are within the ROPE (``P_rope``), or that
    ``b`` is better (``P_left``) -- no multiple-comparison correction needed.
    """
    z = np.asarray(a, float) - np.asarray(b, float)
    n = z.size
    z_aug = np.concatenate(([0.0], z))                  # DP pseudo-observation at 0
    walsh = (z_aug[:, None] + z_aug[None, :]) / 2.0     # Walsh averages (m x m)
    left = (walsh < -rope).astype(float)
    right = (walsh > rope).astype(float)
    alpha = np.ones(n + 1)
    alpha[0] = s
    ws = np.random.default_rng(seed).dirichlet(alpha, size=n_samples)
    # p_region = w^T M_region w  (quadratic form; the three regions partition
    # all (i, j) pairs, so they sum to 1 for every sample).
    pl = np.einsum("si,ij,sj->s", ws, left, ws)
    pr = np.einsum("si,ij,sj->s", ws, right, ws)
    prope = 1.0 - pl - pr
    return {
        "rope": rope,
        "P_left": float(pl.mean()),    # posterior P(b practically better)
        "P_rope": float(prope.mean()), # posterior P(practically equivalent)
        "P_right": float(pr.mean()),   # posterior P(a practically better)
        "P_left_is_max": float(np.mean((pl > pr) & (pl > prope))),
        "P_rope_is_max": float(np.mean((prope > pl) & (prope > pr))),
        "P_right_is_max": float(np.mean((pr > pl) & (pr > prope))),
    }


def format_apv_table(tab: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round every numeric column of an APV / p-value table for compact,
    readable display (the raw tables carry ~15 digits)."""
    out = tab.copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(decimals)
    return out


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


def ttest_holm_pairwise(
    matrix: pd.DataFrame,
    *,
    higher_is_better: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """All-pairs PAIRED Student's t-test + Holm step-down adjustment.

    The parametric companion to :func:`wilcoxon_holm_pairwise`: each pair's
    per-dataset performance differences are compared with a paired t-test
    (assumes roughly normal differences), and the p-values are Holm-corrected
    over all k(k-1)/2 pairs to control the family-wise error. This t-test +
    Holm procedure is the one adopted by several recent tabular-ML benchmarks
    (e.g. Ye et al., 2024; Liu et al., 2024). Returns one row per pair with
    wins/losses, ``mean_diff``, ``t_stat``, ``p_unadjusted``, ``p_holm`` and
    ``significant``.
    """
    sign = 1.0 if higher_is_better else -1.0
    rows = []
    for m1, m2 in itertools.combinations(matrix.columns, 2):
        a, b = matrix[m1].to_numpy(float), matrix[m2].to_numpy(float)
        d = sign * (a - b)
        if np.allclose(a, b) or np.std(a - b) == 0:
            t_stat, p = 0.0, 1.0
        else:
            res = ss.ttest_rel(a, b)
            t_stat, p = float(res.statistic), float(res.pvalue)
        rows.append({"method_1": m1, "method_2": m2,
                     "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                     "mean_diff": float(np.mean(a - b)), "t_stat": t_stat,
                     "p_unadjusted": p})
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
    row_h = 0.58                             # vertical gap between label rows
    cl_h = 0.30                              # vertical gap between clique bars
    axis_y = 0.0
    clique_top = axis_y - 0.35
    label_top = clique_top - n_cl * cl_h - 0.45
    y_bottom = label_top - half * row_h
    margin = max(0.28 * span, 3.0)           # x room for the labels

    # More compact horizontally so the diagram fits a paper column/page
    # (was 1.7*span+8). The rank axis is simply drawn on a narrower canvas;
    # labels still grow outward onto their own rows so nothing overlaps.
    fig_w = max(9.0, 1.1 * span + 6)
    fig_h = max(3.4, 1.0 + n_cl * 0.26 + half * 0.45)
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
    # The connector ends exactly at the text anchor and the text grows AWAY
    # from the plot (left labels right-aligned, right labels left-aligned), so
    # a name can never cross its own connector line. bbox_inches="tight" on the
    # save captures any name that extends past the axes.
    fnd = _foundation_methods()
    x_left = lo - margin * 0.50
    x_right = hi + margin * 0.50
    for i, (name, r) in enumerate(ranks.items()):
        left_side = i < half
        row = i if left_side else (k - 1 - i)        # best/worst closest to top
        y = label_top - row * row_h
        anchor = x_left if left_side else x_right
        ax.plot([r, r], [axis_y, y], c="0.25", lw=0.8)                 # stem
        ax.plot([r, anchor], [y, y], c="0.25", lw=0.8)                 # connector to anchor
        ax.text(anchor + (-0.05 * span if left_side else 0.05 * span), y,
                f"{_display_name(name)} ({r:.2f})",
                ha="right" if left_side else "left", va="center",
                fontsize=12, color="crimson" if name in fnd else "black",
                fontweight="bold" if name in fnd else "normal")

    ax.set_title(title, fontweight="bold", fontsize=TITLE_FS, pad=14)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_significance_matrix(
    apv_table: pd.DataFrame,
    methods: Sequence[str],
    *,
    procedure: str = "shaffer",
    alpha: float = 0.05,
    order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    out_path: Optional[Path] = None,
):
    """k x k heatmap of adjusted p-values: **green = the pair differs
    significantly** (APV < alpha), red = not. Methods are shown in ``order``
    (pass the rank order so the best sit top-left, like every other figure);
    foundation-model names are crimson."""
    import seaborn as sns
    methods = list(order) if order is not None else list(methods)
    mat = pd.DataFrame(np.nan, index=methods, columns=methods, dtype=float)
    for r in apv_table.itertuples():
        if r.method_1 in mat.index and r.method_2 in mat.columns:
            v = getattr(r, procedure)
            mat.loc[r.method_1, r.method_2] = v
            mat.loc[r.method_2, r.method_1] = v
    k = len(methods)
    fig, ax = plt.subplots(figsize=(0.52 * k + 3.5, 0.46 * k + 2.8))
    annot = mat.map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    # RdYlGn_r so small APV (significant) is GREEN, large is red.
    sns.heatmap(mat, annot=annot, fmt="", cmap="RdYlGn_r", vmin=0, vmax=2 * alpha,
                center=alpha, linewidths=0.5, annot_kws={"fontsize": 9},
                cbar_kws={"label": "adjusted p-value"}, ax=ax)
    ax.set_title(title or f"Pairwise adjusted p-values; "
                 f"green = significantly different (< {alpha})",
                 fontweight="bold", fontsize=TITLE_FS)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(labelsize=TICK_FS)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_horizontalalignment("right")
    _color_foundation_ticks(ax, axis="x")
    _color_foundation_ticks(ax, axis="y")
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
    k = len(order)
    fig, ax = plt.subplots(figsize=(0.52 * k + 3.5, 0.46 * k + 2.8))
    sns.heatmap(W, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                annot_kws={"fontsize": 11}, cbar_kws={"label": "# datasets won"}, ax=ax)
    ax.set_title(title, fontweight="bold", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS + 1)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_horizontalalignment("right")
    _color_foundation_ticks(ax, axis="x")
    _color_foundation_ticks(ax, axis="y")
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
    nm = _pretty_metric(metric_name)
    n = len(t)
    # Same look as the experiment bar plots: best-first (top), green->red
    # gradient + black border. Data reversed for barh (so index[0] sits on top),
    # colours reversed to match (green = best stays at the top).
    colors = _best_to_worst_colors(n)[::-1]
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * n + 4), max(4.5, 0.42 * n + 2)))
    ax.set_axisbelow(True)
    ax.barh(t.index[::-1], t["PctOfMax_%"][::-1], color=colors,
            edgecolor="black", linewidth=0.8, zorder=3)
    ax.set_xlabel(f"Mean % of best-per-dataset {nm}".strip(),
                  fontsize=LABEL_FS, fontweight="bold")
    ax.set_xlim(max(0.0, t["PctOfMax_%"].min() - 5), 104)
    ax.set_title(f"Percentage of maximum performance ({nm})"
                 if nm else "Percentage of maximum performance",
                 fontweight="bold", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    _color_foundation_ticks(ax, axis="y")     # foundation names in crimson
    for y, v in enumerate(t["PctOfMax_%"][::-1]):
        ax.text(v + 0.4, y, f"{v:.1f}", va="center", fontsize=10,
                fontweight="bold", color="0.15")
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
    n = len(t)
    # Same look as the experiment bar plots: best-first (top), green->red
    # gradient + black border. (Foundation models are flagged by their crimson
    # NAME, like every other figure -- not by a special bar colour.) Data is
    # reversed for barh so index[0] sits on top; colours reversed to match.
    colors = _best_to_worst_colors(n)[::-1]
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * n + 4), max(4.5, 0.42 * n + 2)))
    ax.set_axisbelow(True)
    ax.barh(t.index[::-1], t["PAMA_%"][::-1], color=colors,
            edgecolor="black", linewidth=0.8, zorder=3)
    nm = _pretty_metric(metric_name or metric)
    ax.set_xlabel(f"PAMA: % of fold-level observations with the top {nm}",
                  fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(f"Probability of Achieving MAximal accuracy ({nm}, "
                 f"n = {int(t['n_folds'].iloc[0])} folds)", fontweight="bold", fontsize=TITLE_FS)
    # Headroom so the "(wins)" annotation stays INSIDE the axes.
    ax.set_xlim(0, t["PAMA_%"].max() * 1.20 + 1)
    for y, (v, w) in enumerate(zip(t["PAMA_%"][::-1], t["wins"][::-1])):
        ax.text(v + 0.4, y, f"{v:.1f}%  ({w})", va="center", fontsize=10,
                fontweight="bold", color="0.15")
    ax.tick_params(labelsize=TICK_FS)
    # Foundation-model names in the shared crimson, like every other figure.
    _color_foundation_ticks(ax, axis="y")
    if fm:
        share = t.loc[t.index.isin(fm), "PAMA_%"].sum()
        ax.text(0.98, 0.04, f"foundation models collectively: {share:.1f}%",
                transform=ax.transAxes, ha="right", fontsize=11,
                bbox=dict(boxstyle="round", fc="#ffe9e9", ec="crimson"))
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
    for r in tab.itertuples():
        # Just the Win/Loss ratio -- significance is reported separately as
        # text (see significant_pairs_text), per the cleaner two-part layout.
        annot.loc[r.method_1, r.method_2] = f"{r.wins}/{r.losses}"
        annot.loc[r.method_2, r.method_1] = f"{r.losses}/{r.wins}"
        margin.loc[r.method_1, r.method_2] = r.wins - r.losses
        margin.loc[r.method_2, r.method_1] = r.losses - r.wins
    vmax = float(np.nanmax(np.abs(margin.values))) or 1.0
    fig, ax = plt.subplots(figsize=(0.52 * k + 3.5, 0.46 * k + 2.8))
    sns.heatmap(margin, annot=annot, fmt="", cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, linewidths=0.4, linecolor="white",
                cbar_kws={"label": "win - loss margin (row vs column)"},
                annot_kws={"fontsize": 11}, ax=ax)
    ax.set_title(f"Pairwise Win/Loss of row vs column ({_pretty_metric(metric_name)})",
                 fontweight="bold", fontsize=TITLE_FS)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(labelsize=TICK_FS + 1)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_horizontalalignment("right")
    _color_foundation_ticks(ax, axis="x")
    _color_foundation_ticks(ax, axis="y")
    fig.tight_layout()
    return _finish(fig, out_path)


def significant_pairs_text(
    matrix: pd.DataFrame,
    *,
    test: str = "wilcoxon",
    higher_is_better: bool = True,
    alpha: float = 0.05,
    metric_name: str = "",
) -> str:
    """Plain-text list of the method pairs whose performance differs
    significantly, Holm-corrected over all pairs. ``test`` selects the pairwise
    test: ``"wilcoxon"`` (signed-rank, non-parametric -- default) or ``"ttest"``
    (paired Student's t-test). Printed as its own section so the W/L matrix
    stays uncluttered."""
    runner = ttest_holm_pairwise if test == "ttest" else wilcoxon_holm_pairwise
    label = "paired t-test" if test == "ttest" else "Wilcoxon signed-rank"
    tab = runner(matrix, higher_is_better=higher_is_better, alpha=alpha)
    sig = tab[tab["significant"]].copy()
    nm = _pretty_metric(metric_name) or "metric"
    lines = [f"Significant pairwise differences in {nm} "
             f"({label}, Holm-corrected, alpha = {alpha}):",
             f"  {int(sig.shape[0])} of {len(tab)} pairs significant.", ""]
    if sig.empty:
        lines.append("  (none reach significance after Holm correction)")
    else:
        sign = 1 if higher_is_better else -1
        for r in sig.sort_values("p_holm").itertuples():
            better = r.method_1 if sign * (r.wins - r.losses) > 0 else r.method_2
            worse = r.method_2 if better == r.method_1 else r.method_1
            lines.append(f"  {better}  >  {worse}    "
                         f"(W/L {r.wins}/{r.losses}, p_holm = {r.p_holm:.4f})")
    text = "\n".join(lines)
    print(text)
    return text


def statistical_report(
    df: pd.DataFrame,
    per_fold_df: pd.DataFrame,
    *,
    metric: str,
    task_name: str = "",
    higher_is_better: bool = True,
    focus: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> str:
    """One copy-pasteable plain-text report of EVERY test in the statistical
    section: PAMA, average ranks, Friedman/Iman-Davenport, Friedman Aligned
    Ranks, Quade, the Nemenyi critical difference, the Holm-corrected Wilcoxon
    AND paired-t significant pairs, the García & Herrera all-pairwise adjusted
    p-values, and the Bayesian signed-rank verdicts for the focus methods. Built
    to be pasted into another tool/LLM for interpretation."""
    matrix = metric_matrix(df, metric)
    k, N = matrix.shape[1], matrix.shape[0]
    pm = _pretty_metric(metric)
    direction = "higher is better" if higher_is_better else "lower is better"
    out: List[str] = []
    rule = "=" * 78
    out += [rule,
            f"STATISTICAL SUMMARY — {task_name} {pm}  "
            f"({N} datasets x {k} methods; {direction}; alpha = {alpha})",
            rule]

    try:
        pama = pama_fold_level(per_fold_df, metric, higher_is_better=higher_is_better)
        out.append("\n[1] PAMA — probability of achieving the maximal score "
                   "(% of (dataset, fold) observations where the method is the single best):")
        for m, r in pama.head(15).iterrows():
            out.append(f"      {m:20s} {r['PAMA_%']:5.1f}%   ({int(r['wins'])}/{int(r['n_folds'])})")
    except Exception as exc:  # noqa: BLE001
        out.append(f"\n[1] PAMA — unavailable ({exc})")

    ranks = average_ranks(matrix, higher_is_better=higher_is_better)
    out.append("\n[2] Average ranks across datasets (1 = best, lower is better):")
    out += [f"      {m:20s} {v:5.2f}" for m, v in ranks.items()]

    f = friedman_test(matrix, higher_is_better=higher_is_better)
    verdict = "REJECT H0 -> methods differ" if f["p_iman_davenport"] < alpha else "cannot reject H0"
    out.append(f"\n[3] Friedman omnibus (Iman-Davenport): F_F = {f['iman_davenport_F']:.3f}, "
               f"p = {f['p_iman_davenport']:.3g}  ->  {verdict}.")
    fa = friedman_aligned_ranks_test(matrix, higher_is_better=higher_is_better)
    qu = quade_test(matrix, higher_is_better=higher_is_better)
    out.append(f"[4] More powerful omnibus alternatives: Friedman Aligned Ranks "
               f"chi2 = {fa['aligned_ranks_chi2']:.2f}, p = {fa['p']:.3g};  "
               f"Quade F = {qu['quade_F']:.3f}, p = {qu['p']:.3g}.")
    cd = nemenyi_cd(k, N, alpha)
    out.append(f"[5] Nemenyi critical difference (alpha = {alpha}): CD = {cd:.3f} "
               f"(two methods differ significantly iff their mean ranks differ by more than CD).")

    sgn = 1 if higher_is_better else -1
    for tag, label, runner in (("6a", "Wilcoxon signed-rank", wilcoxon_holm_pairwise),
                               ("6b", "paired Student t-test", ttest_holm_pairwise)):
        tab = runner(matrix, higher_is_better=higher_is_better, alpha=alpha)
        sig = tab[tab["significant"]]
        out.append(f"\n[{tag}] Significant pairwise differences — {label} "
                   f"(Holm-corrected, alpha = {alpha}): {len(sig)} of {len(tab)} pairs.")
        for r in sig.sort_values("p_holm").itertuples():
            better = r.method_1 if sgn * (r.wins - r.losses) > 0 else r.method_2
            worse = r.method_2 if better == r.method_1 else r.method_1
            out.append(f"      {better:20s} > {worse:20s} "
                       f"(W/L {r.wins}/{r.losses}, p_holm = {r.p_holm:.4f})")

    try:
        apv = pairwise_apv_table(matrix, higher_is_better=higher_is_better)
        out.append("\n[7] All-pairwise adjusted p-values (García & Herrera) — "
                   "significant pairs per procedure:")
        for col in ("nemenyi", "holm", "shaffer", "bergmann_hommel"):
            if col in apv.columns and apv[col].notna().any():
                out.append(f"      {col:18s}: {int((apv[col] < alpha).sum())} / {len(apv)} significant")
    except Exception as exc:  # noqa: BLE001
        out.append(f"\n[7] All-pairwise APVs — unavailable ({exc})")

    foc = [m for m in (focus or []) if m in matrix.columns]
    if len(foc) >= 2:
        out.append(f"\n[8] Bayesian signed-rank test (ROPE = +/-0.01) for focus methods {foc}:")
        for x, y in itertools.combinations(foc, 2):
            r = bayesian_signed_rank_test(matrix[x], matrix[y], rope=0.01)
            out.append(f"      {x:14s} vs {y:14s}: P({x} better) = {r['P_right']:.3f}, "
                       f"P(equivalent) = {r['P_rope']:.3f}, P({y} better) = {r['P_left']:.3f}")

    out.append(rule)
    text = "\n".join(out)
    print(text)
    return text


def _finish(fig, out_path: Optional[Path]):
    saved = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=200)   # crisp on disk
        saved = out_path
    try:
        from src.visualizations.data_exploration import _display_inline
        _display_inline(fig, dpi=96)                          # small inline PNG
    except Exception:
        pass
    plt.close(fig)
    return saved


__all__ = [
    "metric_matrix", "average_ranks",
    "friedman_test", "friedman_aligned_ranks_test", "quade_test",
    "nemenyi_cd", "bonferroni_dunn_cd",
    "wilcoxon_signed_rank", "sign_test",
    "bayesian_sign_test", "bayesian_signed_rank_test",
    "control_apv_table", "pairwise_apv_table", "format_apv_table",
    "wilcoxon_holm_pairwise", "ttest_holm_pairwise", "significant_pairs_text",
    "statistical_report",
    "win_loss_tie", "wlt_summary",
    "percent_of_max", "pama_fold_level",
    "plot_cd_diagram", "plot_significance_matrix",
    "plot_win_loss_matrix", "plot_pama_bars",
    "plot_percent_of_max_bars", "plot_wilcoxon_wl_matrix",
]
