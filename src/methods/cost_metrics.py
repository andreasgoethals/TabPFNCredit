"""Cost-sensitive credit-risk metrics (B9).

Statistical metrics like F1 / AUC do not tell you whether a model would
make money. Credit decisions are inherently asymmetric: missing a default
(false negative) is much more expensive than mis-flagging a healthy
customer (false positive). This module reports metrics that scale with
the **monetary cost** of each prediction.

Three views:

* :func:`expected_loss` -- the optimal expected loss attainable by the
  model's score, given a fixed cost ratio. The denominator is the
  no-skill baseline (random predictions), so values <1 mean the model
  beats the baseline; >1 means it underperforms it.

* :func:`profit_curve` -- profit as a function of decision threshold,
  parametrised by per-prediction cost / benefit. Returns the optimal
  threshold (max profit) and a thresholds-array suitable for plotting.

* :func:`cost_sensitive_summary` -- one-shot dict containing both of the
  above plus the Hand H-measure when ``hmeasure`` is installed.

All functions accept ``y_true`` (0/1) and ``y_proba`` (positive-class
probability) -- the same shape TALENT's ``RunResult.predict_proba[:, 1]``
produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# ============================================================================
#  Default cost matrix
# ============================================================================
# Credit-risk convention: missing a default (FN) costs roughly the exposure;
# a false alarm (FP) costs only the foregone interest margin. Default ratio
# below is "miss is 5x more expensive than a false alarm" which is a common
# starting point and easy to override per-experiment.

@dataclass(frozen=True)
class CostMatrix:
    """Per-prediction cost of each confusion-matrix outcome.

    Values are *relative*; only ratios matter. Defaults:

    * ``cost_fn`` = 5.0  (missed default: the bank loses ~5 units)
    * ``cost_fp`` = 1.0  (false alarm: the bank loses 1 unit of revenue)
    * ``benefit_tn`` = 0.1 (correctly approving a healthy customer earns a bit)
    * ``benefit_tp`` = 0.0 (correctly rejecting a defaulter avoids loss but no upside)
    """
    cost_fn: float = 5.0
    cost_fp: float = 1.0
    benefit_tn: float = 0.1
    benefit_tp: float = 0.0


DEFAULT_COSTS = CostMatrix()


# ============================================================================
#  Expected loss (cost-normalised)
# ============================================================================

def expected_loss(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_ratio: float = 5.0,
) -> float:
    """Expected total cost under the optimal decision rule for ``y_proba``.

    Implements (Hand 2009) cost-weighted misclassification:

    .. math::
        L = \\frac{
            c \\cdot P(D=1) \\cdot \\sum I[\\hat y = 0, y = 1]
            + (1-c) \\cdot P(D=0) \\cdot \\sum I[\\hat y = 1, y = 0]
        }{
            c \\cdot P(D=1) + (1-c) \\cdot P(D=0)
        }

    where ``c = cost_ratio / (1 + cost_ratio)`` is the FN weight.

    Returns a value normalised to the random-prediction baseline; lower
    is better, < 1 means the model beats random.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    proba = np.asarray(y_proba).ravel()

    # Optimal threshold = c / (1 + c) where c is the cost ratio (Elkan 2001).
    # Here cost_ratio = cost_fn / cost_fp, so threshold = 1 / (1 + cost_ratio).
    threshold = 1.0 / (1.0 + cost_ratio)
    y_pred = (proba >= threshold).astype(int)

    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    total = len(y_true)

    weighted_cost = (cost_ratio * fn + 1.0 * fp) / total

    # Random-baseline: a Bernoulli classifier with rate p picks the
    # cost-minimising rate analytically; this is `min(prior, 1 - prior) * effective_cost`.
    prior = float((y_true == 1).mean())
    baseline = min(prior * cost_ratio, (1.0 - prior) * 1.0)
    if baseline == 0.0:
        return float("nan")
    return float(weighted_cost / baseline)


# ============================================================================
#  Profit curve
# ============================================================================

def profit_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    costs: Optional[CostMatrix] = None,
    n_points: int = 101,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Profit as a function of decision threshold.

    Returns
    -------
    thresholds : (n_points,) ndarray
        Threshold grid in ``[0, 1]``.
    profits : (n_points,) ndarray
        Total profit at each threshold (sum over the full population).
    optimal_threshold : float
        Threshold that maximises profit.
    optimal_profit : float
        Profit at the optimal threshold.

    Notes
    -----
    Profit at threshold ``t`` is::

        profit(t) = benefit_tp * TP + benefit_tn * TN
                  - cost_fn  * FN - cost_fp  * FP

    so the optimum is monotone in the cost matrix. Plot ``thresholds`` vs
    ``profits`` to see the operating-point landscape.
    """
    costs = costs or DEFAULT_COSTS
    y_true = np.asarray(y_true).astype(int).ravel()
    proba = np.asarray(y_proba).ravel()
    thresholds = np.linspace(0.0, 1.0, n_points)
    profits = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        y_pred = (proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        profits[i] = (
            costs.benefit_tp * tp + costs.benefit_tn * tn
            - costs.cost_fn * fn - costs.cost_fp * fp
        )
    best_idx = int(np.argmax(profits))
    return thresholds, profits, float(thresholds[best_idx]), float(profits[best_idx])


# ============================================================================
#  H-measure (Hand 2009) -- optional dependency
# ============================================================================

def _h_measure(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        from hmeasure import h_score  # type: ignore
        return float(h_score(y_true, y_proba))
    except ImportError:
        return float("nan")


# ============================================================================
#  One-shot summary
# ============================================================================

def cost_sensitive_summary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    cost_ratio: float = 5.0,
    costs: Optional[CostMatrix] = None,
) -> Dict[str, float]:
    """One-shot cost-sensitive metric dict for a (PD) fold.

    Returns
    -------
    dict
        ``Expected_Loss_Normalized``, ``Optimal_Profit_Threshold``,
        ``Optimal_Profit``, ``H_Measure`` (NaN if ``hmeasure`` not installed).
    """
    proba = np.asarray(y_proba)
    if proba.ndim == 2 and proba.shape[1] == 2:
        proba = proba[:, 1]
    _, _, opt_t, opt_p = profit_curve(y_true, proba, costs=costs)
    return {
        "Expected_Loss_Normalized": expected_loss(y_true, proba, cost_ratio),
        "Optimal_Profit_Threshold": opt_t,
        "Optimal_Profit": opt_p,
        "H_Measure": _h_measure(np.asarray(y_true).astype(int), proba),
    }


__all__ = [
    "CostMatrix", "DEFAULT_COSTS",
    "expected_loss", "profit_curve",
    "cost_sensitive_summary",
]
