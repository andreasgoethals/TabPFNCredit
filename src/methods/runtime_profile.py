"""Per-method runtime profile -- expected wall-clock and SLURM partition pick.

Why this exists
---------------
The 55 TALENT methods span a huge runtime spectrum. A dummy classifier
needs ~1 second per fold; an HPO sweep of GRANDE or TROMPT can need
hours. Submitting one SLURM array task per (dataset, method) treats them
all the same -- which wastes scheduler overhead for the cheap methods
and under-walltimes the expensive ones.

This module groups methods into **tiers** keyed by an estimate of
"seconds per (dataset, fold)" on a single VSC GPU node. The SLURM
generator in :mod:`src.utils.slurm_generator` uses these tiers to:

* Pick the partition (CPU vs P100 vs A100 vs H100) and the per-task
  walltime.
* Decide how many (dataset, method) cells to **pack** into a single
  array slot -- cheap methods get bundled to reduce scheduler load.

Tiers
-----
* ``FAST``      -- < 30 s / fold (most classical CPU methods)
* ``MEDIUM``    -- 30 s -- 5 min / fold (small deep nets, gradient boosting)
* ``SLOW``      -- 5 -- 60 min / fold (large deep transformers, HPO trials)
* ``FOUNDATION``-- 30 s -- 60 min / fold (TabPFN, TabICL, Mitra, TabDPT ...)
  These need a GPU and substantial RAM but rarely need HPO.

The numbers are deliberately conservative -- they're upper bounds, not
medians. The runtime per fold also depends on dataset size; the
profile assumes "medium" credit datasets (1k-100k rows). Override
explicitly via :func:`set_profile_override` if you know a specific
method needs more (or less).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

# ============================================================================
#  Tier definitions
# ============================================================================

class Tier(str, Enum):
    """Runtime tier of a method. Order matters: SLOW > MEDIUM > FAST."""

    FAST = "fast"          # < 30 s / fold (most classical CPU)
    MEDIUM = "medium"      # 30 s - 5 min / fold (small deep, gradient boosting)
    SLOW = "slow"          # 5 - 60 min / fold (large deep, HPO trials)
    FOUNDATION = "foundation"  # GPU foundation models, runtime varies


@dataclass(frozen=True)
class Profile:
    """Runtime + hardware preference for a method."""

    tier: Tier
    seconds_per_fold_estimate: int
    """Conservative upper bound for one fold on a credit dataset (~10k rows)."""
    prefers_gpu: bool
    """Whether GPU placement is preferred (vs CPU fallback)."""
    needs_foundation_gpu: bool = False
    """If True, prefers the foundation partition (A100/H100) over P100."""


# ============================================================================
#  The profile table
# ============================================================================
#
# These numbers are conservative and based on empirical observation of
# Experiment 1 in the original repo. They should be overridden per-experiment
# if a particular (method, dataset_size) combination needs more time.

_PROFILES: Dict[str, Profile] = {
    # ---- Cheap classical baselines ----
    "dummy":           Profile(Tier.FAST, 5,   prefers_gpu=False),
    "LogReg":          Profile(Tier.FAST, 15,  prefers_gpu=False),
    "LinearRegression":Profile(Tier.FAST, 10,  prefers_gpu=False),
    "NaiveBayes":      Profile(Tier.FAST, 10,  prefers_gpu=False),
    "NCM":             Profile(Tier.FAST, 10,  prefers_gpu=False),
    "knn":             Profile(Tier.FAST, 30,  prefers_gpu=False),
    "svm":             Profile(Tier.MEDIUM, 90, prefers_gpu=False),

    # ---- Gradient-boosting (CPU; can be GPU-accelerated) ----
    "xgboost":         Profile(Tier.MEDIUM, 120, prefers_gpu=False),
    "lightgbm":        Profile(Tier.MEDIUM, 120, prefers_gpu=False),
    "catboost":        Profile(Tier.MEDIUM, 180, prefers_gpu=False),
    "RandomForest":    Profile(Tier.MEDIUM, 120, prefers_gpu=False),

    # ---- Kernel methods (GPU-accelerated) ----
    "rfm":             Profile(Tier.MEDIUM, 180, prefers_gpu=True),
    "xrfm":            Profile(Tier.MEDIUM, 240, prefers_gpu=True),

    # ---- Basic neural ----
    "mlp":             Profile(Tier.MEDIUM, 180, prefers_gpu=True),
    "resnet":          Profile(Tier.MEDIUM, 240, prefers_gpu=True),
    "snn":             Profile(Tier.MEDIUM, 180, prefers_gpu=True),
    "realmlp":         Profile(Tier.MEDIUM, 240, prefers_gpu=True),
    "mlp_plr":         Profile(Tier.MEDIUM, 240, prefers_gpu=True),

    # ---- Transformer-based / token-based ----
    "autoint":         Profile(Tier.SLOW, 600, prefers_gpu=True),
    "saint":           Profile(Tier.SLOW, 900, prefers_gpu=True),
    "ftt":             Profile(Tier.SLOW, 600, prefers_gpu=True),
    "tabtransformer":  Profile(Tier.SLOW, 600, prefers_gpu=True),
    "excelformer":     Profile(Tier.SLOW, 600, prefers_gpu=True),
    "amformer":        Profile(Tier.SLOW, 900, prefers_gpu=True),
    "trompt":          Profile(Tier.SLOW, 1800, prefers_gpu=True),
    "t2gformer":       Profile(Tier.SLOW, 900, prefers_gpu=True),

    # ---- Tree-mimic deep nets ----
    "tabnet":          Profile(Tier.SLOW, 600, prefers_gpu=True),
    "node":            Profile(Tier.SLOW, 900, prefers_gpu=True),
    "grownet":         Profile(Tier.SLOW, 1200, prefers_gpu=True),
    "dcn2":            Profile(Tier.MEDIUM, 360, prefers_gpu=True),
    "tabm":            Profile(Tier.SLOW, 900, prefers_gpu=True),
    "grande":          Profile(Tier.SLOW, 1800, prefers_gpu=True),

    # ---- Retrieval / NCA-style ----
    "tabr":            Profile(Tier.SLOW, 900, prefers_gpu=True),
    "modernNCA":       Profile(Tier.SLOW, 900, prefers_gpu=True),
    "dnnr":            Profile(Tier.MEDIUM, 300, prefers_gpu=True),

    # ---- Regularization-based / capsule / prototype ----
    "tangos":          Profile(Tier.SLOW, 600, prefers_gpu=True),
    "switchtab":       Profile(Tier.SLOW, 600, prefers_gpu=True),
    "ptarl":           Profile(Tier.SLOW, 600, prefers_gpu=True),
    "bishop":          Profile(Tier.SLOW, 600, prefers_gpu=True),
    "protogate":       Profile(Tier.SLOW, 600, prefers_gpu=True),
    "tabcaps":         Profile(Tier.MEDIUM, 300, prefers_gpu=True),
    "tabautopnpnet":   Profile(Tier.MEDIUM, 360, prefers_gpu=True),
    "danets":          Profile(Tier.MEDIUM, 360, prefers_gpu=True),

    # ---- Foundation models (in-context learners, no HPO) ----
    "tabpfn":          Profile(Tier.FOUNDATION, 60,   prefers_gpu=True, needs_foundation_gpu=True),
    "tabpfn_v2":       Profile(Tier.FOUNDATION, 180,  prefers_gpu=True, needs_foundation_gpu=True),
    "tabpfn_v2_5":     Profile(Tier.FOUNDATION, 600,  prefers_gpu=True, needs_foundation_gpu=True),
    "tabpfn_v3":       Profile(Tier.FOUNDATION, 1800, prefers_gpu=True, needs_foundation_gpu=True),
    "tabpfn_real":     Profile(Tier.FOUNDATION, 180,  prefers_gpu=True, needs_foundation_gpu=True),
    "tabicl":          Profile(Tier.FOUNDATION, 300,  prefers_gpu=True, needs_foundation_gpu=True),
    "tabicl_v2":       Profile(Tier.FOUNDATION, 900,  prefers_gpu=True, needs_foundation_gpu=True),
    "tabdpt":          Profile(Tier.FOUNDATION, 600,  prefers_gpu=True, needs_foundation_gpu=True),
    # TabFM is by far the most expensive method here: 32 ensemble members, and
    # its cost is dominated by INFERENCE (each member's sequence is its context
    # plus the whole evaluation split), so it scales with the test-split size.
    # With one dataset per array task (TABPFN_MAX_CELLS_PER_SLOT=1) the slot
    # walltime comes from a SINGLE dataset's estimate, so this has to cover the
    # largest dataset. Measured on H100-80GB at a 10k context (job 61519948),
    # seconds per *single* model fit + evaluation:
    #     taiwan_creditcard  6.0k test rows,  23 feat ->  194 s
    #     heloc (LGD)       11.6k,             8      ->  147 s
    #     cobranded         16.0k,            47      ->  434 s
    #     loan_default      21.1k,           500*     ->  697 s
    #     vehicle_loan      46.6k,            35      -> 1401 s
    #     (*capped by max_num_features)
    # Extrapolating to the largest split (hackerearth, 106k x 35) gives ~3200 s
    # per fold. 6000 s keeps ~2x headroom on that -- a ~11 h request rather than
    # the 25 h the old 14000 asked for. The old number was calibrated against
    # runs that unknowingly refit 15x per fold (TALENT's packaged seed_num=15,
    # now pinned to 1 in CONFIG_EXPERIMENT.yaml), so it over-requested by ~15x.
    # Overestimating only costs queue priority: the job exits when done and
    # skip-if-done makes any overrun resumable.
    "tabfm":           Profile(Tier.FOUNDATION, 6000, prefers_gpu=True, needs_foundation_gpu=True),
    "mitra":           Profile(Tier.FOUNDATION, 600,  prefers_gpu=True, needs_foundation_gpu=True),
    "limix":           Profile(Tier.FOUNDATION, 1800, prefers_gpu=True, needs_foundation_gpu=True),
    "hyperfast":       Profile(Tier.FOUNDATION, 60,   prefers_gpu=True),
    "tabptm":          Profile(Tier.FOUNDATION, 180,  prefers_gpu=True),
}

# ============================================================================
#  Validation
# ============================================================================

_UNKNOWN_DEFAULT = Profile(Tier.MEDIUM, 300, prefers_gpu=True)


def _validate_coverage() -> None:
    """Ensure every method in the registry has a profile, and vice-versa."""
    from TALENT.model.method_registry import METHOD_REGISTRY

    missing = sorted(set(METHOD_REGISTRY) - set(_PROFILES))
    extra = sorted(set(_PROFILES) - set(METHOD_REGISTRY))
    if missing:
        # Don't crash -- just default to MEDIUM and warn at first lookup.
        import logging
        logging.getLogger(__name__).warning(
            "runtime_profile: %d methods have no profile entry "
            "(will default to MEDIUM/300s): %s",
            len(missing), missing,
        )
    if extra:
        import logging
        logging.getLogger(__name__).debug(
            "runtime_profile: %d obsolete profile entries (not in TALENT registry): %s",
            len(extra), extra,
        )


_validate_coverage()


# ============================================================================
#  Public API
# ============================================================================

def get_profile(method: str) -> Profile:
    """Return the :class:`Profile` for ``method``. Defaults to MEDIUM if unknown."""
    return _PROFILES.get(method, _UNKNOWN_DEFAULT)


def set_profile_override(method: str, profile: Profile) -> None:
    """Override the profile for ``method`` at runtime (e.g. per-experiment)."""
    _PROFILES[method] = profile


def tier_of(method: str) -> Tier:
    return get_profile(method).tier


def estimate_walltime_seconds(method: str, *, n_folds: int = 1, n_sweep_points: int = 1) -> int:
    """Pessimistic walltime estimate for one (dataset, method) cell.

    Adds 30% overhead for data loading / SLURM startup, plus a 60-second
    base floor so trivial methods still get a reasonable wall budget.
    """
    p = get_profile(method)
    base = int(p.seconds_per_fold_estimate * n_folds * n_sweep_points)
    return max(60, int(base * 1.3) + 60)


# Reference training-set size the per-fold budgets are calibrated against.
# A learning-curve point at ``row_limit`` rows is scaled by row_limit/REFERENCE
# (clamped) so small-row points pack as much cheaper than full-size ones.
ROW_REFERENCE = 30_000


def estimate_point_seconds(
    method: str,
    *,
    n_folds: int = 1,
    row_limit: Optional[int] = None,
    tune: bool = False,
    n_trials: int = 1,
) -> int:
    """Per-SWEEP-POINT walltime estimate, used to BALANCE points across SLURM slots.

    Unlike :func:`estimate_walltime_seconds` (one whole cell), this estimates a
    single sweep point so the generator can shard a cell's many points across
    array tasks. It scales the per-fold budget by ``row_limit / ROW_REFERENCE``
    (Experiment 2 -- a 100-row point is far cheaper than a 30 000-row one) and
    multiplies by the HPO trial count when tuning (Experiment 1). It need not be
    exact -- only monotone and relative -- since it is used purely for
    load-balancing; the per-slot walltime request is capped at the partition's
    hard limit and skip-if-done makes any overrun resumable.
    """
    p = get_profile(method)
    per_fold = float(p.seconds_per_fold_estimate)
    if row_limit:
        per_fold *= max(0.05, min(1.0, row_limit / float(ROW_REFERENCE)))
    trials = max(1, n_trials) if tune else 1
    base = per_fold * n_folds * trials
    return max(5, int(base * 1.3) + 30)


def methods_by_tier(tier: Tier, *, hardware: Optional[str] = None) -> set:
    """Return all method names with the given tier (and optional hardware)."""
    names = {m for m, p in _PROFILES.items() if p.tier == tier}
    if hardware == "cpu":
        names &= {m for m, p in _PROFILES.items() if not p.prefers_gpu}
    elif hardware == "gpu":
        names &= {m for m, p in _PROFILES.items() if p.prefers_gpu}
    return names


# ============================================================================
#  Partition resolution -- maps (method, hardware) -> SLURM partition name
# ============================================================================

def recommended_partition(method: str, *, prefer_h100: bool = True) -> str:
    """Return the partition string for ``method`` on VSC.

    Mapping
    -------
    * CPU-only methods                  -> ``"batch"`` (Genius / wICE)
    * GPU methods, not foundation       -> ``"gpu_p100"`` on Genius
    * Foundation models                 -> ``"gpu_h100"`` on wICE (or A100 if H100 down)
    """
    profile = get_profile(method)
    if not profile.prefers_gpu:
        return "batch"
    if profile.needs_foundation_gpu:
        return "gpu_h100" if prefer_h100 else "gpu_a100"
    return "gpu_p100"


__all__ = [
    "Tier",
    "Profile",
    "get_profile",
    "set_profile_override",
    "tier_of",
    "estimate_walltime_seconds",
    "methods_by_tier",
    "recommended_partition",
]
