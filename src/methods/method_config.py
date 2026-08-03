"""TabPFNCredit method configuration -- thin layer over TALENT's registry.

A thin layer over TALENT's own `MethodSpec` registry and `build_args` API,
holding only what is **wrapper-specific**:

* :class:`PreprocessingConfig`: TabPFNCredit policy defaults that differ from
  TALENT's stock defaults (e.g. ``cat_policy='ohe'`` for methods that
  forbid ``indices`` -- credit-risk categoricals are never ordinal).
* :func:`derive_method_set`: build any subset (``DEEP_METHODS``,
  ``GPU_METHODS``, ``FOUNDATION_METHODS``, ...) by filtering TALENT's
  registry. No manual maintenance, no risk of desync.
* A handful of named module-level constants that the rest of the codebase
  imports today; they are *computed* from TALENT at import time so they
  stay in sync automatically.

Drop-in compatibility
---------------------
The public names ``DEEP_METHODS``, ``CLASSICAL_METHODS``, ``GPU_METHODS``,
``LOGIT_METHODS``, ``PROBABILITY_METHODS``, ``CLASS_LABEL_METHODS``,
``NO_HPO_METHODS``, ``FOUNDATION_METHODS``, ``METHOD_ROW_LIMITS``,
``METHOD_TEST_VAL_LIMITS``, ``REQUIRES_NO_NORMALIZATION``,
``REQUIRES_STANDARD_NORMALIZATION``, ``REQUIRES_NO_NUM_ENCODING``,
``REQUIRES_CAT_INDICES``, ``REQUIRES_CAT_OHE``, ``REQUIRES_CAT_TABR_OHE``,
``FORBIDS_CAT_INDICES``, ``TABPFN_VARIANTS``, and
:func:`apply_preprocessing_policies` / :func:`apply_method_row_limit`
remain importable -- everywhere the wrapper used them, the call site is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dataclass_replace
from typing import Any, Optional, Set

# Re-export TALENT's enums so legacy `from src.methods.method_config import
# OutputType` keeps working.
from TALENT.model.method_registry import (
    Architecture,
    Hardware,
    METHOD_REGISTRY,
    OutputType,
)

# Canonical figure labels (TALENT-free module). Re-exported here so the standard
# display name for a registry key is discoverable next to the registry itself.
from src.methods.method_names import METHOD_DISPLAY_NAMES, display_name

MethodSet = Set[str]


# ============================================================================
#  Wrapper-policy defaults
# ============================================================================

@dataclass(frozen=True)
class PreprocessingConfig:
    """TabPFNCredit defaults that differ from TALENT's stock defaults.

    Rationale: credit-risk categoricals are never ordinal, so the wrapper
    opts for one-hot encoding by default whenever a method does **not**
    require ``indices``. NaN policies favour deterministic behaviour over
    training-set statistics.
    """

    cat_policy_for_ohe_methods: str = "ohe"
    cat_policy_for_indices_methods: str = "indices"
    cat_policy_for_tabr_methods: str = "tabr_ohe"
    normalization_default: str = "standard"
    num_policy_default: str = "none"
    num_nan_policy_default: str = "median"
    cat_nan_policy_default: str = "new"


DEFAULTS = PreprocessingConfig()


# ============================================================================
#  Registry-derived sets (computed once at import time)
# ============================================================================

def derive_method_set(**predicate: Any) -> MethodSet:
    """Return the set of method names matching the given filter.

    Keys map directly onto :class:`MethodSpec` fields; values are compared
    with equality (or membership if the value is a tuple/list/set).

    Examples
    --------
    >>> derive_method_set(architecture=Architecture.DEEP)
    {'mlp', 'resnet', 'tabpfn_v3', ...}
    """
    out: MethodSet = set()
    for spec in METHOD_REGISTRY.values():
        ok = True
        for key, value in predicate.items():
            field_value = getattr(spec, key, None)
            if isinstance(value, (tuple, list, set)):
                if field_value not in value:
                    ok = False
                    break
            else:
                if field_value != value:
                    ok = False
                    break
        if ok:
            out.add(spec.name)
    return out


# Architecture / hardware partitions
DEEP_METHODS: MethodSet = derive_method_set(architecture=Architecture.DEEP)
CLASSICAL_METHODS: MethodSet = derive_method_set(architecture=Architecture.CLASSICAL)
GPU_METHODS: MethodSet = derive_method_set(hardware=Hardware.GPU)
CPU_METHODS: MethodSet = derive_method_set(hardware=Hardware.CPU)

# Output-type partitions
LOGIT_METHODS: MethodSet = derive_method_set(output_type=OutputType.LOGITS)
PROBABILITY_METHODS: MethodSet = derive_method_set(output_type=OutputType.PROBABILITIES)
CLASS_LABEL_METHODS: MethodSet = derive_method_set(output_type=OutputType.CLASS_LABELS)

# ----------------------------------------------------------------------------
#  Registry correction: methods TALENT marks tunable but cannot be tuned
# ----------------------------------------------------------------------------
#
# TALENT's registry sets supports_hpo=True for NCM and NaiveBayes, but their
# implementation opens with ``assert(not args.tune)`` -- so every ``__HPO`` point
# for them fails with a bare AssertionError. It is not a run that needs retrying:
# the point can never succeed, and the planner would keep requesting it forever
# ("missing" on every gap scan).
#
# Corrected here rather than in the fork so it deploys with a plain git pull, and
# in the REGISTRY rather than only in the derived sets so every consumer agrees
# (the resubmit planner, the SLURM generator and spec.validate_args all read it).
_HPO_UNSUPPORTED: MethodSet = {"NCM", "NaiveBayes"}
for _name in sorted(_HPO_UNSUPPORTED):
    _spec = METHOD_REGISTRY.get(_name)
    if _spec is not None and _spec.supports_hpo:
        METHOD_REGISTRY[_name] = _dataclass_replace(_spec, supports_hpo=False)

# ----------------------------------------------------------------------------
#  Capacity row caps: methods too slow to train on the largest datasets
# ----------------------------------------------------------------------------
#
# A TRAINING-side ceiling, applied through the same path as TALENT's own
# ``train_row_limit`` (see method_runner: it is passed to fold construction, so
# validation and TEST folds keep every row). Every method is therefore still
# scored on identical observations -- the property METHOD_TEST_VAL_LIMITS exists
# to protect -- while the fit stays affordable.
#
# ``tangos`` needs this. At 20 HPO trials it is 21 fits per fold; on Hackerearth
# (340,753 training rows) one fold had not finished in 37 h, so a 5-fold tuned run
# is ~150-200 h and no wall-time extension reaches it. It previously "worked" only
# under a 100,000-row WHOLE-DATASET cap, which also shrank the test folds to
# 20,000 against 106,486 for every other method -- exactly the comparability bug
# this replaces. 50,000 training rows lands near 27 h; 100,000 would be ~54 h and
# miss the wall again.
#
# A cap is a ceiling, so it needs no dataset threshold: it binds only where the
# training split is already larger (above ~78k rows, since train = 64% of N). It
# binds on 8 PD datasets and no LGD dataset, and must be disclosed in the paper
# next to the in-context caps.
_CAPACITY_ROW_CAPS: dict = {"tangos": 50_000}
for _name, _cap in sorted(_CAPACITY_ROW_CAPS.items()):
    _spec = METHOD_REGISTRY.get(_name)
    if _spec is not None and _spec.train_row_limit != _cap:
        METHOD_REGISTRY[_name] = _dataclass_replace(_spec, train_row_limit=_cap)

# HPO support
NO_HPO_METHODS: MethodSet = derive_method_set(supports_hpo=False)
HPO_METHODS: MethodSet = derive_method_set(supports_hpo=True)

# Foundation models = in-context learners with bounded context, unified
# with the well-known foundation family names so users can still filter
# by "is it a foundation model".
_FOUNDATION_NAMES = {
    "tabpfn", "tabpfn_v2", "tabpfn_v2_5", "tabpfn_v3", "tabpfn_real",
    "tabicl", "tabicl_v2", "mitra", "limix", "tabdpt", "tabfm", "hyperfast", "tabptm",
}
# A train_row_limit normally means "in-context learner with a bounded context",
# hence the inference below. Methods in _CAPACITY_ROW_CAPS carry one purely
# because they are slow, so they must be excluded: without this, capping tangos
# would relabel it a foundation model in every figure, legend and family-level
# statistic.
FOUNDATION_METHODS: MethodSet = {
    name for name in METHOD_REGISTRY
    if name in _FOUNDATION_NAMES
    or (METHOD_REGISTRY[name].train_row_limit is not None
        and name not in _CAPACITY_ROW_CAPS)
}

# The TabPFN family, derived from the canonical name prefix.
TABPFN_VARIANTS: MethodSet = {n for n in METHOD_REGISTRY if n.startswith("tabpfn")}

# Preprocessing requirements -- queried from MethodSpec
REQUIRES_CAT_INDICES: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.cat_policy is not None and tuple(spec.cat_policy) == ("indices",)
}
REQUIRES_CAT_OHE: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.cat_policy is not None and tuple(spec.cat_policy) == ("ohe",)
}
REQUIRES_CAT_TABR_OHE: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.cat_policy is not None and tuple(spec.cat_policy) == ("tabr_ohe",)
}

# Fix the historical bug: FORBIDS_CAT_INDICES is *any* method whose
# allowed `cat_policy` set excludes 'indices'. The old definition aliased
# this to REQUIRES_CAT_OHE, silently omitting tabr_ohe methods.
FORBIDS_CAT_INDICES: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.cat_policy is not None and "indices" not in spec.cat_policy
}

REQUIRES_NO_NORMALIZATION: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items() if spec.normalization == "none"
}
REQUIRES_STANDARD_NORMALIZATION: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items() if spec.normalization == "standard"
}
REQUIRES_NO_NUM_ENCODING: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items() if spec.num_policy == "none"
}

# Row-limit dictionaries: pulled directly from `MethodSpec.train_row_limit`.
METHOD_ROW_LIMITS: dict = {
    name: spec.train_row_limit
    for name, spec in METHOD_REGISTRY.items()
    if spec.train_row_limit is not None
}

# Inference-OOM escape hatch ONLY -- NOT a fairness/normalisation device.
#
# The val/test cap exists solely to prevent inference OOM (cross-attention is
# O(N_train * N_test) on a *non-batching* path; see _apply_val_test_caps in
# method_runner.py). It is decoupled from the per-method TRAIN cap, which is a
# separate data-regime concern (METHOD_ROW_LIMITS / TALENT `general.sample_size`).
#
# DEFAULT POLICY: empty -> every method is scored on the FULL test/val fold, so
# all methods are evaluated on the identical test set (required for cross-method
# comparability in the benchmark). Add an entry HERE *only* for a method that is
# actually observed to OOM at inference on the largest folds (~60k rows), and set
# the value to a generous safe cap (the most the GPU reliably handles, e.g. the
# in-context family on an 80GB H100). Any method that ends up capped is scored on
# fewer test points than the rest on that dataset and MUST be disclosed in the
# paper's methods section.
#
# Do NOT re-derive this from METHOD_ROW_LIMITS: that aliased the inference-OOM
# cap to whichever methods happened to carry a registry train_row_limit (only
# the TabPFN family), which is an accident of mechanism, not a design choice.
METHOD_TEST_VAL_LIMITS: dict[str, int] = {}


# ============================================================================
#  Args helpers
# ============================================================================

_MISSING_SENTINELS = {None, "", "none", "None", "NONE", "default"}


def _is_missing(value: Any) -> bool:
    """Return True iff the user did not specify a value."""
    try:
        return value in _MISSING_SENTINELS
    except TypeError:
        return False


def apply_preprocessing_policies(args: Any, method: str, user_specified: Optional[dict] = None) -> None:
    """Fill in TabPFNCredit's preferred defaults on ``args`` before calling TALENT.

    TALENT validates and enforces method preprocessing constraints itself
    (via :meth:`MethodSpec.validate_args`); this helper is reduced to
    *default-filling*. It does not raise on conflicts -- if the user set
    something incompatible, ``TALENT.build_args`` will raise a readable
    ``ValueError``.
    """
    user_specified = user_specified or {}
    spec = METHOD_REGISTRY[method]

    def _set_if_missing(attr: str, value: Any) -> None:
        if user_specified.get(attr, False):
            return
        current = getattr(args, attr, None)
        if _is_missing(current):
            setattr(args, attr, value)

    # cat_policy
    if spec.cat_policy is not None:
        _set_if_missing("cat_policy", spec.cat_policy[0])
    else:
        _set_if_missing("cat_policy", DEFAULTS.cat_policy_for_ohe_methods)

    # normalization
    if spec.normalization is not None:
        _set_if_missing("normalization", spec.normalization)
    else:
        _set_if_missing("normalization", DEFAULTS.normalization_default)

    # num_policy
    if spec.num_policy is not None:
        _set_if_missing("num_policy", spec.num_policy)
    else:
        _set_if_missing("num_policy", DEFAULTS.num_policy_default)

    # NaN policies -- not constrained by TALENT, wrapper picks safe defaults
    _set_if_missing("num_nan_policy", DEFAULTS.num_nan_policy_default)
    _set_if_missing("cat_nan_policy", DEFAULTS.cat_nan_policy_default)


def apply_method_row_limit(args: Any, method: str) -> Optional[int]:
    """Return the training-set row limit for ``method`` (or None)."""
    return METHOD_ROW_LIMITS.get(method)


# ============================================================================
#  Validation
# ============================================================================

def _validate_registry_invariants() -> None:
    """Sanity-check the derived sets at import time."""
    assert DEEP_METHODS.isdisjoint(CLASSICAL_METHODS), (
        f"Method appears in both DEEP and CLASSICAL: "
        f"{DEEP_METHODS & CLASSICAL_METHODS}"
    )
    assert FOUNDATION_METHODS <= (DEEP_METHODS | CLASSICAL_METHODS), (
        "All foundation methods must be registered as deep or classical"
    )
    assert "tabpfn_v3" in TABPFN_VARIANTS, "TabPFN v3 missing from family"
    assert "tabicl_v2" in METHOD_REGISTRY, "TabICL v2 missing from registry"
    assert "tabpfn_v2_5" in METHOD_REGISTRY, "TabPFN v2.5 missing from registry"
    assert "tabdpt" in METHOD_REGISTRY, "TabDPT missing from registry"


_validate_registry_invariants()


__all__ = [
    "Architecture", "Hardware", "OutputType",
    "PreprocessingConfig", "DEFAULTS",
    "DEEP_METHODS", "CLASSICAL_METHODS",
    "GPU_METHODS", "CPU_METHODS",
    "LOGIT_METHODS", "PROBABILITY_METHODS", "CLASS_LABEL_METHODS",
    "NO_HPO_METHODS", "HPO_METHODS",
    "FOUNDATION_METHODS", "TABPFN_VARIANTS",
    "METHOD_DISPLAY_NAMES", "display_name",
    "REQUIRES_CAT_INDICES", "REQUIRES_CAT_OHE", "REQUIRES_CAT_TABR_OHE",
    "FORBIDS_CAT_INDICES",
    "REQUIRES_NO_NORMALIZATION", "REQUIRES_STANDARD_NORMALIZATION",
    "REQUIRES_NO_NUM_ENCODING",
    "METHOD_ROW_LIMITS", "METHOD_TEST_VAL_LIMITS",
    "derive_method_set",
    "apply_preprocessing_policies", "apply_method_row_limit",
    "_is_missing",
]
