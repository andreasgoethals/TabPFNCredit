"""Tests for the registry-derived sets in src.methods.method_config."""

from __future__ import annotations

import pytest

from src.methods.method_config import (
    CLASSICAL_METHODS,
    CPU_METHODS,
    DEEP_METHODS,
    FORBIDS_CAT_INDICES,
    FOUNDATION_METHODS,
    GPU_METHODS,
    METHOD_ROW_LIMITS,
    NO_HPO_METHODS,
    REQUIRES_CAT_INDICES,
    REQUIRES_CAT_OHE,
    REQUIRES_CAT_TABR_OHE,
    TABPFN_VARIANTS,
    derive_method_set,
)
from TALENT.model.method_registry import METHOD_REGISTRY


class TestRegistryConsistency:
    """The derived sets must stay in lock-step with TALENT's MethodSpec."""

    def test_deep_and_classical_are_disjoint(self):
        assert DEEP_METHODS.isdisjoint(CLASSICAL_METHODS)

    def test_every_method_appears_in_exactly_one_partition(self):
        all_methods = set(METHOD_REGISTRY.keys())
        assert DEEP_METHODS | CLASSICAL_METHODS == all_methods

    def test_gpu_and_cpu_are_disjoint(self):
        assert GPU_METHODS.isdisjoint(CPU_METHODS)

    def test_every_method_appears_in_exactly_one_hardware_set(self):
        all_methods = set(METHOD_REGISTRY.keys())
        assert GPU_METHODS | CPU_METHODS == all_methods


class TestNewMethodsRegistered:
    """Regression tests for v3 / v2.5 / TabICL v2 / TabDPT."""

    @pytest.mark.parametrize(
        "name", ["tabpfn_v3", "tabpfn_v2_5", "tabicl_v2", "tabdpt"]
    )
    def test_method_is_in_registry(self, name):
        assert name in METHOD_REGISTRY
        assert name in DEEP_METHODS

    def test_tabpfn_family_complete(self):
        assert "tabpfn" in TABPFN_VARIANTS
        assert "tabpfn_v2" in TABPFN_VARIANTS
        assert "tabpfn_v2_5" in TABPFN_VARIANTS
        assert "tabpfn_v3" in TABPFN_VARIANTS

    def test_new_methods_have_row_limits_or_none(self):
        # tabpfn v3 has a million-row limit, v2.5 has 50k, others (tabdpt) may be None.
        assert METHOD_ROW_LIMITS.get("tabpfn_v2_5") == 50_000


class TestForbidsCatIndicesBugFix:
    """Regression test: FORBIDS_CAT_INDICES used to alias REQUIRES_CAT_OHE,
    silently omitting tabr_ohe methods."""

    def test_tabr_methods_forbid_indices(self):
        # tabr / modernNCA / mlp_plr require tabr_ohe -> they forbid indices.
        for m in ("tabr", "modernNCA", "mlp_plr"):
            if m in METHOD_REGISTRY:
                assert m in FORBIDS_CAT_INDICES, f"{m} should forbid 'indices'"

    def test_forbids_includes_both_ohe_and_tabr_ohe(self):
        assert REQUIRES_CAT_OHE <= FORBIDS_CAT_INDICES
        assert REQUIRES_CAT_TABR_OHE <= FORBIDS_CAT_INDICES

    def test_indices_methods_do_not_forbid_indices(self):
        assert REQUIRES_CAT_INDICES.isdisjoint(FORBIDS_CAT_INDICES)


class TestDeriveMethodSet:

    def test_filter_by_supports_hpo(self):
        from TALENT.model.method_registry import Hardware
        hpo_off = derive_method_set(supports_hpo=False)
        assert hpo_off == NO_HPO_METHODS

    def test_filter_by_hardware(self):
        from TALENT.model.method_registry import Hardware
        gpu = derive_method_set(hardware=Hardware.GPU)
        assert gpu == GPU_METHODS

    def test_filter_with_tuple_value(self):
        # Membership semantics for iterable values
        from TALENT.model.method_registry import Architecture
        deep_only = derive_method_set(architecture=(Architecture.DEEP,))
        assert deep_only == DEEP_METHODS


class TestFoundationMethods:

    def test_foundation_contains_all_tabpfn_variants(self):
        assert TABPFN_VARIANTS <= FOUNDATION_METHODS

    def test_foundation_contains_tabicl_and_tabdpt(self):
        for m in ("tabicl", "tabicl_v2", "tabdpt", "mitra", "limix"):
            if m in METHOD_REGISTRY:
                assert m in FOUNDATION_METHODS, f"{m} should be a foundation model"


# ---------------------------------------------------------------------------
#  Registry corrections: HPO support and capacity row caps
# ---------------------------------------------------------------------------

def test_ncm_and_naivebayes_are_not_tunable():
    """TALENT marks them supports_hpo=True but they assert `not args.tune`.

    Left uncorrected, every __HPO point for them fails with a bare
    AssertionError and the resubmit planner requests it again on every gap scan.
    """
    from src.methods.method_config import HPO_METHODS, NO_HPO_METHODS, METHOD_REGISTRY

    for name in ("NCM", "NaiveBayes"):
        assert METHOD_REGISTRY[name].supports_hpo is False, name
        assert name not in HPO_METHODS and name in NO_HPO_METHODS, name


def test_tangos_has_a_training_row_cap():
    """tangos cannot train on the largest datasets within any wall time.

    21 fits per fold at n_trials=20; one fold on Hackerearth's 340,753 training
    rows had not finished in 37 h. The cap is on TRAINING rows only, so test and
    validation folds keep every row and all methods stay comparable.
    """
    from src.methods.method_config import METHOD_ROW_LIMITS

    assert METHOD_ROW_LIMITS.get("tangos") == 50_000


def test_a_capacity_cap_does_not_relabel_a_method_as_a_foundation_model():
    """FOUNDATION_METHODS infers from train_row_limit; capped-for-speed methods
    must be excluded or they appear as foundation models in every figure."""
    from src.methods.method_config import (FOUNDATION_METHODS, METHOD_REGISTRY,
                                           _CAPACITY_ROW_CAPS)

    assert _CAPACITY_ROW_CAPS, "fixture is wrong -- no capacity caps declared"
    for name in _CAPACITY_ROW_CAPS:
        assert METHOD_REGISTRY[name].train_row_limit is not None
        assert name not in FOUNDATION_METHODS, (
            f"{name} is capped for runtime, not an in-context learner")
    # the real foundation family is untouched
    for name in ("tabpfn_v3", "tabicl_v2", "mitra", "tabdpt"):
        assert name in FOUNDATION_METHODS, name


def test_capacity_cap_applies_to_tuned_and_untuned_alike():
    """Both variants must train on the same rows, or the HPO-effect number
    would mix 'tuning helped' with 'trained on less data'."""
    from src.methods.method_config import HPO_METHODS, METHOD_ROW_LIMITS

    # the cap lives on the method, not on the __HPO suffix, so it necessarily
    # applies to both -- assert the method is still tunable so both exist
    assert "tangos" in HPO_METHODS
    assert METHOD_ROW_LIMITS.get("tangos") == 50_000
