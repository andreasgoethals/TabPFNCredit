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
