"""Lightweight import smoke tests.

Verifies that the refactor didn't break any of the public entry points
the experiment drivers and notebooks use.
"""


def test_method_config_imports():
    from src.methods.method_config import (
        DEEP_METHODS,
        CLASSICAL_METHODS,
        GPU_METHODS,
        CPU_METHODS,
        LOGIT_METHODS,
        PROBABILITY_METHODS,
        CLASS_LABEL_METHODS,
        NO_HPO_METHODS,
        HPO_METHODS,
        FOUNDATION_METHODS,
        TABPFN_VARIANTS,
        REQUIRES_CAT_INDICES,
        REQUIRES_CAT_OHE,
        REQUIRES_CAT_TABR_OHE,
        FORBIDS_CAT_INDICES,
        REQUIRES_NO_NORMALIZATION,
        REQUIRES_STANDARD_NORMALIZATION,
        REQUIRES_NO_NUM_ENCODING,
        METHOD_ROW_LIMITS,
        METHOD_TEST_VAL_LIMITS,
        apply_preprocessing_policies,
        apply_method_row_limit,
        derive_method_set,
    )
    assert len(DEEP_METHODS) > 0
    assert len(CLASSICAL_METHODS) > 0


def test_method_runner_imports():
    from src.methods.method_runner import (
        run_talent_method,
        get_available_methods,
        validate_method,
    )
    methods = get_available_methods()
    assert "classical" in methods
    assert "deep" in methods


def test_method_metrics_imports():
    from src.methods.method_metrics import (
        calculate_pd_metrics,
        calculate_lgd_metrics,
        enrich_pd_metrics,
        enrich_lgd_metrics,
        gini_from_auc,
        ks_statistic,
    )


def test_cost_metrics_imports():
    from src.methods.cost_metrics import (
        CostMatrix,
        DEFAULT_COSTS,
        expected_loss,
        profit_curve,
        cost_sensitive_summary,
    )


def test_cli_imports():
    from src.utils.cli import app
    # Typer apps expose a `.registered_commands` attribute -- check it has entries
    assert len(app.registered_commands) >= 4


def test_talent_registry_reachable():
    """The wrapper depends on TALENT's MethodSpec registry being importable."""
    from TALENT.model.method_registry import METHOD_REGISTRY, get_method_spec
    assert "tabpfn_v3" in METHOD_REGISTRY
    assert "tabicl_v2" in METHOD_REGISTRY
    assert "tabpfn_v2_5" in METHOD_REGISTRY
    assert "tabdpt" in METHOD_REGISTRY
    spec = get_method_spec("tabpfn_v3")
    assert spec.supports_hpo is False
