"""Tests for the sklearn >= 1.6 ``_validate_data`` shim.

The bug this guards is environment-specific: scikit-learn 1.6 removed
``BaseEstimator._validate_data``, and TALENT's vendored TabICL v1 calls it in both
arms of its own version check, so every tabicl fit dies with AttributeError on the
cluster (which runs sklearn >= 1.6 because TabFM requires it). A developer machine
on sklearn < 1.6 cannot reproduce that, so these tests SIMULATE both versions
instead of depending on the installed one -- otherwise the shim would only ever be
exercised where it is not needed.
"""

from __future__ import annotations

import pytest
from sklearn.base import BaseEstimator

import src.methods.sklearn_compat as compat


@pytest.fixture(autouse=True)
def _fresh_install_flag(monkeypatch):
    """The shim is install-once; reset that between tests."""
    monkeypatch.setattr(compat, "_INSTALLED", False)


def test_no_op_when_sklearn_still_has_the_method(monkeypatch):
    """On sklearn < 1.6 the real method exists and must not be replaced."""
    sentinel = object()
    monkeypatch.setattr(BaseEstimator, "_validate_data", sentinel, raising=False)
    assert compat.install_sklearn_validate_data_shim() is False
    assert BaseEstimator._validate_data is sentinel


def test_shim_forwards_to_the_new_free_function(monkeypatch):
    """Simulated sklearn >= 1.6: the shim must restore the method and forward."""
    import sklearn.utils.validation as skv

    calls = []

    def fake_validate_data(estimator, X="no_validation", y="no_validation", **kwargs):
        calls.append((estimator, X, y, kwargs))
        return X, y

    monkeypatch.delattr(BaseEstimator, "_validate_data", raising=False)
    monkeypatch.setattr(skv, "validate_data", fake_validate_data, raising=False)

    assert compat.install_sklearn_validate_data_shim() is True
    assert hasattr(BaseEstimator, "_validate_data")

    class Dummy(BaseEstimator):
        pass

    estimator = Dummy()
    # exactly the call TabICL v1 makes on the >= 1.6 path
    out_X, out_y = estimator._validate_data([[1.0]], [0], dtype=None,
                                            skip_check_array=True)
    assert out_X == [[1.0]] and out_y == [0]
    assert len(calls) == 1
    got_estimator, got_X, got_y, got_kwargs = calls[0]
    assert got_estimator is estimator, "the estimator must be passed positionally"
    assert got_X == [[1.0]] and got_y == [0]
    assert got_kwargs == {"dtype": None, "skip_check_array": True}


def test_idempotent(monkeypatch):
    """Called once per method run -- it must not reinstall or fail."""
    import sklearn.utils.validation as skv

    monkeypatch.delattr(BaseEstimator, "_validate_data", raising=False)
    monkeypatch.setattr(skv, "validate_data",
                        lambda est, X=None, y=None, **k: (X, y), raising=False)
    assert compat.install_sklearn_validate_data_shim() is True
    assert compat.install_sklearn_validate_data_shim() is True


def test_degrades_when_neither_api_is_available(monkeypatch):
    """An unknown sklearn layout must warn, not raise -- the run may not need it."""
    import sklearn.utils.validation as skv

    monkeypatch.delattr(BaseEstimator, "_validate_data", raising=False)
    monkeypatch.delattr(skv, "validate_data", raising=False)
    # the shim imports the name at call time, so hide it from the import machinery
    monkeypatch.setattr(compat, "_INSTALLED", False)
    assert compat.install_sklearn_validate_data_shim() is False


# The run path calls the composite install_sklearn_compat(); that wiring is
# asserted by test_install_sklearn_compat_runs_both_shims below.


# ---------------------------------------------------------------------------
#  force_all_finite -> ensure_all_finite (renamed 1.6, removed 1.8)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_validator_flag(monkeypatch):
    monkeypatch.setattr(compat, "_VALIDATORS_PATCHED", False)


def _new_style(X=None, y=None, *, ensure_all_finite=True, **kw):
    """Stands in for sklearn >= 1.8: the old kwarg no longer exists."""
    _new_style.seen = {"ensure_all_finite": ensure_all_finite, **kw}
    return X, y


def test_finite_kwarg_is_translated(monkeypatch):
    import sklearn.utils.validation as skv

    monkeypatch.setattr(skv, "check_X_y", _new_style, raising=False)
    assert compat.install_sklearn_finite_kwarg_shim() is True

    # exactly the call TALENT's realmlp / tabpfn wrappers make
    skv.check_X_y([[1.0]], [0], force_all_finite="allow-nan", multi_output=True)
    assert _new_style.seen["ensure_all_finite"] == "allow-nan", (
        "the value is load-bearing: these methods accept NaN, so dropping it "
        "would make validation reject the data")
    assert _new_style.seen["multi_output"] is True


def test_already_imported_module_is_rebound(monkeypatch):
    """The wrappers bind check_X_y as a module global at import time."""
    import sys
    import types

    import sklearn.utils.validation as skv

    vendored = types.ModuleType("_fake_vendored_wrapper")
    vendored.check_X_y = _new_style
    monkeypatch.setitem(sys.modules, "_fake_vendored_wrapper", vendored)
    monkeypatch.setattr(skv, "check_X_y", _new_style, raising=False)

    assert compat.install_sklearn_finite_kwarg_shim() is True
    assert vendored.check_X_y is not _new_style, "stale binding was not rebound"
    vendored.check_X_y([[1.0]], [0], force_all_finite=False)
    assert _new_style.seen["ensure_all_finite"] is False


def test_no_op_when_the_old_kwarg_still_exists(monkeypatch):
    """sklearn < 1.8 accepts force_all_finite; leave it alone."""
    import sklearn.utils.validation as skv

    def old_style(X=None, y=None, *, force_all_finite=True, **kw):
        return X, y

    monkeypatch.setattr(skv, "check_X_y", old_style, raising=False)
    monkeypatch.setattr(skv, "check_array", old_style, raising=False)
    assert compat.install_sklearn_finite_kwarg_shim() is False
    assert skv.check_X_y is old_style


def test_finite_shim_is_idempotent(monkeypatch):
    import sklearn.utils.validation as skv

    monkeypatch.setattr(skv, "check_X_y", _new_style, raising=False)
    assert compat.install_sklearn_finite_kwarg_shim() is True
    first = skv.check_X_y
    monkeypatch.setattr(compat, "_VALIDATORS_PATCHED", False)
    compat.install_sklearn_finite_kwarg_shim()
    assert skv.check_X_y is first, "double-wrapped"


def test_install_sklearn_compat_runs_both_shims():
    import inspect

    from src.methods import method_runner

    source = inspect.getsource(compat.install_sklearn_compat)
    assert "install_sklearn_validate_data_shim()" in source
    assert "install_sklearn_finite_kwarg_shim()" in source
    assert "install_sklearn_compat()" in inspect.getsource(method_runner), (
        "the shims are never called from the run path")
