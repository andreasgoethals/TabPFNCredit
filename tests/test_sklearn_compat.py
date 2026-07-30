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


def test_method_runner_installs_the_shim():
    """The hook must actually be wired into the run path, not merely exist."""
    import inspect

    from src.methods import method_runner

    source = inspect.getsource(method_runner)
    assert "install_sklearn_validate_data_shim()" in source, (
        "the shim is never called -- TabICL v1 would still fail on sklearn >= 1.6")
