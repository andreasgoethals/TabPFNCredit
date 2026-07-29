"""Tests for memory-safe TabFM inference (``src/methods/tabfm_chunked.py``).

Job 61519948 lost 3 of 14 PD datasets to ``torch.OutOfMemoryError`` while
scoring: TabFM puts the WHOLE evaluation split into one sequence per ensemble
member, so peak memory grows with the split size, and ``max_num_rows`` caps
only the in-context half.

Job 61587874 then failed the SAME way at byte-identical allocation sizes,
because the first fix lived in the TALENT fork and never got reinstalled on the
cluster. Hence ``TestRegistryOverride``: the fix now ships with this repo and
is wired in through TALENT's method registry, so ``git pull`` is the whole
deployment.

Everything is exercised with a stub estimator -- no GPU, no ``tabfm`` package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.methods.tabfm_chunked import (
    _MIN_PREDICT_CHUNK,
    ChunkedInference,
    is_cuda_oom,
    is_installed,
)
from src.methods import tabfm_chunked


# ---------------------------------------------------------------------------
#  Stubs
# ---------------------------------------------------------------------------

class _StubModel:
    """Row-independent scorer that OOMs above ``oom_above`` rows at once.

    Mirrors TabFM's real failure mode: the whole split goes through one forward
    pass, so a big split blows up while the same rows in smaller groups fit.
    """

    def __init__(self, oom_above: int | None = None):
        self.oom_above = oom_above
        self.calls: list[int] = []
        self.fitted = False

    def _score(self, X):
        self.calls.append(len(X))
        if self.oom_above is not None and len(X) > self.oom_above:
            raise _make_oom()
        # Deterministic and row-local -> chunking must not change it.
        base = np.asarray(X["num_0"], dtype=np.float64)
        return np.column_stack([1.0 - base / 1000.0, base / 1000.0])

    # NB: delegate at CALL time (``predict_proba = _score`` would bind the
    # function at class-definition time, so patching ``_score`` per instance
    # would silently have no effect).
    def predict_proba(self, X):
        return self._score(X)

    def predict(self, X):
        return self._score(X)[:, 1]

    def fit(self, X, y):
        self.fitted = True
        return self

    @property
    def n_estimators(self):
        return 32


def _make_oom() -> Exception:
    import torch

    exc_type = getattr(torch, "OutOfMemoryError", None)
    if isinstance(exc_type, type):
        return exc_type("CUDA out of memory. Tried to allocate 18.22 GiB.")
    return RuntimeError("CUDA out of memory. Tried to allocate 18.22 GiB.")


@pytest.fixture(autouse=True)
def _reset_announcement():
    """The "chunked inference active" banner prints once per process."""
    tabfm_chunked._ANNOUNCED = False
    yield
    tabfm_chunked._ANNOUNCED = False


def _proxy(chunk_size=None, *, oom_above=None):
    return ChunkedInference(_StubModel(oom_above=oom_above), chunk_size=chunk_size)


def _frame(n):
    return pd.DataFrame({"num_0": np.arange(n, dtype=np.float64)})


# ---------------------------------------------------------------------------
#  OOM detection
# ---------------------------------------------------------------------------

class TestOomDetection:

    def test_recognises_torch_oom(self):
        assert is_cuda_oom(_make_oom())

    def test_recognises_legacy_runtime_error(self):
        assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))

    @pytest.mark.parametrize("exc", [
        ValueError("bad shape"),
        RuntimeError("expected scalar type Float but found Half"),
        KeyError("num_0"),
    ])
    def test_rejects_everything_else(self, exc):
        assert not is_cuda_oom(exc)


# ---------------------------------------------------------------------------
#  Transparency -- the proxy must be a drop-in for the estimator
# ---------------------------------------------------------------------------

class TestProxyTransparency:

    def test_forwards_unknown_attributes(self):
        assert _proxy().n_estimators == 32

    def test_forwards_fit(self):
        p = _proxy()
        p.fit(_frame(10), np.zeros(10))
        assert p.wrapped.fitted is True

    def test_exposes_the_wrapped_estimator(self):
        p = _proxy()
        assert isinstance(p.wrapped, _StubModel)


# ---------------------------------------------------------------------------
#  Chunking
# ---------------------------------------------------------------------------

class TestChunking:

    def test_default_is_one_unchunked_call(self):
        """No config, no OOM => exactly the pre-fix computation."""
        p = _proxy()
        out = p.predict_proba(_frame(2000))
        assert p.wrapped.calls == [2000], "split must go through in ONE pass"
        assert out.shape == (2000, 2)

    def test_explicit_chunk_size_splits_the_work(self):
        p = _proxy(chunk_size=512)
        out = p.predict_proba(_frame(1300))
        assert p.wrapped.calls == [512, 512, 276]
        assert out.shape == (1300, 2)

    def test_chunked_equals_unchunked(self):
        """The property the whole fix rests on."""
        X = _frame(1500)
        whole = _proxy().predict_proba(X)
        chunked = _proxy(chunk_size=512).predict_proba(X)
        np.testing.assert_allclose(whole, chunked)

    def test_row_order_is_preserved(self):
        out = _proxy(chunk_size=512).predict_proba(_frame(1300))
        # column 1 == num_0 / 1000 by construction
        np.testing.assert_allclose(out[:, 1], np.arange(1300) / 1000.0)

    def test_regression_path_is_1d(self):
        out = _proxy(chunk_size=512).predict(_frame(1300))
        assert out.shape == (1300,)
        np.testing.assert_allclose(out, np.arange(1300) / 1000.0)

    def test_chunk_larger_than_split_stays_a_single_call(self):
        p = _proxy(chunk_size=99_999)
        p.predict_proba(_frame(700))
        assert p.wrapped.calls == [700]

    def test_small_explicit_chunk_is_honoured_verbatim(self):
        """A chunk size below the OOM-halving floor must still be respected --
        the floor bounds the fallback ladder, not the caller's request. Without
        this, src/utils/verify_inference_chunking.py silently degrades to a
        single pass on a small dataset and can never prove anything."""
        p = _proxy(chunk_size=64)
        p.predict_proba(_frame(200))
        assert p.wrapped.calls == [64, 64, 64, 8]
        assert max(p.wrapped.calls) < _MIN_PREDICT_CHUNK

    def test_zero_and_none_mean_no_chunking(self):
        for value in (0, None):
            p = _proxy(chunk_size=value)
            p.predict_proba(_frame(1500))
            assert p.wrapped.calls == [1500], f"chunk_size={value!r} must not chunk"


# ---------------------------------------------------------------------------
#  OOM recovery -- the cluster failure
# ---------------------------------------------------------------------------

class TestOomRecovery:

    def test_halves_until_it_fits(self):
        """106k rows OOM'd on an H100; a stub that OOMs above 1500 must recover
        by halving rather than killing the whole dataset."""
        p = _proxy(oom_above=1500)
        out = p.predict_proba(_frame(4000))
        # 4000 (OOM) -> 2000 (OOM) -> 1000 x4 (ok)
        assert p.wrapped.calls[:2] == [4000, 2000]
        assert all(c <= 1500 for c in p.wrapped.calls[2:])
        assert sum(p.wrapped.calls[2:]) == 4000
        assert out.shape == (4000, 2)
        np.testing.assert_allclose(out[:, 1], np.arange(4000) / 1000.0)

    def test_recovered_result_matches_the_unchunked_one(self):
        X = _frame(4000)
        expected = _proxy().predict_proba(X)
        recovered = _proxy(oom_above=1500).predict_proba(X)
        np.testing.assert_allclose(expected, recovered)

    def test_non_oom_errors_propagate(self):
        p = _proxy()
        p.wrapped._score = lambda X: (_ for _ in ()).throw(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            p.predict_proba(_frame(1000))

    def test_gives_up_at_the_floor_instead_of_looping(self):
        """A model that OOMs at any size must raise, not spin forever."""
        p = _proxy(oom_above=0)
        with pytest.raises(Exception) as excinfo:
            p.predict_proba(_frame(4000))
        assert is_cuda_oom(excinfo.value)
        assert min(p.wrapped.calls) == _MIN_PREDICT_CHUNK, (
            "must probe all the way down to the floor before giving up"
        )

    def test_fallback_is_announced(self, capsys):
        """The effective chunk size has to be visible in the job log so it can
        be disclosed with the results -- and so `grep '[TabFM]' *.out` answers
        "did the fix actually run on the cluster?"."""
        _proxy(oom_above=1500).predict_proba(_frame(4000))
        out = capsys.readouterr().out
        assert "[TabFM]" in out
        assert "CUDA OOM" in out
        assert "chunks of" in out

    def test_mode_is_announced_even_without_an_oom(self, capsys):
        _proxy().predict_proba(_frame(100))
        assert "[TabFM] chunked inference active" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  Deployment: TALENT must resolve `tabfm` to OUR class
# ---------------------------------------------------------------------------

class TestRegistryOverride:
    """Job 61587874 re-ran the stock wrapper because the fork was never
    reinstalled. These tests pin the mechanism that removes that step."""

    def test_runner_import_installs_the_override(self):
        pytest.importorskip("TALENT", reason="TALENT not installed")
        # Importing the runner is what a real run does.
        import src.methods.method_runner  # noqa: F401

        assert tabfm_chunked.install() is True
        assert is_installed(), "TALENT must resolve tabfm to ChunkedTabFMMethod"

    def test_override_resolves_to_our_subclass(self):
        pytest.importorskip("TALENT", reason="TALENT not installed")
        from TALENT.model.method_registry import get_method_spec
        from TALENT.model.methods.tabfm import TabFMMethod

        tabfm_chunked.install()
        cls = get_method_spec("tabfm").get_class()
        assert cls is tabfm_chunked.ChunkedTabFMMethod
        assert issubclass(cls, TabFMMethod), (
            "must remain a TabFMMethod so the parent's metric/reporting code "
            "stays the single implementation"
        )

    def test_override_preserves_the_rest_of_the_spec(self):
        """Only module/class_name may change -- preprocessing constraints,
        supports_hpo, architecture etc. must survive."""
        pytest.importorskip("TALENT", reason="TALENT not installed")
        from TALENT.model.method_registry import get_method_spec

        tabfm_chunked.install()
        spec = get_method_spec("tabfm")
        assert spec.name == "tabfm"
        assert spec.supports_hpo is False
        assert spec.normalization == "none"
        assert spec.num_policy == "none"
        assert spec.cat_policy == ("indices",)

    def test_install_is_idempotent(self):
        pytest.importorskip("TALENT", reason="TALENT not installed")
        from TALENT.model.method_registry import get_method_spec

        tabfm_chunked.install()
        first = get_method_spec("tabfm")
        tabfm_chunked.install()
        assert get_method_spec("tabfm") == first

    def test_construct_model_wraps_the_estimator(self):
        """``construct_model`` must hand the parent's estimator to the proxy --
        this is what makes the parent's untouched ``predict`` chunk."""
        pytest.importorskip("TALENT", reason="TALENT not installed")
        from types import SimpleNamespace

        cls = tabfm_chunked.ChunkedTabFMMethod
        obj = object.__new__(cls)
        obj.args = SimpleNamespace(config={"general": {"predict_chunk_size": 4096}})
        obj.is_regression = False
        stub = _StubModel()

        # Stand in for the parent's construct_model, which needs the real
        # `tabfm` package. Patched onto the class, so it takes ``self``.
        def fake_super_construct(self, model_config=None, cat_indices=None):
            self.model = stub

        import unittest.mock as mock

        with mock.patch.object(
            tabfm_chunked._TabFMMethod, "construct_model", fake_super_construct
        ):
            cls.construct_model(obj)

        assert isinstance(obj.model, ChunkedInference)
        assert obj.model.wrapped is stub
        assert obj.model._chunk_size == 4096
