"""Tests for TabFM's chunked / OOM-resilient inference path.

Job 61519948 lost 3 of 14 PD datasets to ``torch.OutOfMemoryError`` inside
``TabFMMethod.predict``: TabFM scores the WHOLE evaluation split in one
sequence per ensemble member, so peak memory grows with the split size, and
``max_num_rows`` only caps the in-context half. The wrapper now falls back to
chunked scoring on OOM.

These tests exercise that logic with a stub model (no GPU, no ``tabfm``
package), covering:

1. the default path is still a single unchunked call -- results for every
   split that already fit are unchanged;
2. an explicit ``predict_chunk_size`` splits the work and reassembles it in
   the original row order;
3. an OOM triggers halving and eventually succeeds -- the cluster failure;
4. a non-OOM error is never swallowed;
5. an OOM that persists down to the floor is re-raised rather than looping.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

tabfm_mod = pytest.importorskip(
    "TALENT.model.methods.tabfm",
    reason="TALENT not installed",
)

if not hasattr(tabfm_mod, "_predict_chunked") and not hasattr(
    tabfm_mod.TabFMMethod, "_predict_chunked"
):  # pragma: no cover
    pytest.skip(
        "installed TALENT predates the TabFM chunked-inference fix "
        "(reinstall TALENT from the fork)",
        allow_module_level=True,
    )

TabFMMethod = tabfm_mod.TabFMMethod
_is_cuda_oom = tabfm_mod._is_cuda_oom
_MIN_PREDICT_CHUNK = tabfm_mod._MIN_PREDICT_CHUNK


# ---------------------------------------------------------------------------
#  Stubs
# ---------------------------------------------------------------------------

class _StubModel:
    """Row-independent scorer that OOMs above ``oom_above`` rows at once.

    Mirrors TabFM's real failure mode: the whole split goes through one
    forward pass, so a big split blows up while the same rows in smaller
    groups are fine.
    """

    def __init__(self, oom_above: int | None = None):
        self.oom_above = oom_above
        self.calls: list[int] = []

    def _score(self, X):
        self.calls.append(len(X))
        if self.oom_above is not None and len(X) > self.oom_above:
            raise _make_oom()
        # Deterministic, row-local -> chunking must not change it.
        base = np.asarray(X["num_0"], dtype=np.float64)
        return np.column_stack([1.0 - base / 1000.0, base / 1000.0])

    # NB: delegate at CALL time (``predict_proba = _score`` would bind the
    # function at class-definition time, so patching ``_score`` per instance
    # would silently have no effect).
    def predict_proba(self, X):
        return self._score(X)

    def predict(self, X):
        return self._score(X)[:, 1]


def _make_oom() -> Exception:
    import torch

    exc_type = getattr(torch, "OutOfMemoryError", None)
    if isinstance(exc_type, type):
        return exc_type("CUDA out of memory. Tried to allocate 18.22 GiB.")
    return RuntimeError("CUDA out of memory. Tried to allocate 18.22 GiB.")


def _method(chunk_size=None, *, is_regression=False, oom_above=None):
    """A TabFMMethod with just enough state for the predict path."""
    obj = object.__new__(TabFMMethod)
    general = {}
    if chunk_size is not None:
        general["predict_chunk_size"] = chunk_size
    obj.args = SimpleNamespace(config={"general": general})
    obj.is_regression = is_regression
    obj.model = _StubModel(oom_above=oom_above)
    return obj


def _frame(n):
    return pd.DataFrame({"num_0": np.arange(n, dtype=np.float64)})


# ---------------------------------------------------------------------------
#  OOM detection
# ---------------------------------------------------------------------------

class TestOomDetection:

    def test_recognises_torch_oom(self):
        assert _is_cuda_oom(_make_oom())

    def test_recognises_legacy_runtime_error(self):
        assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))

    @pytest.mark.parametrize("exc", [
        ValueError("bad shape"),
        RuntimeError("expected scalar type Float but found Half"),
        KeyError("num_0"),
    ])
    def test_rejects_everything_else(self, exc):
        assert not _is_cuda_oom(exc)


# ---------------------------------------------------------------------------
#  Chunking
# ---------------------------------------------------------------------------

class TestChunking:

    def test_default_is_one_unchunked_call(self):
        """No config, no OOM => exactly the pre-fix behaviour."""
        m = _method()
        X = _frame(2000)
        out = m._predict_with_oom_fallback(X)
        assert m.model.calls == [2000], "split must go through in ONE pass"
        assert out.shape == (2000, 2)

    def test_explicit_chunk_size_splits_the_work(self):
        m = _method(chunk_size=512)
        X = _frame(1300)
        out = m._predict_with_oom_fallback(X)
        assert m.model.calls == [512, 512, 276]
        assert out.shape == (1300, 2)

    def test_chunked_equals_unchunked(self):
        """The property the whole fix rests on."""
        X = _frame(1500)
        whole = _method()._predict_with_oom_fallback(X)
        chunked = _method(chunk_size=512)._predict_with_oom_fallback(X)
        np.testing.assert_allclose(whole, chunked)

    def test_row_order_is_preserved(self):
        m = _method(chunk_size=512)
        out = m._predict_with_oom_fallback(_frame(1300))
        # column 1 == num_0 / 1000 by construction
        np.testing.assert_allclose(out[:, 1], np.arange(1300) / 1000.0)

    def test_regression_path_is_1d(self):
        m = _method(chunk_size=512, is_regression=True)
        out = m._predict_with_oom_fallback(_frame(1300))
        assert out.shape == (1300,)
        np.testing.assert_allclose(out, np.arange(1300) / 1000.0)

    def test_chunk_larger_than_split_stays_a_single_call(self):
        m = _method(chunk_size=99_999)
        m._predict_with_oom_fallback(_frame(700))
        assert m.model.calls == [700]

    def test_small_explicit_chunk_is_honoured_verbatim(self):
        """A chunk size below the OOM-halving floor must still be respected --
        the floor bounds the fallback ladder, not the caller's request. Without
        this, scripts/verify_inference_chunking.py silently degrades to a
        single pass on a small dataset and can never prove anything."""
        m = _method(chunk_size=64)
        m._predict_with_oom_fallback(_frame(200))
        assert m.model.calls == [64, 64, 64, 8]
        assert max(m.model.calls) < _MIN_PREDICT_CHUNK

    def test_zero_and_none_mean_no_chunking(self):
        for value in (0, None):
            m = _method(chunk_size=value)
            m._predict_with_oom_fallback(_frame(1500))
            assert m.model.calls == [1500], f"chunk_size={value!r} must not chunk"


# ---------------------------------------------------------------------------
#  OOM recovery -- the cluster failure
# ---------------------------------------------------------------------------

class TestOomRecovery:

    def test_halves_until_it_fits(self):
        """106k rows OOM'd on an H100; a stub that OOMs above 1500 must
        recover by halving rather than killing the whole dataset."""
        m = _method(oom_above=1500)
        X = _frame(4000)
        out = m._predict_with_oom_fallback(X)
        # 4000 (OOM) -> 2000 (OOM) -> 1000 x4 (ok)
        assert m.model.calls[:2] == [4000, 2000]
        assert all(c <= 1500 for c in m.model.calls[2:])
        assert sum(m.model.calls[2:]) == 4000
        assert out.shape == (4000, 2)
        np.testing.assert_allclose(out[:, 1], np.arange(4000) / 1000.0)

    def test_recovered_result_matches_the_unchunked_one(self):
        X = _frame(4000)
        expected = _method()._predict_with_oom_fallback(X)
        recovered = _method(oom_above=1500)._predict_with_oom_fallback(X)
        np.testing.assert_allclose(expected, recovered)

    def test_non_oom_errors_propagate(self):
        m = _method()
        m.model._score = lambda X: (_ for _ in ()).throw(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            m._predict_with_oom_fallback(_frame(1000))

    def test_gives_up_at_the_floor_instead_of_looping(self):
        """A model that OOMs at any size must raise, not spin forever."""
        m = _method(oom_above=0)
        with pytest.raises(Exception) as excinfo:
            m._predict_with_oom_fallback(_frame(4000))
        assert _is_cuda_oom(excinfo.value)
        assert min(m.model.calls) == _MIN_PREDICT_CHUNK, (
            "must probe all the way down to the floor before giving up"
        )

    def test_fallback_is_announced(self, capsys):
        """The effective chunk size has to be visible in the job log so it can
        be disclosed with the results."""
        _method(oom_above=1500)._predict_with_oom_fallback(_frame(4000))
        out = capsys.readouterr().out
        assert "CUDA OOM" in out
        assert "chunks of" in out
