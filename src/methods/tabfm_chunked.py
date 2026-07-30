"""Memory-safe TabFM inference: chunk the evaluation split, recover from OOM.

WHY THIS LIVES HERE AND NOT IN THE TALENT FORK
----------------------------------------------
The obvious home for this is TALENT's ``TabFMMethod``. It is here instead
because TALENT is installed from a git URL (see ``pyproject.toml``), so a fix
there only reaches the cluster after a *separate* push + ``pip install
--force-reinstall``. That step was missed once and cost two more H100 jobs
(61587874): the code was correct locally while the cluster kept running the
old wrapper and OOM'd at byte-identical allocation sizes. Everything in this
file ships with a plain ``git pull``.

It also sits next to the other half of the same policy: ``method_runner``
already injects TabFM's ``max_num_rows`` context cap. Memory budgeting is this
repo's concern -- how much GPU we are willing to spend -- while TALENT's job is
just knowing how to call TabFM.

WHAT THE PROBLEM IS
-------------------
TabFM builds ONE sequence per ensemble member: that member's in-context train
rows followed by the ENTIRE evaluation split
(``Xs: (n_members, train + test, n_features)``), and ``batch_size`` is how many
members share a forward pass. Peak activation memory scales with
``(context + n_eval) * n_features``.

``max_num_rows`` caps only the CONTEXT. Measured on an 80 GB H100 with a 10k
context (jobs 61519948 / 61587874), every split up to 30k rows completed while
these three died in the row interactor's feed-forward:

    hackerearth      106 485 rows x  35 features  ->  +18.22 GiB on top of 60.8
    home_credit       61 502 rows x 120 features  ->  +27.00 GiB
    algorithmwatch    31 740 rows x 500 features  ->  +17.04 GiB

WHY CHUNKING IS EQUIVALENT
--------------------------
Evaluation rows are conditionally independent given the context: the row
attention mask is derived from ``train_size``, which is precisely what allows
the train context to be KV-cached (``use_cache``). Scoring the split in
row-chunks and concatenating therefore reproduces the single-pass result.
``src/utils/verify_inference_chunking.py`` checks that empirically.

DEFAULT IS UNCHANGED BEHAVIOUR
------------------------------
One pass, exactly as before, so every split that already fits reproduces
bit-identically. Chunking engages only when
``general.predict_chunk_size`` is set, or when a pass raises CUDA OOM -- then
the chunk halves and retries down to ``_MIN_PREDICT_CHUNK``. Both paths print
the chunk size actually used, so the job log answers "did the fix run?" with a
single grep for ``[TabFM]``.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import replace
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Smallest chunk tried before a CUDA OOM is allowed to propagate. An OOM at
#: this size means something other than the split is exhausting the GPU.
_MIN_PREDICT_CHUNK = 512

#: Set once per process so the mode is stated in the log without repeating it
#: for all 5 folds x 2 splits.
_ANNOUNCED = False

#: Per-process OOM memory: feature count -> smallest row count whose single
#: forward pass raised CUDA OOM. TabFM's activation memory scales with
#: rows x features, and a new ``ChunkedInference`` proxy is built for EVERY
#: fold, so without this the OOM ladder re-probed the exact same failing size
#: ten times per dataset (5 folds x test+val on job 61590876) -- each probe a
#: wasted forward attempt that runs until the allocator gives up. Keyed by
#: feature count because one SLURM slot can run several datasets in one
#: process; row count only transfers between splits of the same width.
_OOM_ROWS_BY_WIDTH: Dict[int, int] = {}

__all__ = [
    "ChunkedTabFMMethod",
    "ChunkedInference",
    "install",
    "is_installed",
]


def is_cuda_oom(exc: BaseException) -> bool:
    """True iff ``exc`` is a CUDA out-of-memory error.

    torch >= 2.5 raises ``torch.OutOfMemoryError``; older builds raise a plain
    ``RuntimeError``. Both are recognised.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover -- torch is a hard dependency
        return False
    oom_types = tuple(
        t
        for t in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        )
        if isinstance(t, type)
    )
    if oom_types and isinstance(exc, oom_types):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_gpu() -> None:
    """Release the partially-allocated activations of a failed pass, otherwise
    the smaller chunk hits the same ceiling."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


class ChunkedInference:
    """Forwarding proxy around a fitted TabFM estimator that scores in chunks.

    Everything except ``predict`` / ``predict_proba`` is delegated untouched, so
    the proxy is a drop-in for the estimator TALENT's wrapper holds in
    ``self.model`` (it calls only ``fit``, ``predict`` and ``predict_proba``).
    """

    def __init__(self, model: Any, chunk_size: Optional[int] = None):
        self._model = model
        self._chunk_size = chunk_size

    # -- transparency ----------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this class does not define.
        return getattr(self._model, name)

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid
        return f"ChunkedInference({self._model!r}, chunk_size={self._chunk_size})"

    @property
    def wrapped(self) -> Any:
        """The underlying estimator (for tests / introspection)."""
        return self._model

    # -- scoring ---------------------------------------------------------
    def predict(self, X, *args, **kwargs):
        return self._scored(self._model.predict, X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        return self._scored(self._model.predict_proba, X, *args, **kwargs)

    def _in_chunks(self, fn, X, chunk_size: Optional[int], *args, **kwargs):
        n = len(X)
        if chunk_size is None or chunk_size >= n:
            # Hand the frame over untouched -- no slicing, no copy.
            return np.asarray(fn(X, *args, **kwargs))
        parts = []
        for start in range(0, n, chunk_size):
            block = X.iloc[start:start + chunk_size].reset_index(drop=True)
            parts.append(np.asarray(fn(block, *args, **kwargs)))
        return np.concatenate(parts, axis=0)

    def _scored(self, fn, X, *args, **kwargs):
        global _ANNOUNCED
        n = len(X)
        width = int(getattr(X, "shape", (n, 0))[1] or 0)
        requested = self._chunk_size
        # An explicit request is honoured verbatim, bounded only by the split.
        # _MIN_PREDICT_CHUNK is the floor of the OOM ladder below, NOT a minimum
        # on what the caller may ask for -- clamping here would silently ignore a
        # small chunk size on a small split, which is exactly the setup
        # src/utils/verify_inference_chunking.py needs.
        chunk = n if requested in (None, 0) else max(1, int(requested))
        chunk = min(chunk, n)

        if not _ANNOUNCED:
            _ANNOUNCED = True
            mode = ("one pass, halving on CUDA OOM" if requested in (None, 0)
                    else f"fixed chunks of {chunk}")
            print(f"[TabFM] chunked inference active ({mode}); split = {n} rows.")

        # Skip pass sizes this process has ALREADY seen OOM at this feature
        # width: walk the same halving ladder down, just without burning a
        # forward attempt per step. A smaller split than any known failure
        # still gets its one-pass try -- an OOM at 106k rows says nothing
        # about 85k (which fit, on the same dataset, in job 61590876).
        known_oom = _OOM_ROWS_BY_WIDTH.get(width)
        if (requested in (None, 0) and known_oom is not None
                and chunk >= known_oom and chunk > _MIN_PREDICT_CHUNK):
            first = chunk
            while chunk >= known_oom and chunk > _MIN_PREDICT_CHUNK:
                chunk = max(_MIN_PREDICT_CHUNK, chunk // 2)
            print(
                f"[TabFM] starting at chunks of {chunk} (split = {first} rows; "
                f"a {known_oom}-row pass hit CUDA OOM earlier in this run)."
            )

        while True:
            try:
                out = self._in_chunks(fn, X, chunk, *args, **kwargs)
            except Exception as exc:
                if not is_cuda_oom(exc) or chunk <= _MIN_PREDICT_CHUNK:
                    raise
                _OOM_ROWS_BY_WIDTH[width] = min(
                    _OOM_ROWS_BY_WIDTH.get(width, chunk), chunk)
                nxt = max(_MIN_PREDICT_CHUNK, chunk // 2)
                print(
                    f"[TabFM] CUDA OOM scoring {chunk} rows in one pass "
                    f"(split = {n} rows); retrying in chunks of {nxt}."
                )
                chunk = nxt
                _free_gpu()
                continue
            if chunk < n:
                print(f"[TabFM] scored {n} rows in chunks of {chunk}.")
            return out


def _base_class():
    """TALENT's stock TabFM method class."""
    from TALENT.model.methods.tabfm import TabFMMethod

    return TabFMMethod


try:
    _TabFMMethod = _base_class()
except ImportError:  # pragma: no cover -- TALENT absent (docs build, fresh clone)
    _TabFMMethod = None
    ChunkedTabFMMethod = None  # type: ignore[assignment]
else:

    class ChunkedTabFMMethod(_TabFMMethod):  # type: ignore[misc,valid-type]
        """TabFM with memory-safe inference.

        Wraps the fitted estimator instead of overriding ``predict``, so the
        parent's metric/loss/reporting code stays the single implementation and
        this class cannot drift from it.
        """

        def construct_model(self, model_config=None, cat_indices=None):
            super().construct_model(model_config, cat_indices)
            general = (self.args.config or {}).get("general", {}) or {}
            self.model = ChunkedInference(
                self.model, chunk_size=general.get("predict_chunk_size")
            )


def is_installed() -> bool:
    """True if TALENT's registry currently resolves ``tabfm`` to this module."""
    try:
        from TALENT.model.method_registry import get_method_spec

        return get_method_spec("tabfm").module == __name__
    except Exception:  # pragma: no cover -- TALENT absent or no tabfm spec
        return False


def install() -> bool:
    """Point TALENT's ``tabfm`` spec at :class:`ChunkedTabFMMethod`.

    ``MethodSpec`` is a frozen dataclass that lazy-imports ``module.class_name``
    in ``get_class()``, so swapping those two fields is enough -- every other
    field (architecture, preprocessing constraints, ``supports_hpo``, ...) is
    preserved by ``dataclasses.replace``. Idempotent; returns True if the
    override is in place afterwards.
    """
    if ChunkedTabFMMethod is None:
        return False
    if is_installed():
        return True
    try:
        from TALENT.model.method_registry import METHOD_REGISTRY, get_method_spec

        spec = get_method_spec("tabfm")
        METHOD_REGISTRY[spec.name] = replace(
            spec, module=__name__, class_name="ChunkedTabFMMethod"
        )
    except Exception as exc:  # pragma: no cover -- never block a run over this
        logger.warning(
            "Could not install TabFM chunked inference (%s); TabFM will run "
            "TALENT's stock wrapper and may OOM on large evaluation splits.",
            exc,
        )
        return False
    logger.info("TabFM: chunked inference installed (%s.ChunkedTabFMMethod)", __name__)
    return True
