#!/usr/bin/env python
"""Check that a method's predictions do not depend on how the evaluation
split is batched.

In-context learners (TabFM, TabPFN, TabICL, TabDPT, Mitra, ...) build one
sequence per ensemble member consisting of the in-context training rows
followed by the rows to score, so peak GPU memory grows with the size of the
evaluation split. The standard remedy is to score the split in row-chunks --
which is only legitimate if the rows being scored are conditionally
independent given the context (i.e. they attend to the context but not to each
other).

This script verifies exactly that, on a dataset small enough that a single
unchunked pass fits comfortably: it fits once, then scores the same split
whole and in chunks, and reports the largest disagreement. A clean run is
evidence that chunked inference on the large datasets is a memory
optimisation and not a change of model.

Run it on a GPU node, e.g.::

    python scripts/verify_inference_chunking.py --method tabfm \\
        --task pd --dataset 0008.german --chunk 64

Exits non-zero if the two paths disagree by more than ``--tol``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--method", default="tabfm", help="registry method name")
    p.add_argument("--task", default="pd", choices=("pd", "lgd"))
    p.add_argument(
        "--dataset",
        default="0008.german",
        help="processed dataset slug; keep it SMALL so one pass fits",
    )
    p.add_argument(
        "--chunk", type=int, default=64,
        help="rows per chunk for the chunked pass (must be < split size)",
    )
    p.add_argument(
        "--tol", type=float, default=1e-5,
        help="largest tolerated absolute difference between the two passes",
    )
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    import TALENT

    from src.methods.method_runner import _prepare_folds_uncached

    # One fold is enough -- we are testing an inference property, not a score.
    folds = _prepare_folds_uncached(
        task=args.task, dataset=args.dataset,
        test_size=0.2, val_size=0.2, cv_splits=1, seed=args.seed,
        row_limit=None, train_row_limit=None, sampling=None,
    )
    (N, C, y), info = next(iter(folds.values()))
    if isinstance(C, dict) and C.get("train") is None:
        C = None

    train_val = (
        {"train": N["train"], "val": N["val"]} if N is not None else None,
        {"train": C["train"], "val": C["val"]} if C is not None else None,
        {"train": y["train"], "val": y["val"]},
    )
    test = (
        {"test": N["test"]} if N is not None else None,
        {"test": C["test"]} if C is not None else None,
        {"test": y["test"]},
    )
    n_test = len(y["test"])
    if args.chunk >= n_test:
        print(
            f"ERROR: --chunk {args.chunk} >= split size {n_test}; the two passes "
            f"would be identical by construction. Pick a smaller --chunk or a "
            f"bigger --dataset.",
            file=sys.stderr,
        )
        return 2

    is_regression = args.task == "lgd"

    def score(chunk: int | None) -> tuple[np.ndarray, str]:
        talent_args = TALENT.build_args(
            args.method, seed=args.seed, seed_num=1, tune=False, n_trials=1,
        )
        gen = talent_args.config.setdefault("general", {})
        if chunk is None:
            gen.pop("predict_chunk_size", None)
        else:
            gen["predict_chunk_size"] = chunk
        # Tee the method's own chatter so we can confirm it really chunked --
        # a method that silently ignores ``predict_chunk_size`` would otherwise
        # score both passes identically and "pass" for the wrong reason.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = TALENT.run(
                args.method, train_val, test, info,
                args=talent_args, seed=args.seed, seed_num=1, tune=False,
            )
        out = result.predict_proba if not is_regression else result.predictions
        arr = np.asarray(out, dtype=np.float64).reshape(len(y["test"]), -1)
        return arr, buf.getvalue()

    print(
        f"{args.method} on {args.task}/{args.dataset}: "
        f"{n_test} rows to score, chunk={args.chunk}"
    )
    whole, _ = score(None)
    chunked, chatter = score(args.chunk)

    if f"chunks of {args.chunk}" not in chatter:
        print(
            f"INCONCLUSIVE: {args.method} did not report chunked scoring, so it "
            f"ignores general.predict_chunk_size -- the two passes above are "
            f"the same computation and prove nothing. Only methods whose "
            f"wrapper implements chunking can be checked here.",
            file=sys.stderr,
        )
        return 2

    if whole.shape != chunked.shape:
        print(
            f"FAIL: shape mismatch {whole.shape} vs {chunked.shape}",
            file=sys.stderr,
        )
        return 1

    diff = float(np.max(np.abs(whole - chunked)))
    print(f"max |unchunked - chunked| = {diff:.3e}   (tol {args.tol:.1e})")
    if diff > args.tol:
        print(
            "FAIL: predictions depend on the batching, so chunked inference is "
            "NOT equivalent for this method. Do not chunk it -- reduce the "
            "context (max_num_rows) or the evaluation split instead.",
            file=sys.stderr,
        )
        return 1
    print("OK: chunked inference reproduces the single-pass predictions.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
