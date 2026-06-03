"""Tests for src.visualizations.calibration_plots.

The reliability diagram / grid now save in **PDF only** and additionally
display the figure inline in Jupyter. The tests verify the on-disk side
of that contract -- any extension passed via ``out_path`` is normalised
to ``.pdf`` by the implementation.
"""

from __future__ import annotations

import numpy as np

from src.visualizations.calibration_plots import (
    reliability_diagram,
    reliability_grid,
)


def test_reliability_diagram_writes_pdf(tmp_path, synthetic_probas):
    y_true, y_proba = synthetic_probas
    # Pass a .png path on purpose -- the implementation must rewrite it
    # to .pdf, since we only save PDFs now.
    out = tmp_path / "reliability.png"
    result = reliability_diagram(y_true, y_proba, out_path=out)
    expected = out.with_suffix(".pdf")
    assert result == expected
    assert expected.exists()
    assert expected.stat().st_size > 0
    # The .png MUST NOT have been written.
    assert not out.exists()


def test_reliability_diagram_pdf_extension_preserved(tmp_path, synthetic_probas):
    y_true, y_proba = synthetic_probas
    out = tmp_path / "reliability.pdf"
    result = reliability_diagram(y_true, y_proba, out_path=out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_reliability_grid(tmp_path, synthetic_probas):
    y_true, y_proba = synthetic_probas
    runs = {
        "method_a": {"y_true": y_true, "y_proba": y_proba},
        "method_b": {"y_true": y_true, "y_proba": y_proba * 0.5 + 0.25},
        "method_c": {"y_true": y_true, "y_proba": y_proba},
    }
    out = tmp_path / "grid.pdf"
    result = reliability_grid(runs, out_path=out, ncols=2)
    assert result == out
    assert out.exists()


def test_handles_1d_proba(tmp_path):
    rng = np.random.default_rng(0)
    y_true = (rng.random(100) < 0.5).astype(int)
    proba_1d = rng.random(100)
    out = tmp_path / "single_proba.pdf"
    reliability_diagram(y_true, proba_1d, out_path=out)
    assert out.exists()
