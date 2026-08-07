"""Every test the report promises must actually be printed.

``statistical_report`` is the single source of the statistical numbers that reach
``results/All_Results.md``. It used to emit the Bayesian ROPE block only when the
caller passed ``focus=`` with at least two matching methods -- and no notebook
passed it, so the battery silently stopped at [7] while the section header
promised "omnibus tests, post-hoc, Bayesian ROPE". The numbers were visible in
the notebook (rendered through IPython's display channel) but absent from the
report, so nothing in the paper could be audited against them.

A missing block is invisible to ordinary tests: the report still "works", it is
just short. These tests assert the presence of each numbered section, which is
the only thing that would have caught it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.utils.statistical_testing as st


@pytest.fixture
def synthetic():
    """8 datasets x 4 methods with a stable ranking, plus per-fold rows."""
    rng = np.random.default_rng(0)
    methods = ["tabpfn_v3", "catboost", "tabicl_v2", "LogReg"]
    offset = {"tabpfn_v3": 0.020, "catboost": 0.015, "tabicl_v2": 0.010, "LogReg": 0.0}
    per_method, per_fold = [], []
    for i in range(8):
        dataset, base = f"d{i}", rng.uniform(0.6, 0.8)
        for m in methods:
            value = base + rng.normal(offset[m], 0.01)
            per_method.append({"dataset": dataset, "method": m,
                               "metric.AUC_mean": value})
            for fold in range(1, 6):
                per_fold.append({"dataset": dataset, "method": m, "fold": fold,
                                 "metric.AUC": value + rng.normal(0, 0.005)})
    return pd.DataFrame(per_method), pd.DataFrame(per_fold), methods


def _report(synthetic, **kwargs) -> str:
    df, folds, _ = synthetic
    return st.statistical_report(df, folds, metric="AUC", task_name="T", **kwargs)


def test_every_numbered_section_is_present(synthetic, capsys):
    """[1] through [8] must all appear -- [8] is the one that went missing."""
    text = _report(synthetic, focus=synthetic[2])
    capsys.readouterr()
    for tag in ("[1]", "[2]", "[3]", "[4]", "[5]", "[6a]", "[6b]", "[7]", "[8]"):
        assert tag in text, f"section {tag} missing from the report"


def test_rope_block_is_printed_without_an_explicit_focus(synthetic, capsys):
    """The old bug exactly: no focus passed -> block silently dropped."""
    text = _report(synthetic)
    capsys.readouterr()
    assert "[8]" in text and "Bayesian" in text
    assert "auto-selected" in text, (
        "when no focus is given the report must say which methods it chose")
    assert "P(" in text, "the actual ROPE probabilities must be printed"


def test_rope_probabilities_are_real_numbers(synthetic, capsys):
    """A block that prints only 'nan' would pass a presence check but be useless."""
    import re

    text = _report(synthetic, focus=synthetic[2])
    capsys.readouterr()
    block = text.split("[8]", 1)[1]
    values = [float(v) for v in re.findall(r"= (\d+\.\d+)", block)]
    assert values, "no probabilities parsed out of the ROPE block"
    assert all(0.0 <= v <= 1.0 for v in values), f"not probabilities: {values}"
    assert not any(np.isnan(values)), "nan in the ROPE block"


def test_control_block_is_printed_and_populated(synthetic, capsys):
    """The champion notebooks promise 'control = TabPFN-3'; [9] must deliver it."""
    text = _report(synthetic, focus=synthetic[2], control="tabpfn_v3")
    capsys.readouterr()
    assert "[9]" in text, "control comparison missing"
    block = text.split("[9]", 1)[1]
    assert "Bonferroni-Dunn CD" in block
    assert "bonferroni_dunn" in block, "the adjusted p-value column must be shown"
    assert "nan" not in block.lower(), "control table printed nan p-values"
    for method in ("catboost", "tabicl_v2", "LogReg"):
        assert method in block, f"{method} missing from the control comparison"


def test_control_block_is_absent_when_no_control_is_given(synthetic, capsys):
    """All-learner notebooks have no control; [9] must not appear empty."""
    text = _report(synthetic, focus=synthetic[2])
    capsys.readouterr()
    assert "[9]" not in text


def test_unknown_control_is_reported_not_silently_skipped(synthetic, capsys):
    text = _report(synthetic, focus=synthetic[2], control="does_not_exist")
    capsys.readouterr()
    assert "[9]" in text and "unavailable" in text, (
        "a control that is not in the matrix must be reported, not dropped")


def test_the_report_reaches_stdout(synthetic, capsys):
    """run_notebooks harvests STDOUT only -- a report that merely returns a
    string would never reach All_Results.md."""
    _report(synthetic, focus=synthetic[2])
    printed = capsys.readouterr().out
    assert "[8]" in printed, "the report must be printed, not just returned"
