"""Unit tests for src.utils.run_notebooks (pure logic — no kernel launched)."""
from __future__ import annotations

import json

import pytest

from src.utils import run_notebooks as rn


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _nb(*cells):
    return {"cells": list(cells), "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(stdout=None, *, results=None, error=None):
    outs = []
    if stdout is not None:
        outs.append({"output_type": "stream", "name": "stdout", "text": stdout})
    if results is not None:
        outs.append({"output_type": "execute_result", "data": {"text/plain": results},
                     "execution_count": 1, "metadata": {}})
    if error is not None:
        outs.append({"output_type": "error", "ename": error, "evalue": "boom", "traceback": []})
    return {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": outs,
            "source": ["print('x')\n"]}


# --------------------------------------------------------------------------- #
#  ordering / discovery
# --------------------------------------------------------------------------- #
def test_natural_sort_orders_experiments_correctly():
    names = ["Experiment2.1-PD.ipynb", "Experiment1.10-X.ipynb", "Experiment1.2-PD-Stat.ipynb",
             "Experiment0.ipynb", "Data_Exploration.ipynb"]
    ordered = sorted(names, key=rn._natural_key)
    assert ordered == ["Data_Exploration.ipynb", "Experiment0.ipynb",
                       "Experiment1.2-PD-Stat.ipynb", "Experiment1.10-X.ipynb",
                       "Experiment2.1-PD.ipynb"]


def test_discover_runs_results_checking_but_not_method_runner(tmp_path):
    nbdir = tmp_path / "notebooks"
    nbdir.mkdir()
    for name in ["Experiment0.ipynb", "Experiment1.1-PD.ipynb",
                 "Results_Checking.ipynb", "Individual_Method_Runner.ipynb"]:
        (nbdir / name).write_text(json.dumps(_nb()), encoding="utf-8")
    stems = [p.stem for p in rn.discover_notebooks(nbdir)]
    # Results_Checking IS discovered (it gets re-run); the method runner is not.
    assert stems == ["Experiment0", "Experiment1.1-PD", "Results_Checking"]
    assert "Individual_Method_Runner" not in stems
    # ...but Results_Checking is never collected into All_Results.md.
    assert "Results_Checking" in rn.NO_COLLECT and "Individual_Method_Runner" in rn.NO_COLLECT
    assert rn.RUN_SKIP == {"Individual_Method_Runner"}
    stems_all = [p.stem for p in rn.discover_notebooks(nbdir, include_exempt=True)]
    assert "Individual_Method_Runner" in stems_all


# --------------------------------------------------------------------------- #
#  clear
# --------------------------------------------------------------------------- #
def test_clear_notebook_strips_outputs(tmp_path):
    p = tmp_path / "n.ipynb"
    p.write_text(json.dumps(_nb(_code("hello\n"), _md("# Title"))), encoding="utf-8")
    rn.clear_notebook(p)
    nb = json.loads(p.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"][0]
    assert code["outputs"] == [] and code["execution_count"] is None


# --------------------------------------------------------------------------- #
#  harvest
# --------------------------------------------------------------------------- #
def test_harvest_collects_only_stdout_by_default():
    nb = _nb(_md("# Heading"),
             _code("first table\n"),
             _code(stdout="second\n", results="'IGNORED REPR'"))
    body = rn.harvest_stdout(nb)
    assert "first table" in body and "second" in body
    assert "IGNORED REPR" not in body          # text/plain skipped by default


def test_harvest_can_include_results_and_flags_errors():
    nb = _nb(_code(stdout="out\n", results="42"), _code(error="ValueError"))
    body = rn.harvest_stdout(nb, include_results=True)
    assert "42" in body
    assert "ValueError" in body and "cell raised" in body


def test_harvest_handles_str_and_list_text():
    nb = _nb(_code(stdout=["a\n", "b\n"]), _code(stdout="c\n"))
    body = rn.harvest_stdout(nb)
    assert body.splitlines() == ["a", "b", "c"]


def test_notebook_title_prefers_first_heading():
    nb = _nb(_md("intro\n## 1. PAMA\n"), _code("x\n"))
    assert rn.notebook_title(nb, "fallback") == "1. PAMA"
    assert rn.notebook_title(_nb(_code("x\n")), "fallback") == "fallback"


# --------------------------------------------------------------------------- #
#  markdown block round-trip + merge ordering + idempotency
# --------------------------------------------------------------------------- #
def test_block_roundtrips_through_parser():
    blk = rn.render_block("Experiment0", "Exp 0", "Experiment0.ipynb", "the output", "2026-01-01 00:00")
    parsed = rn.parse_existing_blocks("junk\n" + blk + "\nmore junk")
    assert set(parsed) == {"Experiment0"}
    assert parsed["Experiment0"] == blk


def test_fence_survives_backticks_in_body():
    body = "a code ``` fence inside"
    blk = rn.render_block("X", "t", "X.ipynb", body, "s")
    assert "````text" in blk           # bumped to 4 backticks so it doesn't break


def test_update_md_orders_and_replaces_only_rerun_section(tmp_path):
    md = tmp_path / "All_Results.md"
    order = ["Experiment0", "Experiment1.1-PD", "Experiment2.1-PD"]

    # First write: two of three sections.
    rn.update_all_results_md(
        md,
        {"Experiment1.1-PD": rn.render_block("Experiment1.1-PD", "1.1", "Experiment1.1-PD.ipynb", "AAA", "s"),
         "Experiment0": rn.render_block("Experiment0", "0", "Experiment0.ipynb", "ZERO", "s")},
        order, stamp="s")
    txt1 = md.read_text(encoding="utf-8")
    # Sections appear in folder order regardless of insertion order.
    assert txt1.index("nb:START Experiment0") < txt1.index("nb:START Experiment1.1-PD")
    assert "Experiment2.1-PD" not in txt1

    # Re-run ONLY Experiment0 with new content + add Experiment2.1-PD.
    rn.update_all_results_md(
        md,
        {"Experiment0": rn.render_block("Experiment0", "0", "Experiment0.ipynb", "ZERO-v2", "s2"),
         "Experiment2.1-PD": rn.render_block("Experiment2.1-PD", "2.1", "Experiment2.1-PD.ipynb", "TWO", "s2")},
        order, stamp="s2")
    txt2 = md.read_text(encoding="utf-8")

    assert "ZERO-v2" in txt2 and "ZERO\n" not in txt2.replace("ZERO-v2", "")   # old content gone
    assert "AAA" in txt2                                                        # untouched section preserved
    assert "TWO" in txt2                                                        # new section added
    # Canonical order maintained across all three.
    assert (txt2.index("nb:START Experiment0")
            < txt2.index("nb:START Experiment1.1-PD")
            < txt2.index("nb:START Experiment2.1-PD"))


def test_update_md_is_idempotent(tmp_path):
    md = tmp_path / "All_Results.md"
    order = ["Experiment0", "Experiment1.1-PD"]
    blocks = {"Experiment0": rn.render_block("Experiment0", "0", "Experiment0.ipynb", "z", "s"),
              "Experiment1.1-PD": rn.render_block("Experiment1.1-PD", "1", "Experiment1.1-PD.ipynb", "o", "s")}
    rn.update_all_results_md(md, blocks, order, stamp="s")
    first = md.read_text(encoding="utf-8")
    # Re-collecting the same blocks (preserving the rest) yields a byte-identical file.
    rn.update_all_results_md(md, blocks, order, stamp="s")
    assert md.read_text(encoding="utf-8") == first


def test_venv_python_detection(tmp_path):
    # POSIX layout
    (tmp_path / "bin").mkdir()
    py = tmp_path / "bin" / "python"
    py.write_text("")
    assert rn.venv_python(tmp_path) == py
    # absent
    assert rn.venv_python(tmp_path / "nope") is None
