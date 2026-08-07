"""Static checks on the notebooks, so a broken cell fails in a second.

A notebook cell that references a name nobody imported only shows up when the
notebook is actually executed -- four minutes into a 12-notebook run, which then
cancels everything queued behind it. That happened twice while consolidating the
notebooks' printed output: once for ``_pd`` and once for ``display_name``, both
used by a newly added ``preview(...)`` call in notebooks whose setup cell did not
import them.

pyflakes over the notebook's concatenated cells catches exactly that. Cells run
in order and share one namespace, so concatenating them is a faithful model of
the kernel's view -- good enough to spot an undefined name, which is the bug
class worth guarding.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))

pytestmark = pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks found")


#: Names an IPython kernel injects into the user namespace, so a notebook may
#: use them without importing anything. Declared here so the check models the
#: kernel accurately instead of reporting them as undefined.
_KERNEL_BUILTINS = (
    "from IPython.display import display, display_html, display_markdown, HTML, Markdown",
    "get_ipython = None",
)


def _flatten(nb_path: Path) -> str:
    """All code cells, in order, as one module, after the kernel's own injections.

    IPython-only lines (``%magic``, ``!shell``) are blanked -- they are not
    Python and would be reported as syntax errors.
    """
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out: list[str] = [*_KERNEL_BUILTINS, ""]
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            out.append("" if re.match(r"\s*[%!]", line) else line)
        out.append("")
    return "\n".join(out) + "\n"


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.stem)
def test_every_code_cell_parses(nb_path: Path):
    """A saved notebook must never contain unparseable Python."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if re.match(r"\s*[%!]", source):        # magic-only cell
            continue
        try:
            compile(_blank_magics(source), f"{nb_path.name}#cell{index}", "exec")
        except SyntaxError as exc:              # pragma: no cover -- the point
            pytest.fail(f"{nb_path.name} cell {index}: {exc.msg} (line {exc.lineno})")


def _blank_magics(source: str) -> str:
    return "\n".join("" if re.match(r"\s*[%!]", ln) else ln
                     for ln in source.splitlines())


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_undefined_names(nb_path: Path, tmp_path: Path):
    """No cell may use a name that no earlier cell imports or defines."""
    pyflakes = pytest.importorskip("pyflakes", reason="pyflakes not installed")
    del pyflakes

    flat = tmp_path / f"{nb_path.stem}.py"
    flat.write_text(_flatten(nb_path), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", str(flat)],
        capture_output=True, text=True,
    )
    undefined = [
        line for line in (proc.stdout + proc.stderr).splitlines()
        if "undefined name" in line
    ]
    assert not undefined, (
        f"{nb_path.name} uses names nothing imports or defines:\n  "
        + "\n  ".join(u.replace(str(flat), nb_path.name) for u in undefined)
    )


# ---------------------------------------------------------------------------
#  A report section must deliver what its title promises
# ---------------------------------------------------------------------------
#
# The printed report is what reaches results/All_Results.md, so a section titled
# "... Bayesian ROPE" that prints no ROPE block makes the paper unauditable. That
# happened: statistical_report only emitted the block when `focus=` named >= 2
# methods, and no notebook passed it.

def _report_calls(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", []))
                     for c in nb.get("cells", []) if c.get("cell_type") == "code")


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.stem)
def test_statistical_sections_pass_what_they_promise(nb_path: Path):
    src = _report_calls(nb_path)
    if "statistical_report" not in src:
        pytest.skip("no statistical report in this notebook")

    call = src[src.index("st.statistical_report"):]
    call = call[:call.index("))") + 2] if "))" in call else call

    titles = " ".join(re.findall(r"section\(\s*['\"]([^'\"]+)", src)).lower()

    if "rope" in titles or "bayesian" in titles:
        assert "focus=" in call, (
            f"{nb_path.name} promises a Bayesian ROPE section but does not pass "
            f"focus= to statistical_report, so the block is auto-selected rather "
            f"than the methods the notebook actually shows")
    if "control" in titles:
        assert "control=" in call, (
            f"{nb_path.name} promises a control-based section but does not pass "
            f"control= to statistical_report, so the control table is never printed")
