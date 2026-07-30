"""One consolidated text report per analysis notebook.

Why
---
``run_notebooks`` harvests each notebook's **stdout** into
``results/All_Results.md``. When printing was scattered across a notebook, that
file became a pile of unlabelled fragments -- and anything shown with
``display(...)`` never arrived at all, because a rich display is not stdout.

So every analysis notebook now ends with exactly ONE call to
:func:`notebook_report`, which prints a single sectioned block: a header saying
which notebook produced it and what it contains, then one numbered section per
piece of the analysis. Everything a reader needs is in that one block, in text,
and therefore in ``All_Results.md``.

Usage
-----
::

    from src.utils.report import notebook_report, capture, section

    notebook_report(
        "Experiment 1 - PD",
        notebook="Experiment1.1-PD.ipynb",
        about="5-fold CV headline benchmark on the 14 PD datasets ...",
        sections=[
            section("Coverage", f"{n_methods} methods x {n_datasets} datasets"),
            section("Headline metrics", capture(pd_summary_text, df, task_name="PD")),
            section("Calibration", capture(calibration_summary_text, tab, ...)),
        ],
    )

:func:`capture` runs an existing *printing* helper and returns what it printed
instead, so the report reuses the very same text the individual helpers produce
-- no duplicated formatting logic, and no signature churn in the plot module.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional,
                    Sequence, Tuple)

if TYPE_CHECKING:                    # pandas is imported lazily where it is used
    import pandas as pd

#: A report section: ``(heading, body)``.
Section = Tuple[str, str]

_RULE = "=" * 78
_SUB = "-" * 78


def capture(fn: Callable, *args, **kwargs) -> str:
    """Call ``fn`` and return its printed output as text (nothing is printed).

    Used to fold the existing ``*_summary_text`` / ``statistical_report``
    helpers into the single report without changing their signatures.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        fn(*args, **kwargs)
    return buffer.getvalue().rstrip("\n")


def section(heading: str, body: object) -> Section:
    """Build one report section from a heading and any stringifiable body.

    A DataFrame / Series body is rendered with ``to_string()`` so tables that
    used to be shown with ``display(...)`` -- and were therefore invisible in
    ``All_Results.md`` -- end up in the text report.
    """
    if body is None:
        text = "(nothing to report)"
    elif hasattr(body, "to_string"):
        text = body.to_string()
    else:
        text = str(body)
    return (str(heading), text.rstrip("\n") or "(empty)")


def notebook_report(
    title: str,
    *,
    notebook: str,
    about: str,
    sections: Sequence[Section],
    echo: bool = True,
) -> str:
    """Print (and return) ONE labelled text report for a whole notebook.

    Args:
        title: Report title, e.g. ``"Experiment 1 - PD"``.
        notebook: File name of the producing notebook, for provenance.
        about: One or two sentences describing what this report contains, so a
            reader of ``All_Results.md`` knows what they are looking at without
            opening the notebook.
        sections: ``(heading, body)`` pairs, in reading order; build them with
            :func:`section` / :func:`capture`. Empty-bodied sections are kept
            (with a placeholder) so a missing piece is visible, never silent.
        echo: Print the report (default). ``False`` only returns the string.

    Returns:
        The full report text.
    """
    lines: List[str] = [
        _RULE,
        f"{title.upper()} -- COMPLETE TEXT SUMMARY",
        _RULE,
        f"Source notebook : notebooks/{notebook}",
        f"Contents        : {about}",
        f"Sections        : {len(sections)}",
        "",
        "This is the notebook's ONLY printed output: every table and figure-backing",
        "number it produces is reproduced below, so results/All_Results.md is a",
        "complete text record of the analysis.",
    ]
    for index, (heading, body) in enumerate(sections, start=1):
        lines += ["", _SUB, f"[{index}] {heading}", _SUB, body]
    lines += ["", _RULE, f"END OF {title.upper()} SUMMARY", _RULE]
    text = "\n".join(lines)
    if echo:
        print(text)
    return text


__all__ = ["Section", "capture", "section", "notebook_report"]


# ---------------------------------------------------------------------------
#  In-notebook previews (rendered, NOT printed)
# ---------------------------------------------------------------------------
#
# Every notebook has exactly ONE printed report, at the end, and that is what
# ``run_notebooks`` harvests into results/All_Results.md. Consolidating to that
# single report meant deleting the per-cell ``print`` calls -- which left cells
# that compute a result and show nothing, sitting under a markdown header
# promising an analysis. They read as "this cell did not run".
#
# ``preview`` closes that gap without breaking the one-report rule: it goes
# through IPython's DISPLAY channel, and the harvester collects stdout only
# (``harvest_stdout(..., include_results=False)``). So the value is visible in
# the notebook and absent from All_Results.md, which is exactly what we want --
# the report remains the single source of the printed record.

def preview(value: Any, *, label: Optional[str] = None) -> None:
    """Render ``value`` in the notebook without printing it.

    Falls back to no-op-safe behaviour outside IPython (e.g. under pytest), so
    calling it from a plain Python process is harmless.
    """
    try:
        from IPython.display import display  # noqa: PLC0415 -- optional dep
    except ImportError:                       # pragma: no cover -- not in a kernel
        return
    try:
        if label:
            from IPython.display import Markdown
            display(Markdown(f"**{label}**"))
        display(value)
    except Exception:                         # pragma: no cover -- never break a run
        pass


def omnibus_frame(**tests: Dict[str, float]) -> "pd.DataFrame":
    """Tidy one or more omnibus-test result dicts into a comparable table.

    ``omnibus_frame(Friedman=f, **{"Aligned ranks": fa}, Quade=qu)`` -> one row
    per test, so the notebook can show them side by side under its header.
    """
    import pandas as pd

    rows = {name: dict(result) for name, result in tests.items() if result}
    frame = pd.DataFrame(rows).T
    return frame.round(6)


__all__ += ["preview", "omnibus_frame"]
