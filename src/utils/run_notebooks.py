"""Clear, restart-and-run the analysis notebooks, and collect their printed
output into one ``results/All_Results.md``.

What it does
------------
For every *included* notebook (all of ``notebooks/`` except the per-method
interactive tool ``Individual_Method_Runner``; ``Results_Checking`` IS re-run
but its printed output is not collected into ``All_Results.md``), in the same
order they appear in the folder:

1. **Clear** every output + execution count (a clean slate on disk).
2. **Restart & run** the whole notebook with a FRESH kernel -- nbconvert's
   ``--execute`` starts a brand-new kernel per notebook, so this is exactly the
   "Restart Kernel and Run All Cells" you get in the IDE, then writes the fresh
   outputs back in place.
3. **Harvest** the printed (stdout) text of every cell and write it as that
   notebook's section in ``results/All_Results.md``. Sections are ordered like
   the notebooks folder; re-running a notebook deletes and rewrites only its own
   section, leaving the others untouched.

Notebooks run **in parallel** (``-j``, default ``min(4, CPUs)``): each one is an
independent nbconvert subprocess with its own kernel, its own figure directory,
and its own All_Results.md section, so nothing they write overlaps. The one
shared artifact -- the per-experiment summary CSVs that every notebook refreshes
on kernel start -- is built ONCE up front by this controller instead
(``TABPFNCREDIT_SKIP_AUTO_SUMMARIZE`` tells the kernels to skip their own
refresh), which removes both the write race and the redundant re-summarizing
that made even sequential runs slow (six experiment1 notebooks used to mean six
identical rebuilds). ``-j 1`` restores strictly sequential runs; ``-v`` implies
it (live-streamed kernel output cannot be interleaved).

The kernel is bound to the project venv (default ``<repo>/tabpfncreditvenv``):
the notebooks are executed by *that* interpreter regardless of which Python runs
this script, because we shell out to ``<venv>/python -m nbconvert`` with
``--ExecutePreprocessor.kernel_name=python3`` (which resolves to the venv's own
ipykernel). The controller logic below uses only the standard library, so it is
safe to launch with any interpreter.

CLI
---
    # everything: clear -> restart+run -> rebuild All_Results.md
    python -m src.utils.run_notebooks

    # just one/some notebooks (their All_Results.md sections are rewritten)
    python -m src.utils.run_notebooks Experiment1.2-PD-Stat Experiment2.1-PD

    python -m src.utils.run_notebooks --list           # show run order, do nothing
    python -m src.utils.run_notebooks -j 6             # more parallel kernels
    python -m src.utils.run_notebooks -j 1             # strictly sequential
    python -m src.utils.run_notebooks --md-only        # only (re)collect output -> md
    python -m src.utils.run_notebooks --clear-only     # only clear outputs
    python -m src.utils.run_notebooks --no-md          # run but don't touch the md
    python -m src.utils.run_notebooks --venv D:/envs/x # different venv
    python -m src.utils.run_notebooks --timeout 1800   # per-cell timeout (default: none)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import PROJECT_ROOT, results_root  # noqa: E402

# The per-method tool is never auto-run (it's parametric/interactive).
RUN_SKIP = {"Individual_Method_Runner"}
# These never get a section in All_Results.md. Results_Checking IS run (a QA
# audit), but its output is not a result to collect; the method runner is neither.
NO_COLLECT = {"Individual_Method_Runner", "Results_Checking"}
# Back-compat alias (older callers / tests referenced this name).
EXEMPT_STEMS = NO_COLLECT
DEFAULT_VENV = PROJECT_ROOT / "tabpfncreditvenv"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_MD_NAME = "All_Results.md"


# ============================================================================
#  Notebook discovery / ordering
# ============================================================================

def _natural_key(name: str):
    """Sort key so Experiment1.2 < Experiment1.10 < Experiment2.1 (folder order)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


def discover_notebooks(notebooks_dir: Path = NOTEBOOKS_DIR,
                       *, include_exempt: bool = False) -> List[Path]:
    """All runnable ``.ipynb`` in folder (natural-sort) order. ``Results_Checking``
    IS included (it gets re-run); only the per-method tool is dropped, unless
    ``include_exempt``. Hidden/checkpoint notebooks are ignored."""
    nbs = [p for p in notebooks_dir.glob("*.ipynb")
           if ".ipynb_checkpoints" not in p.parts]
    if not include_exempt:
        nbs = [p for p in nbs if p.stem not in RUN_SKIP]
    return sorted(nbs, key=lambda p: _natural_key(p.name))


def resolve_targets(names: Sequence[str], notebooks_dir: Path = NOTEBOOKS_DIR) -> List[Path]:
    """Map user-supplied names/stems/paths to notebook paths, preserving folder
    order. Accepts ``Experiment2.1-PD``, ``Experiment2.1-PD.ipynb`` or a path."""
    if not names:
        return discover_notebooks(notebooks_dir)
    by_stem = {p.stem: p for p in notebooks_dir.glob("*.ipynb")}
    chosen: List[Path] = []
    for n in names:
        cand = Path(n)
        if cand.exists() and cand.suffix == ".ipynb":
            chosen.append(cand.resolve())
            continue
        stem = cand.stem if cand.suffix == ".ipynb" else n
        if stem in by_stem:
            chosen.append(by_stem[stem])
        else:
            raise SystemExit(f"Unknown notebook: {n!r}. Known: {sorted(by_stem)}")
    # de-dup, keep folder order
    uniq = {p.resolve(): p for p in chosen}
    return sorted(uniq.values(), key=lambda p: _natural_key(p.name))


# ============================================================================
#  Notebook clearing (stdlib only -- no nbformat dependency on the controller)
# ============================================================================

# ============================================================================
#  Notebook file I/O that survives a cloud-sync client / AV scanner
# ============================================================================
#
# This repo is often checked out inside a synced folder (OneDrive, Google Drive).
# Those clients open a file for a few tens of milliseconds after it changes, and
# so do AV scanners. With ``-j N`` several kernels rewrite their own notebook at
# the same moment (nbconvert ``--inplace``), so a read or write that lands in
# that window fails -- on Windows with ``OSError: [Errno 22] Invalid argument``
# naming the .ipynb. Serial runs almost never see it; parallel runs did, seven
# notebooks into a 12-notebook batch.
#
# Retrying is the right response: the condition is transient by construction.
# Losing a 100-second notebook (and cancelling the queue behind it) is not.
#
# How long "transient" is depends on the sync client, not on us. A 5 MB notebook
# replaced while Google Drive is uploading it can stay locked for tens of
# seconds -- the earlier 5 attempts from 0.2 s (3 s of total patience) gave up
# well inside that window and failed two notebooks with WinError 5.

#: Attempts, and the base for the exponential backoff between them. Doubling
#: from 0.4 s over 8 attempts waits ~51 s in total before giving up.
_IO_ATTEMPTS = 8
_IO_BACKOFF_S = 0.4

#: Serialises the single synced-file write each notebook makes. The workers are
#: threads in one process, so one lock removes the whole contention spike: with
#: -j 4, four multi-megabyte replaces used to land on the sync client at once.
#: The write itself takes ~50 ms, so holding this costs nothing measurable while
#: execution -- the part that actually takes minutes -- stays fully parallel.
_WRITE_LOCK = threading.Lock()


def _retry_io(action, path: Path, what: str):
    """Run ``action`` , retrying transient OSErrors on ``path``."""
    delay = _IO_BACKOFF_S
    for attempt in range(1, _IO_ATTEMPTS + 1):
        try:
            return action()
        except OSError as exc:
            if attempt == _IO_ATTEMPTS:
                # Do not leave the sibling temp file behind as debris.
                tmp = path.with_name(path.name + ".tmp")
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:                  # pragma: no cover -- best effort
                    pass
                raise OSError(
                    f"could not {what} {path.name} after {_IO_ATTEMPTS} attempts "
                    f"over ~{_IO_BACKOFF_S * (2 ** _IO_ATTEMPTS - 2):.0f}s "
                    f"({type(exc).__name__}: {exc}). If this repo sits in a "
                    f"OneDrive/Google Drive folder, pause syncing or exclude it "
                    f"-- or rerun with -j 1."
                ) from exc
            time.sleep(delay)
            delay *= 2


def read_notebook_text(path: Path) -> str:
    """``path.read_text`` with retries (see :func:`_retry_io`)."""
    return _retry_io(lambda: path.read_text(encoding="utf-8"), path, "read")


def write_notebook_text(path: Path, text: str) -> None:
    """Atomically replace ``path``'s contents, with retries.

    ``Path.write_text`` truncates the file and then fills it, so a concurrent
    reader can see an empty or partial notebook. Writing a sibling temp file and
    ``os.replace``-ing it is atomic on Windows and POSIX alike, which removes
    that window entirely.
    """
    def _do() -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    with _WRITE_LOCK:                # see _WRITE_LOCK: never two replaces at once
        _retry_io(_do, path, "write")


def clear_notebook(path: Path) -> None:
    """Strip every code cell's outputs and reset its execution count, in place."""
    nb = json.loads(read_notebook_text(path))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    write_notebook_text(path, json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


# ============================================================================
#  Output harvesting -> Markdown
# ============================================================================

def _as_text(value) -> str:
    """nbformat stores stream/text either as a str or a list of str lines."""
    if isinstance(value, list):
        return "".join(value)
    return value or ""


def harvest_stdout(nb: dict, *, include_results: bool = False,
                   include_errors: bool = True) -> str:
    """Concatenate the printed (stdout) text of every code cell, in cell order.

    ``include_results`` also pulls ``text/plain`` execute_result / display_data
    (off by default -- it tends to duplicate printed text as an ugly repr).
    ``include_errors`` appends a one-line marker for any cell that raised, so a
    failed run is never silently blank.
    """
    chunks: List[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            ot = out.get("output_type")
            if ot == "stream" and out.get("name") == "stdout":
                chunks.append(_as_text(out.get("text")))
            elif ot in ("execute_result", "display_data") and include_results:
                txt = (out.get("data") or {}).get("text/plain")
                if txt is not None:
                    chunks.append(_as_text(txt) + "\n")
            elif ot == "error" and include_errors:
                ename, evalue = out.get("ename", "Error"), out.get("evalue", "")
                chunks.append(f"\n[!] cell raised {ename}: {evalue}\n")
    return "".join(chunks).strip("\n")


def notebook_title(nb: dict, fallback: str) -> str:
    """First top-level Markdown heading (``# ...``) in the notebook, else fallback."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            for line in _as_text("".join(cell.get("source", []))).splitlines():
                m = re.match(r"#{1,3}\s+(.*\S)", line.strip())
                if m:
                    return m.group(1).strip()
    return fallback


def _fence_for(body: str) -> str:
    """A code fence longer than any backtick run in ``body`` (so it never breaks)."""
    longest = max((len(m) for m in re.findall(r"`+", body)), default=0)
    return "`" * max(3, longest + 1)


def render_block(stem: str, title: str, rel_path: str, body: str, stamp: str) -> str:
    """One self-delimited All_Results.md section, fenced as monospace text."""
    body = body.strip("\n") or "(no printed output)"
    fence = _fence_for(body)
    return (
        f"<!-- nb:START {stem} -->\n"
        f"## {title}\n\n"
        f"*Source: `notebooks/{rel_path}` — collected {stamp}*\n\n"
        f"{fence}text\n{body}\n{fence}\n"
        f"<!-- nb:END {stem} -->"
    )


_BLOCK_RE = re.compile(r"<!-- nb:START (?P<stem>\S+) -->.*?<!-- nb:END (?P=stem) -->",
                       re.DOTALL)


def parse_existing_blocks(md_text: str) -> Dict[str, str]:
    """Map ``stem -> full block text`` (markers included) from an All_Results.md."""
    return {m.group("stem"): m.group(0) for m in _BLOCK_RE.finditer(md_text)}


def update_all_results_md(md_path: Path, fresh_blocks: Dict[str, str],
                          ordered_stems: List[str], *, stamp: str) -> None:
    """Rewrite ``All_Results.md`` so each notebook keeps one section in folder
    order: freshly-run notebooks get a new block, the rest are preserved verbatim,
    and sections whose stem is no longer in ``ordered_stems`` (a renamed or removed
    notebook) are pruned so the file never accumulates stale sections.

    ``ordered_stems`` is the canonical (folder-order) list of *collected* notebook
    stems; a section appears only if its stem is in that list and it was either
    just run or already present.
    """
    existing = parse_existing_blocks(md_path.read_text(encoding="utf-8")) if md_path.exists() else {}
    merged = {**existing, **fresh_blocks}

    blocks = [merged[s] for s in ordered_stems if s in merged]
    # Drop any pre-existing section whose stem is no longer a collected notebook
    # (e.g. a renamed or removed notebook) so the file never accumulates stale
    # sections -- but report what was pruned, never silently.
    orphans = sorted(s for s in existing if s not in ordered_stems)
    if orphans:
        print(f"   (pruned {len(orphans)} stale All_Results.md section(s): {', '.join(orphans)})")

    header = (
        "# TabPFNCredit — All Results\n\n"
        f"*Auto-generated by `src/utils/run_notebooks.py` — last updated {stamp}.*\n\n"
        "## What this file is\n\n"
        "The **complete text record of every analysis in this repository**: one "
        "section per analysis notebook, in the same order as the `notebooks/` "
        "folder. Each notebook ends with a single call to "
        "`src.utils.report.notebook_report`, so its section here is one labelled "
        "report containing *every* table and every figure-backing number that "
        "notebook produces — you can read the results without opening a notebook "
        "or a PDF.\n\n"
        "## How to read a section\n\n"
        "Each report starts with a header naming the source notebook and "
        "summarising its contents, then numbered sections `[1] … [n]` — coverage "
        "(methods × datasets), per-method metric tables, ranks, tuning effects, "
        "head-to-head comparisons, calibration numbers, sweep evolutions and the "
        "statistical batteries, depending on the notebook. Dataset names are the "
        "paper display names (proprietary datasets appear anonymised); method "
        "names are the standard figure labels.\n\n"
        "The corresponding figures live under `figures/<experiment>/`, with "
        "paper-ready captions in `figures/CAPTIONS.md`.\n\n"
        "Re-running one notebook rewrites only its own section.\n\n"
        "<!-- This file is generated; edit the notebooks, not this file. -->\n"
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(header + "\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")


# ============================================================================
#  Execution (shell out to the venv interpreter so the kernel is the venv)
# ============================================================================

def venv_python(venv_dir: Path) -> Optional[Path]:
    """The interpreter inside ``venv_dir`` (Windows ``Scripts`` or POSIX ``bin``)."""
    for cand in (venv_dir / "Scripts" / "python.exe", venv_dir / "bin" / "python"):
        if cand.exists():
            return cand
    return None


def _check_nbconvert(py: Path) -> None:
    """Fail early, naming exactly which notebook-execution package is missing."""
    probe = subprocess.run(
        [str(py), "-c",
         "import importlib.util as u;"
         "print(' '.join(m for m in ('nbconvert', 'ipykernel') if u.find_spec(m) is None))"],
        capture_output=True, text=True)
    missing = (probe.stdout or "").strip() or ("nbconvert ipykernel" if probe.returncode else "")
    if missing:
        raise SystemExit(
            f"The venv at {py} is missing: {missing.replace(' ', ', ')} "
            "(needed to run notebooks).\n"
            f"   With the venv active:  python -m pip install {missing}\n"
            f"   Or explicitly:         \"{py}\" -m pip install {missing}")


def execute_notebook(path: Path, py: Path, *, timeout: int, allow_errors: bool,
                     kernel_name: str, out_dir: Path, verbose: bool = False,
                     extra_env: Optional[Dict[str, str]] = None) -> Path:
    """Restart-and-run ``path`` with a fresh venv kernel; return the executed copy.

    The executed notebook is written into ``out_dir`` (keep that OFF a synced
    filesystem) and the input is left untouched -- see the comment on ``cmd``.

    Quiet by default: nbconvert's own chatter and the harmless Windows
    ``zmq`` ``add_reader`` RuntimeWarning are captured (shown only on failure).
    ``verbose=True`` streams everything live for debugging. ``extra_env`` adds
    variables to the kernel's environment (e.g. the skip-auto-summarize flag)."""
    # NOT --inplace. nbconvert would truncate + refill the .ipynb in place, and
    # this repo commonly lives in a synced folder (OneDrive / Google Drive) whose
    # client opens every changed file: with -j N that window is where reads and
    # writes fail with EINVAL. Writing the executed copy to `out_dir` (a temp
    # directory off the synced tree) means the synced notebook is replaced once,
    # atomically, by the caller after execution -- never mid-run.
    #
    # The INPUT path stays the real notebook, so nbconvert still runs the kernel
    # with cwd = notebooks/. That is required: the notebooks derive PROJECT_ROOT
    # from Path.cwd(), so executing a copy elsewhere would break every relative
    # path in them.
    cmd = [str(py), "-m", "nbconvert", "--to", "notebook", "--execute",
           "--output-dir", str(out_dir), "--output", path.name,
           "--log-level", "WARN",
           f"--ExecutePreprocessor.timeout={timeout}",
           f"--ExecutePreprocessor.kernel_name={kernel_name}"]
    if allow_errors:
        cmd.append("--ExecutePreprocessor.allow_errors=True")
    cmd.append(str(path))
    # Silence the cosmetic "Proactor event loop ... add_reader" zmq warning the
    # kernel emits on Windows (kept alongside any filter the user already set).
    zmq_filter = "ignore::RuntimeWarning:zmq._future"
    prior = os.environ.get("PYTHONWARNINGS")
    env = {
        **os.environ,
        "PYTHONWARNINGS": f"{zmq_filter},{prior}" if prior else zmq_filter,
        # Direct Jupyter/VS Code notebook runs refresh captions after each
        # saved project figure. Here we run headlessly and refresh once below,
        # so suppress the per-save hook to avoid repeated CAPTIONS.md writes.
        "TABPFNCREDIT_AUTO_CAPTIONS": "0",
        **(extra_env or {}),
    }
    if verbose:
        subprocess.run(cmd, check=True, env=env)
    else:
        subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    executed = out_dir / path.name
    if not executed.exists():                 # nbconvert changed its naming
        candidates = sorted(out_dir.glob("*.ipynb"))
        if not candidates:
            raise FileNotFoundError(
                f"nbconvert reported success but wrote no notebook into {out_dir}")
        executed = candidates[0]
    return executed


# ============================================================================
#  Up-front summary build (once per experiment, instead of once per kernel)
# ============================================================================

_EXP_STEM_RE = re.compile(r"^Experiment(\d+)", re.IGNORECASE)


def _experiments_of(targets: Sequence[Path]) -> List[str]:
    """The ``experimentN`` result folders the target notebooks read from."""
    return sorted({f"experiment{m.group(1)}"
                   for p in targets if (m := _EXP_STEM_RE.match(p.stem))})


def presummarize_experiments(py: Path, targets: Sequence[Path], *,
                             jobs: int = 1, verbose: bool = False) -> bool:
    """Build each target experiment's summary CSVs ONCE, before any kernel starts.

    Every notebook otherwise refreshes its experiment's CSVs on kernel start
    (``load_summary(auto_summarize=True)``): on a full run that is six identical
    experiment1 rebuilds, and under parallel execution it is a write race on the
    shared CSVs. Building them here (in venv subprocesses, so this controller
    stays stdlib-only) and setting ``TABPFNCREDIT_SKIP_AUTO_SUMMARIZE`` for the
    kernels fixes both. Experiments with no local result files are skipped (a
    CSV-only download reads the existing CSVs as-is). Returns True when the
    pass ran (even if some experiment failed -- failures are reported loudly and
    the kernels' missing-CSV fallback still applies).
    """
    root = results_root()
    todo = [exp for exp in _experiments_of(targets)
            if (root / exp).is_dir() and next((root / exp).rglob("*.json"), None) is not None]
    if not todo:
        return False
    code = ("import sys; from src.utils.result_summary import summarize_to_csv; "
            "from src.utils.paths import results_root; base = results_root(); "
            "summarize_to_csv(base=base, experiment=sys.argv[1], out_dir=base / 'summaries')")

    def _one(exp: str) -> None:
        subprocess.run([str(py), "-c", code, exp], check=True, cwd=str(PROJECT_ROOT),
                       capture_output=not verbose, text=True)

    t0 = time.perf_counter()
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(todo)))) as pool:
        futs = {pool.submit(_one, exp): exp for exp in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except subprocess.CalledProcessError as exc:
                failures.append(futs[fut])
                tail = ((exc.stderr or "") + (exc.stdout or "")).strip().splitlines()[-4:]
                print(f"   [warn] pre-summarize failed for {futs[fut]}: " + " | ".join(tail))
    kept = [e for e in todo if e not in failures]
    print(f"summaries: rebuilt {', '.join(kept) or '(none)'} once up front "
          f"({time.perf_counter() - t0:.1f}s)"
          + (f" -- FAILED: {', '.join(failures)}" if failures else ""))
    return True


def _print_error_tail(exc: subprocess.CalledProcessError, n: int = 18) -> None:
    """Show the tail of a failed nbconvert run (the actual cell traceback)."""
    txt = ((exc.stderr or "") + (exc.stdout or "")).strip()
    if not txt:
        return
    print("        ---- nbconvert output (tail) ----")
    for line in txt.splitlines()[-n:]:
        print(f"        {line}")


# ============================================================================
#  Orchestration
# ============================================================================

def _process_notebook(nb_path: Path, *, py: Optional[Path], do_clear: bool,
                      do_execute: bool, do_md: bool, is_collected: bool,
                      timeout: int, allow_errors: bool, kernel_name: str,
                      verbose: bool, include_results: bool, stamp: str,
                      extra_env: Optional[Dict[str, str]]) -> Tuple[List[str], Optional[str], float]:
    """Clear + execute + harvest ONE notebook (the unit of parallel work).

    Everything here touches only this notebook's own files (its .ipynb, its
    figure directory); the shared All_Results.md is assembled by the caller.
    Returns ``(step descriptions, md block or None, seconds)``; raises on failure.
    """
    t0 = time.perf_counter()
    steps: List[str] = []
    block: Optional[str] = None
    # Clearing on disk is only worth a synced-file write when we are NOT about to
    # execute: --execute re-runs every cell and replaces every output, and the
    # result is copied back below, so a pre-clear would be a second write for no
    # observable difference.
    if do_clear and not do_execute:
        clear_notebook(nb_path)
        steps.append("cleared")
    source = nb_path
    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    try:
        if do_execute:
            # tempfile uses %TEMP% / $TMPDIR -- local disk, never synced.
            tmp_dir = tempfile.TemporaryDirectory(prefix="tabpfncredit_nb_")
            source = execute_notebook(
                nb_path, py, timeout=timeout, allow_errors=allow_errors,
                kernel_name=kernel_name, out_dir=Path(tmp_dir.name),
                verbose=verbose, extra_env=extra_env)
            steps.append("ran")
        if do_md and is_collected:
            nb = json.loads(read_notebook_text(source))
            body = harvest_stdout(nb, include_results=include_results)
            block = render_block(nb_path.stem, notebook_title(nb, nb_path.stem),
                                 nb_path.name, body, stamp)
            steps.append(f"{len(body):,} chars")
        elif do_md:
            steps.append("not collected")
        if do_execute:
            # The ONE write the sync client sees for this notebook, and it is an
            # atomic replace rather than a truncate-then-fill.
            write_notebook_text(nb_path, read_notebook_text(source))
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir.name, ignore_errors=True)
    return steps, block, time.perf_counter() - t0


def run(targets: List[Path], *, py: Optional[Path], do_clear: bool, do_execute: bool,
        do_md: bool, md_path: Path, timeout: int, allow_errors: bool,
        kernel_name: str, continue_on_error: bool, include_results: bool,
        verbose: bool = False, do_captions: bool = True, jobs: int = 1) -> int:
    """Drive clear/execute/collect across ``targets``. Returns a process exit code.

    One concise line per notebook: ``[i/N] <name> … ✓  12.3s  (cleared, ran, 5,512 chars)``.
    On failure the tail of nbconvert's output is shown so the error is visible.
    ``jobs > 1`` runs that many notebooks concurrently (each is its own kernel
    subprocess); completion lines then appear in finish order, and on a failure
    without ``--continue-on-error`` the queued notebooks are cancelled while the
    already-running ones finish.
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    # Folder-order list of the notebooks that DO get an All_Results.md section
    # (Results_Checking is run but excluded here, so it isn't collected).
    included_order = [p.stem for p in discover_notebooks() if p.stem not in NO_COLLECT]
    fresh: Dict[str, str] = {}
    ok, failed, cancelled = [], [], []
    n = len(targets)
    width = max((len(p.stem) for p in targets), default=10)
    # Status glyphs degrade to ASCII on a non-UTF-8 console (Windows cp1252)
    # so a tick/cross never raises UnicodeEncodeError mid-run.
    _uni = "utf" in (getattr(sys.stdout, "encoding", "") or "").lower()
    M_OK, M_FAIL = ("✓", "✗") if _uni else ("ok", "FAIL")

    # Build each target experiment's summary CSVs once, up front, and tell the
    # kernels to skip their own per-session refresh (see presummarize_experiments).
    extra_env: Dict[str, str] = {}
    if do_execute and py is not None:
        if presummarize_experiments(py, targets, jobs=max(1, jobs), verbose=verbose):
            extra_env["TABPFNCREDIT_SKIP_AUTO_SUMMARIZE"] = "1"

    def _kwargs(nb_path: Path) -> dict:
        return dict(py=py, do_clear=do_clear, do_execute=do_execute, do_md=do_md,
                    is_collected=nb_path.stem in included_order, timeout=timeout,
                    allow_errors=allow_errors, kernel_name=kernel_name, verbose=verbose,
                    include_results=include_results, stamp=stamp, extra_env=extra_env)

    def _tag(nb_path: Path) -> str:
        return "" if nb_path.stem in included_order else " (run-only)"

    t_all = time.perf_counter()
    if jobs <= 1:
        for i, nb_path in enumerate(targets, 1):
            stem = nb_path.stem
            print(f"[{i}/{n}] {stem:<{width}}{_tag(nb_path)} ... ",
                  end="\n" if verbose else "", flush=True)
            try:
                steps, block, dt = _process_notebook(nb_path, **_kwargs(nb_path))
                if block is not None:
                    fresh[stem] = block
                print(f"{M_OK} {dt:5.1f}s  ({', '.join(steps) or 'ok'})")
                ok.append(stem)
            except subprocess.CalledProcessError as exc:
                print(f"{M_FAIL}  (exit {exc.returncode})")
                failed.append(stem)
                if not verbose:                     # in verbose the tail already streamed
                    _print_error_tail(exc)
                if not continue_on_error:
                    print("        stopped -- fix the error, or pass --continue-on-error.")
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"{M_FAIL}  {type(exc).__name__}: {exc}")
                failed.append(stem)
                if not continue_on_error:
                    break
    else:
        # Submit in folder order; print in completion order (prefixed by name, so
        # interleaving is unambiguous). All printing happens on this thread.
        stopping = False
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_process_notebook, p, **_kwargs(p)): p for p in targets}
            done_count = 0
            for fut in as_completed(futs):
                nb_path = futs[fut]
                stem = nb_path.stem
                done_count += 1
                prefix = f"[{done_count}/{n}] {stem:<{width}}{_tag(nb_path)} ... "
                try:
                    steps, block, dt = fut.result()
                except CancelledError:
                    cancelled.append(stem)
                    print(f"{prefix}-- cancelled (earlier failure)")
                    continue
                except subprocess.CalledProcessError as exc:
                    print(f"{prefix}{M_FAIL}  (exit {exc.returncode})")
                    failed.append(stem)
                    _print_error_tail(exc)
                    if not continue_on_error and not stopping:
                        stopping = True
                        n_cancelled = sum(f.cancel() for f in futs)
                        if n_cancelled:
                            print(f"        cancelling {n_cancelled} queued notebook(s); "
                                  "already-running ones finish. "
                                  "(--continue-on-error disables this.)")
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f"{prefix}{M_FAIL}  {type(exc).__name__}: {exc}")
                    failed.append(stem)
                    if not continue_on_error and not stopping:
                        stopping = True
                        for f in futs:
                            f.cancel()
                    continue
                if block is not None:
                    fresh[stem] = block
                print(f"{prefix}{M_OK} {dt:5.1f}s  ({', '.join(steps) or 'ok'})")
                ok.append(stem)

    if do_md and fresh:
        update_all_results_md(md_path, fresh, included_order, stamp=stamp)
        print(f"\nAll_Results.md  ->  {md_path}  ({len(fresh)} section{'s' if len(fresh) != 1 else ''})")

    # Refresh the consolidated figure captions from whatever figures the run wrote.
    if do_captions and do_execute and ok:
        try:
            from src.utils.generate_captions import generate_captions
            written = generate_captions(PROJECT_ROOT / "figures")
            print(f"captions      ->  {written[0] if written else '(no figures found)'}")
        except Exception as exc:  # noqa: BLE001 -- captions are non-critical
            print(f"(caption regeneration skipped: {type(exc).__name__}: {exc})")

    summary = f"\nDone: {len(ok)}/{n} ok in {time.perf_counter() - t_all:.0f}s"
    if failed:
        summary += f", {len(failed)} FAILED: {', '.join(failed)}"
    if cancelled:
        summary += f", {len(cancelled)} cancelled"
    print(summary)
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    # Print UTF-8 regardless of the console code page (Windows cp1252 would
    # otherwise crash on a tick/arrow). errors="replace" keeps it crash-proof.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover -- not all streams support it
            pass

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebooks", nargs="*",
                    help="specific notebooks (name/stem/path); default: all included, in folder order")
    ap.add_argument("--list", action="store_true", help="print the run order and exit")
    ap.add_argument("--md-only", action="store_true",
                    help="don't clear/run; only (re)collect current outputs into All_Results.md")
    ap.add_argument("--clear-only", action="store_true", help="only clear outputs; don't run or collect")
    ap.add_argument("--no-md", action="store_true", help="run but leave All_Results.md untouched")
    ap.add_argument("--no-clear", action="store_true", help="skip the pre-run clear step")
    ap.add_argument("--allow-errors", action="store_true",
                    help="let a notebook finish past a cell error (errors still flagged in the md)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep processing the remaining notebooks if one fails")
    ap.add_argument("--include-results", action="store_true",
                    help="also collect text/plain Out[] results, not just printed stdout")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="notebooks to run concurrently, each in its own kernel process "
                         "(default: 0 = auto, min(4, CPUs); 1 = strictly sequential)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="stream raw nbconvert/kernel output live (default: quiet, shown only on failure; implies -j 1)")
    ap.add_argument("--no-captions", action="store_true",
                    help="don't regenerate figures/CAPTIONS.md after running")
    ap.add_argument("--venv", type=Path, default=DEFAULT_VENV,
                    help=f"project venv whose kernel runs the notebooks (default: {DEFAULT_VENV})")
    ap.add_argument("--python", type=Path, default=None,
                    help="explicit interpreter to run the notebooks (overrides --venv)")
    ap.add_argument("--kernel-name", default="python3", help="ipykernel spec name (default: python3)")
    ap.add_argument("--timeout", type=int, default=-1, help="per-cell timeout in s (default: -1 = none)")
    ap.add_argument("--results-md", type=Path, default=None,
                    help=f"All_Results.md location (default: <results_root>/{RESULTS_MD_NAME})")
    args = ap.parse_args(argv)

    targets = resolve_targets(args.notebooks)
    if args.list:
        print("Run order:")
        for p in targets:
            note = "  (run-only, not collected)" if p.stem in NO_COLLECT else ""
            print(f"  {p.stem}{note}")
        print(f"\nNever auto-run: {sorted(RUN_SKIP)}")
        return 0

    do_clear = not (args.md_only or args.no_clear)
    do_execute = not (args.md_only or args.clear_only)
    do_md = not (args.no_md or args.clear_only)
    md_path = args.results_md or (results_root() / RESULTS_MD_NAME)

    py: Optional[Path] = None
    if do_execute:
        py = args.python or venv_python(args.venv)
        if py is None or not Path(py).exists():
            raise SystemExit(
                f"No interpreter found (looked in {args.venv}). "
                f"Pass --venv <dir> or --python <path>.")
        _check_nbconvert(Path(py))

    # Parallelism: auto = min(4, CPUs) kernels. Verbose streams raw kernel
    # output, which cannot be interleaved -> forced sequential. Non-execute
    # passes (md-only / clear-only) are I/O-trivial -> sequential too.
    jobs = args.jobs if args.jobs > 0 else min(4, os.cpu_count() or 2)
    jobs = max(1, min(jobs, len(targets)))
    if args.verbose and jobs > 1:
        print("(-v streams kernel output live; forcing -j 1)")
        jobs = 1
    if not do_execute:
        jobs = 1

    steps = "+".join(s for s, on in (("clear", do_clear), ("run", do_execute), ("collect", do_md)) if on)
    # Report the interpreter and the KERNELSPEC separately. This line used to
    # print ``py.name`` under the label "kernel:", i.e. "kernel: python.exe",
    # which is the interpreter filename and not a kernelspec at all -- copying
    # it into --kernel-name fails with NoSuchKernel.
    print(f"{len(targets)} notebook(s) | {steps} | python: {py.name if py else '(none)'} "
          f"| kernel: {args.kernel_name} "
          f"| jobs: {jobs} | md: {md_path.name if do_md else 'skipped'}")

    return run(targets, py=py, do_clear=do_clear, do_execute=do_execute, do_md=do_md,
               md_path=md_path, timeout=args.timeout, allow_errors=args.allow_errors,
               kernel_name=args.kernel_name, continue_on_error=args.continue_on_error,
               include_results=args.include_results, verbose=args.verbose,
               do_captions=not args.no_captions, jobs=jobs)


__all__ = [
    "discover_notebooks", "resolve_targets", "clear_notebook", "harvest_stdout",
    "notebook_title", "render_block", "parse_existing_blocks", "update_all_results_md",
    "venv_python", "execute_notebook", "presummarize_experiments", "run", "main",
    "RUN_SKIP", "NO_COLLECT", "EXEMPT_STEMS", "DEFAULT_VENV", "RESULTS_MD_NAME",
]


if __name__ == "__main__":
    raise SystemExit(main())
