"""Clear, restart-and-run the analysis notebooks, and collect their printed
output into one ``results/All_Results.md``.

What it does
------------
For every *included* notebook (all of ``notebooks/`` EXCEPT the two interactive
tools -- ``Results_Checking`` and ``Individual_Method_Runner``), in the same
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
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import PROJECT_ROOT, results_root  # noqa: E402

# Interactive tools that are NOT result notebooks: never auto-run, never collected.
EXEMPT_STEMS = {"Results_Checking", "Individual_Method_Runner"}
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
    """All included ``.ipynb`` in folder (natural-sort) order; exempt ones dropped
    unless ``include_exempt``. Hidden/checkpoint notebooks are ignored."""
    nbs = [p for p in notebooks_dir.glob("*.ipynb")
           if ".ipynb_checkpoints" not in p.parts]
    if not include_exempt:
        nbs = [p for p in nbs if p.stem not in EXEMPT_STEMS]
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

def clear_notebook(path: Path) -> None:
    """Strip every code cell's outputs and reset its execution count, in place."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


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
    order: freshly-run notebooks get a new block, the rest are preserved verbatim.

    ``ordered_stems`` is the canonical (folder-order) list of *included* notebook
    stems. A section appears only if it was just run OR already existed.
    """
    existing = parse_existing_blocks(md_path.read_text(encoding="utf-8")) if md_path.exists() else {}
    merged = {**existing, **fresh_blocks}

    blocks = [merged[s] for s in ordered_stems if s in merged]
    # Any pre-existing section whose notebook is no longer included is kept at the
    # end rather than silently dropped.
    for stem, blk in existing.items():
        if stem not in ordered_stems and stem not in fresh_blocks:
            blocks.append(blk)

    toc = "\n".join(
        f"- [{s}](#{s.lower().replace('.', '').replace('_', '-')})"
        for s in ordered_stems if s in merged)
    header = (
        "# TabPFNCredit — All Results\n\n"
        f"*Auto-generated by `src/utils/run_notebooks.py` — last updated {stamp}.*\n\n"
        "Each section below is the printed output of one analysis notebook, in the "
        "same order as the `notebooks/` folder. Re-running a notebook rewrites only "
        "its own section.\n\n"
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
                     kernel_name: str) -> None:
    """Restart-and-run ``path`` in place with a fresh venv kernel (raises on failure)."""
    cmd = [str(py), "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           f"--ExecutePreprocessor.timeout={timeout}",
           f"--ExecutePreprocessor.kernel_name={kernel_name}"]
    if allow_errors:
        cmd.append("--ExecutePreprocessor.allow_errors=True")
    cmd.append(str(path))
    subprocess.run(cmd, check=True)


# ============================================================================
#  Orchestration
# ============================================================================

def run(targets: List[Path], *, py: Optional[Path], do_clear: bool, do_execute: bool,
        do_md: bool, md_path: Path, timeout: int, allow_errors: bool,
        kernel_name: str, continue_on_error: bool, include_results: bool) -> int:
    """Drive clear/execute/collect across ``targets``. Returns a process exit code."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    included_order = [p.stem for p in discover_notebooks()]
    fresh: Dict[str, str] = {}
    ok, failed = [], []

    for nb_path in targets:
        stem = nb_path.stem
        is_collected = stem in included_order  # exempt notebooks run but aren't collected
        print(f"\n=== {stem} {'(run only, not collected)' if not is_collected else ''} ===")
        try:
            if do_clear:
                print("  • clearing outputs")
                clear_notebook(nb_path)
            if do_execute:
                print(f"  • restart & run  ({py})")
                execute_notebook(nb_path, py, timeout=timeout, allow_errors=allow_errors,
                                 kernel_name=kernel_name)
            if do_md and is_collected:
                nb = json.loads(nb_path.read_text(encoding="utf-8"))
                body = harvest_stdout(nb, include_results=include_results)
                title = notebook_title(nb, stem)
                fresh[stem] = render_block(stem, title, nb_path.name, body, stamp)
                print(f"  • collected {len(body)} chars of printed output")
            ok.append(stem)
        except subprocess.CalledProcessError as exc:
            failed.append(stem)
            print(f"  ✗ execution FAILED (exit {exc.returncode})")
            if not continue_on_error:
                print("  (stopping; pass --continue-on-error to keep going)")
                break
        except Exception as exc:  # noqa: BLE001
            failed.append(stem)
            print(f"  ✗ {type(exc).__name__}: {exc}")
            if not continue_on_error:
                break

    if do_md and fresh:
        update_all_results_md(md_path, fresh, included_order, stamp=stamp)
        print(f"\nAll_Results.md updated ({len(fresh)} section(s)) -> {md_path}")

    print(f"\nDone. ok={ok or '-'}  failed={failed or '-'}")
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
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
        print("Run order (included notebooks):")
        for p in targets:
            print(f"  {p.stem}")
        print(f"\nExempt (never auto-run/collected): {sorted(EXEMPT_STEMS)}")
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

    print(f"Notebooks   : {[p.stem for p in targets]}")
    print(f"Interpreter : {py or '(not running — md/clear only)'}")
    print(f"All_Results : {md_path if do_md else '(skipped)'}")

    return run(targets, py=py, do_clear=do_clear, do_execute=do_execute, do_md=do_md,
               md_path=md_path, timeout=args.timeout, allow_errors=args.allow_errors,
               kernel_name=args.kernel_name, continue_on_error=args.continue_on_error,
               include_results=args.include_results)


__all__ = [
    "discover_notebooks", "resolve_targets", "clear_notebook", "harvest_stdout",
    "notebook_title", "render_block", "parse_existing_blocks", "update_all_results_md",
    "venv_python", "execute_notebook", "run", "main",
    "EXEMPT_STEMS", "DEFAULT_VENV", "RESULTS_MD_NAME",
]


if __name__ == "__main__":
    raise SystemExit(main())
