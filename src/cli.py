"""TabPFNCredit command-line interface.

One entry point for everything that matters:

::

    tabpfncredit experiment Experiment1 [--task pd|lgd]
                                        [--dataset 0001.gmsc]
                                        [--method tabpfn_v3]
                                        [--no-submit]
                                        [--after <SLURM_JOB_ID>]

What `experiment` does
----------------------
1. Load the experiment's three YAML configs.
2. Filter the (dataset, method, task) cells by the optional
   ``--task`` / ``--dataset`` / ``--method`` flags.
3. Auto-preprocess any dataset that is needed but not yet cached under
   ``data/processed/<task>/<dataset>/``.
4. **Locally** (no ``$VSC_INSTITUTE_CLUSTER``): run each cell in-process,
   then summarize. No SLURM, no scratch directories.
5. **On the VSC** (``$VSC_INSTITUTE_CLUSTER`` set): wipe any stale
   scripts under ``scripts/<Experiment>/_generated/``, regenerate fresh
   SLURM scripts, and ``sbatch`` them (unless ``--no-submit``). A final
   summarize job is submitted with ``--dependency=afterok:<arrays>``
   so the CSVs land automatically once every array slot finishes.

Helper commands
---------------
* ``resubmit`` -- scan results and submit ONLY the not-yet-done points
* ``summarize`` -- aggregate fold results into per-fold / per-method CSVs
  (user-facing; also invoked by the generated SLURM summarize job)
* ``list``   -- enumerate registered methods + their runtime profile
* ``doctor`` -- environment / VSC sanity check

Internal command (you should rarely type this by hand; it is called by the
generated SLURM scripts):

* ``slurm-task`` -- workhorse for one array slot
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import typer
from rich.console import Console
from rich.table import Table

# Path injection so `import src.*` works whether or not the package is editable-installed.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset_inventory import list_datasets  # noqa: E402
from src.methods.method_config import (  # noqa: E402
    CLASSICAL_METHODS,
    CPU_METHODS,
    DEEP_METHODS,
    FOUNDATION_METHODS,
    GPU_METHODS,
    HPO_METHODS,
)
from src.methods.method_runner import run_talent_method  # noqa: E402
from src.methods.runtime_profile import get_profile, estimate_point_seconds  # noqa: E402
from src.utils.slurm_generator import (  # noqa: E402
    PARTITIONS,
    generate_scripts_for_experiment,
    generate_summarize_script,
    load_plan,
    partition_for_method,
)
from src.utils.config_reader import load_config  # noqa: E402
from src.utils.paths import (  # noqa: E402
    describe as _describe_paths,
    results_root as _paths_results_root,
)
from src.utils.result_io import (  # noqa: E402
    build_method_name,
    has_complete_packed_point,
    has_complete_result,
    load_method,
    save_method,
    save_packed_point,
)
from src.utils.runtime_quiet import configure_quiet_runtime  # noqa: E402

app = typer.Typer(
    add_completion=False,
    help="TabPFNCredit: credit-risk tabular benchmarking on top of TALENT.",
)
console = Console()
logger = logging.getLogger("tabpfncredit")


# ============================================================================
#  Results-root resolution + VSC detection
# ============================================================================

def _results_root() -> Path:
    """Return the directory where result JSON/npz files live.

    Resolution (see :mod:`src.utils.paths`): ``$TABPFN_RESULTS_ROOT`` (set by
    the generated SLURM scripts to point at the shared project storage), else
    the project-storage ``results/`` when available, else ``<project>/results``
    for local runs. The directory is created if missing.
    """
    path = _paths_results_root()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _on_vsc() -> bool:
    """Best-effort detect whether we're running on a VSC cluster.

    Looks at ``$VSC_INSTITUTE_CLUSTER`` (set on Genius / wICE) and
    ``$VSC_HOME``. Returns ``False`` on workstations.
    """
    return bool(
        os.environ.get("VSC_INSTITUTE_CLUSTER") or os.environ.get("VSC_HOME")
    )


def _have_sbatch() -> bool:
    """Return True iff ``sbatch`` is on the PATH (best-effort)."""
    return shutil.which("sbatch") is not None


# ============================================================================
#  Cell list helpers
# ============================================================================

def _enabled_methods_for_task(config: dict, task: str) -> List[str]:
    return sorted(config.get("methods", {}).get(task, {}).keys())


def _enabled_datasets_for_task(config: dict, task: str) -> List[str]:
    return list(config.get("datasets", {}).get(task, {}).keys())


def _build_task_list(
    config: dict,
    *,
    task_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    method_filter: Optional[str] = None,
) -> List[dict]:
    """All ``(dataset, method, task)`` cells in ``config``, optionally filtered."""
    tasks = ("pd", "lgd") if task_filter is None else (task_filter,)
    cells: List[dict] = []
    for task in tasks:
        for dataset in _enabled_datasets_for_task(config, task):
            if dataset_filter and dataset != dataset_filter:
                continue
            for method in _enabled_methods_for_task(config, task):
                if method_filter and method != method_filter:
                    continue
                cells.append({"dataset": dataset, "method": method, "task": task})
    return cells


# ============================================================================
#  Preprocessing (auto-trigger if data/processed is missing)
# ============================================================================

def _preprocess_if_needed(cells: Sequence[dict]) -> set:
    """Ensure ``data/processed/<task>/<dataset>/y.npy`` exists for every cell.

    Calls :func:`src.data.preprocessing.preprocess_dataset` once per
    ``(task, dataset)`` tuple. Cached datasets are a no-op so calling this
    on every ``experiment`` invocation costs almost nothing.

    A dataset whose **raw** file is not present on this machine (common on
    a fresh cluster checkout -- ``data/`` is gitignored and must be copied
    separately) is skipped with a warning rather than crashing the whole
    sweep. Returns the set of ``(task, dataset)`` tuples that could not be
    made available, so the caller can drop their cells and run the rest.
    """
    from src.data.preprocessing import preprocess_dataset  # local import keeps startup snappy
    from src.utils.paths import find_processed_dir

    needed = sorted({(c["task"], c["dataset"]) for c in cells})
    to_make = [
        (task, dataset) for task, dataset in needed
        if find_processed_dir(task, dataset) is None
    ]

    unavailable: set = set()
    if not to_make:
        return unavailable

    console.print(f"[blue]Preprocessing {len(to_make)} dataset(s)...[/blue]")
    for task, dataset in to_make:
        try:
            preprocess_dataset(task, dataset)
            console.print(f"  [green]ok[/green]   {task}/{dataset}")
        except FileNotFoundError:
            # The RAW file genuinely isn't on this machine -- the dataset
            # cannot run anywhere, so drop its cells (with a warning).
            console.print(
                f"  [yellow]skip[/yellow] {task}/{dataset} "
                f"-- raw file not found under data/raw/{task}/ on this machine"
            )
            logger.warning("Skipping %s/%s: raw data file missing.", task, dataset)
            unavailable.add((task, dataset))
        except Exception as exc:  # pragma: no cover -- defensive
            # The raw file EXISTS but preprocessing failed HERE -- typically a
            # login node running out of memory on a large dataset. Keep the
            # cell: a compute node (much more RAM) preprocesses it inside
            # DataFeeder at run time. preprocess_dataset writes atomically and
            # is idempotent, so concurrent compute-node preprocessing is safe.
            # Run with --verbose for the full traceback.
            console.print(
                f"  [yellow]defer[/yellow] {task}/{dataset} -- could not preprocess "
                f"here ({type(exc).__name__}); the compute node will do it at run time"
            )
            logger.debug("Login-node preprocessing deferred for %s/%s", task, dataset, exc_info=True)
    return unavailable


# ============================================================================
#  Sweep expansion -- one (dataset, method, task) cell -> N sweep points
# ============================================================================
#
# This is what makes Experiment 1 (NO_HPO + HPO), Experiment 2 (training-size
# learning curve) and Experiment 3 (minority-proportion sweep) ACTUALLY sweep
# instead of running a single point. Each sweep point becomes its own
# ``<method>__<suffix>.{json,npz}`` result file so the points never collide.
# Both the local runner and the SLURM workhorse expand cells the same way
# (the config is re-read on the compute node), so the per-slot plan only
# needs to carry the bare {dataset, method, task}.

def _frange_desc(hi: float, lo: float, step: float) -> List[float]:
    """Inclusive descending range ``hi, hi-step, ..., >= lo`` (float-safe)."""
    out: List[float] = []
    v = float(hi)
    while v >= lo - 1e-9:
        out.append(round(v, 6))
        v -= step
    return out


def _sweep_points(experiment: str, config: dict, cell: dict) -> List[dict]:
    """Expand a cell into its sweep points.

    Each point is a dict with:
      * ``name``    -- result filename stem (method + sweep suffix)
      * ``tune``    -- HPO on/off for this point
      * ``row_limit`` / ``sampling`` -- per-point overrides for run_talent_method
    """
    method, task = cell["method"], cell["task"]
    exp = experiment.lower()
    base_row = config["split"].get("row_limit")
    base_sampling = config["split"].get("sampling")

    if exp == "experiment1":
        # Every method gets a NO_HPO point AND an HPO point, so the HPO-vs-NO_HPO
        # comparison is complete for all methods. Methods that support tuning get
        # a real tuned run; methods that can't be tuned (TabICL, TabPFN, ... are
        # NO_HPO-only) get an HPO point that simply COPIES their NO_HPO result --
        # their "tuned" performance is, by definition, their default performance.
        hpo_name = build_method_name(method, {"HPO": True})
        points = [{"name": method, "tune": False,
                   "row_limit": base_row, "sampling": base_sampling}]
        if method in HPO_METHODS:
            points.append({"name": hpo_name, "tune": True,
                           "row_limit": base_row, "sampling": base_sampling})
        else:
            points.append({"name": hpo_name, "tune": False, "copy_from": method,
                           "row_limit": base_row, "sampling": base_sampling})
        return points

    if exp == "experiment2":
        lc = (config.get("learning_curve") or {}).get(task) or {}
        rmax, rmin, rstep = lc.get("row_max"), lc.get("row_min"), lc.get("row_step")
        if not (rmax and rstep):
            return [{"name": method, "tune": False, "row_limit": base_row, "sampling": base_sampling}]
        rows = [int(r) for r in _frange_desc(rmax, rmin or 0, rstep)]
        return [{"name": build_method_name(method, {"row": r}), "tune": False,
                 "row_limit": r, "sampling": base_sampling} for r in rows]

    if exp == "experiment3":
        imb = config.get("imbalance") or {}
        pmax = imb.get("minority_proportion_max")
        pmin = imb.get("minority_proportion_min")
        pstep = imb.get("minority_proportion_step")
        if not (pmax and pstep):
            return [{"name": method, "tune": False, "row_limit": base_row, "sampling": base_sampling}]
        props = _frange_desc(pmax, pmin or 0, pstep)
        # One point per minority proportion. The whole dataset (train, val AND
        # test) is subsampled to the target rate, so evaluation happens on the
        # subsampled distribution. Removal is nested across the sweep (see
        # DataFeeder._nested_minority_keep_mask): a lower target's kept minority
        # is a strict subset of any higher target's, so the only thing changing
        # between points is how many minority rows remain.
        return [{"name": build_method_name(method, {"min": p}), "tune": False,
                 "row_limit": base_row, "sampling": p} for p in props]

    # Experiment 0 (and any unknown experiment): a single point, no sweep.
    return [{"name": method, "tune": False, "row_limit": base_row, "sampling": base_sampling}]


def _run_one_point(
    experiment: str, config: dict, cell: dict, point: dict, results_root: Path,
    *, verbose: bool = False,
) -> str:
    """Run + save ONE sweep point. Returns 'skip' / 'done' / 'fail'."""
    task, dataset, method = cell["task"], cell["dataset"], cell["method"]
    name = point["name"]
    cv = config["split"]["cv_splits"]
    # Experiments 2 & 3 PACK all of a cell's sweep points into one
    # <method>.json (metrics only), instead of one file per point — otherwise
    # the sweep exhausts the project-storage inode quota. The SLURM generator
    # keeps a cell's points in a single array task, so the packed file has a
    # single writer (no lock needed).
    packed = experiment.lower() in ("experiment2", "experiment3")
    if packed:
        already_done = has_complete_packed_point(
            base=results_root, experiment=experiment.lower(), task=task,
            dataset=dataset, method_base=method, point_name=name, expected_folds=cv,
        )
    else:
        already_done = has_complete_result(
            base=results_root, experiment=experiment.lower(),
            task=task, dataset=dataset, method=name, expected_folds=cv,
        )
    if already_done:
        console.print(f"  [yellow]skip[/yellow] {task}/{dataset}/{name} (already done)")
        return "skip"

    # HPO mode for a method that can't be tuned: its HPO result is just a copy of
    # its NO_HPO result. The NO_HPO point shares this cell (same array task) and
    # runs first, so the source result is already on disk -- no recompute.
    copy_from = point.get("copy_from")
    if copy_from:
        try:
            src_results = load_method(
                base=results_root, experiment=experiment.lower(),
                task=task, dataset=dataset, method=copy_from,
            )
        except FileNotFoundError:
            console.print(
                f"  [red]fail[/red] {task}/{dataset}/{name}: "
                f"NO_HPO source '{copy_from}' not found to copy"
            )
            return "fail"
        save_method(
            src_results, base=results_root, experiment=experiment.lower(),
            task=task, dataset=dataset, method=name,
        )
        console.print(
            f"  [bold]copy[/bold] {task}/{dataset}/{name} <- {copy_from} (method has no HPO)"
        )
        return "done"

    console.print(f"  [bold]run[/bold]  {task}/{dataset}/{name}")
    try:
        fold_results = run_talent_method(
            task=task, dataset=dataset, method=method,
            test_size=config["split"]["test_size"],
            val_size=config["split"]["val_size"],
            cv_splits=cv,
            seed=config["split"]["seed"],
            row_limit=point.get("row_limit"),
            sampling=point.get("sampling"),
            max_epoch=config["training"]["max_epochs"],
            batch_size=config["training"]["batch_size"],
            tune=point.get("tune", False),
            n_trials=config["tuning"]["n_trials"] if point.get("tune") else 1,
            early_stopping=config["training"]["early_stopping"],
            early_stopping_patience=config["training"]["early_stopping_patience"],
            verbose=verbose,
        )
        if packed:
            # Metrics only: drop the per-fold prediction arrays (no npz) and
            # append this point into the cell's single packed <method>.json.
            for _fold in fold_results.values():
                for _k in ("y_true", "y_prob", "y_pred", "val_y_true", "val_y_prob"):
                    if _k in _fold:
                        _fold[_k] = None
            save_packed_point(
                fold_results, base=results_root, experiment=experiment.lower(),
                task=task, dataset=dataset, method_base=method, point_name=name,
            )
        else:
            # One file per (dataset, method) point (Experiment 0/1).
            save_method(
                fold_results, base=results_root, experiment=experiment.lower(),
                task=task, dataset=dataset, method=name,
            )
        return "done"
    except Exception as exc:  # pragma: no cover -- defensive
        console.print(f"  [red]fail[/red] {task}/{dataset}/{name}: {exc}")
        logger.exception("Cell %s point %s failed", cell, name)
        return "fail"


# ============================================================================
#  Local runner (no SLURM)
# ============================================================================

def _run_cells_locally(experiment: str, cells: Sequence[dict], config: dict) -> int:
    """Run every cell (expanded into its sweep points) in-process.

    Returns the number of failed sweep points.
    """
    results_root = _results_root()
    n_done = n_skipped = n_failed = 0

    for cell in cells:
        for point in _sweep_points(experiment, config, cell):
            status = _run_one_point(experiment, config, cell, point, results_root)
            n_done += status == "done"
            n_skipped += status == "skip"
            n_failed += status == "fail"

    console.print(
        f"\n[green]done {n_done}[/green]  "
        f"[yellow]skipped {n_skipped}[/yellow]  "
        f"[red]failed {n_failed}[/red]  "
        f"out of {len(cells)} cell(s)"
    )
    return n_failed


# ============================================================================
#  VSC runner (auto-sbatch)
# ============================================================================

def _wipe_generated_dir(out_dir: Path) -> None:
    """Remove every ``*.slurm`` / ``*_plan.json`` file under ``out_dir``."""
    if not out_dir.exists():
        return
    for pattern in ("*.slurm", "*_plan.json"):
        for path in out_dir.glob(pattern):
            path.unlink()


def _sbatch(script: Path, *, dependency: Optional[str] = None) -> str:
    """Submit ``script`` via ``sbatch --parsable``; return the NUMERIC job ID.

    On the VSC, ``sbatch --parsable`` returns ``<jobid>;<cluster>`` when the
    job is sent to a non-default cluster (e.g. submitting a wICE script from
    a Genius login node yields ``60188739;wice``). We strip the ``;cluster``
    suffix and return just the numeric id, because SLURM ``afterok``
    dependencies are within a single cluster and want the bare number
    (the dependent job carries its own ``--clusters=`` directive).

    Raises :class:`RuntimeError` if the submission fails. ``dependency`` is
    a job-id spec (e.g. ``"afterok:12345:12346"``) added as
    ``--dependency=<dependency>``.
    """
    cmd: List[str] = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script))
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    raw = out.stdout.strip()
    if not raw:
        raise RuntimeError(f"sbatch produced no job id: {' '.join(shlex.quote(c) for c in cmd)}")
    # Keep only the numeric job id, dropping any ``;cluster`` suffix.
    return raw.split(";", 1)[0]


def _run_experiment_vsc(
    experiment: str,
    cells: Sequence[dict],
    config: dict,
    *,
    submit: bool,
    after_job_id: Optional[str],
) -> Optional[str]:
    """Generate scripts under ``scripts/<exp>/_generated/`` and (optionally) submit.

    Returns the job ID of the summarize step if anything was submitted,
    else ``None``.
    """
    out_dir = _PROJECT_ROOT / "scripts" / experiment / "_generated"

    # 1) Wipe stale scripts/plans so an old _generated/ directory doesn't
    # poison the next sbatch run.
    _wipe_generated_dir(out_dir)

    # 2) Expand every cell into its sweep POINTS and estimate each point's cost.
    #    The point -- not the cell -- is the scheduling unit, so a cell with
    #    thousands of points (Experiment 2's row sweep, Experiment 3's minority
    #    sweep) gets sharded across many array slots and each slot stays under
    #    the 72 h wall instead of one slot running the whole cell serially.
    cv = config["split"]["cv_splits"]
    n_trials = (config.get("tuning") or {}).get("n_trials", 1)
    work_items: List[dict] = []
    for cell in cells:
        for point in _sweep_points(experiment, config, cell):
            # An HPO point for a non-tunable method is a pure file COPY of its
            # NO_HPO result (see _run_one_point), so copy_from must travel into
            # the SLURM plan: the cluster then copies the result (seconds)
            # instead of re-running the model, which would double the GPU cost
            # and yield a non-identical result under GPU nondeterminism.
            copy_from = point.get("copy_from")
            work_items.append({
                "dataset": cell["dataset"],
                "method": cell["method"],
                "task": cell["task"],
                "name": point["name"],
                "tune": point.get("tune", False),
                "row_limit": point.get("row_limit"),
                "sampling": point.get("sampling"),
                "copy_from": copy_from,
                "est_seconds": 5 if copy_from else estimate_point_seconds(
                    cell["method"], n_folds=cv,
                    row_limit=point.get("row_limit"),
                    tune=point.get("tune", False), n_trials=n_trials,
                ),
            })

    # 3) Emit per-partition .slurm + .json plans (one array slot = a balanced
    #    bundle of points, packed under the 72 h wall).
    jobs = generate_scripts_for_experiment(
        experiment=experiment,
        work_items=work_items,
        out_dir=out_dir,
        n_folds=cv,
    )
    if not jobs:
        console.print("[red]No SLURM scripts were generated (no cells).[/red]")
        return None

    summarize_script = generate_summarize_script(
        experiment=experiment, out_dir=out_dir,
    )

    table = Table(title=f"Generated {len(jobs)} SLURM script(s) + 1 summarize")
    table.add_column("Script")
    table.add_column("Partition")
    table.add_column("Array slots")
    table.add_column("Walltime / slot")
    for j in jobs:
        table.add_row(str(j.path.name), j.partition_key, str(j.n_array_slots), j.walltime)
    table.add_row(str(summarize_script.name), "cpu", "1", "00:15:00")
    console.print(table)
    console.print(f"Written under: [blue]{out_dir}[/blue]")

    if not submit:
        console.print("\n[bold]--no-submit[/bold]: not submitting. To submit:")
        for j in jobs:
            console.print(f"  sbatch {j.path}")
        console.print(f"  sbatch --dependency=afterok:<ARRAY_IDS> {summarize_script}")
        return None

    if not _have_sbatch():
        console.print("[red]sbatch not found on PATH -- run this on the VSC.[/red]")
        return None

    # 4) Submit. Per-partition arrays go first (optionally chained to a
    # caller-supplied job via --after); the summarize job depends on them.
    # SLURM ``afterok`` CANNOT cross clusters, so arrays on a different
    # cluster than the summarize job (e.g. Genius V100/P100 offload) are
    # excluded from the dependency -- and cannot themselves wait on --after.
    summarize_cluster = PARTITIONS["cpu"].cluster
    array_dep: Optional[str] = f"afterok:{after_job_id}" if after_job_id else None
    dep_ids: List[str] = []
    cross_ids: List[str] = []
    for j in jobs:
        cluster = PARTITIONS[j.partition_key].cluster
        try:
            jid = _sbatch(j.path,
                          dependency=array_dep if cluster == summarize_cluster else None)
            console.print(f"  [green]submitted[/green] {j.path.name} -> {jid}")
            (dep_ids if cluster == summarize_cluster else cross_ids).append(jid)
        except subprocess.CalledProcessError as exc:
            console.print(f"[red]sbatch failed for {j.path}:[/red] {exc.stderr}")
            raise

    if cross_ids:
        console.print(
            f"  [yellow]note:[/yellow] job(s) {', '.join(cross_ids)} run on another "
            f"cluster; the summarize job cannot wait for them (afterok is "
            f"per-cluster). Re-run [bold]tabpfncredit summarize[/bold] once they finish."
        )
    summarize_dep = ("afterok:" + ":".join(dep_ids)) if dep_ids else None
    summarize_id = _sbatch(summarize_script, dependency=summarize_dep)
    console.print(f"  [green]submitted[/green] {summarize_script.name} -> {summarize_id}")
    console.print(f"\n[bold]Final job id (summarize):[/bold] {summarize_id}")
    return summarize_id


# ============================================================================
#  `experiment` -- the only command most users need to know
# ============================================================================

@app.command("experiment")
def cmd_experiment(
    name: str = typer.Argument(..., help="e.g. 'Experiment0' .. 'Experiment3'."),
    task: Optional[str] = typer.Option(None, help="Run only 'pd' or 'lgd' cells."),
    dataset: Optional[str] = typer.Option(None, help="Run only cells for this dataset name."),
    method: Optional[str] = typer.Option(None, help="Run only cells for this method name."),
    submit: Optional[bool] = typer.Option(
        None,
        help="Force on/off the auto-sbatch step. Default: on when on VSC, off locally.",
    ),
    after: Optional[str] = typer.Option(
        None,
        help="SLURM job id to chain this experiment after (via --dependency=afterok:JOBID).",
    ),
    verbose: bool = typer.Option(False, help="DEBUG-level logs."),
) -> None:
    """Run one experiment end-to-end (auto-preprocess + auto-SLURM + auto-summarize)."""
    configure_quiet_runtime()
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    config = load_config(name)
    cells = _build_task_list(
        config, task_filter=task, dataset_filter=dataset, method_filter=method,
    )
    if not cells:
        console.print(
            f"[red]No cells to run for {name} "
            f"(task={task}, dataset={dataset}, method={method}).[/red]"
        )
        raise typer.Exit(code=1)

    console.print(
        f"[bold]{name}[/bold]: {len(cells)} cell(s) to run "
        f"(filters: task={task or 'all'}, dataset={dataset or 'all'}, method={method or 'all'})"
    )

    # 1) Preprocess any missing dataset. Datasets whose raw file isn't on
    #    this machine are skipped (with a warning) and their cells dropped,
    #    so a missing dataset can't abort the whole sweep.
    unavailable = _preprocess_if_needed(cells)
    if unavailable:
        cells = [c for c in cells if (c["task"], c["dataset"]) not in unavailable]
        console.print(
            f"[yellow]Skipped {len(unavailable)} unavailable dataset(s); "
            f"{len(cells)} cell(s) remain.[/yellow]"
        )
    if not cells:
        console.print(
            "[red]No runnable cells remain -- every requested dataset is missing "
            "its raw file under data/raw/. Copy the data to this machine first.[/red]"
        )
        raise typer.Exit(code=1)

    # 2) Run locally or via SLURM.
    use_slurm = submit if submit is not None else _on_vsc()

    if use_slurm:
        _run_experiment_vsc(
            name, cells, config,
            submit=submit if submit is not None else True,
            after_job_id=after,
        )
    else:
        if after is not None:
            console.print("[yellow]--after is ignored when running locally.[/yellow]")
        _run_cells_locally(name, cells, config)
        # Local: summarize right away (results are on disk).
        _summarize_now(name)


def _summarize_now(experiment: str) -> None:
    """Run summarization synchronously (used at the end of a local run)."""
    from src.utils.result_summary import summarize_to_csv

    out_dir = _results_root() / "summaries"
    try:
        paths = summarize_to_csv(
            base=_results_root(), experiment=experiment.lower(), out_dir=out_dir,
        )
    except Exception as exc:  # pragma: no cover -- defensive
        console.print(f"[red]summarize failed:[/red] {exc}")
        return
    for p in paths:
        console.print(f"[green]wrote[/green] {p}")


# ============================================================================
#  `resubmit` -- pack ONLY the not-yet-done points into fresh dense arrays
# ============================================================================

@app.command("resubmit")
def cmd_resubmit(
    names: Optional[List[str]] = typer.Argument(
        None, help="Experiment names (e.g. Experiment1 Experiment2). Omit with --all."
    ),
    all_experiments: bool = typer.Option(
        False, "--all", help="Scan and resubmit every experiment (0-3) at once."
    ),
    submit: Optional[bool] = typer.Option(
        None, help="Force on/off the auto-sbatch step. Default: on when on VSC, off locally.",
    ),
    verbose: bool = typer.Option(False, help="DEBUG-level logs."),
) -> None:
    """Scan results for missing (task, dataset, method[, sweep/HPO]) points and
    submit ONLY those.

    Unlike re-running ``experiment`` (which re-shards done + missing points and
    queues slots that have nothing left to do), this packs the missing points
    into the smallest possible dense array. Works locally (reports + writes
    scripts) and on the VSC (also submits). Previous ``_generated/`` scripts
    and plans are wiped first so stale plans can never be picked up.
    """
    configure_quiet_runtime()
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    from src.utils.resubmit_planner import find_missing_work_items

    targets = (
        ["Experiment0", "Experiment1", "Experiment2", "Experiment3"]
        if all_experiments else list(names or [])
    )
    if not targets:
        console.print("[red]Pass experiment name(s) or --all.[/red]")
        raise typer.Exit(code=1)

    do_submit = submit if submit is not None else (_on_vsc() and _have_sbatch())
    console.print(
        "[yellow]Note:[/yellow] this wipes scripts/<Exp>/_generated/. If arrays from a "
        "previous submission are still PENDING they would read the new plans -- "
        "check [bold]squeue -u $USER[/bold] and scancel leftovers first.\n"
    )

    for exp in targets:
        console.print(f"[bold]== {exp} ==[/bold]")
        try:
            items, summary = find_missing_work_items(exp)
        except Exception as exc:
            console.print(f"[red]scan failed for {exp}: {exc}[/red]")
            continue
        console.print(
            f"expected {summary['expected']}  done {summary['done']}  "
            f"[bold]missing {summary['missing']}[/bold]  "
            f"(results root: {summary['results_root']})"
        )
        for m, n in summary["missing_by_method"].items():
            console.print(f"    {m}: {n}")

        out_dir = _PROJECT_ROOT / "scripts" / exp / "_generated"
        _wipe_generated_dir(out_dir)
        if not items:
            console.print("  [green]nothing to do.[/green]\n")
            continue

        cv = load_config(exp)["split"]["cv_splits"]
        jobs = generate_scripts_for_experiment(
            experiment=exp, work_items=items, out_dir=out_dir, n_folds=cv,
        )
        summarize_script = generate_summarize_script(experiment=exp, out_dir=out_dir)
        for j in jobs:
            console.print(
                f"  generated {j.path.name}  ({j.partition_key}, "
                f"{j.n_array_slots} slot(s), {j.walltime}/slot)"
            )

        if not do_submit:
            console.print("  not submitting (local / --no-submit). To submit on the VSC:")
            for j in jobs:
                console.print(f"    sbatch {j.path.name}")
            console.print("")
            continue

        summarize_cluster = PARTITIONS["cpu"].cluster
        dep_ids: List[str] = []
        cross_ids: List[str] = []
        for j in jobs:
            jid = _sbatch(j.path)
            console.print(f"  [green]submitted[/green] {j.path.name} -> {jid}")
            (dep_ids if PARTITIONS[j.partition_key].cluster == summarize_cluster
             else cross_ids).append(jid)
        if cross_ids:
            console.print(
                f"  [yellow]note:[/yellow] {', '.join(cross_ids)} run on another cluster; "
                f"re-run [bold]tabpfncredit summarize[/bold] after they finish."
            )
        summarize_id = _sbatch(
            summarize_script,
            dependency=("afterok:" + ":".join(dep_ids)) if dep_ids else None,
        )
        console.print(f"  [green]submitted[/green] {summarize_script.name} -> {summarize_id}\n")


# ============================================================================
#  `summarize` -- public helper (also called by the SLURM summarize job)
# ============================================================================

@app.command("summarize")
def cmd_summarize(
    experiment: str = typer.Option(..., help="e.g. 'Experiment1'."),
    out_dir: Optional[Path] = typer.Option(None, help="Where to write the CSVs."),
    results_root: Optional[Path] = typer.Option(
        None,
        help="Results root to read (e.g. a locally downloaded copy). "
             "Default: the auto-resolved root ($TABPFN_RESULTS_ROOT / project "
             "storage / ./results).",
    ),
) -> None:
    """Aggregate every fold result into per-fold and per-method CSVs.

    Safe to run at ANY time, including mid-run: it summarizes whatever result
    files exist right now, and the next invocation simply overwrites the two
    CSVs. Partial summaries are just missing the not-yet-finished rows --
    nothing breaks downstream (notebooks show fewer points).
    """
    from src.utils.result_summary import summarize_to_csv

    base = Path(results_root) if results_root else _results_root()
    out_dir = out_dir or (base / "summaries")
    paths = summarize_to_csv(
        base=base,
        experiment=experiment.lower(),
        out_dir=out_dir,
    )
    for p in paths:
        console.print(f"[green]wrote[/green] {p}")


# ============================================================================
#  `slurm-task` -- workhorse called by the generated SLURM scripts
# ============================================================================

@app.command("slurm-task", hidden=True)
def cmd_slurm_task(
    experiment: str = typer.Option(...),
    partition: str = typer.Option(..., help="Partition key, e.g. cpu / cpu_genius / gpu_v100 / gpu_a100 / gpu_h100 (any key in slurm_generator.PARTITIONS)."),
    array_id: int = typer.Option(..., help="SLURM_ARRAY_TASK_ID."),
    plan_path: Optional[Path] = typer.Option(
        None,
        help="Path to the JSON plan file. Defaults to "
             "scripts/<exp>/_generated/<exp>_<partition>_plan.json",
    ),
    verbose: bool = typer.Option(False),
) -> None:
    """Run the (dataset, method, task) cells for ONE array slot. (Internal.)"""
    if plan_path is None:
        plan_path = (
            _PROJECT_ROOT / "scripts" / experiment / "_generated"
            / f"{experiment.lower()}_{partition}_plan.json"
        )
    if not plan_path.exists():
        console.print(f"[red]plan file missing: {plan_path}[/red]")
        raise typer.Exit(code=1)
    slots = load_plan(plan_path)
    if not slots:
        console.print(f"[red]Empty plan for {experiment}/{partition}.[/red]")
        raise typer.Exit(code=1)
    if array_id < 0 or array_id >= len(slots):
        console.print(
            f"[red]array_id={array_id} out of range [0, {len(slots) - 1}].[/red]"
        )
        raise typer.Exit(code=2)

    slot = slots[array_id]
    console.print(f"[bold]Slot {array_id}/{len(slots) - 1}:[/bold] {len(slot)} point(s)")

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    configure_quiet_runtime()
    config = load_config(experiment)
    results_root = _results_root()

    # The plan already holds the exact sweep POINTS assigned to this slot
    # (the generator sharded each cell's points across slots so no slot blows
    # the 72 h wall). Run each point directly -- no re-expansion -- so a cell's
    # thousands of points really do run in parallel across array tasks. Each
    # point is saved under its own ``<method>__<suffix>`` file; skip-if-done
    # makes a re-submit resume any point a timed-out slot didn't finish.
    for item in slot:
        cell = {"dataset": item["dataset"], "method": item["method"], "task": item["task"]}
        point = {
            "name": item.get("name") or item["method"],
            "tune": item.get("tune", False),
            "row_limit": item.get("row_limit"),
            "sampling": item.get("sampling"),
            "copy_from": item.get("copy_from"),
        }
        _run_one_point(experiment, config, cell, point, results_root, verbose=verbose)


# ============================================================================
#  Helper commands
# ============================================================================

@app.command("list")
def cmd_list(
    architecture: Optional[str] = typer.Option(None, help="'deep' or 'classical'."),
    hardware: Optional[str] = typer.Option(None, help="'cpu' or 'gpu'."),
    show_profile: bool = typer.Option(False, help="Also show runtime tier + partition."),
) -> None:
    """List registered methods (optionally with runtime profile)."""
    pool = DEEP_METHODS | CLASSICAL_METHODS
    if architecture == "deep":
        pool &= DEEP_METHODS
    elif architecture == "classical":
        pool &= CLASSICAL_METHODS
    if hardware == "cpu":
        pool &= CPU_METHODS
    elif hardware == "gpu":
        pool &= GPU_METHODS

    table = Table(title=f"{len(pool)} method(s)")
    table.add_column("Method")
    table.add_column("Arch")
    table.add_column("HW")
    table.add_column("Foundation")
    if show_profile:
        table.add_column("Tier")
        table.add_column("~sec/fold")
        table.add_column("Partition")
    for m in sorted(pool):
        row = [
            m,
            "deep" if m in DEEP_METHODS else "classical",
            "gpu" if m in GPU_METHODS else "cpu",
            "yes" if m in FOUNDATION_METHODS else "no",
        ]
        if show_profile:
            profile = get_profile(m)
            row.extend([
                profile.tier.value,
                f"{profile.seconds_per_fold_estimate}",
                partition_for_method(m),
            ])
        table.add_row(*row)
    console.print(table)


@app.command("doctor")
def cmd_doctor() -> None:
    """Quick environment / VSC sanity check."""
    table = Table(title="Environment")
    table.add_column("Key")
    table.add_column("Value")
    for key in ("VSC_HOME", "VSC_DATA", "VSC_SCRATCH", "VSC_INSTITUTE_CLUSTER",
                "TABPFN_STAGING_ROOT", "TABPFN_RESULTS_ROOT", "TABPFN_CACHE_ROOT",
                "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "CUDA_VISIBLE_DEVICES"):
        table.add_row(key, os.environ.get(key, "(unset)"))
    console.print(table)

    try:
        import torch
        gpu_info = (
            f"cuda available: {torch.cuda.is_available()}; "
            f"device count: {torch.cuda.device_count()}"
        )
    except ImportError:
        gpu_info = "(torch not installed)"
    console.print(f"[bold]Torch[/bold]: {gpu_info}")
    console.print(f"[bold]On VSC?[/bold] {_on_vsc()}")
    console.print(f"[bold]sbatch?[/bold] {'yes' if _have_sbatch() else 'no'}")
    console.print("[bold]Resolved paths[/bold] (data/checkpoints = repo first, then project storage):")
    for k, v in _describe_paths().items():
        console.print(f"  {k}: {v}")
    for task in ("pd", "lgd"):
        console.print(f"[bold]Datasets available ({task})[/bold]: {len(list_datasets(task))}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
