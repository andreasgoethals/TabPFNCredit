"""TabPFNCredit command-line interface.

Single entry point for the entire benchmark workflow. Replaces the
twelve+ legacy per-experiment driver scripts with a Typer-powered CLI
whose method list is derived from the TALENT registry. On VSC the
``slurm-generate`` + ``slurm-task`` pair drive the whole sweep:

::

    # 1) Generate SLURM scripts (right-sized per VSC docs)
    tabpfncredit slurm-generate --experiment Experiment1

    # 2) Submit
    sbatch scripts/Experiment1/_generated/experiment1_gpu_h100.slurm
    sbatch scripts/Experiment1/_generated/experiment1_cpu.slurm

Locally the same machinery works without the SLURM layer:

::

    tabpfncredit run --experiment Experiment1 \\
        --dataset 0001.gmsc --method tabpfn_v3 --task pd

Results root resolution
-----------------------
Result paths default to ``./results`` (local dev) but honour
``$TABPFN_RESULTS_ROOT`` when set -- the generated SLURM scripts point
this at ``$VSC_DATA/TabPFNCredit/results`` (permanent + backed up).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

# Path injection so `import src.*` works whether or not the package is editable-installed.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.methods.method_config import (  # noqa: E402
    CLASSICAL_METHODS,
    CPU_METHODS,
    DEEP_METHODS,
    FOUNDATION_METHODS,
    GPU_METHODS,
)
from src.methods.method_runner import run_talent_method  # noqa: E402
from src.methods.runtime_profile import (  # noqa: E402
    Tier,
    estimate_walltime_seconds,
    get_profile,
    tier_of,
)
from src.slurm.generator import (  # noqa: E402
    PARTITIONS,
    generate_scripts_for_experiment,
    load_plan,
    partition_for_method,
)
from src.utils.config_reader import load_config  # noqa: E402
from src.utils.result_io import (  # noqa: E402
    has_complete_result,
    save_method,
)

app = typer.Typer(
    add_completion=False,
    help="TabPFNCredit: credit-risk tabular benchmarking on top of TALENT.",
)
console = Console()
logger = logging.getLogger("tabpfncredit")


# ============================================================================
#  Results-root resolution
# ============================================================================

def _results_root() -> Path:
    """Return the directory where result JSON/npz files live.

    Honours ``$TABPFN_RESULTS_ROOT`` (set by the SLURM scripts to point at
    ``$VSC_SCRATCH``) and falls back to ``<project>/results``.
    """
    env = os.environ.get("TABPFN_RESULTS_ROOT")
    if env:
        path = Path(env)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return _PROJECT_ROOT / "results"


# ============================================================================
#  Partition -> method-filter (legacy convenience)
# ============================================================================

def _methods_for_partition(partition: str) -> set[str]:
    """Filter the TALENT registry by SLURM partition key.

    Accepts the new partition keys (``cpu``, ``gpu_p100``, ``gpu_a100``,
    ``gpu_h100``) used by ``slurm-generate``, plus the legacy aliases
    (``gpu_foundation``, ``gpu_standard``, ``all``) for backwards
    compatibility.
    """
    partition = partition.lower()
    if partition in ("cpu", "cpu_genius"):
        return CPU_METHODS | {m for m in DEEP_METHODS if not get_profile(m).prefers_gpu}
    if partition == "gpu_foundation":
        return GPU_METHODS & FOUNDATION_METHODS
    if partition == "gpu_standard":
        return GPU_METHODS - FOUNDATION_METHODS
    if partition == "all":
        return CPU_METHODS | GPU_METHODS
    # New keys: derive from runtime profile
    if partition == "gpu_p100":
        return {m for m in GPU_METHODS if not get_profile(m).needs_foundation_gpu}
    if partition in ("gpu_a100", "gpu_h100"):
        return {m for m in GPU_METHODS if get_profile(m).needs_foundation_gpu}
    raise typer.BadParameter(
        f"Unknown partition {partition!r}; choose from cpu / gpu_p100 / "
        f"gpu_a100 / gpu_h100 (or legacy: gpu_foundation / gpu_standard / all)."
    )


def _enabled_methods_for_task(config: dict, task: str) -> set[str]:
    return set(config.get("methods", {}).get(task, {}).keys())


def _enabled_datasets_for_task(config: dict, task: str) -> List[str]:
    return list(config.get("datasets", {}).get(task, {}).keys())


def _build_task_list(config: dict, partition: str) -> List[dict]:
    """Build ``[{dataset, method, task}, ...]`` for one (config, partition)."""
    partition_methods = _methods_for_partition(partition)
    cells: List[dict] = []
    for task in ("pd", "lgd"):
        enabled = _enabled_methods_for_task(config, task) & partition_methods
        for dataset in _enabled_datasets_for_task(config, task):
            for method in sorted(enabled):
                cells.append({"dataset": dataset, "method": method, "task": task})
    return cells


# ============================================================================
#  `list` command -- enumerate methods, optionally with runtime profile
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


# ============================================================================
#  `run` command -- one (dataset, method, task) cell
# ============================================================================

@app.command("run")
def cmd_run(
    experiment: str = typer.Option(..., help="e.g. 'Experiment0'."),
    dataset: str = typer.Option(..., help="e.g. '0001.gmsc'."),
    method: str = typer.Option(..., help="e.g. 'tabpfn_v3'."),
    task: str = typer.Option("pd", help="'pd' or 'lgd'."),
    tune: bool = typer.Option(False, help="Per-fold HPO."),
    n_trials: int = typer.Option(50, help="Optuna trials when --tune."),
    cv_splits: Optional[int] = typer.Option(None, help="Override cv_splits."),
    row_limit: Optional[int] = typer.Option(None, help="Cap total dataset rows."),
    verbose: bool = typer.Option(False),
    write_results: bool = typer.Option(True),
    force: bool = typer.Option(False, help="Re-run even if a result already exists."),
) -> None:
    """Run one (dataset, method, task) cell of an experiment."""
    config = load_config(experiment)
    cv = cv_splits if cv_splits is not None else config["split"]["cv_splits"]

    results_root = _results_root()
    if not force and has_complete_result(
        base=results_root,
        experiment=experiment.lower(),
        task=task,
        dataset=dataset,
        method=method,
        expected_folds=cv,
    ):
        console.print(
            f"[yellow]SKIP[/yellow] {experiment}/{task}/{dataset}/{method} "
            f"(already complete; pass --force to re-run)"
        )
        return

    if verbose:
        logging.basicConfig(level=logging.INFO)

    fold_results = run_talent_method(
        task=task, dataset=dataset, method=method,
        test_size=config["split"]["test_size"],
        val_size=config["split"]["val_size"],
        cv_splits=cv,
        seed=config["split"]["seed"],
        row_limit=row_limit if row_limit is not None else config["split"].get("row_limit"),
        sampling=config["split"].get("sampling"),
        max_epoch=config["training"]["max_epochs"],
        batch_size=config["training"]["batch_size"],
        tune=tune,
        n_trials=n_trials,
        early_stopping=config["training"]["early_stopping"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        verbose=verbose,
    )
    console.print(f"[green]Completed {len(fold_results)} folds.[/green]")

    if write_results:
        save_method(
            fold_results, base=results_root,
            experiment=experiment.lower(),
            task=task, dataset=dataset, method=method,
        )
        console.print(f"[blue]Results written under {results_root / experiment.lower()}[/blue]")


# ============================================================================
#  `slurm-generate` -- produce VSC-optimised SLURM scripts for an experiment
# ============================================================================

@app.command("slurm-generate")
def cmd_slurm_generate(
    experiment: str = typer.Option(..., help="e.g. 'Experiment1'."),
    out_dir: Optional[Path] = typer.Option(
        None, help="Where to write the .slurm files (default: scripts/<exp>/_generated)."
    ),
    prefer_h100: bool = typer.Option(True, help="Use H100 (wICE) for foundation models."),
    gpu_cmode: str = typer.Option("shared", help="shared or exclusive."),
    max_concurrent: int = typer.Option(16, help="SLURM array %% throttle."),
    mail_email: str = typer.Option("", help="Failure/timeout email notifications."),
) -> None:
    """Generate VSC-compliant SLURM array scripts for an experiment.

    Splits the (dataset, method) cells by partition (CPU vs P100 vs H100
    foundation), packs cheap methods so each array slot has ~10 min of
    work, sizes CPU/memory per the VSC per-GPU caps, and writes results
    to ``$VSC_SCRATCH`` (never ``$VSC_DATA``).
    """
    config = load_config(experiment)
    tasks: List[dict] = []
    for task in ("pd", "lgd"):
        enabled = _enabled_methods_for_task(config, task)
        for dataset in _enabled_datasets_for_task(config, task):
            for method in sorted(enabled):
                tasks.append({"dataset": dataset, "method": method, "task": task})

    if not tasks:
        console.print(f"[red]No enabled cells in {experiment} configs.[/red]")
        raise typer.Exit(code=1)

    out_dir = out_dir or (_PROJECT_ROOT / "scripts" / experiment / "_generated")
    jobs = generate_scripts_for_experiment(
        experiment=experiment,
        tasks=tasks,
        out_dir=out_dir,
        n_folds=config["split"]["cv_splits"],
        prefer_h100=prefer_h100,
        gpu_cmode=gpu_cmode,
        max_concurrent=max_concurrent,
        mail_email=mail_email,
    )
    table = Table(title=f"Generated {len(jobs)} SLURM script(s)")
    table.add_column("Script")
    table.add_column("Partition")
    table.add_column("Array slots")
    table.add_column("Walltime / slot")
    for j in jobs:
        table.add_row(str(j.path.name), j.partition_key, str(j.n_array_slots), j.walltime)
    console.print(table)
    console.print(f"\nWritten under: [blue]{out_dir}[/blue]")
    console.print("Submit with: [bold]sbatch <script.slurm>[/bold]")


# ============================================================================
#  `slurm-task` -- the workhorse called by every array slot
# ============================================================================

@app.command("slurm-task")
def cmd_slurm_task(
    experiment: str = typer.Option(...),
    partition: str = typer.Option(..., help="Partition key (cpu / gpu_p100 / gpu_a100 / gpu_h100)."),
    array_id: int = typer.Option(..., help="SLURM_ARRAY_TASK_ID."),
    plan_path: Optional[Path] = typer.Option(
        None,
        help="Path to the JSON plan file. Defaults to "
             "scripts/<exp>/_generated/<exp>_<partition>_plan.json",
    ),
    verbose: bool = typer.Option(False),
) -> None:
    """Run the (dataset, method, task) cells for ONE array slot.

    Reads the per-slot plan written by ``slurm-generate``. If the plan
    file is missing (legacy path) falls back to building the task list
    deterministically from the experiment config.
    """
    if plan_path is None:
        plan_path = (
            _PROJECT_ROOT / "scripts" / experiment / "_generated"
            / f"{experiment.lower()}_{partition}_plan.json"
        )

    if plan_path.exists():
        slots = load_plan(plan_path)
    else:
        # Legacy fallback -- one (dataset, method, task) cell per slot,
        # no packing. Used by old hand-written SLURM scripts.
        config = load_config(experiment)
        slots = [[c] for c in _build_task_list(config, partition)]

    if not slots:
        console.print(f"[red]Empty plan for {experiment}/{partition}.[/red]")
        raise typer.Exit(code=1)
    if array_id < 0 or array_id >= len(slots):
        console.print(
            f"[red]array_id={array_id} out of range [0, {len(slots) - 1}].[/red]"
        )
        raise typer.Exit(code=2)

    slot = slots[array_id]
    console.print(
        f"[bold]Slot {array_id}/{len(slots) - 1}:[/bold] "
        f"{len(slot)} cell(s)"
    )

    # Sequentially run each cell in this slot. The skip-already-done check
    # inside `cmd_run` keeps re-runs safe.
    for cell in slot:
        try:
            cmd_run(
                experiment=experiment,
                dataset=cell["dataset"],
                method=cell["method"],
                task=cell["task"],
                tune=False,                # generator only emits NO_HPO plans
                n_trials=1,
                cv_splits=None, row_limit=None, verbose=verbose,
                write_results=True,
                force=False,
            )
        except typer.Exit:
            raise
        except Exception as exc:  # pragma: no cover -- defensive
            logger.exception("Cell %s failed: %s", cell, exc)
            # Continue with the rest of the slot -- one bad cell shouldn't
            # waste the whole slot's wall-clock.
            continue


# ============================================================================
#  `summarize` -- per-fold + per-method CSVs
# ============================================================================

@app.command("summarize")
def cmd_summarize(
    experiment: str = typer.Option(...),
    out_dir: Optional[Path] = typer.Option(None, help="Where to write the CSVs."),
) -> None:
    """Aggregate every fold result into per-fold and per-method CSVs."""
    from src.utils.summarize_results_polars import summarize_to_csv
    out_dir = out_dir or (_results_root() / "summaries")
    paths = summarize_to_csv(
        base=_results_root(),
        experiment=experiment.lower(),
        out_dir=out_dir,
    )
    for p in paths:
        console.print(f"[green]wrote[/green] {p}")


# ============================================================================
#  `config` -- pretty-print one experiment's merged config
# ============================================================================

@app.command("config")
def cmd_config(experiment: str = typer.Option(...)) -> None:
    """Pretty-print the merged config for an experiment."""
    cfg = load_config(experiment)
    console.print_json(json.dumps(cfg, default=str))


# ============================================================================
#  Misc utility
# ============================================================================

@app.command("doctor")
def cmd_doctor() -> None:
    """Quick environment / VSC sanity check."""
    table = Table(title="Environment")
    table.add_column("Key")
    table.add_column("Value")
    for key in ("VSC_HOME", "VSC_DATA", "VSC_SCRATCH", "TABPFN_RESULTS_ROOT",
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

    console.print(f"[bold]Results root[/bold]: {_results_root()}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
