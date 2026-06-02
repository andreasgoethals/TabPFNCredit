"""TabPFNCredit command-line interface (B3 + B2).

A Typer-powered CLI that replaces the six near-identical experiment
driver scripts (``Experiment0_CPU.py``, ``Experiment0_GPU.py``,
``Experiment1_CPU.py``, ``Experiment1_GPU.py``, ``Experiment2_GPU_Foundation.py``,
``Experiment2_GPU_Standard.py``, ...) with one driver whose method list
is *derived* from the TALENT registry filtered by the requested partition.

Examples
--------
List every method TALENT exposes::

    $ tabpfncredit list

Run one (dataset, method) on the GPU::

    $ tabpfncredit run --experiment Experiment0 --dataset 0001.gmsc \\
        --method tabpfn_v3 --task pd

Run the entire SLURM-array task plan for one partition::

    $ tabpfncredit slurm-task --experiment Experiment0 \\
        --partition gpu_foundation --array-id $SLURM_ARRAY_TASK_ID

Aggregate fold results into CSVs::

    $ tabpfncredit summarize --experiment Experiment0

The legacy ``scripts/Experiment*/Experiment*_CPU.py``/``_GPU.py``
drivers still work, but new code should use this entry point.
"""

from __future__ import annotations

import json
import logging
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
from src.utils.config_reader import load_config  # noqa: E402
from src.utils.result_io import save_method  # noqa: E402

app = typer.Typer(
    add_completion=False,
    help="TabPFNCredit: credit-risk tabular benchmarking on top of TALENT.",
)
console = Console()
logger = logging.getLogger("tabpfncredit")


# ============================================================================
#  Partition -> method-filter
# ============================================================================

def _methods_for_partition(partition: str) -> set[str]:
    """Filter the TALENT registry by SLURM partition."""
    partition = partition.lower()
    if partition == "cpu":
        return CPU_METHODS
    if partition == "gpu_foundation":
        return GPU_METHODS & FOUNDATION_METHODS
    if partition == "gpu_standard":
        return GPU_METHODS - FOUNDATION_METHODS
    if partition == "all":
        return CPU_METHODS | GPU_METHODS
    raise typer.BadParameter(
        f"Unknown partition {partition!r}; choose from cpu / gpu_foundation / gpu_standard / all."
    )


def _enabled_methods_for_task(config: dict, task: str) -> set[str]:
    """Methods toggled to true in the experiment's CONFIG_METHOD.yaml for `task`."""
    return set(config.get("methods", {}).get(task, {}).keys())


def _enabled_datasets_for_task(config: dict, task: str) -> List[str]:
    return list(config.get("datasets", {}).get(task, {}).keys())


# ============================================================================
#  Commands
# ============================================================================

@app.command("list")
def cmd_list(
    architecture: Optional[str] = typer.Option(
        None, help="Filter: 'deep' or 'classical' (default: show all)."
    ),
    hardware: Optional[str] = typer.Option(
        None, help="Filter: 'cpu' or 'gpu' (default: show all)."
    ),
) -> None:
    """List every TALENT method registered in this project."""
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
    for m in sorted(pool):
        table.add_row(
            m,
            "deep" if m in DEEP_METHODS else "classical",
            "gpu" if m in GPU_METHODS else "cpu",
            "yes" if m in FOUNDATION_METHODS else "no",
        )
    console.print(table)


@app.command("run")
def cmd_run(
    experiment: str = typer.Option(..., help="Experiment name, e.g. 'Experiment0'."),
    dataset: str = typer.Option(..., help="Dataset name, e.g. '0001.gmsc'."),
    method: str = typer.Option(..., help="TALENT method name, e.g. 'tabpfn_v3'."),
    task: str = typer.Option("pd", help="Task: 'pd' or 'lgd'."),
    tune: bool = typer.Option(False, help="Run per-fold hyperparameter tuning."),
    n_trials: int = typer.Option(50, help="Optuna trials when --tune is set."),
    cv_splits: Optional[int] = typer.Option(None, help="Override cv_splits from config."),
    row_limit: Optional[int] = typer.Option(None, help="Cap total dataset rows."),
    verbose: bool = typer.Option(False, help="Verbose progress output."),
    write_results: bool = typer.Option(
        True, help="Persist results via the new JSON+npz layout."
    ),
) -> None:
    """Run one (dataset, method, task) cell of an experiment."""
    config = load_config(experiment)
    cv = cv_splits if cv_splits is not None else config["split"]["cv_splits"]

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
        results_root = _PROJECT_ROOT / "results"
        save_method(
            fold_results,
            base=results_root,
            experiment=experiment.lower(),
            task=task,
            dataset=dataset,
            method=method,
        )
        console.print(
            f"[blue]Results written under {results_root / experiment.lower()}[/blue]"
        )


@app.command("slurm-task")
def cmd_slurm_task(
    experiment: str = typer.Option(...),
    partition: str = typer.Option(..., help="cpu / gpu_foundation / gpu_standard / all"),
    array_id: int = typer.Option(..., help="SLURM_ARRAY_TASK_ID"),
    hpo_mode: str = typer.Option("NO_HPO", help="NO_HPO or HPO"),
    verbose: bool = typer.Option(False),
) -> None:
    """Run the single (dataset, method, task) cell assigned to this SLURM array slot.

    Builds the deterministic task list from
    ``(enabled datasets) x (enabled methods filtered by partition)`` and
    indexes into it with ``array_id``.
    """
    config = load_config(experiment)
    partition_methods = _methods_for_partition(partition)

    tasks: List[tuple] = []
    for task in ("pd", "lgd"):
        enabled = _enabled_methods_for_task(config, task) & partition_methods
        for dataset in _enabled_datasets_for_task(config, task):
            for method in sorted(enabled):
                tasks.append((dataset, method, task))

    if not tasks:
        console.print(f"[red]No tasks for partition={partition!r}[/red]")
        raise typer.Exit(code=1)
    if array_id < 0 or array_id >= len(tasks):
        console.print(
            f"[red]array_id={array_id} out of range [0, {len(tasks) - 1}][/red]"
        )
        raise typer.Exit(code=2)

    dataset, method, task = tasks[array_id]
    console.print(
        f"[bold]Task {array_id}/{len(tasks) - 1}:[/bold] "
        f"dataset={dataset} method={method} task={task}"
    )

    cmd_run(
        experiment=experiment, dataset=dataset, method=method, task=task,
        tune=(hpo_mode.upper() == "HPO"),
        n_trials=config.get("tuning", {}).get("n_trials", 50),
        cv_splits=None, row_limit=None, verbose=verbose,
        write_results=True,
    )


@app.command("summarize")
def cmd_summarize(
    experiment: str = typer.Option(...),
    out_dir: Optional[Path] = typer.Option(None, help="Where to write the CSVs."),
) -> None:
    """Aggregate every fold result into per-fold and per-method CSVs."""
    from src.utils.summarize_results_polars import summarize_to_csv
    out_dir = out_dir or (_PROJECT_ROOT / "results" / "summaries")
    paths = summarize_to_csv(
        base=_PROJECT_ROOT / "results",
        experiment=experiment.lower(),
        out_dir=out_dir,
    )
    for p in paths:
        console.print(f"[green]wrote[/green] {p}")


@app.command("config")
def cmd_config(experiment: str = typer.Option(...)) -> None:
    """Pretty-print the merged config for an experiment."""
    cfg = load_config(experiment)
    console.print_json(json.dumps(cfg, default=str))


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
