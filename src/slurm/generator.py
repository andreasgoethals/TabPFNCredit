"""SLURM script generator for the VSC (Genius + wICE) clusters.

Designed against the VSC documentation. Highlights:

* **Results land on ``$VSC_DATA``** (small, permanent, backed up, 75 GB
  quota). Per-sweep result files total ~40 MB so the quota is never the
  bottleneck. ``$VSC_SCRATCH`` is reserved for heavy intermediate I/O
  (which this benchmark does not produce); we deliberately keep results
  off scratch because scratch files are auto-purged after 30 days of
  inactivity.
* **CPU + memory right-sized per partition** so the job uses the
  documented per-GPU caps efficiently (P100: 9c/45 GB, A100: 18c/126 GB,
  H100: 16c/187 GB).
* **Shebang ``#!/bin/bash -l``** so ``~/.bashrc`` and the cluster module
  are sourced (the docs are explicit about this).
* **``--clusters=`` (plural)** instead of the legacy ``--cluster=``.
* **``module --force purge`` then explicit load** -- docs warn that
  loading modules in ``~/.bashrc`` is fragile.
* **``--gpu_cmode=shared``** (default) lets multiple short methods share
  one GPU when packed together.
* **Deterministic stagger** based on ``SLURM_ARRAY_TASK_ID`` (no random
  thundering-herd).
* **Per-array-slot packing**: cheap methods (FAST/MEDIUM tier) get
  bundled to reduce scheduler overhead -- the VSC docs explicitly
  recommend merging work items < 3-4 min.
* **Worker-framework hint**: when an experiment has more than ~500 cells
  the generator prints a notice suggesting Worker-NG as a more efficient
  submission path.

Layout
------
``tabpfncredit slurm-generate --experiment ExperimentN`` emits one
``.slurm`` file per partition under ``scripts/ExperimentN/_generated/``.
Each script:

1. Sets up the environment (``module --force purge``, conda activate).
2. ``cd "$VSC_DATA/TabPFNCredit"`` and exports
   ``TABPFN_RESULTS_ROOT="$VSC_DATA/TabPFNCredit/results"``.
3. Reads its global task ID from ``SLURM_ARRAY_TASK_ID``.
4. Invokes ``tabpfncredit slurm-task`` with the partition and array id;
   the CLI looks up the cell assigned to that slot in the partition's
   ``_plan.json`` and runs it through ``run_talent_method`` +
   ``save_method``.

The user only ever calls ``tabpfncredit slurm-generate`` then ``sbatch``
on the generated files.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import List, Optional

from src.methods.runtime_profile import (
    Profile,
    Tier,
    estimate_walltime_seconds,
    get_profile,
)


# ============================================================================
#  Partition presets (right-sized per VSC docs)
# ============================================================================

@dataclass(frozen=True)
class PartitionSpec:
    """Right-sized resources for one VSC partition."""

    cluster: str
    partition: str
    cpus_per_gpu: int          # max cores per GPU (docs)
    mem_per_gpu_mb: int        # max memory per GPU in MiB (docs)
    gpus_per_node: int         # 0 for CPU-only
    max_walltime_hours: int    # partition wall-clock cap
    cost_weight: float         # TRES billing weight (lower = cheaper)


# These numbers come from the VSC cpu_resource_limits_in_gpu_jobs table.
# Memory is set just below the cap so the scheduler doesn't multiply cores.
PARTITIONS: dict[str, PartitionSpec] = {
    "cpu": PartitionSpec(
        cluster="wice", partition="batch_sapphirerapids",
        cpus_per_gpu=36,       # whole-node CPU job
        mem_per_gpu_mb=160_000,  # ~160 GB total
        gpus_per_node=0,
        max_walltime_hours=72,
        cost_weight=2.55,
    ),
    "cpu_genius": PartitionSpec(
        cluster="genius", partition="batch",
        cpus_per_gpu=36,
        mem_per_gpu_mb=140_000,
        gpus_per_node=0,
        max_walltime_hours=72,
        cost_weight=1.0,
    ),
    "gpu_p100": PartitionSpec(
        cluster="genius", partition="gpu_p100",
        cpus_per_gpu=9,
        mem_per_gpu_mb=44_000,  # below 45000 to stay clear of the cap
        gpus_per_node=1,
        max_walltime_hours=72,
        cost_weight=41.67,
    ),
    "gpu_a100": PartitionSpec(
        cluster="wice", partition="gpu_a100",
        cpus_per_gpu=18,
        mem_per_gpu_mb=125_000,
        gpus_per_node=1,
        max_walltime_hours=72,
        cost_weight=141.67,
    ),
    "gpu_h100": PartitionSpec(
        cluster="wice", partition="gpu_h100",
        cpus_per_gpu=16,
        mem_per_gpu_mb=186_000,
        gpus_per_node=1,
        max_walltime_hours=72,
        cost_weight=569.44,
    ),
}


def partition_for_method(method: str, *, prefer_h100: bool = True) -> str:
    """Return the partition KEY (not the SLURM partition name) for ``method``."""
    profile = get_profile(method)
    if not profile.prefers_gpu:
        return "cpu"
    if profile.needs_foundation_gpu:
        return "gpu_h100" if prefer_h100 else "gpu_a100"
    return "gpu_p100"


# ============================================================================
#  Batch packing
# ============================================================================
#
# The VSC docs warn that any work item under ~3-4 minutes wastes scheduler
# time. We pack cells so each array slot runs at least PACK_TARGET_SECONDS of
# work. The packing only applies to CPU partitions and the FAST tier --
# foundation models always get one slot each (their walltime varies wildly
# and GPU memory is exclusive).

PACK_TARGET_SECONDS = 600    # ~10 minutes per array slot for FAST cells
PACK_MAX_PER_SLOT = 64       # never bundle more than this many cells per slot


def pack_cells(cells: List[dict], *, target_seconds: int = PACK_TARGET_SECONDS) -> List[List[dict]]:
    """Group cells into array slots so each slot has roughly ``target_seconds`` of work.

    ``cells`` is a list of dicts with at least ``{"method": str, "dataset": str,
    "task": str}``. Foundation-model cells are never packed (one slot each).
    """
    bins: List[List[dict]] = []
    current: List[dict] = []
    current_sec = 0
    for cell in cells:
        profile = get_profile(cell["method"])
        sec = profile.seconds_per_fold_estimate
        # Foundation models always go solo so a slow run doesn't block a
        # fast one in the same slot.
        if profile.tier == Tier.FOUNDATION:
            if current:
                bins.append(current)
                current, current_sec = [], 0
            bins.append([cell])
            continue
        if current and (current_sec + sec > target_seconds or len(current) >= PACK_MAX_PER_SLOT):
            bins.append(current)
            current, current_sec = [], 0
        current.append(cell)
        current_sec += sec
    if current:
        bins.append(current)
    return bins


# ============================================================================
#  Script body assembly
# ============================================================================

_SHEBANG = "#!/bin/bash -l"

def _prologue(*, cluster: str, partition: str) -> str:
    """Return the per-script prologue, parameterised by VSC cluster + partition.

    On KU Leuven VSC, you must load ``cluster/<cluster>/<partition>``
    before any toolchain (incl. Python) module becomes visible. We do
    this here so users never need to type the chain by hand.
    """
    return dedent(
        f"""\
        set -euo pipefail

        # Deterministic stagger (0-29 s) based on array index -- avoids the
        # thundering-herd I/O storm that RANDOM%60 would create.
        sleep $((${{SLURM_ARRAY_TASK_ID:-0}} % 30))

        # Unbuffered Python output so SLURM streams stdout in real time.
        export PYTHONUNBUFFERED=1

        # Memory-fragmentation mitigation for long-running PyTorch jobs.
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        # HPC convention: never load modules in ~/.bashrc; do it here.
        module --force purge

        # VSC quirk: you have to load a cluster module BEFORE the
        # toolchain modules (incl. Python) become visible. Override
        # by exporting TABPFN_CLUSTER_MODULE before sbatch if needed.
        : "${{TABPFN_CLUSTER_MODULE:=cluster/{cluster}/{partition}}}"
        module load "${{TABPFN_CLUSTER_MODULE}}" 2>/dev/null || \\
            echo "WARN: could not module load ${{TABPFN_CLUSTER_MODULE}}" >&2

        # Load the Python module the venv was built against. Override
        # per cluster by exporting TABPFN_PYTHON_MODULE before sbatch.
        # On your own cluster: ``module spider Python/3.12`` to find names.
        : "${{TABPFN_PYTHON_MODULE:=Python/3.12.3-GCCcore-13.3.0}}"
        module load "${{TABPFN_PYTHON_MODULE}}" 2>/dev/null || \\
            echo "WARN: could not module load ${{TABPFN_PYTHON_MODULE}}" >&2

        # Activate the project's Python env. Prefer a plain ``python -m venv``
        # at the repo root (fast to create, no conda solver) and fall back
        # to a conda env named ``tabpfncreditvenv`` if that's what you set up.
        if [ -f "${{VSC_DATA}}/TabPFNCredit/tabpfncreditvenv/bin/activate" ]; then
            source "${{VSC_DATA}}/TabPFNCredit/tabpfncreditvenv/bin/activate"
        elif command -v conda >/dev/null 2>&1 || [ -d "${{VSC_DATA}}/miniforge3" ] || [ -d "${{VSC_DATA}}/miniconda3" ]; then
            if [ -d "${{VSC_DATA}}/miniforge3" ]; then
                source "${{VSC_DATA}}/miniforge3/etc/profile.d/conda.sh"
            else
                source "${{VSC_DATA}}/miniconda3/etc/profile.d/conda.sh"
            fi
            conda activate tabpfncreditvenv 2>/dev/null || conda activate TabPFNCredit
            export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
        else
            echo "ERROR: no Python env found (looked for tabpfncreditvenv venv and conda)." >&2
            exit 1
        fi"""
    )


_EPILOGUE = dedent(
    """\
    EXIT_CODE=$?
    echo "Task done, exit=${EXIT_CODE}"
    exit ${EXIT_CODE}
    """
)


def _sbatch_header(
    *,
    job_name: str,
    spec: PartitionSpec,
    n_gpus: int,
    array_range: str,
    walltime: str,
    log_dir: str,
    mail_email: str = "",
    gpu_cmode: Optional[str] = None,
) -> str:
    """Build the #SBATCH block for one batch."""
    cpus = spec.cpus_per_gpu if n_gpus else spec.cpus_per_gpu  # full-node CPU
    mem_mb = spec.mem_per_gpu_mb * max(1, n_gpus) if n_gpus else spec.mem_per_gpu_mb

    lines = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --clusters={spec.cluster}",
        "#SBATCH --account=lp_verbekelab",
        f"#SBATCH --partition={spec.partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
    ]
    if n_gpus:
        lines.append(f"#SBATCH --gpus-per-node={n_gpus}")
        if gpu_cmode:
            lines.append(f"#SBATCH --gpu_cmode={gpu_cmode}")
    lines.extend([
        f"#SBATCH --mem={mem_mb}M",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --output={log_dir}/{job_name}_%A_%a.out",
        f"#SBATCH --error={log_dir}/{job_name}_%A_%a.err",
        "#SBATCH --requeue",
    ])
    email = (mail_email or os.environ.get("TABPFN_SLURM_NOTIFY_EMAIL", "")).strip()
    if email:
        lines.append("#SBATCH --mail-type=FAIL,TIME_LIMIT,REQUEUE")
        lines.append(f"#SBATCH --mail-user={email}")
    lines.append(f"#SBATCH --array={array_range}")
    return "\n".join(lines) + "\n"


def _python_invocation(experiment: str, partition_key: str) -> str:
    """The actual work line -- delegates to the Typer CLI."""
    # Results land under $VSC_DATA (permanent + backed up + 75 GB quota).
    # Per-fold scratch (TALENT's internal save_path) goes to a tempfile
    # directory created inside ``method_runner._run_method``, which is
    # auto-cleaned at fold exit. Nothing else needs scratch storage.
    return dedent(f"""\
        cd "${{VSC_DATA}}/TabPFNCredit"

        export TABPFN_RESULTS_ROOT="${{VSC_DATA}}/TabPFNCredit/results"
        mkdir -p "${{TABPFN_RESULTS_ROOT}}"

        tabpfncredit slurm-task \\
            --experiment {experiment} \\
            --partition {partition_key} \\
            --array-id "${{SLURM_ARRAY_TASK_ID}}"
        """)


# ============================================================================
#  Public entry point
# ============================================================================

@dataclass
class GeneratedJob:
    """One generated SLURM script (one partition + batch tuple)."""

    path: Path
    partition_key: str
    n_array_slots: int
    walltime: str


def generate_scripts_for_experiment(
    *,
    experiment: str,
    tasks: List[dict],
    out_dir: Path,
    n_folds: int = 5,
    n_sweep_points: int = 1,
    prefer_h100: bool = True,
    gpu_cmode: str = "shared",
    max_concurrent: int = 16,
    mail_email: str = "",
) -> List[GeneratedJob]:
    """Write SLURM scripts for one experiment, split by partition + tier.

    Parameters
    ----------
    experiment : str
        ``"Experiment0"`` etc. (case-insensitive); used in the job name.
    tasks : list of dict
        One entry per (dataset, method, task). Each dict must have the keys
        ``"dataset"``, ``"method"``, ``"task"``.
    out_dir : Path
        Directory to write the generated scripts into.
    n_folds : int
        For walltime estimation.
    n_sweep_points : int
        For walltime estimation (Experiment 2/3 sweep multiple points per cell).
    prefer_h100 : bool
        Use H100 wICE partition for foundation models; fall back to A100 if False.
    gpu_cmode : str
        ``shared`` (default; multiple processes per GPU OK) or ``exclusive``.
    max_concurrent : int
        SLURM array ``%`` throttle. Pick this so total concurrent slots stay
        under the per-partition GPU count (e.g. wICE has 20 H100s total).
    mail_email : str
        If non-empty, add ``--mail-type=FAIL,TIME_LIMIT,REQUEUE`` notifications.

    Returns
    -------
    list of :class:`GeneratedJob`
    """
    if not tasks:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Group tasks by partition
    by_partition: dict[str, list[dict]] = {}
    for cell in tasks:
        key = partition_for_method(cell["method"], prefer_h100=prefer_h100)
        by_partition.setdefault(key, []).append(cell)

    generated: List[GeneratedJob] = []

    for partition_key, cells in sorted(by_partition.items()):
        if partition_key not in PARTITIONS:
            continue
        spec = PARTITIONS[partition_key]
        n_gpus = 1 if spec.gpus_per_node else 0

        # 2) Pack cheap cells together; foundation models go solo
        slots = pack_cells(cells)

        # 3) Walltime: dominated by the slowest cell in any slot, capped at
        #    the partition's hard walltime.
        max_seconds_in_any_slot = max(
            sum(
                estimate_walltime_seconds(
                    c["method"], n_folds=n_folds, n_sweep_points=n_sweep_points
                )
                for c in slot
            )
            for slot in slots
        )
        cap_seconds = spec.max_walltime_hours * 3600
        walltime_seconds = min(max_seconds_in_any_slot, cap_seconds)
        walltime = _format_walltime(walltime_seconds)

        # 4) Persist the per-slot task plan as a JSON sibling so slurm-task
        #    knows what to run.
        plan_path = out_dir / f"{experiment.lower()}_{partition_key}_plan.json"
        _write_plan(plan_path, slots)

        # 5) Emit the SLURM script
        job_name = f"{experiment.lower()}_{partition_key}"
        array_range = (
            f"0-{len(slots) - 1}%{max_concurrent}" if len(slots) > 1 else "0"
        )
        # Logs live alongside the results under $VSC_DATA (the result
        # files are tiny so $VSC_DATA's 75 GB quota is plenty). No
        # separate slurm/ subfolder -- everything in <exp>/logs/.
        log_dir = (
            f"${{VSC_DATA}}/TabPFNCredit/results/{experiment.lower()}/logs"
        )
        header = _sbatch_header(
            job_name=job_name,
            spec=spec,
            n_gpus=n_gpus,
            array_range=array_range,
            walltime=walltime,
            log_dir=log_dir,
            mail_email=mail_email,
            gpu_cmode=gpu_cmode if n_gpus else None,
        )
        banner = "\n".join([
            f'mkdir -p "{log_dir}"',
            'echo "============================================="',
            f'echo "{experiment} on {spec.cluster}/{spec.partition}"',
            'echo "JobID:   ${SLURM_JOB_ID}"',
            'echo "ArrayID: ${SLURM_ARRAY_TASK_ID}"',
            'echo "Node:    ${SLURMD_NODENAME}"',
            'echo "GPU:     ${CUDA_VISIBLE_DEVICES:-N/A}"',
            f'echo "Plan:    {plan_path.name} (slot ${{SLURM_ARRAY_TASK_ID}})"',
            'echo "============================================="',
        ]) + "\n"
        script = "\n".join([
            _SHEBANG,
            header,
            _prologue(cluster=spec.cluster, partition=spec.partition),
            banner,
            _python_invocation(experiment, partition_key),
            _EPILOGUE,
        ])

        script_path = out_dir / f"{job_name}.slurm"
        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o755)

        generated.append(GeneratedJob(
            path=script_path,
            partition_key=partition_key,
            n_array_slots=len(slots),
            walltime=walltime,
        ))

    # 6) Worker-framework hint
    total_slots = sum(j.n_array_slots for j in generated)
    if total_slots > 500:
        import logging
        logging.getLogger(__name__).warning(
            "slurm-generate: %d array slots requested. VSC docs recommend the "
            "Worker framework (module load worker-ng/1.0.11-GCCcore-10.3.0) "
            "for sweeps >500 work items. Job arrays still work but may load "
            "the scheduler.",
            total_slots,
        )

    return generated


def _format_walltime(seconds: int) -> str:
    """Format seconds as ``HH:MM:SS`` or ``D-HH:MM:SS`` (SLURM-accepted)."""
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 24:
        days, hh = divmod(hours, 24)
        return f"{days}-{hh:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _write_plan(path: Path, slots: List[List[dict]]) -> None:
    """Persist the per-slot task plan as JSON. ``slurm-task`` reads it."""
    import json
    payload = {
        "slots": [
            [{"dataset": c["dataset"], "method": c["method"], "task": c["task"]} for c in slot]
            for slot in slots
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================================
#  Plan loader (used by the CLI)
# ============================================================================

def load_plan(path: Path) -> List[List[dict]]:
    """Inverse of :func:`_write_plan`."""
    import json
    payload = json.loads(Path(path).read_text())
    return payload["slots"]


# ============================================================================
#  Summarize SLURM script
# ============================================================================
#
# Emitted alongside the per-partition arrays. The caller submits this with
# ``--dependency=afterok:<array_ids>`` so it runs once *every* array slot
# has finished -- producing the per-fold + per-method CSVs.

def generate_summarize_script(
    *,
    experiment: str,
    out_dir: Path,
    mail_email: str = "",
) -> Path:
    """Write a tiny single-task SLURM script that runs ``tabpfncredit summarize``.

    Uses the ``batch`` (CPU) partition since aggregation is pure pandas /
    polars work -- no GPU needed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = PARTITIONS["cpu"]
    job_name = f"{experiment.lower()}_summarize"
    log_dir = f"${{VSC_DATA}}/TabPFNCredit/results/{experiment.lower()}/logs"
    # 15 minutes is plenty -- polars scans all <method>.json files in
    # seconds even on multi-thousand-row sweeps.
    walltime = "00:15:00"

    header = _sbatch_header(
        job_name=job_name,
        spec=spec,
        n_gpus=0,
        array_range="0",        # single task, not an array
        walltime=walltime,
        log_dir=log_dir,
        mail_email=mail_email,
        gpu_cmode=None,
    )
    # Strip the array directive from a header that uses array_range="0":
    # the helper still emits ``#SBATCH --array=0`` which we don't want here.
    header = "\n".join(
        line for line in header.splitlines() if not line.startswith("#SBATCH --array=")
    ) + "\n"

    banner = "\n".join([
        f'mkdir -p "{log_dir}"',
        'echo "============================================="',
        f'echo "{experiment} -- summarize step"',
        'echo "JobID:   ${SLURM_JOB_ID}"',
        'echo "Node:    ${SLURMD_NODENAME}"',
        'echo "============================================="',
    ]) + "\n"

    invocation = dedent(f"""\
        cd "${{VSC_DATA}}/TabPFNCredit"

        export TABPFN_RESULTS_ROOT="${{VSC_DATA}}/TabPFNCredit/results"

        tabpfncredit summarize --experiment {experiment}
        """)

    script = "\n".join([
        _SHEBANG,
        header,
        _prologue(cluster=spec.cluster, partition=spec.partition),
        banner,
        invocation,
        _EPILOGUE,
    ])

    script_path = out_dir / f"{job_name}.slurm"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


__all__ = [
    "PartitionSpec",
    "PARTITIONS",
    "partition_for_method",
    "pack_cells",
    "GeneratedJob",
    "generate_scripts_for_experiment",
    "load_plan",
]
