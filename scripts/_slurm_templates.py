#!/usr/bin/env python3
"""
Shared SLURM script-template helpers for the ``scripts/ExperimentN/``
Setup generators.

Centralises the directives common to every batch submission on the VSC
clusters:

* Absolute SLURM ``--output`` / ``--error`` paths under ``${VSC_DATA}``
  (previously the generators used relative paths, so stderr/stdout landed
  wherever ``sbatch`` was invoked from).
* ``#SBATCH --requeue`` so a node failure re-queues the array slot rather
  than requiring a manual rerun.
* Optional ``#SBATCH --mail-type=FAIL --mail-user=…`` driven by the
  ``TABPFN_SLURM_NOTIFY_EMAIL`` environment variable at generation time.
* ``set -euo pipefail`` + ``mkdir -p`` prologue for robust execution.
* Conda + ``LD_LIBRARY_PATH`` activation consistent across every script.
* Soft-isolation memory strategy for foundation models (100 G on wICE H100
  nodes without ``--exclusive``).
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple


# --------------------------------------------------------------------------- #
# Resource-tier helpers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SlurmResources:
    """Per-batch SLURM resource request.

    Attributes:
        cluster:   SLURM cluster name (e.g. ``"genius"``, ``"wice"``).
        partition: Partition name (e.g. ``"gpu_p100"``, ``"batch"``).
        cpus:      Cores per task.
        gpus:      GPUs per node (0 for CPU-only jobs).
        memory:    Memory request (SLURM-formatted string, e.g. ``"45G"``).
        time:      Wall-clock limit (``HH:MM:SS``).
    """
    cluster: str
    partition: str
    cpus: int
    gpus: int
    memory: str
    time: str


# --------------------------------------------------------------------------- #
# Header assembly
# --------------------------------------------------------------------------- #

def _notify_directive() -> str:
    """Emit ``--mail-type/--mail-user`` lines if the env-var is set."""
    email = os.environ.get("TABPFN_SLURM_NOTIFY_EMAIL", "").strip()
    if not email:
        return ""
    return (
        f"#SBATCH --mail-type=FAIL,TIME_LIMIT,REQUEUE\n"
        f"#SBATCH --mail-user={email}\n"
    )


def _array_range(n_tasks: int, max_concurrent: int) -> str:
    if n_tasks == 0:
        return "0"
    return f"0-{n_tasks-1}%{max_concurrent}"


def slurm_header(
    *,
    job_name: str,
    resources: SlurmResources,
    log_subdir: str,
    array_range: str,
    soft_isolation: bool = False,
) -> str:
    """Assemble the ``#SBATCH`` block shared by every batched array job.

    Args:
        job_name:       SLURM ``--job-name`` value.
        resources:      :class:`SlurmResources` instance.
        log_subdir:     Subdirectory under ``results/`` to put logs in
                        (e.g. ``"experiment1/logs/slurm"``). Paths are made
                        absolute via ``${VSC_DATA}``.
        array_range:    SLURM ``--array`` value (e.g. ``"0-399%16"``).
        soft_isolation: If ``True``, request 100 GiB RAM to dominate a wICE
                        H100 node without ``--exclusive`` (pseudo-isolation).
    """
    mem_line = "#SBATCH --mem=100G" if soft_isolation else f"#SBATCH --mem={resources.memory}"
    gpus_line = f"#SBATCH --gpus-per-node={resources.gpus}\n" if resources.gpus else ""
    notify = _notify_directive()

    return textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --cluster={resources.cluster}
        #SBATCH --account=lp_verbekelab
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={resources.cpus}
        {gpus_line}{mem_line}
        #SBATCH --partition={resources.partition}
        #SBATCH --time={resources.time}
        #SBATCH --output=${{VSC_DATA}}/TabPFNCredit/results/{log_subdir}/{job_name}_%A_%a.out
        #SBATCH --error=${{VSC_DATA}}/TabPFNCredit/results/{log_subdir}/{job_name}_%A_%a.err
        #SBATCH --requeue
        {notify}#SBATCH --array={array_range}
        """
    )


# --------------------------------------------------------------------------- #
# Prologue / environment setup
# --------------------------------------------------------------------------- #

SLURM_PROLOGUE = textwrap.dedent(
    """\
    # Fail fast on any error; treat unset vars as errors; pipefail.
    set -euo pipefail

    # Stagger array-slot starts by up to 60 s to avoid I/O thundering-herd
    # on the shared filesystem when hundreds of tasks launch together.
    sleep $((RANDOM % 60 + 1))

    # Force unbuffered I/O so SLURM streams Python output in real time.
    export PYTHONUNBUFFERED=1

    # Memory-fragmentation mitigation for PyTorch (critical for foundation models).
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

    # Activate conda environment installed under $VSC_DATA.
    export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
    source activate TabPFNCredit

    # Use conda's C++ runtime libs (avoids glibc++ mismatches on VSC CentOS nodes).
    export LD_LIBRARY_PATH="${VSC_DATA}/miniconda3/envs/TabPFNCredit/lib:${LD_LIBRARY_PATH:-}"

    # Move to the project root and ensure log directories exist.
    cd "${VSC_DATA}/TabPFNCredit"
    mkdir -p "${VSC_DATA}/TabPFNCredit/results"
    """
)


SLURM_EPILOGUE = textwrap.dedent(
    """\
    EXIT_CODE=$?
    echo "=========================================="
    echo "Task completed with exit code: ${EXIT_CODE}"
    echo "=========================================="
    exit ${EXIT_CODE}
    """
)


# --------------------------------------------------------------------------- #
# Full-script assembly
# --------------------------------------------------------------------------- #

def assemble_array_script(
    *,
    header: str,
    banner_title: str,
    log_subdir: str,
    python_command: str,
    task_offset: int,
) -> str:
    """Glue ``header`` + prologue + body + epilogue into a full SLURM script."""
    banner = textwrap.dedent(
        f"""\
        mkdir -p "${{VSC_DATA}}/TabPFNCredit/results/{log_subdir}"
        echo "=========================================="
        echo "{banner_title}"
        echo "=========================================="
        echo "Job ID:       ${{SLURM_JOB_ID}}"
        echo "Array ID:     ${{SLURM_ARRAY_TASK_ID}}"
        echo "Task offset:  {task_offset}"
        echo "Node:         ${{SLURMD_NODENAME}}"
        echo "GPU:          ${{CUDA_VISIBLE_DEVICES:-N/A}}"
        echo "=========================================="

        GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + {task_offset}))
        """
    )
    return header + "\n" + SLURM_PROLOGUE + "\n" + banner + "\n" + python_command + "\n" + SLURM_EPILOGUE


# --------------------------------------------------------------------------- #
# Batch generation helper
# --------------------------------------------------------------------------- #

def write_batched_scripts(
    *,
    total_tasks: int,
    max_tasks_per_batch: int,
    max_concurrent: int,
    prefix: str,
    scripts_dir: Path,
    build_script: Callable[[int, int, int, int], str],
) -> List[Tuple[str, int, int]]:
    """Split ``total_tasks`` into batches and write one SLURM file per batch.

    ``build_script`` is a callable ``(batch_id, start, end, max_concurrent) -> str``
    that returns the full script body for that batch. Returns a list of
    ``(filename, start_task, end_task)`` tuples for user-facing reporting.
    """
    import math

    if total_tasks == 0:
        return []

    scripts_dir.mkdir(parents=True, exist_ok=True)
    n_batches = math.ceil(total_tasks / max_tasks_per_batch)
    written: List[Tuple[str, int, int]] = []

    for batch_id in range(n_batches):
        start = batch_id * max_tasks_per_batch
        end = min(start + max_tasks_per_batch, total_tasks)
        script = build_script(batch_id, start, end, max_concurrent)

        filename = f"{prefix}{batch_id}.slurm"
        filepath = scripts_dir / filename
        with open(filepath, "w", newline="\n") as fh:
            fh.write(script)
        filepath.chmod(0o755)
        written.append((filename, start, end))

    return written


__all__ = [
    "SlurmResources",
    "slurm_header",
    "SLURM_PROLOGUE",
    "SLURM_EPILOGUE",
    "assemble_array_script",
    "write_batched_scripts",
]
