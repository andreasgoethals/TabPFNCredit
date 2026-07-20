"""SLURM script generator for the VSC (Genius + wICE) clusters.

Designed against the VSC documentation. Highlights:

* **Results land on the shared project storage** (``$TABPFN_STAGING_ROOT``,
  large and non-purged) so the small general data storage cannot fill mid-run.
  The **regenerable joblib cache goes to ``$VSC_SCRATCH``** -- the project
  storage is inode-limited (~150k inodes/TB) and a cache of many tiny files
  would exhaust it. **Logs deliberately stay on the general data storage**
  (the repo root) -- tiny and must persist. Datasets and checkpoints are read
  from the repo first, then the project storage.
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
2. ``cd "$VSC_DATA/TabPFNCredit"`` and exports ``TABPFN_RESULTS_ROOT`` /
   ``TABPFN_CACHE_ROOT`` onto the shared project storage.
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
import shlex
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

# Absolute path to the repo root, resolved at generation time (the generator
# runs on the login node, where this is the real on-cluster path, e.g.
# /vsc-hard-mounts/leuven-data/<group>/<vsc-id>/TabPFNCredit). We bake this
# ABSOLUTE path into the generated scripts instead of ``$VSC_DATA/...``.
# Reason: SLURM does NOT expand environment variables in ``#SBATCH --output``
# / ``--error`` directives, so ``${VSC_DATA}`` there is taken LITERALLY and
# SLURM creates a bogus directory literally named ``${VSC_DATA}`` under the
# submit dir. An absolute path sidesteps that entirely. Since the repo lives
# at ``$VSC_DATA/TabPFNCredit`` on the cluster, ``<repo>/logs`` ==
# ``$VSC_DATA/TabPFNCredit/logs`` -- on the general data storage, no surprises.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])

# Shared project-storage root (env-overridable): this project's own subfolder
# of the shared ``stg_00211`` allocation. Results live here so they never fill
# the small general data storage; logs and the (inode-heavy) joblib cache do
# NOT (the cache goes to $VSC_SCRATCH).
_DEFAULT_STAGING_ROOT = "/lustre1/project/stg_00211/TabPFNCredit"


def _staging_bash() -> str:
    """Bash expression for the project-storage root, expanded at run time.

    Safe to use in the script BODY (env vars expand there) -- NOT in #SBATCH
    directives, which SLURM does not env-expand.
    """
    return f"${{TABPFN_STAGING_ROOT:-{_DEFAULT_STAGING_ROOT}}}"


def _results_root_bash() -> str:
    """Quoted bash expr for the results root on the shared project storage."""
    return f'"{_staging_bash()}/results"'


def _cache_root_bash() -> str:
    """Quoted bash expr for the regenerable-cache root on ``$VSC_SCRATCH``.

    Caches are many tiny files; the project storage is inode-limited
    (~150k inodes/TB), so the joblib folds cache goes to scratch (the
    purge-after-28-days parallel FS meant for regenerable job I/O) instead.
    """
    return '"${VSC_SCRATCH:-/tmp}/tabpfncredit/cache"'


def _log_dir(experiment: str) -> str:
    """SLURM log directory on the general data storage (repo root), NEVER project storage.

    Returned as an ABSOLUTE literal path: SLURM does not expand env vars in
    ``#SBATCH --output`` / ``--error``. The repo lives at
    ``$VSC_DATA/TabPFNCredit``, so ``<repo>/logs`` is on the general data storage.
    """
    return f"{_REPO_ROOT}/logs/{experiment.lower()}"


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
    "gpu_v100": PartitionSpec(
        cluster="genius", partition="gpu_v100",
        cpus_per_gpu=4,          # 2 nodes x 8 V100 32GB; ~36 cores/node
        mem_per_gpu_mb=20_000,   # small-row offload work needs little host RAM
        gpus_per_node=1,
        max_walltime_hours=72,
        cost_weight=59.58,
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
    # Tier-2 Mindwell (production since 2026-06): 3 nodes x 8 NVIDIA B200 SXM6
    # (192 GiB, Blackwell sm_100), 2x 96-core AMD EPYC 9655, 1536 GiB RAM ->
    # 24 cores & ~192 GB per GPU. Brand-new GPUs: the venv's torch build must
    # ship Blackwell (sm_100) kernels and a 2025a+ Python module, and the
    # project account must have Mindwell credits/storage. Used ONLY as a
    # cross-cluster REPLICA target (never the afterok primary), so if any of
    # those prerequisites are missing the replica simply fails harmlessly and
    # wICE still completes the work. Opt in via TABPFN_ALL_CLUSTERS / the
    # replicate set.
    "gpu_b200": PartitionSpec(
        cluster="mindwell", partition="gpu_b200",
        cpus_per_gpu=16,
        mem_per_gpu_mb=180_000,
        gpus_per_node=1,
        max_walltime_hours=72,
        cost_weight=600.0,
    ),
}

# GPU partitions across ALL clusters, used by the "all clusters at once" mode
# (TABPFN_ALL_CLUSTERS). wICE A100/H100 stay the afterok primary; Genius V100
# and Mindwell B200 are added as replicas. P100 is excluded (torch 2.8 ships
# sm_70+ only; Pascal sm_60 is unusable).
ALL_GPU_PARTITIONS = ("gpu_h100", "gpu_a100", "gpu_v100", "gpu_b200")


def partition_for_method(method: str, *, prefer_h100: bool = True) -> str:
    """Return the partition KEY (not the SLURM partition name) for ``method``.

    IMPORTANT -- single-cluster rule
    --------------------------------
    Everything maps to **wICE** partitions (``cpu`` = batch_sapphirerapids,
    ``gpu_a100``, ``gpu_h100``). We deliberately do NOT route anything to
    Genius (``gpu_p100`` / ``cpu_genius``): SLURM ``afterok`` dependencies
    cannot cross clusters, so the primary experiment dependency chain would
    be invalid if one experiment's arrays were split between Genius and wICE.
    Keeping a whole experiment on one cluster makes the dependency chain valid
    and the job-id bookkeeping trivial. (The Genius partition specs remain in
    ``PARTITIONS`` for manual/advanced use, but the generator never picks them.)
    """
    profile = get_profile(method)
    if not profile.prefers_gpu:
        return "cpu"
    if profile.needs_foundation_gpu:
        return "gpu_h100" if prefer_h100 else "gpu_a100"
    # Standard (non-foundation) GPU methods: A100 on wICE -- NOT P100 on
    # Genius, to keep the whole sweep on a single cluster (see above).
    return "gpu_a100"


# ============================================================================
#  Work-item packing (point-level sharding)
# ============================================================================
#
# The scheduling unit is a SWEEP POINT (one result file), not a whole
# (dataset, method) cell. This matters for Experiment 2/3: a single cell can
# expand into thousands of points (the row / minority sweep), which would blow
# past the 72 h wall if run sequentially in one slot. We therefore bin-pack the
# *points* across array slots so each slot's estimated work fits under the
# partition wall, while honouring the VSC per-user submit limit.
#
# TWO HARD CONSTRAINTS (VSC docs, confirmed):
#   * Wall-clock <= 72 h on every wICE partition (gpu_a100/gpu_h100/
#     batch_sapphirerapids); there is NO `_long` GPU partition on wICE.
#   * MaxSubmitJobsPerUser (~500 on the `normal` QOS) and EACH array element
#     counts. So #slots is capped at MAX_ARRAY_SLOTS per partition.
#
# Packing: n_slots = clamp(ceil(total_est / cap_seconds), 1, max_slots), then
# greedy longest-processing-time-first (heaviest point into the least-loaded
# slot) for an even spread. When the work needs MORE slots than the cap allows,
# we still emit max_slots slots (each then exceeds the 72 h estimate) and the
# caller warns -- skip-if-done makes a timed-out slot resumable on re-submit.

# Hard cap on array tasks PER PARTITION. Default 40 keeps the full
# run_all_experiments.sh chain (4 experiments pre-submitted at once via afterok,
# up to 3 partitions each) under the ~500 `normal`-QOS submit ceiling:
# 4 x 3 x 40 = 480 < 500. For a big STANDALONE sweep (e.g. Experiment 2) raise
# it via $TABPFN_MAX_ARRAY_SLOTS -- e.g. 150 gives ~450 tasks across 3
# partitions, still < 500 -- to shorten each slot and parallelise harder.
MAX_ARRAY_SLOTS = 40


def _split_group_by_cap(group: List[dict], cap_seconds: int) -> List[List[dict]]:
    """Split one cell's points into contiguous sub-groups each ``<= cap_seconds``.

    Preserves sweep order within a sub-group (so a shard holds a contiguous
    slice of the curve). A single point costing more than the cap becomes its
    own sub-group. Used only for packed, splittable experiments (2/3).
    """
    subs: List[List[dict]] = []
    cur: List[dict] = []
    cur_est = 0
    for it in group:
        e = int(it.get("est_seconds", 0))
        if cur and cur_est + e > cap_seconds:
            subs.append(cur)
            cur, cur_est = [], 0
        cur.append(it)
        cur_est += e
    if cur:
        subs.append(cur)
    return subs or [group]


def pack_work_items(
    items: List[dict],
    *,
    cap_seconds: int,
    max_slots: int = MAX_ARRAY_SLOTS,
    split_cells: bool = False,
    max_cells_per_slot: Optional[int] = None,
) -> tuple[List[List[dict]], int]:
    """Greedy LPT bin-pack work items into ``<= max_slots`` slots.

    Points are grouped by their ``(task, dataset, method)`` **cell**. By default
    each whole cell is packed as a unit (Experiment 0/1: a cell is 1-2 points,
    and Experiment 1's HPO-copy point must share its NO_HPO point's slot).

    When ``split_cells`` is True (Experiment 2/3), a cell whose estimate exceeds
    ``cap_seconds`` is split into contiguous sub-groups so its sweep points fan
    out across MULTIPLE array tasks and run in parallel — each task writes its
    own per-task packed shard file, so there is still a single writer per file.
    Cheap cells (estimate within one slot) stay whole, i.e. one shard, keeping
    the file/inode count low.

    ``items`` are dicts each carrying an ``est_seconds`` cost plus the point
    payload (``dataset``, ``method``, ``task``, ``name``, ``tune``,
    ``row_limit``, ``sampling``). The slot count is
    ``clamp(ceil(total / cap_seconds), 1, max_slots)`` so the per-slot load
    targets ``cap_seconds`` (~95% of the wall) but never produces more tasks
    than the submit limit allows.

    ``max_cells_per_slot`` (default: unbounded) forces finer parallelism than
    the pure time-packing would choose: no slot gets more than this many cells,
    so e.g. ``max_cells_per_slot=1`` runs every (task, dataset, method) cell as
    its OWN array task -- one dataset per GPU, each with the full walltime. This
    is what you want for an expensive in-context method (e.g. TabFM) whose
    per-dataset cost is hard to estimate: isolate each dataset so a slow one
    can't drag the others past the wall. Honoured on a best-effort basis -- if
    ``max_slots`` is too small to give every cell its own slot, the leftover
    cells pack into the least-loaded slots.

    Returns ``(slots, max_slot_seconds)``. ``max_slot_seconds`` lets the caller
    detect when the work could not be squeezed under ``cap_seconds`` (i.e. it
    needed more than ``max_slots`` slots, or a single cell exceeds the cap) and
    warn accordingly.
    """
    items = [it for it in items if it]
    if not items:
        return [], 0

    # Group points by cell. By default a cell stays whole; with split_cells the
    # cell's points may fan out across tasks (each task -> its own shard file).
    groups: dict = {}
    for it in items:
        groups.setdefault((it.get("task"), it.get("dataset"), it.get("method")), []).append(it)
    group_list = list(groups.values())

    if split_cells:
        split_list: List[List[dict]] = []
        for g in group_list:
            g_est = sum(int(it.get("est_seconds", 0)) for it in g)
            if len(g) <= 1 or g_est <= cap_seconds:
                split_list.append(g)            # cheap cell -> one shard
            else:
                split_list.extend(_split_group_by_cap(g, cap_seconds))
        group_list = split_list

    group_est = [sum(int(it.get("est_seconds", 0)) for it in g) for g in group_list]

    total = sum(group_est)
    needed = max(1, math.ceil(total / max(1, cap_seconds)))
    cap_cells = max_cells_per_slot if (max_cells_per_slot and max_cells_per_slot > 0) else None
    if cap_cells is not None:
        # Need at least this many slots to keep every slot within the cell cap.
        needed = max(needed, math.ceil(len(group_list) / cap_cells))
    n_slots = max(1, min(needed, max_slots, len(group_list)))

    slots: List[List[dict]] = [[] for _ in range(n_slots)]
    loads = [0] * n_slots
    counts = [0] * n_slots
    for gi in sorted(range(len(group_list)), key=lambda j: group_est[j], reverse=True):
        # Prefer slots still under the per-slot cell cap; if every slot is at the
        # cap (max_slots too small to honour it), fall back to the least loaded.
        candidates = [j for j in range(n_slots) if cap_cells is None or counts[j] < cap_cells]
        if not candidates:
            candidates = list(range(n_slots))
        k = min(candidates, key=lambda j: loads[j])
        slots[k].extend(group_list[gi])
        loads[k] += group_est[gi]
        counts[k] += 1
    return [s for s in slots if s], (max(loads) if loads else 0)


# ============================================================================
#  Script body assembly
# ============================================================================

_SHEBANG = "#!/bin/bash -l"

def _prologue(*, cluster: str, partition: str) -> str:
    """Return the per-script prologue (runs on the assigned compute node).

    VSC module rules (from the docs):
      * The ``cluster`` module is *sticky* and is auto-loaded for the node
        the job lands on -- we must NOT load one ourselves, and we must NOT
        use ``module --force purge`` (that nukes the sticky cluster module
        and breaks $MODULEPATH). Plain ``module purge`` keeps it.
      * After ``module purge`` the node's own software stack is on
        $MODULEPATH, so a plain ``module load Python/...`` resolves to that
        node's build. Override the Python module name by exporting
        ``TABPFN_PYTHON_MODULE`` before sbatch if your cluster differs.
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

        # ---- Foundation-model weight caches (OFFLINE) ----
        # Compute nodes have NO outbound internet, so foundation models
        # (TabPFN v2/v2.5/v3, TabICL, TabDPT, ...) must read pre-staged weights
        # instead of downloading at run time. We look for a populated
        # repo-local `checkpoints/` FIRST, then fall back to the shared project
        # storage (`$TABPFN_STAGING_ROOT/checkpoints`). Offline mode is forced
        # so a missing weight fails fast with a clear error rather than hanging
        # on a blocked network call.
        export TABPFN_STAGING_ROOT="{_staging_bash()}"
        if [ -d "{_REPO_ROOT}/checkpoints" ] && [ -n "$(ls -A "{_REPO_ROOT}/checkpoints" 2>/dev/null)" ]; then
            CKPT_ROOT="{_REPO_ROOT}/checkpoints"
        else
            CKPT_ROOT="${{TABPFN_STAGING_ROOT}}/checkpoints"
        fi
        echo "Checkpoints root: ${{CKPT_ROOT}}"
        export HF_HOME="${{CKPT_ROOT}}/huggingface"
        export HUGGINGFACE_HUB_CACHE="${{CKPT_ROOT}}/huggingface/hub"
        export TORCH_HOME="${{CKPT_ROOT}}/torch"
        export XDG_CACHE_HOME="${{CKPT_ROOT}}/xdg"
        export TABPFN_MODEL_CACHE_DIR="${{CKPT_ROOT}}/tabpfn"
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

        # Silence tqdm progress bars in batch logs (e.g. Optuna HPO) -- they
        # otherwise emit megabytes of partial-line spam to the .out file.
        export TQDM_DISABLE=1

        # Plain ``module purge`` (NOT --force) -- keeps the sticky cluster
        # module that's auto-loaded for this node, then load the toolchain.
        module purge
        : "${{TABPFN_PYTHON_MODULE:=Python/3.12.3-GCCcore-13.3.0}}"
        module load "${{TABPFN_PYTHON_MODULE}}" 2>/dev/null || \\
            echo "WARN: could not module load ${{TABPFN_PYTHON_MODULE}}" >&2

        # Activate the project's Python env. Prefer a plain ``python -m venv``
        # at the repo root (absolute path baked in at generation time) and
        # fall back to a conda env named ``tabpfncreditvenv`` if that's what
        # you set up.
        if [ -f "{_REPO_ROOT}/tabpfncreditvenv/bin/activate" ]; then
            source "{_REPO_ROOT}/tabpfncreditvenv/bin/activate"
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
    if n_gpus:
        cpus = spec.cpus_per_gpu
        mem_mb = spec.mem_per_gpu_mb * max(1, n_gpus)
    else:
        # CPU arrays default to HALF a node (18 of 36 cores) so two tasks pack
        # per node -- classical methods scale poorly past ~18 threads, so this
        # ~doubles CPU throughput for the same node count. Memory scales with
        # the core share. Override with $TABPFN_CPU_CORES_PER_TASK (e.g. 36
        # for a whole node when one task genuinely needs all the RAM).
        cpus = int(os.environ.get("TABPFN_CPU_CORES_PER_TASK", 18))
        cpus = max(1, min(cpus, spec.cpus_per_gpu))
        mem_mb = int(spec.mem_per_gpu_mb * cpus / spec.cpus_per_gpu)

    lines = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --clusters={spec.cluster}",
        # Slurm credit account: defaults to the project's account; override for
        # another project with TABPFN_SLURM_ACCOUNT.
        f"#SBATCH --account={os.environ.get('TABPFN_SLURM_ACCOUNT', 'lp_verbekelab')}",
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


def _python_invocation(experiment: str, partition_key: str, plan_path: Path) -> str:
    """The actual work line -- delegates to the Typer CLI."""
    # Results live on the shared project storage (large, non-purged); the
    # regenerable joblib cache goes to $VSC_SCRATCH (the project storage is
    # inode-limited, bad for many tiny cache files). These run in the script
    # BODY, where env vars DO expand (unlike #SBATCH directives). Per-fold
    # scratch (TALENT's internal save_path) is a tempfile dir cleaned at exit.
    return dedent(f"""\
        cd "{_REPO_ROOT}"

        export TABPFN_STAGING_ROOT="{_staging_bash()}"
        export TABPFN_RESULTS_ROOT={_results_root_bash()}
        export TABPFN_CACHE_ROOT={_cache_root_bash()}
        mkdir -p "${{TABPFN_RESULTS_ROOT}}" "${{TABPFN_CACHE_ROOT}}"

        tabpfncredit slurm-task \\
            --experiment {experiment} \\
            --partition {partition_key} \\
            --array-id "${{SLURM_ARRAY_TASK_ID}}" \\
            --plan-path {shlex.quote(str(plan_path))}
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
    work_items: List[dict],
    out_dir: Path,
    n_folds: int = 5,
    prefer_h100: bool = True,
    gpu_cmode: str = "shared",
    max_concurrent: Optional[int] = None,
    mail_email: str = "",
    max_slots: Optional[int] = None,
    max_cells_per_slot: Optional[int] = None,
    run_id: Optional[str] = None,
) -> List[GeneratedJob]:
    """Write SLURM scripts for one experiment, sharding sweep POINTS across slots.

    Parameters
    ----------
    experiment : str
        ``"Experiment0"`` etc. (case-insensitive); used in the job name.
    work_items : list of dict
        One entry per SWEEP POINT (not per cell). Each dict must carry
        ``"dataset"``, ``"method"``, ``"task"``, ``"name"`` (the result-file
        stem incl. sweep suffix), ``"tune"``, ``"row_limit"``, ``"sampling"``
        and ``"est_seconds"`` (per-point cost, from
        :func:`runtime_profile.estimate_point_seconds`). The points are
        bin-packed across array slots so each slot fits under the 72 h wall.
    out_dir : Path
        Directory to write the generated scripts into.
    n_folds : int
        Unused here (kept for API stability); point costs already fold in folds.
    prefer_h100 : bool
        Use H100 wICE partition for foundation models; fall back to A100 if False.
    gpu_cmode : str
        ``shared`` (default; multiple processes per GPU OK) or ``exclusive``.
    max_concurrent : int
        SLURM array ``%`` throttle.
    mail_email : str
        If non-empty, add ``--mail-type=FAIL,TIME_LIMIT,REQUEUE`` notifications.
    max_slots : int, optional
        Per-partition array-task cap. Defaults to ``$TABPFN_MAX_ARRAY_SLOTS``
        or :data:`MAX_ARRAY_SLOTS`.
    run_id : str, optional
        Suffix for generated script/plan filenames. Use a unique value for
        submitted jobs so later submissions cannot overwrite the plan that a
        pending SLURM array will read.

    Returns
    -------
    list of :class:`GeneratedJob`
    """
    if not work_items:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if max_slots is None:
        max_slots = int(os.environ.get("TABPFN_MAX_ARRAY_SLOTS", MAX_ARRAY_SLOTS))

    # Optional per-slot cell cap -> finer parallelism than time-packing alone.
    # e.g. TABPFN_MAX_CELLS_PER_SLOT=1 runs one (dataset, method) cell per array
    # task (one dataset per GPU), so a slow dataset can't drag others past the
    # wall. Unset = pure time-packing (unchanged default behaviour).
    if max_cells_per_slot is None:
        _mcps_env = os.environ.get("TABPFN_MAX_CELLS_PER_SLOT")
        max_cells_per_slot = int(_mcps_env) if _mcps_env else None

    # Experiment 2/3 pack a cell's sweep points into per-task shard files, so a
    # big cell's points may be split across array tasks (intra-cell parallelism).
    # Experiment 0/1 keep cells whole (1-2 points; Exp1's HPO-copy needs its
    # NO_HPO sibling in the same task).
    split_cells = experiment.lower() in ("experiment2", "experiment3")

    import logging
    _log = logging.getLogger(__name__)

    # 1) Group POINTS by partition (keyed on the point's method).
    by_partition: dict[str, list[dict]] = {}
    for item in work_items:
        key = partition_for_method(item["method"], prefer_h100=prefer_h100)
        by_partition.setdefault(key, []).append(item)

    # 1a-bis) Optional AGGRESSIVE replication: COPY small-data GPU work to
    # every partition listed in TABPFN_REPLICATE_PARTITIONS (incl. "cpu").
    # Runtime skip-if-done dedupes; first replica to finish a point wins.
    # Takes precedence over the Genius offload (which MOVES instead).
    replicated = _replicate_small_data(by_partition, log=_log)

    # 1a) Optional Genius offload: route SMALL-DATA GPU work (Experiment 2's
    # row-capped sweeps, Experiment 3's subsampled sweeps) to the old-but-idle
    # Genius V100/P100 fleet. Opt-in via TABPFN_GENIUS_GPUS -- see the helper.
    if not replicated:
        _genius_offload(by_partition, log=_log)

    # 1b) GPU spillover: when one GPU partition's work exceeds what its slot
    # cap can finish inside the wall while the other GPU partition has spare
    # capacity, move whole CELLS across (both wICE GPU partitions run every
    # GPU method; A100/H100 both have 80 GB). Disable with TABPFN_GPU_SPREAD=0
    # -- e.g. to keep deep-method work off the ~4x-more-expensive H100.
    _spillover_gpu(by_partition, max_slots=max_slots, log=_log)

    generated: List[GeneratedJob] = []
    filename_suffix = f"_{run_id}" if run_id else ""

    for partition_key, items in sorted(by_partition.items()):
        if partition_key not in PARTITIONS:
            continue
        spec = PARTITIONS[partition_key]
        n_gpus = 1 if spec.gpus_per_node else 0

        # 2) Shard the partition's points across <= max_slots array tasks,
        #    targeting ~95% of the wall per slot so each slot finishes inside
        #    the 72 h cap whenever the work fits in the slot budget.
        cap_seconds = int(0.95 * spec.max_walltime_hours * 3600)
        slots, max_slot_seconds = pack_work_items(
            items, cap_seconds=cap_seconds, max_slots=max_slots,
            split_cells=split_cells, max_cells_per_slot=max_cells_per_slot,
        )
        if not slots:
            continue

        # 3) Walltime = the heaviest slot's estimate, floored at 10 min and
        #    capped at the partition's hard wall.
        hard_cap = spec.max_walltime_hours * 3600
        walltime_seconds = min(max(max_slot_seconds, 600), hard_cap)
        walltime = _format_walltime(walltime_seconds)

        # If the work could not be squeezed under the wall with the available
        # slots, warn: the slot will hit the 72 h limit and need a re-submit
        # (skip-if-done resumes), OR the user can raise the slot cap.
        if max_slot_seconds > cap_seconds:
            needed = math.ceil(
                sum(int(it.get("est_seconds", 0)) for it in items) / max(1, cap_seconds)
            )
            _log.warning(
                "slurm-generate: %s/%s needs ~%d slots to fit every slot under "
                "%dh but is capped at %d. Slots may hit the wall and require a "
                "re-submit to resume (safe -- skip-if-done). To parallelise "
                "harder in one shot, raise the cap, e.g. "
                "TABPFN_MAX_ARRAY_SLOTS=%d (keep TOTAL submitted array tasks < 500).",
                experiment, partition_key, needed, spec.max_walltime_hours,
                len(slots), min(needed, 450),
            )

        # 4) Persist the per-slot point plan as a JSON sibling so slurm-task
        #    knows what to run.
        plan_path = out_dir / f"{experiment.lower()}_{partition_key}{filename_suffix}_plan.json"
        _write_plan(plan_path, slots)

        # 5) Emit the SLURM script. NO ``%N`` concurrency throttle by default:
        # the throttle limited how many array elements RUN at once (it never
        # affected the ~500-job SUBMIT limit -- every element counts as
        # submitted regardless), so it only kept allocatable nodes idle.
        # SLURM fairshare governs concurrency; re-throttle for I/O reasons via
        # $TABPFN_MAX_CONCURRENT or the max_concurrent parameter.
        job_name = f"{experiment.lower()}_{partition_key}"
        throttle = max_concurrent or int(os.environ.get("TABPFN_MAX_CONCURRENT", 0))
        array_range = (
            f"0-{len(slots) - 1}" + (f"%{throttle}" if throttle else "")
            if len(slots) > 1 else "0"
        )
        # Logs stay on the general data storage (repo root), NOT on project
        # storage. ABSOLUTE literal path (SLURM does not env-expand
        # --output/--error).
        log_dir = _log_dir(experiment)
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
            _python_invocation(experiment, partition_key, plan_path),
            _EPILOGUE,
        ])

        script_path = out_dir / f"{job_name}{filename_suffix}.slurm"
        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o755)

        generated.append(GeneratedJob(
            path=script_path,
            partition_key=partition_key,
            n_array_slots=len(slots),
            walltime=walltime,
        ))

    # 6) Submit-limit guard. EACH array element counts toward the VSC
    #    MaxSubmitJobsPerUser (~500 on the `normal` QOS), and the
    #    run_all_experiments.sh chain pre-submits every experiment at once via
    #    afterok -- so the TOTAL across all submitted arrays must stay < 500.
    total_slots = sum(j.n_array_slots for j in generated)
    if total_slots > 450:
        _log.warning(
            "slurm-generate: %d array tasks for this experiment alone. EACH "
            "counts toward the VSC per-user submit limit (~500 on the `normal` "
            "QOS); if you also chain other experiments you may hit "
            "QOSMaxSubmitJobPerUserLimit. Lower $TABPFN_MAX_ARRAY_SLOTS or "
            "submit experiments one at a time.",
            total_slots,
        )

    return generated


def _small_data_eligible(it: dict, row_cap: int) -> bool:
    """Small-data = an explicit row cap at/under ``row_cap`` (Experiment 2's
    capped sweeps) or a sampling target (Experiment 3's subsampled sweeps)."""
    rl, sp = it.get("row_limit"), it.get("sampling")
    return (rl is not None and rl <= row_cap) or (sp is not None)


def _replicate_small_data(by_partition: dict, *, log) -> bool:
    """AGGRESSIVE mode: duplicate small-data GPU cells onto EVERY partition in
    ``TABPFN_REPLICATE_PARTITIONS`` (comma-separated keys; may include "cpu").

    Every replica array runs the same points; each point is skipped at RUN
    time when its result already exists, so whichever queue starts first does
    the work and the others skip through. To minimise duplicate compute the
    replicas traverse the work in different orders (rotation + reversal), so
    queues eat the list from different ends and meet in the middle. Races on
    the same point are harmless: result writes are atomic, a lost packed
    update is re-detected as missing by the next ``tabpfncredit resubmit``.

    CPU replicas additionally require an explicit row cap at/under
    ``TABPFN_CPU_FOUNDATION_ROW_CAP`` (default 10000) -- in-context inference
    on CPU is viable for small fits only -- and their cost estimates are
    scaled by ``TABPFN_CPU_FOUNDATION_SLOWDOWN`` (default 10x) for packing.

    Returns True when replication ran (the Genius MOVE offload is then skipped).
    """
    targets = [p.strip() for p in os.environ.get("TABPFN_REPLICATE_PARTITIONS", "").split(",")
               if p.strip()]
    all_clusters = os.environ.get("TABPFN_ALL_CLUSTERS", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if not targets and all_clusters:
        # "Use every cluster at once": replicate small-data GPU work onto every
        # GPU partition across wICE + Genius + Mindwell. wICE A100/H100 remain
        # the afterok primary; Genius V100 / Mindwell B200 are pure accelerators.
        targets = list(ALL_GPU_PARTITIONS)
    if not targets:
        return False
    if "gpu_p100" in targets:
        log.warning("slurm-generate: gpu_p100 cannot run the installed torch 2.8 "
                    "wheels (sm_70+ only); dropping it from the replica set.")
    targets = [p for p in targets if p in PARTITIONS and p != "gpu_p100"]
    if "gpu_b200" in targets:
        log.warning("slurm-generate: replicating to Mindwell gpu_b200 (B200, "
                    "Blackwell sm_100). Ensure the venv torch ships sm_100 kernels "
                    "+ a 2025a Python module (export TABPFN_PYTHON_MODULE) and the "
                    "account has Mindwell credits. These are REPLICA tasks -- if a "
                    "prerequisite is missing they fail harmlessly and wICE still "
                    "completes the work.")
    if not targets:
        return False

    row_cap = int(os.environ.get("TABPFN_GENIUS_ROW_CAP", 60_000))
    cpu_row_cap = int(os.environ.get("TABPFN_CPU_FOUNDATION_ROW_CAP", 10_000))
    cpu_slowdown = float(os.environ.get("TABPFN_CPU_FOUNDATION_SLOWDOWN", 10.0))

    # Source points: small-data work currently homed on the wICE GPU queues.
    # Track WHICH wICE GPU partition each point is homed on, so we only skip
    # *that* partition as a replica target -- the other wICE GPU (e.g. A100 when
    # foundation work homes on H100) is free capacity and should also get a
    # replica rather than sit idle.
    source: List[dict] = []
    source_homes: set = set()
    for src in ("gpu_h100", "gpu_a100"):
        src_pts = [it for it in by_partition.get(src, [])
                   if _small_data_eligible(it, row_cap)]
        if src_pts:
            source_homes.add(src)
        source.extend(src_pts)
    if not source:
        return False

    n_added = {}
    replica_idx = 0
    for dst in targets:
        if dst in source_homes:
            continue  # don't duplicate points onto the partition they're homed on
        if dst == "cpu" or PARTITIONS[dst].gpus_per_node == 0:
            rep = [dict(it) for it in source
                   if it.get("row_limit") is not None
                   and it["row_limit"] <= cpu_row_cap]
            for it in rep:
                it["est_seconds"] = int(it.get("est_seconds", 0) * cpu_slowdown)
        else:
            rep = [dict(it) for it in source]
        if not rep:
            continue
        # Distinct traversal per replica: rotate by a different fraction and
        # reverse every other replica, so queues start at different regions.
        replica_idx += 1
        k = (replica_idx * len(rep)) // (len(targets) + 1)
        rep = rep[k:] + rep[:k]
        if replica_idx % 2 == 1:
            rep.reverse()
        by_partition.setdefault(dst, []).extend(rep)
        n_added[dst] = len(rep)

    if n_added:
        log.warning(
            "slurm-generate: AGGRESSIVE replication copied small-data points to "
            "%s (runtime skip-if-done dedupes; run `tabpfncredit resubmit` once "
            "more after completion to mop up any raced packed points).",
            ", ".join(f"{p}:+{n}" for p, n in n_added.items()),
        )
    return bool(n_added)


def _genius_offload(by_partition: dict, *, log) -> None:
    """Opt-in: move SMALL-DATA GPU cells to the Genius V100/P100 fleet.

    Enabled by ``TABPFN_GENIUS_GPUS`` (comma-separated partition keys, e.g.
    ``gpu_v100`` or ``gpu_v100,gpu_p100``). Only cells whose EVERY point is
    small-data qualify: a ``row_limit`` at or under ``TABPFN_GENIUS_ROW_CAP``
    (default 60000 -- covers Experiment 2's capped sweeps) or a ``sampling``
    value (Experiment 3's minority sweeps, whose datasets are small by
    construction). Big-data work (Experiment 0/1 full-dataset foundation
    fits) NEVER moves: P100 has 16 GB and V100 32 GB.

    Moved cells are balanced across the enabled Genius partitions by fleet
    throughput (16 V100s vs 52 ~half-speed P100s). NOTE: SLURM ``afterok``
    cannot cross clusters, so the caller excludes Genius arrays from the
    summarize dependency (the CLI prints a reminder).
    """
    targets = [p.strip() for p in os.environ.get("TABPFN_GENIUS_GPUS", "").split(",")
               if p.strip()]
    if "gpu_p100" in targets:
        # Verified 2026-06-12: the venv's torch 2.8 CUDA wheels ship sm_70+
        # kernels only ("no kernel image" on Pascal). P100 = sm_60 -> unusable.
        log.warning("slurm-generate: gpu_p100 requested but Pascal (sm_60) is "
                    "NOT supported by the installed torch 2.8 wheels (sm_70+). "
                    "Ignoring gpu_p100; use gpu_v100.")
        targets = [p for p in targets if p != "gpu_p100"]
    targets = [p for p in targets if p in PARTITIONS and PARTITIONS[p].cluster == "genius"]
    if not targets:
        return
    row_cap = int(os.environ.get("TABPFN_GENIUS_ROW_CAP", 60_000))
    # Relative throughput per partition: #GPUs x speed vs V100.
    fleet = {"gpu_v100": 16 * 1.0, "gpu_p100": 52 * 0.5}
    weights = {p: fleet.get(p, 1.0) for p in targets}

    def _eligible(it: dict) -> bool:
        return _small_data_eligible(it, row_cap)

    loads = {p: 0.0 for p in targets}
    moved = 0
    for src in ("gpu_h100", "gpu_a100"):
        src_items = by_partition.get(src)
        if not src_items:
            continue
        cells: dict = {}
        for it in src_items:
            cells.setdefault((it.get("task"), it.get("dataset"), it.get("method")), []).append(it)
        for _key, cell_items in sorted(
                cells.items(),
                key=lambda kv: -sum(int(i.get("est_seconds", 0)) for i in kv[1])):
            if not all(_eligible(it) for it in cell_items):
                continue
            cell_est = sum(int(i.get("est_seconds", 0)) for i in cell_items)
            # least-loaded weighted bin
            dst = min(targets, key=lambda p: loads[p] / weights[p])
            loads[dst] += cell_est
            for it in cell_items:
                src_items.remove(it)
            by_partition.setdefault(dst, []).extend(cell_items)
            moved += len(cell_items)
    if moved:
        log.warning(
            "slurm-generate: Genius offload moved %d small-data point(s) to %s "
            "(row cap %d; disable by unsetting TABPFN_GENIUS_GPUS). Reminder: "
            "summarize cannot depend on cross-cluster arrays -- re-run "
            "`tabpfncredit summarize` after the Genius jobs finish.",
            moved, "+".join(targets), row_cap,
        )


def _spillover_gpu(by_partition: dict, *, max_slots: int, log) -> None:
    """Rebalance overflow between the two wICE GPU partitions (in place).

    For each direction (a100->h100, h100->a100): if the source partition's
    total estimated work exceeds its slot capacity (``0.95 * wall * max_slots``)
    and the destination has headroom, whole cells (largest first) are moved
    until the source fits or the destination is full. Cells stay whole so the
    packed-results invariant (one array task owns a whole cell) holds.
    """
    if os.environ.get("TABPFN_GPU_SPREAD", "1") == "0":
        return

    def _total(items: List[dict]) -> int:
        return sum(int(it.get("est_seconds", 0)) for it in items)

    def _capacity(p: str) -> int:
        return int(0.95 * PARTITIONS[p].max_walltime_hours * 3600) * max_slots

    for src, dst in (("gpu_a100", "gpu_h100"), ("gpu_h100", "gpu_a100")):
        src_items = by_partition.get(src)
        if not src_items:
            continue
        over = _total(src_items) - _capacity(src)
        if over <= 0:
            continue
        dst_items = by_partition.setdefault(dst, [])
        headroom = _capacity(dst) - _total(dst_items)
        if headroom <= 0:
            continue

        cells: dict = {}
        for it in src_items:
            cells.setdefault((it.get("task"), it.get("dataset"), it.get("method")), []).append(it)

        moved = 0
        for _key, cell_items in sorted(cells.items(), key=lambda kv: -_total(kv[1])):
            cell_est = _total(cell_items)
            if over <= 0:
                break
            if cell_est > headroom:
                continue
            for it in cell_items:
                src_items.remove(it)
            dst_items.extend(cell_items)
            over -= cell_est
            headroom -= cell_est
            moved += len(cell_items)
        if moved:
            log.warning(
                "slurm-generate: GPU spillover moved %d point(s) %s -> %s to "
                "balance the queues (disable with TABPFN_GPU_SPREAD=0).",
                moved, src, dst,
            )


def _format_walltime(seconds: int) -> str:
    """Format seconds as ``HH:MM:SS`` or ``D-HH:MM:SS`` (SLURM-accepted)."""
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 24:
        days, hh = divmod(hours, 24)
        return f"{days}-{hh:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


_PLAN_ITEM_KEYS = ("dataset", "method", "task", "name", "tune", "row_limit", "sampling", "copy_from")


def _write_plan(path: Path, slots: List[List[dict]]) -> None:
    """Persist the per-slot POINT plan as JSON. ``slurm-task`` reads it.

    Each item keeps the full sweep-point payload (``name``/``tune``/
    ``row_limit``/``sampling``) so the worker runs exactly the point assigned to
    its slot -- no re-expansion, so a cell's thousands of points can be split
    across many slots.
    """
    import json
    payload = {
        "slots": [
            [{k: item.get(k) for k in _PLAN_ITEM_KEYS} for item in slot]
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


__all__ = [
    "PartitionSpec",
    "PARTITIONS",
    "partition_for_method",
    "pack_work_items",
    "MAX_ARRAY_SLOTS",
    "GeneratedJob",
    "generate_scripts_for_experiment",
    "load_plan",
]
