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

# Absolute path to the repo root, resolved at generation time (the generator
# runs on the login node, where this is the real on-cluster path, e.g.
# /vsc-hard-mounts/leuven-data/383/vsc38338/TabPFNCredit). We bake this
# ABSOLUTE path into the generated scripts instead of ``$VSC_DATA/...``.
# Reason: SLURM does NOT expand environment variables in ``#SBATCH --output``
# / ``--error`` directives, so ``${VSC_DATA}`` there is taken LITERALLY and
# SLURM creates a bogus directory literally named ``${VSC_DATA}`` under the
# submit dir. An absolute path sidesteps that entirely. Since the repo lives
# at ``$VSC_DATA/TabPFNCredit`` on the cluster, ``<repo>/results`` ==
# ``$VSC_DATA/TabPFNCredit/results`` -- same location, no env-var surprises.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _results_dir() -> str:
    """Absolute results root baked into generated scripts (``<repo>/results``)."""
    return f"{_REPO_ROOT}/results"


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
    """Return the partition KEY (not the SLURM partition name) for ``method``.

    IMPORTANT -- single-cluster rule
    --------------------------------
    Everything maps to **wICE** partitions (``cpu`` = batch_sapphirerapids,
    ``gpu_a100``, ``gpu_h100``). We deliberately do NOT route anything to
    Genius (``gpu_p100`` / ``cpu_genius``): SLURM ``afterok`` dependencies
    cannot cross clusters, so the auto-submitted summarize job would fail
    if one experiment's arrays were split between Genius and wICE. Keeping
    a whole experiment on one cluster makes the dependency chain valid and
    the job-id bookkeeping trivial. (The Genius partition specs remain in
    ``PARTITIONS`` for manual/advanced use, but the generator never picks
    them.)
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


def pack_work_items(
    items: List[dict],
    *,
    cap_seconds: int,
    max_slots: int = MAX_ARRAY_SLOTS,
) -> tuple[List[List[dict]], int]:
    """Greedy LPT bin-pack work items into ``<= max_slots`` slots.

    ``items`` are dicts each carrying an ``est_seconds`` cost plus the point
    payload (``dataset``, ``method``, ``task``, ``name``, ``tune``,
    ``row_limit``, ``sampling``). The slot count is
    ``clamp(ceil(total / cap_seconds), 1, max_slots)`` so the per-slot load
    targets ``cap_seconds`` (~95% of the wall) but never produces more tasks
    than the submit limit allows.

    Returns ``(slots, max_slot_seconds)``. ``max_slot_seconds`` lets the caller
    detect when the work could not be squeezed under ``cap_seconds`` (i.e. it
    needed more than ``max_slots`` slots) and warn accordingly.
    """
    items = [it for it in items if it]
    if not items:
        return [], 0
    total = sum(int(it.get("est_seconds", 0)) for it in items)
    needed = max(1, math.ceil(total / max(1, cap_seconds)))
    n_slots = max(1, min(needed, max_slots, len(items)))

    slots: List[List[dict]] = [[] for _ in range(n_slots)]
    loads = [0] * n_slots
    for it in sorted(items, key=lambda x: int(x.get("est_seconds", 0)), reverse=True):
        k = min(range(n_slots), key=lambda j: loads[j])
        slots[k].append(it)
        loads[k] += int(it.get("est_seconds", 0))
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

        # ---- Foundation-model weight caches (uploaded checkpoints/, OFFLINE) ----
        # wICE compute nodes have NO outbound internet, so foundation models
        # (TabPFN v2/v2.5/v3, TabICL, TabDPT, ...) cannot download their weights
        # at run time. Download them ONCE on your LOCAL machine with
        # `python scripts/fetch_weights.py`, upload the resulting `checkpoints/`
        # folder to `$VSC_DATA/TabPFNCredit/checkpoints/`, then run
        # `bash scripts/setup_vsc_checkpoints.sh` once to provision the few
        # models that load from a package-internal path (Mitra, HyperFast).
        # Here we point the caches at that uploaded folder and force offline
        # mode so a missing weight fails fast with a clear error instead of
        # hanging on a blocked network call.
        #
        # We bake the ABSOLUTE repo path ({_REPO_ROOT}) rather than expanding
        # ${{VSC_DATA}} at run time -- it is the same location (the repo lives at
        # $VSC_DATA/TabPFNCredit) and avoids any env-var-expansion ambiguity.
        export HF_HOME="{_REPO_ROOT}/checkpoints/huggingface"
        export HUGGINGFACE_HUB_CACHE="{_REPO_ROOT}/checkpoints/huggingface/hub"
        export TORCH_HOME="{_REPO_ROOT}/checkpoints/torch"
        export XDG_CACHE_HOME="{_REPO_ROOT}/checkpoints/xdg"
        export TABPFN_MODEL_CACHE_DIR="{_REPO_ROOT}/checkpoints/tabpfn"
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

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
    # Results land under the repo's ``results/`` (which on the cluster IS
    # ``$VSC_DATA/TabPFNCredit/results`` -- permanent, backed up). We use the
    # ABSOLUTE path baked in at generation time, not ``$VSC_DATA``, so there
    # is no env-var-expansion ambiguity. Per-fold scratch (TALENT's internal
    # save_path) goes to a tempfile dir cleaned up at fold exit.
    return dedent(f"""\
        cd "{_REPO_ROOT}"

        export TABPFN_RESULTS_ROOT="{_results_dir()}"
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
    work_items: List[dict],
    out_dir: Path,
    n_folds: int = 5,
    prefer_h100: bool = True,
    gpu_cmode: str = "shared",
    max_concurrent: int = 16,
    mail_email: str = "",
    max_slots: Optional[int] = None,
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

    import logging
    _log = logging.getLogger(__name__)

    # 1) Group POINTS by partition (keyed on the point's method).
    by_partition: dict[str, list[dict]] = {}
    for item in work_items:
        key = partition_for_method(item["method"], prefer_h100=prefer_h100)
        by_partition.setdefault(key, []).append(item)

    generated: List[GeneratedJob] = []

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
        plan_path = out_dir / f"{experiment.lower()}_{partition_key}_plan.json"
        _write_plan(plan_path, slots)

        # 5) Emit the SLURM script
        job_name = f"{experiment.lower()}_{partition_key}"
        array_range = (
            f"0-{len(slots) - 1}%{max_concurrent}" if len(slots) > 1 else "0"
        )
        # Logs live alongside the results. ABSOLUTE path (NOT ${VSC_DATA}):
        # SLURM does not expand env vars in --output/--error, so a literal
        # ${VSC_DATA} there would create a bogus directory.
        log_dir = f"{_results_dir()}/{experiment.lower()}/logs"
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


def _format_walltime(seconds: int) -> str:
    """Format seconds as ``HH:MM:SS`` or ``D-HH:MM:SS`` (SLURM-accepted)."""
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 24:
        days, hh = divmod(hours, 24)
        return f"{days}-{hh:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


_PLAN_ITEM_KEYS = ("dataset", "method", "task", "name", "tune", "row_limit", "sampling")


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
    log_dir = f"{_results_dir()}/{experiment.lower()}/logs"
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
        cd "{_REPO_ROOT}"

        export TABPFN_RESULTS_ROOT="{_results_dir()}"

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
    "pack_work_items",
    "MAX_ARRAY_SLOTS",
    "GeneratedJob",
    "generate_scripts_for_experiment",
    "load_plan",
]
