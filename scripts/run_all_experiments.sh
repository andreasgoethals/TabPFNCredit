#!/bin/bash -l
# ============================================================================
# run_all_experiments.sh -- chain every TabPFNCredit experiment on the VSC
# ============================================================================
#
# This is the single script you `scp` to the VSC (or check out via git) and
# invoke as the orchestrator for a full benchmark sweep. It is *idempotent*:
# experiments that are already complete skip their cells, so re-running this
# after a partial failure is safe.
#
# Default behaviour (no arguments) -- run all four experiments in sequence,
# each waiting on the previous via SLURM ``--dependency=afterok``:
#
#     bash scripts/run_all_experiments.sh
#
# Run a subset (one or more experiment names). Order matters -- each
# experiment chains after the previous one in the list:
#
#     bash scripts/run_all_experiments.sh Experiment0
#     bash scripts/run_all_experiments.sh Experiment0 Experiment1
#     bash scripts/run_all_experiments.sh Experiment2 Experiment3
#
# Under the hood
# --------------
# For each requested experiment, the script invokes
# ``tabpfncredit experiment <NAME>`` which:
#   1. Reads the experiment's YAML configs.
#   2. Auto-preprocesses any dataset that is missing under data/processed.
#   3. Wipes scripts/<Experiment>/_generated/ to drop stale SLURM scripts.
#   4. Regenerates fresh SLURM scripts sized to each VSC partition.
#   5. Submits every per-partition array via ``sbatch`` (with --dependency
#      pointing at the previous experiment's final summarize job).
#   6. Submits a one-shot summarize SLURM job that runs once all arrays
#      finish (``--dependency=afterok:<arrays>``); its job ID becomes the
#      "previous job" for the next experiment in the chain.
#
# Each experiment's methods come from its own
# ``scripts/<Experiment>/config/CONFIG_METHOD.yaml`` -- they are NOT
# auto-derived from Experiment 0's outcomes. After Experiment 0 finishes,
# inspect ``results/experiment0/summaries/`` and edit
# ``scripts/Experiment1/config/CONFIG_METHOD.yaml`` to enable only the
# methods that actually worked.
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# Land in the repo root regardless of where the user invoked us from.
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ----------------------------------------------------------------------------
# Make sure the `tabpfncredit` console script is on PATH.
#
# If you ran this in a fresh shell that hasn't activated the project venv,
# we set it up for you: on an HPC cluster we load the Python toolchain
# module (the venv's interpreter links against it), then activate the
# venv at the repo root. Override the module names by exporting
# TABPFN_CLUSTER_MODULE / TABPFN_PYTHON_MODULE before running. If the venv
# is already active (tabpfncredit found) we touch nothing.
# ----------------------------------------------------------------------------
if ! command -v tabpfncredit >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        # Plain `module purge` (NOT --force) keeps the sticky cluster module.
        module purge 2>/dev/null || true
        PYMOD="${TABPFN_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
        # Try loading Python directly (works when a cluster module is active,
        # which is the norm on a login node). If it fails, the cluster module
        # was force-purged at some point -- restore the login cluster module
        # then retry.
        if ! module load "${PYMOD}" 2>/dev/null; then
            module load "${TABPFN_CLUSTER_MODULE:-cluster/genius/login}" 2>/dev/null || true
            module load "${PYMOD}" 2>/dev/null || true
        fi
    fi
    if [ -f "${REPO_ROOT}/tabpfncreditvenv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${REPO_ROOT}/tabpfncreditvenv/bin/activate"
    fi
fi

if ! command -v tabpfncredit >/dev/null 2>&1; then
    echo "ERROR: 'tabpfncredit' is not on PATH and could not be auto-activated." >&2
    echo "Activate your environment first, e.g.:" >&2
    echo "    source ${REPO_ROOT}/tabpfncreditvenv/bin/activate" >&2
    echo "(and run 'pip install -e \".[hpc]\"' if you haven't installed yet)." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Default order: 0 -> 1 -> 2 -> 3. Override by passing experiments as args.
# ----------------------------------------------------------------------------
if [ "$#" -eq 0 ]; then
    EXPERIMENTS=(Experiment0 Experiment1 Experiment2 Experiment3)
else
    EXPERIMENTS=("$@")
fi

# ----------------------------------------------------------------------------
# tabpfncredit's `experiment` command auto-detects "am I on VSC?" and runs
# sbatch when on VSC. Each invocation prints "Final job id (summarize): N"
# so we can pull that out and pass to --after on the next call.
# ----------------------------------------------------------------------------
PREV_JOB=""
for EXP in "${EXPERIMENTS[@]}"; do
    echo "============================================================"
    echo "  Submitting ${EXP}"
    if [ -n "${PREV_JOB}" ]; then
        echo "  Depends on previous experiment's summarize job: ${PREV_JOB}"
    fi
    echo "============================================================"

    if [ -n "${PREV_JOB}" ]; then
        OUTPUT=$(tabpfncredit experiment "${EXP}" --after "${PREV_JOB}")
    else
        OUTPUT=$(tabpfncredit experiment "${EXP}")
    fi
    echo "${OUTPUT}"

    # Extract the final summarize job id printed by the CLI.
    # Line looks like:  Final job id (summarize): 12345678
    NEXT_JOB=$(echo "${OUTPUT}" | awk '/Final job id \(summarize\):/ {print $NF}' | tail -n1)
    if [ -z "${NEXT_JOB}" ]; then
        echo "WARNING: could not extract a summarize job ID for ${EXP};" \
             "later experiments in the chain will not block on it."
        PREV_JOB=""
    else
        PREV_JOB="${NEXT_JOB}"
    fi
done

echo
echo "All experiments submitted. Final summarize job id: ${PREV_JOB:-<none>}"
echo "Monitor with: squeue -u \$USER  or  sacct -j <jobid>"
