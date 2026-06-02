#!/usr/bin/env python3
"""
Setup Script for Experiment 2: Learning Curve Analysis

This script:
1. Uses method_config.py for GPU/CPU/Foundation categorization (no hardcoding)
2. Reads enabled methods and datasets from config
3. Splits GPU methods into Standard (genius) and Foundation (wICE) groups
4. Counts tasks for each hardware category
5. Generates THREE sets of BATCHED SLURM scripts:
    - Experiment2_GPU_Standard*.slurm  (genius cluster, gpu_p100)
    - Experiment2_GPU_Foundation*.slurm (wICE cluster, gpu_h100)
    - Experiment2_CPU*.slurm           (genius cluster, batch)
6. Provides instructions for submission

Key differences from Experiment1:
- Each SLURM task = ONE method + ONE dataset (no HPO variation)
- Loop over row_limits happens INSIDE Python (not as separate SLURM jobs)
- No HPO mode - all use default parameters
- Longer time limits due to multiple row_limit iterations
"""

import os
import sys
import math
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.methods.method_config import GPU_METHODS, CPU_METHODS, FOUNDATION_METHODS

# VSC-safe constants
MAX_TASKS_PER_SLURM = 400 


# ======================================================================================
#                     SLURM SCRIPT TEMPLATES (PARAMETERIZED)
# ======================================================================================

def generate_gpu_slurm_script(
    batch_id: int,
    start_task: int,
    end_task: int,
    max_concurrent: int,
    cluster: str,
    partition: str,
    memory: str,
    job_type: str,
    orchestrator_script: str
):
    """
    Generate GPU SLURM script for a batch of learning curve tasks.
    
    UPDATES:
    - Implements Soft Isolation for Foundation models (100G mem, no exclusive).
    - Adds fragmentation fix.
    """
    n_tasks = end_task - start_task

    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"

    time_limit = "48:00:00" 
    
    # --- SOFT ISOLATION STRATEGY ---
    if "Foundation" in job_type:
        cpus = 16
        gpus = 1
        # CRITICAL: Request 100G to 'dominate' the node. 
        # This prevents other large jobs from running here, acting as pseudo-isolation
        # without the long wait times of --exclusive.
        mem_flag = "#SBATCH --mem=100G" 
        exclusive_flag = "" 
    else:
        # Standard Models: Share resources efficiently
        cpus = 4
        gpus = 1
        mem_flag = f"#SBATCH --mem={memory}"
        exclusive_flag = ""

    notify_email = os.environ.get("TABPFN_SLURM_NOTIFY_EMAIL", "").strip()
    notify_block = (
        f"#SBATCH --mail-type=FAIL,TIME_LIMIT,REQUEUE\n"
        f"#SBATCH --mail-user={notify_email}\n"
    ) if notify_email else ""

    return f"""#!/bin/bash
#SBATCH --job-name=exp2_{job_type.lower()}{batch_id}
#SBATCH --cluster={cluster}
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=${{VSC_DATA}}/TabPFNCredit/results/experiment2/logs/{job_type.lower()}{batch_id}_%A_%a.out
#SBATCH --error=${{VSC_DATA}}/TabPFNCredit/results/experiment2/logs/{job_type.lower()}{batch_id}_%A_%a.err
#SBATCH --time={time_limit}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus-per-node={gpus}
{mem_flag}
#SBATCH --partition={partition}
#SBATCH --requeue
{notify_block}#SBATCH --array={array_range}
{exclusive_flag}

# ---------------------------------------------------------
# EXPERIMENT 2: {job_type.upper()} GPU - BATCH {batch_id}
# Tasks {start_task}-{end_task-1}
# Memory strategy: {mem_flag}
# ---------------------------------------------------------

set -euo pipefail

# Stagger starts to avoid I/O thundering-herd.
sleep $((${SLURM_ARRAY_TASK_ID:-0} % 30))

# Force unbuffered I/O + PyTorch alloc fragmentation mitigation.
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Activate conda env.
export PATH="${{VSC_DATA}}/miniconda3/bin:${{PATH}}"
source activate TabPFNCredit
export LD_LIBRARY_PATH="${{VSC_DATA}}/miniconda3/envs/TabPFNCredit/lib:${{LD_LIBRARY_PATH:-}}"

cd "${{VSC_DATA}}/TabPFNCredit"
mkdir -p "${{VSC_DATA}}/TabPFNCredit/results/experiment2/logs/slurm"

echo "=========================================="
echo "EXP 2 - {job_type.upper()} - BATCH {batch_id}"
echo "=========================================="
echo "Job ID:       ${{SLURM_JOB_ID}}"
echo "Array ID:     ${{SLURM_ARRAY_TASK_ID}}"
echo "Node:         ${{SLURMD_NODENAME}}"
echo "GPU:          ${{CUDA_VISIBLE_DEVICES:-N/A}}"
echo "Memory:       {'100G (soft isolation)' if 'Foundation' in job_type else memory}"
echo "=========================================="

GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + {start_task}))

python -u scripts/Experiment2/{orchestrator_script} --array_id="${{GLOBAL_TASK_ID}}" --verbose

EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: ${{EXIT_CODE}}"
echo "=========================================="
exit ${{EXIT_CODE}}
"""


def generate_cpu_slurm_script(batch_id, start_task, end_task, max_concurrent):
    """Generate CPU SLURM script for a batch of learning curve tasks."""

    n_tasks = end_task - start_task

    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"

    notify_email = os.environ.get("TABPFN_SLURM_NOTIFY_EMAIL", "").strip()
    notify_block = (
        f"#SBATCH --mail-type=FAIL,TIME_LIMIT,REQUEUE\n"
        f"#SBATCH --mail-user={notify_email}\n"
    ) if notify_email else ""

    return f"""#!/bin/bash
#SBATCH --job-name=exp2_cpu{batch_id}
#SBATCH --cluster=genius
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=${{VSC_DATA}}/TabPFNCredit/results/experiment2/logs/cpu{batch_id}_%A_%a.out
#SBATCH --error=${{VSC_DATA}}/TabPFNCredit/results/experiment2/logs/cpu{batch_id}_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=batch
#SBATCH --requeue
{notify_block}#SBATCH --array={array_range}

# ---------------------------------------------------------
# EXPERIMENT 2: LEARNING CURVE - CPU - BATCH {batch_id}
# Tasks {start_task}-{end_task-1}
# ---------------------------------------------------------

set -euo pipefail
sleep $((${SLURM_ARRAY_TASK_ID:-0} % 30))
export PYTHONUNBUFFERED=1

export PATH="${{VSC_DATA}}/miniconda3/bin:${{PATH}}"
source activate TabPFNCredit

cd "${{VSC_DATA}}/TabPFNCredit"
mkdir -p "${{VSC_DATA}}/TabPFNCredit/results/experiment2/logs/slurm"

echo "=========================================="
echo "EXPERIMENT 2 - CPU - BATCH {batch_id}"
echo "=========================================="
echo "Job ID:       ${{SLURM_JOB_ID}}"
echo "Array ID:     ${{SLURM_ARRAY_TASK_ID}}"
echo "Node:         ${{SLURMD_NODENAME}}"
echo "=========================================="

GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + {start_task}))

python -u scripts/Experiment2/Experiment2_CPU.py --array_id="${{GLOBAL_TASK_ID}}" --verbose

EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: ${{EXIT_CODE}}"
echo "=========================================="
exit ${{EXIT_CODE}}
"""


# ======================================================================================
#                     TASK COUNTING AND BATCHING
# ======================================================================================

def count_tasks_exp2(methods, datasets):
    """
    Count total tasks for Experiment 2 (Learning Curve).
    """
    return len(methods) * len(datasets)


def generate_batched_slurm_files_gpu(
    total_tasks: int,
    prefix: str,
    scripts_dir: Path,
    max_concurrent: int,
    cluster: str,
    partition: str,
    memory: str,
    job_type: str,
    orchestrator_script: str
):
    """Generate multiple GPU SLURM files."""
    if total_tasks == 0:
        return []

    n_batches = math.ceil(total_tasks / MAX_TASKS_PER_SLURM)
    generated_files = []

    for batch_id in range(n_batches):
        start_task = batch_id * MAX_TASKS_PER_SLURM
        end_task = min(start_task + MAX_TASKS_PER_SLURM, total_tasks)

        # Generate script content
        script_content = generate_gpu_slurm_script(
            batch_id=batch_id,
            start_task=start_task,
            end_task=end_task,
            max_concurrent=max_concurrent,
            cluster=cluster,
            partition=partition,
            memory=memory,
            job_type=job_type,
            orchestrator_script=orchestrator_script
        )

        # Write to file
        filename = f"{prefix}{batch_id}.slurm"
        filepath = scripts_dir / filename

        with open(filepath, 'w', newline='\n') as f:
            f.write(script_content)

        filepath.chmod(0o755)
        generated_files.append((filename, start_task, end_task))

    return generated_files


def generate_batched_slurm_files_cpu(total_tasks, prefix, scripts_dir, max_concurrent):
    """Generate multiple CPU SLURM files."""
    if total_tasks == 0:
        return []

    n_batches = math.ceil(total_tasks / MAX_TASKS_PER_SLURM)
    generated_files = []

    for batch_id in range(n_batches):
        start_task = batch_id * MAX_TASKS_PER_SLURM
        end_task = min(start_task + MAX_TASKS_PER_SLURM, total_tasks)

        # Generate script content
        script_content = generate_cpu_slurm_script(
            batch_id=batch_id,
            start_task=start_task,
            end_task=end_task,
            max_concurrent=max_concurrent
        )

        # Write to file
        filename = f"{prefix}{batch_id}.slurm"
        filepath = scripts_dir / filename

        with open(filepath, 'w', newline='\n') as f:
            f.write(script_content)

        filepath.chmod(0o755)
        generated_files.append((filename, start_task, end_task))

    return generated_files


def print_method_summary(
    standard_gpu_methods,
    foundation_gpu_methods,
    cpu_methods,
    pd_datasets,
    lgd_datasets,
    lc_config
):
    """Print summary of methods, datasets, and learning curve config."""

    print(f"\n{'='*70}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*70}")

    print(f"\nLearning Curve Parameters:")
    print(f"  row_max:  {lc_config['row_max']:,}")
    print(f"  row_min:  {lc_config['row_min']:,}")
    print(f"  row_step: {lc_config['row_step']:,}")

    # Estimate number of row limit iterations
    max_iterations = (lc_config['row_max'] - lc_config['row_min']) // lc_config['row_step'] + 1
    print(f"  Max iterations per task: ~{max_iterations}")

    print(f"\nDatasets:")
    print(f"  PD ({len(pd_datasets)}):  {', '.join(sorted(pd_datasets)[:5])}{'...' if len(pd_datasets) > 5 else ''}")
    print(f"  LGD ({len(lgd_datasets)}): {', '.join(sorted(lgd_datasets)[:5])}{'...' if len(lgd_datasets) > 5 else ''}")

    print(f"\nStandard GPU Methods ({len(standard_gpu_methods)}) - genius/gpu_p100:")
    for i, method in enumerate(sorted(standard_gpu_methods), 1):
        print(f"  {i:2d}. {method}")

    print(f"\nFoundation GPU Methods ({len(foundation_gpu_methods)}) - wICE/gpu_h100:")
    for i, method in enumerate(sorted(foundation_gpu_methods), 1):
        print(f"  {i:2d}. {method}")

    print(f"\nCPU Methods ({len(cpu_methods)}) - genius/batch:")
    for i, method in enumerate(sorted(cpu_methods), 1):
        print(f"  {i:2d}. {method}")

    print(f"{'='*70}\n")


def main():
    print(f"\n{'='*70}")
    print("EXPERIMENT 2 SETUP - LEARNING CURVE ANALYSIS")
    print(f"{'='*70}\n")

    # Load config
    config = load_config("Experiment2")
    lc_config = config['learning_curve']
    pd_datasets = list(config['datasets']['pd'].keys())
    lgd_datasets = list(config['datasets']['lgd'].keys())
    
    all_pd_methods = list(config['methods']['pd'].keys())
    all_lgd_methods = list(config['methods']['lgd'].keys())

    # ==========================================
    # SPLIT METHODS
    # ==========================================
    standard_gpu_pd_methods = [m for m in all_pd_methods if m in GPU_METHODS and m not in FOUNDATION_METHODS]
    standard_gpu_lgd_methods = [m for m in all_lgd_methods if m in GPU_METHODS and m not in FOUNDATION_METHODS]

    foundation_gpu_pd_methods = [m for m in all_pd_methods if m in GPU_METHODS and m in FOUNDATION_METHODS]
    foundation_gpu_lgd_methods = [m for m in all_lgd_methods if m in GPU_METHODS and m in FOUNDATION_METHODS]

    cpu_pd_methods = [m for m in all_pd_methods if m in CPU_METHODS]
    cpu_lgd_methods = [m for m in all_lgd_methods if m in CPU_METHODS]

    standard_gpu_methods = set(standard_gpu_pd_methods + standard_gpu_lgd_methods)
    foundation_gpu_methods = set(foundation_gpu_pd_methods + foundation_gpu_lgd_methods)
    cpu_methods = set(cpu_pd_methods + cpu_lgd_methods)

    if not standard_gpu_methods and not foundation_gpu_methods and not cpu_methods:
        print("ERROR: No methods enabled in config")
        sys.exit(1)

    print_method_summary(
        standard_gpu_methods, foundation_gpu_methods, cpu_methods,
        pd_datasets, lgd_datasets, lc_config
    )

    # ==========================================
    # COUNT TASKS
    # ==========================================
    n_standard_gpu_tasks = (
        count_tasks_exp2(standard_gpu_pd_methods, pd_datasets) +
        count_tasks_exp2(standard_gpu_lgd_methods, lgd_datasets)
    )

    n_foundation_gpu_tasks = (
        count_tasks_exp2(foundation_gpu_pd_methods, pd_datasets) +
        count_tasks_exp2(foundation_gpu_lgd_methods, lgd_datasets)
    )

    n_cpu_tasks = (
        count_tasks_exp2(cpu_pd_methods, pd_datasets) +
        count_tasks_exp2(cpu_lgd_methods, lgd_datasets)
    )

    max_standard_concurrent = min(12, n_standard_gpu_tasks) if n_standard_gpu_tasks > 0 else 1
    max_foundation_concurrent = min(8, n_foundation_gpu_tasks) if n_foundation_gpu_tasks > 0 else 1
    max_cpu_concurrent = min(24, n_cpu_tasks) if n_cpu_tasks > 0 else 1

    print(f"{'='*70}")
    print("TASK COUNTS")
    print(f"{'='*70}")
    print(f"Standard GPU tasks:   {n_standard_gpu_tasks:4d}")
    print(f"Foundation GPU tasks: {n_foundation_gpu_tasks:4d}")
    print(f"CPU tasks:            {n_cpu_tasks:4d}")
    print(f"{'='*70}\n")

    # ==========================================
    # GENERATE SCRIPTS
    # ==========================================
    slurm_log_dir = PROJECT_ROOT / "results" / "experiment2" / "logs" / "slurm"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = PROJECT_ROOT / "scripts" / "Experiment2"
    scripts_dir.mkdir(exist_ok=True)

    print("Generating SLURM scripts...")

    # Standard GPU
    standard_gpu_files = generate_batched_slurm_files_gpu(
        total_tasks=n_standard_gpu_tasks,
        prefix="Experiment2_GPU_Standard",
        scripts_dir=scripts_dir,
        max_concurrent=max_standard_concurrent,
        cluster="genius",
        partition="gpu_p100",
        memory="45G",
        job_type="Standard",
        orchestrator_script="Experiment2_GPU_Standard.py"
    )

    # Foundation GPU (Uses High Mem / Soft Isolation)
    foundation_gpu_files = generate_batched_slurm_files_gpu(
        total_tasks=n_foundation_gpu_tasks,
        prefix="Experiment2_GPU_Foundation",
        scripts_dir=scripts_dir,
        max_concurrent=max_foundation_concurrent,
        cluster="wice",
        partition="gpu_h100",
        memory="64G", # Ignored in function logic for Foundation, overwritten by 100G
        job_type="Foundation",
        orchestrator_script="Experiment2_GPU_Foundation.py"
    )

    # CPU
    cpu_files = generate_batched_slurm_files_cpu(
        total_tasks=n_cpu_tasks,
        prefix="Experiment2_CPU",
        scripts_dir=scripts_dir,
        max_concurrent=max_cpu_concurrent
    )

    # ==========================================
    # INSTRUCTIONS
    # ==========================================
    print(f"\n{'='*70}")
    print("SETUP COMPLETE")
    print(f"{'='*70}\n")
    print("Next Steps:\n")

    step_num = 1
    if standard_gpu_files:
        print(f"  {step_num}. Submit Standard GPU jobs (genius cluster):")
        for filename, _, _ in standard_gpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        step_num += 1

    if foundation_gpu_files:
        print(f"\n  {step_num}. Submit Foundation GPU jobs (wICE cluster):")
        for filename, _, _ in foundation_gpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        step_num += 1

    if cpu_files:
        print(f"\n  {step_num}. Submit CPU jobs (genius cluster):")
        for filename, _, _ in cpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        step_num += 1

    print(f"\n{step_num}. Check results:")
    print(f"     ls results/experiment2/pd/*.pkl | wc -l")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()