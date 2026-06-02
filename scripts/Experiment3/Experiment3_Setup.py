#!/usr/bin/env python3
"""
Setup Script for Experiment 3: Class Imbalance Analysis

This script:
1. Uses method_config.py for GPU/CPU/Foundation categorization (no hardcoding)
2. Reads enabled methods and datasets from config
3. Splits GPU methods into Standard (genius) and Foundation (wICE) groups
4. Counts tasks for each hardware category (PD only - no LGD)
5. Generates THREE sets of BATCHED SLURM scripts:
   - Experiment3_GPU_Standard*.slurm  (genius cluster, gpu_p100)
   - Experiment3_GPU_Foundation*.slurm (wICE cluster, gpu_a100)
   - Experiment3_CPU*.slurm           (genius cluster, batch)
6. Provides instructions for submission

Key differences from Experiment2:
- Only PD (classification) tasks - LGD not applicable for imbalance analysis
- Loop over minority_proportion happens INSIDE Python (not as separate SLURM jobs)
- No HPO mode - all use default parameters
- All jobs use 72:00:00 time limit (VSC maximum)
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
MAX_TASKS_PER_SLURM = 400  # Safe buffer below 500 limit
MAX_WALLTIME = "72:00:00"  # VSC maximum walltime


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
    Generate GPU SLURM script for a batch of class imbalance analysis tasks.

    UPDATES:
    - Implements Soft Isolation for Foundation models (100G mem, no exclusive).
    - Adds fragmentation fix.

    Args:
        batch_id: Batch number for this SLURM file
        start_task: Starting global task ID
        end_task: Ending global task ID (exclusive)
        max_concurrent: Maximum concurrent array jobs
        cluster: SLURM cluster name ("genius" or "wice")
        partition: SLURM partition ("gpu_p100" or "gpu_a100")
        memory: Memory allocation (e.g., "45G" or "64G")
        job_type: Label for the job ("Standard" or "Foundation")
        orchestrator_script: Python script to run (e.g., "Experiment3_GPU_Standard.py")
    """
    n_tasks = end_task - start_task

    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"

    # All jobs use 72h time limit (VSC maximum)
    time_limit = MAX_WALLTIME

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
#SBATCH --job-name=exp3_{job_type.lower()}{batch_id}
#SBATCH --cluster={cluster}
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=${{VSC_DATA}}/TabPFNCredit/results/experiment3/logs/{job_type.lower()}{batch_id}_%A_%a.out
#SBATCH --error=${{VSC_DATA}}/TabPFNCredit/results/experiment3/logs/{job_type.lower()}{batch_id}_%A_%a.err
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
# EXPERIMENT 3: IMBALANCE - {job_type.upper()} GPU - BATCH {batch_id}
# Tasks {start_task}-{end_task-1}
# Memory strategy: {mem_flag}
# ---------------------------------------------------------

set -euo pipefail
sleep $((${SLURM_ARRAY_TASK_ID:-0} % 30))

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export PATH="${{VSC_DATA}}/miniconda3/bin:${{PATH}}"
source activate TabPFNCredit
export LD_LIBRARY_PATH="${{VSC_DATA}}/miniconda3/envs/TabPFNCredit/lib:${{LD_LIBRARY_PATH:-}}"

cd "${{VSC_DATA}}/TabPFNCredit"
mkdir -p "${{VSC_DATA}}/TabPFNCredit/results/experiment3/logs/slurm"

echo "=========================================="
echo "EXP 3 - {job_type.upper()} - BATCH {batch_id}"
echo "=========================================="
echo "Job ID:       ${{SLURM_JOB_ID}}"
echo "Array ID:     ${{SLURM_ARRAY_TASK_ID}}"
echo "Node:         ${{SLURMD_NODENAME}}"
echo "GPU:          ${{CUDA_VISIBLE_DEVICES:-N/A}}"
echo "Memory:       {'100G (soft isolation)' if 'Foundation' in job_type else memory}"
echo "=========================================="

GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + {start_task}))

python -u scripts/Experiment3/{orchestrator_script} --array_id="${{GLOBAL_TASK_ID}}" --verbose

EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: ${{EXIT_CODE}}"
echo "=========================================="
exit ${{EXIT_CODE}}
"""


def generate_cpu_slurm_script(batch_id, start_task, end_task, max_concurrent):
    """Generate CPU SLURM script for a batch of class imbalance analysis tasks."""

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
#SBATCH --job-name=exp3_cpu{batch_id}
#SBATCH --cluster=genius
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=${{VSC_DATA}}/TabPFNCredit/results/experiment3/logs/cpu{batch_id}_%A_%a.out
#SBATCH --error=${{VSC_DATA}}/TabPFNCredit/results/experiment3/logs/cpu{batch_id}_%A_%a.err
#SBATCH --time={MAX_WALLTIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=batch
#SBATCH --requeue
{notify_block}#SBATCH --array={array_range}

# ---------------------------------------------------------
# EXPERIMENT 3: IMBALANCE - CPU - BATCH {batch_id}
# Tasks {start_task}-{end_task-1}
# ---------------------------------------------------------

set -euo pipefail
sleep $((${SLURM_ARRAY_TASK_ID:-0} % 30))
export PYTHONUNBUFFERED=1

export PATH="${{VSC_DATA}}/miniconda3/bin:${{PATH}}"
source activate TabPFNCredit

cd "${{VSC_DATA}}/TabPFNCredit"
mkdir -p "${{VSC_DATA}}/TabPFNCredit/results/experiment3/logs/slurm"

echo "=========================================="
echo "EXPERIMENT 3 - CPU - BATCH {batch_id}"
echo "=========================================="
echo "Job ID:       ${{SLURM_JOB_ID}}"
echo "Array ID:     ${{SLURM_ARRAY_TASK_ID}}"
echo "Node:         ${{SLURMD_NODENAME}}"
echo "=========================================="

GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + {start_task}))

python -u scripts/Experiment3/Experiment3_CPU.py --array_id="${{GLOBAL_TASK_ID}}" --verbose

EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: ${{EXIT_CODE}}"
echo "=========================================="
exit ${{EXIT_CODE}}
"""


# ======================================================================================
#                     TASK COUNTING AND BATCHING
# ======================================================================================

def count_tasks_exp3(methods, datasets):
    """
    Count total tasks for Experiment 3 (Class Imbalance Analysis).

    Each task = ONE method + ONE dataset (no HPO variation).
    Only PD tasks - LGD not applicable for imbalance analysis.
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
    """
    Generate multiple GPU SLURM files with parameterized hardware settings.

    Args:
        total_tasks: Total number of tasks
        prefix: Filename prefix (e.g., 'Experiment3_GPU_Standard')
        scripts_dir: Directory to save scripts
        max_concurrent: Max concurrent jobs
        cluster: SLURM cluster
        partition: SLURM partition
        memory: Memory allocation
        job_type: Job type label
        orchestrator_script: Python script to run

    Returns:
        List of (filename, start_task, end_task) tuples
    """
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
    """
    Generate multiple CPU SLURM files, each with max 400 tasks.

    Returns:
        List of (filename, start_task, end_task) tuples
    """
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
    imbalance_config
):
    """Print summary of methods, datasets, and imbalance config."""

    print(f"\n{'='*70}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*70}")

    print(f"\nClass Imbalance Parameters:")
    print(f"  minority_proportion_max:  {imbalance_config['minority_proportion_max']}")
    print(f"  minority_proportion_min:  {imbalance_config['minority_proportion_min']}")
    print(f"  minority_proportion_step: {imbalance_config['minority_proportion_step']}")

    # Estimate number of proportion iterations
    prop_range = imbalance_config['minority_proportion_max'] - imbalance_config['minority_proportion_min']
    max_iterations = int(prop_range / imbalance_config['minority_proportion_step']) + 1
    print(f"  Max iterations per task: ~{max_iterations}")

    print(f"\nDatasets (PD only - class imbalance applies to classification):")
    print(f"  PD ({len(pd_datasets)}):  {', '.join(sorted(pd_datasets)[:5])}{'...' if len(pd_datasets) > 5 else ''}")

    print(f"\nStandard GPU Methods ({len(standard_gpu_methods)}) - genius/gpu_p100:")
    for i, method in enumerate(sorted(standard_gpu_methods), 1):
        print(f"  {i:2d}. {method}")

    print(f"\nFoundation GPU Methods ({len(foundation_gpu_methods)}) - wICE/gpu_a100:")
    for i, method in enumerate(sorted(foundation_gpu_methods), 1):
        print(f"  {i:2d}. {method}")

    print(f"\nCPU Methods ({len(cpu_methods)}) - genius/batch:")
    for i, method in enumerate(sorted(cpu_methods), 1):
        print(f"  {i:2d}. {method}")

    print(f"{'='*70}\n")


def main():
    experiment_name = "experiment3"

    print(f"\n{'='*70}")
    print("EXPERIMENT 3 SETUP - CLASS IMBALANCE ANALYSIS")
    print(f"{'='*70}\n")

    # Load config
    config = load_config("Experiment3")

    # Get imbalance parameters
    imbalance_config = config['imbalance']

    # Get enabled datasets (PD only for class imbalance)
    pd_datasets = list(config['datasets']['pd'].keys())

    if not pd_datasets:
        print("ERROR: No PD datasets enabled in config")
        print("NOTE: Experiment 3 (Class Imbalance Analysis) only applies to PD (classification) tasks.")
        sys.exit(1)

    # Get enabled methods from config (PD only)
    all_pd_methods = list(config['methods']['pd'].keys())

    # ==========================================
    # SPLIT METHODS INTO THREE CATEGORIES
    # ==========================================

    # Standard GPU = GPU methods that are NOT foundation methods
    standard_gpu_pd_methods = [m for m in all_pd_methods if m in GPU_METHODS and m not in FOUNDATION_METHODS]

    # Foundation GPU = GPU methods that ARE foundation methods
    foundation_gpu_pd_methods = [m for m in all_pd_methods if m in GPU_METHODS and m in FOUNDATION_METHODS]

    # CPU methods
    cpu_pd_methods = [m for m in all_pd_methods if m in CPU_METHODS]

    # Get unique method names for summary
    standard_gpu_methods = set(standard_gpu_pd_methods)
    foundation_gpu_methods = set(foundation_gpu_pd_methods)
    cpu_methods = set(cpu_pd_methods)

    if not standard_gpu_methods and not foundation_gpu_methods and not cpu_methods:
        print("ERROR: No methods enabled in config")
        sys.exit(1)

    # Print summary
    print_method_summary(
        standard_gpu_methods, foundation_gpu_methods, cpu_methods,
        pd_datasets, imbalance_config
    )

    # ==========================================
    # COUNT TASKS FOR EACH CATEGORY
    # ==========================================

    n_standard_gpu_tasks = count_tasks_exp3(standard_gpu_pd_methods, pd_datasets)
    n_foundation_gpu_tasks = count_tasks_exp3(foundation_gpu_pd_methods, pd_datasets)
    n_cpu_tasks = count_tasks_exp3(cpu_pd_methods, pd_datasets)

    # Determine concurrency (fewer concurrent jobs due to longer runtimes)
    max_standard_concurrent = min(12, n_standard_gpu_tasks) if n_standard_gpu_tasks > 0 else 1
    max_foundation_concurrent = min(8, n_foundation_gpu_tasks) if n_foundation_gpu_tasks > 0 else 1
    max_cpu_concurrent = min(24, n_cpu_tasks) if n_cpu_tasks > 0 else 1

    print(f"{'='*70}")
    print("TASK COUNTS")
    print(f"{'='*70}")
    print(f"Standard GPU tasks (genius/gpu_p100):    {n_standard_gpu_tasks:4d}")
    print(f"Foundation GPU tasks (wICE/gpu_a100):    {n_foundation_gpu_tasks:4d}")
    print(f"CPU tasks (genius/batch):                {n_cpu_tasks:4d}")
    print(f"{'='*70}")
    print(f"Total:                                   {n_standard_gpu_tasks + n_foundation_gpu_tasks + n_cpu_tasks:4d}")
    print(f"\nNote: Each task runs an imbalance curve (multiple minority_proportion values)")
    print(f"      No HPO variation - all tasks use default parameters")
    print(f"      Only PD (classification) tasks - LGD not applicable")
    print(f"\nBatching strategy: {MAX_TASKS_PER_SLURM} tasks per SLURM file (VSC limit workaround)")
    print(f"Time limit: {MAX_WALLTIME} (72 hours - VSC maximum)")
    print(f"{'='*70}\n")

    # Create SLURM log directories
    slurm_log_dir = PROJECT_ROOT / "results" / "experiment3" / "logs" / "slurm"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    # Generate SLURM scripts
    print("Generating SLURM scripts...")
    scripts_dir = PROJECT_ROOT / "scripts" / "Experiment3"
    scripts_dir.mkdir(exist_ok=True)

    # ==========================================
    # GENERATE STANDARD GPU SCRIPTS (genius/gpu_p100)
    # ==========================================
    standard_gpu_files = generate_batched_slurm_files_gpu(
        total_tasks=n_standard_gpu_tasks,
        prefix="Experiment3_GPU_Standard",
        scripts_dir=scripts_dir,
        max_concurrent=max_standard_concurrent,
        cluster="genius",
        partition="gpu_p100",
        memory="45G",
        job_type="Standard",
        orchestrator_script="Experiment3_GPU_Standard.py"
    )

    if standard_gpu_files:
        print(f"\nGenerated {len(standard_gpu_files)} Standard GPU batch script(s) [genius/gpu_p100]:")
        for filename, start, end in standard_gpu_files:
            print(f"  - {filename} (tasks {start}-{end-1})")

    # ==========================================
    # GENERATE FOUNDATION GPU SCRIPTS (wICE/gpu_a100)
    # Uses Soft Isolation: 100G memory request (no --exclusive)
    # ==========================================
    foundation_gpu_files = generate_batched_slurm_files_gpu(
        total_tasks=n_foundation_gpu_tasks,
        prefix="Experiment3_GPU_Foundation",
        scripts_dir=scripts_dir,
        max_concurrent=max_foundation_concurrent,
        cluster="wice",
        partition="gpu_a100",
        memory="64G",  # Ignored in function logic for Foundation, overwritten by 100G
        job_type="Foundation",
        orchestrator_script="Experiment3_GPU_Foundation.py"
    )

    if foundation_gpu_files:
        print(f"\nGenerated {len(foundation_gpu_files)} Foundation GPU batch script(s) [wICE/gpu_a100]:")
        for filename, start, end in foundation_gpu_files:
            print(f"  - {filename} (tasks {start}-{end-1})")

    # ==========================================
    # GENERATE CPU SCRIPTS (genius/batch)
    # ==========================================
    cpu_files = generate_batched_slurm_files_cpu(
        total_tasks=n_cpu_tasks,
        prefix="Experiment3_CPU",
        scripts_dir=scripts_dir,
        max_concurrent=max_cpu_concurrent
    )

    if cpu_files:
        print(f"\nGenerated {len(cpu_files)} CPU batch script(s) [genius/batch]:")
        for filename, start, end in cpu_files:
            print(f"  - {filename} (tasks {start}-{end-1})")

    # ==========================================
    # PRINT SUBMISSION INSTRUCTIONS
    # ==========================================
    print(f"\n{'='*70}")
    print("SETUP COMPLETE")
    print(f"{'='*70}\n")

    print("Next Steps:\n")

    step_num = 1

    # Standard GPU submission instructions
    if standard_gpu_files:
        print(f"  {step_num}. Submit Standard GPU jobs (genius cluster):\n")
        for filename, _, _ in standard_gpu_files:
            print(f"     sbatch scripts/Experiment3/{filename}")
        print()
        step_num += 1

    # Foundation GPU submission instructions
    if foundation_gpu_files:
        print(f"  {step_num}. Submit Foundation GPU jobs (wICE cluster):\n")
        for filename, _, _ in foundation_gpu_files:
            print(f"     sbatch scripts/Experiment3/{filename}")
        print()
        step_num += 1

    # CPU submission instructions
    if cpu_files:
        print(f"  {step_num}. Submit CPU jobs (genius cluster):\n")
        for filename, _, _ in cpu_files:
            print(f"     sbatch scripts/Experiment3/{filename}")
        print()
        step_num += 1

    # Submission script suggestion
    all_files = standard_gpu_files + foundation_gpu_files + cpu_files
    if len(all_files) > 2:
        print(f"  TIP: Create a submission script:\n")
        print(f"     cat > submit_all_exp3.sh << 'EOF'")
        print(f"     #!/bin/bash")
        print(f"     # Standard GPU (genius cluster)")
        for filename, _, _ in standard_gpu_files:
            print(f"     sbatch scripts/Experiment3/{filename}")
        print(f"     # Foundation GPU (wICE cluster)")
        for filename, _, _ in foundation_gpu_files:
            print(f"     sbatch scripts/Experiment3/{filename}")
        print(f"     # CPU (genius cluster)")
        for filename, _, _ in cpu_files:
            print(f"     sbatch scripts/Experiment3/{filename}")
        print(f"     EOF")
        print(f"     chmod +x submit_all_exp3.sh")
        print(f"     ./submit_all_exp3.sh\n")

    print(f"  {step_num}. Monitor:")
    print(f"     squeue -u $USER\n")
    step_num += 1

    print(f"  {step_num}. Check results:")
    print(f"     ls results/experiment3/pd/*.pkl | wc -l\n")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
