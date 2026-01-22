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

    Args:
        batch_id: Batch number for this SLURM file
        start_task: Starting global task ID
        end_task: Ending global task ID (exclusive)
        max_concurrent: Maximum concurrent array jobs
        cluster: SLURM cluster name ("genius" or "wice")
        partition: SLURM partition ("gpu_p100" or "gpu_h100")
        memory: Memory allocation (e.g., "45G" or "64G")
        job_type: Label for the job ("Standard" or "Foundation")
        orchestrator_script: Python script to run (e.g., "Experiment2_GPU.py")
    """
    n_tasks = end_task - start_task

    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"

    # Use longer time for foundation models (they're slower)
    time_limit = "96:00:00" if "Foundation" in job_type else "72:00:00"
    cpus = 8 if "Foundation" in job_type else 4

    return f"""#!/bin/bash
#SBATCH --job-name=exp2_{job_type.lower()}{batch_id}
#SBATCH --cluster={cluster}
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment2/logs/slurm/{job_type.lower()}{batch_id}_%A_%a.out
#SBATCH --error=results/experiment2/logs/slurm/{job_type.lower()}{batch_id}_%A_%a.err
#SBATCH --time={time_limit}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus-per-node=1
#SBATCH --mem={memory}
#SBATCH --partition={partition}
#SBATCH --array={array_range}

# ---------------------------------------------------------
# EXPERIMENT 2: LEARNING CURVE ANALYSIS - {job_type.upper()} GPU
# BATCH {batch_id}: Tasks {start_task}-{end_task-1}
# ---------------------------------------------------------
# Cluster: {cluster} | Partition: {partition} | Memory: {memory}
# Each task runs a learning curve for ONE method + ONE dataset
# Loop over row_limits happens inside Python
# ---------------------------------------------------------
# STAGGERED START TO PREVENT I/O CONGESTION
# ---------------------------------------------------------
sleep $((RANDOM % 60 + 1))
# ---------------------------------------------------------

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup conda from $VSC_DATA
export PATH="${{VSC_DATA}}/miniconda3/bin:${{PATH}}"
source activate TabPFNCredit

# USE CONDA'S C++ LIBRARIES
export LD_LIBRARY_PATH="${{VSC_DATA}}/miniconda3/envs/TabPFNCredit/lib:${{LD_LIBRARY_PATH}}"

# Navigate to project
cd $VSC_DATA/TabPFNCredit

echo "=========================================="
echo "EXPERIMENT 2 - LEARNING CURVE - {job_type.upper()} GPU - BATCH {batch_id}"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Batch:        {batch_id}"
echo "Task offset:  {start_task}"
echo "Cluster:      {cluster}"
echo "Partition:    {partition}"
echo "Memory:       {memory}"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Calculate global task ID from batch offset
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + {start_task}))

# Run GPU orchestrator with global task ID
python -u scripts/Experiment2/{orchestrator_script} --array_id=$GLOBAL_TASK_ID --verbose

# Exit status
EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
"""


def generate_cpu_slurm_script(batch_id, start_task, end_task, max_concurrent):
    """Generate CPU SLURM script for a batch of learning curve tasks."""

    n_tasks = end_task - start_task

    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"

    return f"""#!/bin/bash
#SBATCH --job-name=exp2_cpu{batch_id}
#SBATCH --cluster=genius
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --output=results/experiment2/logs/slurm/cpu{batch_id}_%A_%a.out
#SBATCH --error=results/experiment2/logs/slurm/cpu{batch_id}_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=batch
#SBATCH --array={array_range}

# ---------------------------------------------------------
# EXPERIMENT 2: LEARNING CURVE ANALYSIS - CPU
# BATCH {batch_id}: Tasks {start_task}-{end_task-1}
# ---------------------------------------------------------
# Cluster: genius | Partition: batch | Memory: 40G
# Each task runs a learning curve for ONE method + ONE dataset
# Loop over row_limits happens inside Python
# ---------------------------------------------------------
# STAGGERED START TO PREVENT I/O CONGESTION
# ---------------------------------------------------------
sleep $((RANDOM % 60 + 1))
# ---------------------------------------------------------

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup conda from $VSC_DATA
export PATH="${{VSC_DATA}}/miniconda3/bin:${{PATH}}"
source activate TabPFNCredit

# Navigate to project
cd $VSC_DATA/TabPFNCredit

echo "=========================================="
echo "EXPERIMENT 2 - LEARNING CURVE - CPU - BATCH {batch_id}"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Batch:        {batch_id}"
echo "Task offset:  {start_task}"
echo "Node:         $SLURMD_NODENAME"
echo "=========================================="

# Calculate global task ID from batch offset
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + {start_task}))

# Run CPU orchestrator with global task ID
python -u scripts/Experiment2/Experiment2_CPU.py --array_id=$GLOBAL_TASK_ID --verbose

# Exit status
EXIT_CODE=$?
echo "=========================================="
echo "Task completed with exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
"""


# ======================================================================================
#                     TASK COUNTING AND BATCHING
# ======================================================================================

def count_tasks_exp2(methods, datasets):
    """
    Count total tasks for Experiment 2 (Learning Curve).

    Each task = ONE method + ONE dataset (no HPO variation).
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
        prefix: Filename prefix (e.g., 'Experiment2_GPU_Standard')
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

    # Estimate number of row limit iterations for a typical large dataset
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
    experiment_name = "experiment2"

    print(f"\n{'='*70}")
    print("EXPERIMENT 2 SETUP - LEARNING CURVE ANALYSIS")
    print(f"{'='*70}\n")

    # Load config
    config = load_config("Experiment2")

    # Get learning curve parameters
    lc_config = config['learning_curve']

    # Get enabled datasets
    pd_datasets = list(config['datasets']['pd'].keys())
    lgd_datasets = list(config['datasets']['lgd'].keys())
    all_datasets = pd_datasets + lgd_datasets

    if not all_datasets:
        print("ERROR: No datasets enabled in config")
        sys.exit(1)

    # Get enabled methods from config
    all_pd_methods = list(config['methods']['pd'].keys())
    all_lgd_methods = list(config['methods']['lgd'].keys())

    # ==========================================
    # SPLIT METHODS INTO THREE CATEGORIES
    # ==========================================

    # Standard GPU = GPU methods that are NOT foundation methods
    standard_gpu_pd_methods = [m for m in all_pd_methods if m in GPU_METHODS and m not in FOUNDATION_METHODS]
    standard_gpu_lgd_methods = [m for m in all_lgd_methods if m in GPU_METHODS and m not in FOUNDATION_METHODS]

    # Foundation GPU = GPU methods that ARE foundation methods
    foundation_gpu_pd_methods = [m for m in all_pd_methods if m in GPU_METHODS and m in FOUNDATION_METHODS]
    foundation_gpu_lgd_methods = [m for m in all_lgd_methods if m in GPU_METHODS and m in FOUNDATION_METHODS]

    # CPU methods
    cpu_pd_methods = [m for m in all_pd_methods if m in CPU_METHODS]
    cpu_lgd_methods = [m for m in all_lgd_methods if m in CPU_METHODS]

    # Get unique method names for summary
    standard_gpu_methods = set(standard_gpu_pd_methods + standard_gpu_lgd_methods)
    foundation_gpu_methods = set(foundation_gpu_pd_methods + foundation_gpu_lgd_methods)
    cpu_methods = set(cpu_pd_methods + cpu_lgd_methods)

    if not standard_gpu_methods and not foundation_gpu_methods and not cpu_methods:
        print("ERROR: No methods enabled in config")
        sys.exit(1)

    # Print summary
    print_method_summary(
        standard_gpu_methods, foundation_gpu_methods, cpu_methods,
        pd_datasets, lgd_datasets, lc_config
    )

    # ==========================================
    # COUNT TASKS FOR EACH CATEGORY
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

    # Determine concurrency (fewer concurrent jobs due to longer runtimes)
    max_standard_concurrent = min(12, n_standard_gpu_tasks) if n_standard_gpu_tasks > 0 else 1
    max_foundation_concurrent = min(8, n_foundation_gpu_tasks) if n_foundation_gpu_tasks > 0 else 1
    max_cpu_concurrent = min(24, n_cpu_tasks) if n_cpu_tasks > 0 else 1

    print(f"{'='*70}")
    print("TASK COUNTS")
    print(f"{'='*70}")
    print(f"Standard GPU tasks (genius/gpu_p100):    {n_standard_gpu_tasks:4d}")
    print(f"Foundation GPU tasks (wICE/gpu_h100):    {n_foundation_gpu_tasks:4d}")
    print(f"CPU tasks (genius/batch):                {n_cpu_tasks:4d}")
    print(f"{'='*70}")
    print(f"Total:                                   {n_standard_gpu_tasks + n_foundation_gpu_tasks + n_cpu_tasks:4d}")
    print(f"\nNote: Each task runs a learning curve (multiple row_limits)")
    print(f"      No HPO variation - all tasks use default parameters")
    print(f"\nBatching strategy: {MAX_TASKS_PER_SLURM} tasks per SLURM file (VSC limit workaround)")
    print(f"{'='*70}\n")

    # Create SLURM log directories
    slurm_log_dir = PROJECT_ROOT / "results" / "experiment2" / "logs" / "slurm"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    # Generate SLURM scripts
    print("Generating SLURM scripts...")
    scripts_dir = PROJECT_ROOT / "scripts" / "Experiment2"
    scripts_dir.mkdir(exist_ok=True)

    # ==========================================
    # GENERATE STANDARD GPU SCRIPTS (genius/gpu_p100)
    # ==========================================
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

    if standard_gpu_files:
        print(f"\nGenerated {len(standard_gpu_files)} Standard GPU batch script(s) [genius/gpu_p100]:")
        for filename, start, end in standard_gpu_files:
            print(f"  - {filename} (tasks {start}-{end-1})")

    # ==========================================
    # GENERATE FOUNDATION GPU SCRIPTS (wICE/gpu_h100)
    # ==========================================
    foundation_gpu_files = generate_batched_slurm_files_gpu(
        total_tasks=n_foundation_gpu_tasks,
        prefix="Experiment2_GPU_Foundation",
        scripts_dir=scripts_dir,
        max_concurrent=max_foundation_concurrent,
        cluster="wice",
        partition="gpu_h100",
        memory="64G",
        job_type="Foundation",
        orchestrator_script="Experiment2_GPU_Foundation.py"
    )

    if foundation_gpu_files:
        print(f"\nGenerated {len(foundation_gpu_files)} Foundation GPU batch script(s) [wICE/gpu_h100]:")
        for filename, start, end in foundation_gpu_files:
            print(f"  - {filename} (tasks {start}-{end-1})")

    # ==========================================
    # GENERATE CPU SCRIPTS (genius/batch)
    # ==========================================
    cpu_files = generate_batched_slurm_files_cpu(
        total_tasks=n_cpu_tasks,
        prefix="Experiment2_CPU",
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
            print(f"     sbatch scripts/Experiment2/{filename}")
        print()
        step_num += 1

    # Foundation GPU submission instructions
    if foundation_gpu_files:
        print(f"  {step_num}. Submit Foundation GPU jobs (wICE cluster):\n")
        for filename, _, _ in foundation_gpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        print()
        step_num += 1

    # CPU submission instructions
    if cpu_files:
        print(f"  {step_num}. Submit CPU jobs (genius cluster):\n")
        for filename, _, _ in cpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        print()
        step_num += 1

    # Submission script suggestion
    all_files = standard_gpu_files + foundation_gpu_files + cpu_files
    if len(all_files) > 2:
        print(f"  TIP: Create a submission script:\n")
        print(f"     cat > submit_all_exp2.sh << 'EOF'")
        print(f"     #!/bin/bash")
        print(f"     # Standard GPU (genius cluster)")
        for filename, _, _ in standard_gpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        print(f"     # Foundation GPU (wICE cluster)")
        for filename, _, _ in foundation_gpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        print(f"     # CPU (genius cluster)")
        for filename, _, _ in cpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        print(f"     EOF")
        print(f"     chmod +x submit_all_exp2.sh")
        print(f"     ./submit_all_exp2.sh\n")

    print(f"  {step_num}. Monitor:")
    print(f"     squeue -u $USER\n")
    step_num += 1

    print(f"  {step_num}. Check results:")
    print(f"     ls results/experiment2/pd/*.pkl | wc -l")
    print(f"     ls results/experiment2/lgd/*.pkl | wc -l\n")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
