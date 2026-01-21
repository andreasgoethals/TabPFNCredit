#!/usr/bin/env python3
"""
Setup Script for Experiment 2: Learning Curve Analysis

This script:
1. Uses method_config.py for GPU/CPU categorization (no hardcoding)
2. Reads enabled methods and datasets from config
3. Counts GPU and CPU tasks (method × dataset combinations)
4. Generates BATCHED SLURM scripts (max 400 tasks per file)
5. Provides instructions for submission

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
from src.methods.method_config import GPU_METHODS, CPU_METHODS

# VSC-safe constants
MAX_TASKS_PER_SLURM = 400  # Safe buffer below 500 limit


# ======================================================================================
#                     SLURM SCRIPT TEMPLATES (BATCHED)
# ======================================================================================

def generate_gpu_slurm_script(batch_id, start_task, end_task, max_concurrent):
    """Generate GPU SLURM script for a batch of learning curve tasks."""

    n_tasks = end_task - start_task

    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"

    return f"""#!/bin/bash
#SBATCH --job-name=exp2_gpu{batch_id}
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab"
#SBATCH --nodes="1"
#SBATCH --output=results/experiment2/logs/slurm/gpu{batch_id}_%A_%a.out
#SBATCH --error=results/experiment2/logs/slurm/gpu{batch_id}_%A_%a.err
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=45G
#SBATCH --partition=gpu_p100
#SBATCH --array={array_range}

# ---------------------------------------------------------
# EXPERIMENT 2: LEARNING CURVE ANALYSIS
# BATCH {batch_id}: Tasks {start_task}-{end_task-1}
# ---------------------------------------------------------
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
echo "EXPERIMENT 2 - LEARNING CURVE - GPU - BATCH {batch_id}"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Batch:        {batch_id}"
echo "Task offset:  {start_task}"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Calculate global task ID from batch offset
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + {start_task}))

# Run GPU orchestrator with global task ID
python -u scripts/Experiment2/Experiment2_GPU.py --array_id=$GLOBAL_TASK_ID --verbose
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
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab"
#SBATCH --output=results/experiment2/logs/slurm/cpu{batch_id}_%A_%a.out
#SBATCH --error=results/experiment2/logs/slurm/cpu{batch_id}_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=batch
#SBATCH --array={array_range}

# ---------------------------------------------------------
# EXPERIMENT 2: LEARNING CURVE ANALYSIS
# BATCH {batch_id}: Tasks {start_task}-{end_task-1}
# ---------------------------------------------------------
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


def generate_batched_slurm_files(total_tasks, script_generator, prefix, scripts_dir, max_concurrent):
    """
    Generate multiple SLURM files, each with max 400 tasks.

    Args:
        total_tasks: Total number of tasks
        script_generator: Function to generate script content
        prefix: Filename prefix ('Experiment2_GPU' or 'Experiment2_CPU')
        scripts_dir: Directory to save scripts
        max_concurrent: Max concurrent jobs

    Returns:
        List of generated filenames with task ranges
    """
    if total_tasks == 0:
        return []

    n_batches = math.ceil(total_tasks / MAX_TASKS_PER_SLURM)
    generated_files = []

    for batch_id in range(n_batches):
        start_task = batch_id * MAX_TASKS_PER_SLURM
        end_task = min(start_task + MAX_TASKS_PER_SLURM, total_tasks)

        # Generate script content
        script_content = script_generator(
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


def print_method_summary(gpu_methods, cpu_methods, pd_datasets, lgd_datasets, lc_config):
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

    print(f"\nGPU Methods ({len(gpu_methods)}):")
    for i, method in enumerate(sorted(gpu_methods), 1):
        print(f"  {i:2d}. {method}")

    print(f"\nCPU Methods ({len(cpu_methods)}):")
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

    # Get enabled methods (filtered by GPU/CPU using method_config.py)
    all_pd_methods = list(config['methods']['pd'].keys())
    all_lgd_methods = list(config['methods']['lgd'].keys())

    gpu_pd_methods = [m for m in all_pd_methods if m in GPU_METHODS]
    gpu_lgd_methods = [m for m in all_lgd_methods if m in GPU_METHODS]

    cpu_pd_methods = [m for m in all_pd_methods if m in CPU_METHODS]
    cpu_lgd_methods = [m for m in all_lgd_methods if m in CPU_METHODS]

    # Get unique method names
    gpu_methods = set(gpu_pd_methods + gpu_lgd_methods)
    cpu_methods = set(cpu_pd_methods + cpu_lgd_methods)

    if not gpu_methods and not cpu_methods:
        print("ERROR: No methods enabled in config")
        sys.exit(1)

    # Print summary
    print_method_summary(gpu_methods, cpu_methods, pd_datasets, lgd_datasets, lc_config)

    # Count tasks (no HPO variation for Experiment2)
    n_gpu_tasks = (
        count_tasks_exp2(gpu_pd_methods, pd_datasets) +
        count_tasks_exp2(gpu_lgd_methods, lgd_datasets)
    )

    n_cpu_tasks = (
        count_tasks_exp2(cpu_pd_methods, pd_datasets) +
        count_tasks_exp2(cpu_lgd_methods, lgd_datasets)
    )

    # Determine concurrency (fewer concurrent jobs due to longer runtimes)
    max_gpu_concurrent = min(12, n_gpu_tasks) if n_gpu_tasks > 0 else 1
    max_cpu_concurrent = min(24, n_cpu_tasks) if n_cpu_tasks > 0 else 1

    print(f"{'='*70}")
    print("TASK COUNTS")
    print(f"{'='*70}")
    print(f"GPU tasks:  {n_gpu_tasks:4d}")
    print(f"CPU tasks:  {n_cpu_tasks:4d}")
    print(f"Total:      {n_gpu_tasks + n_cpu_tasks:4d}")
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

    # Generate GPU scripts (batched)
    gpu_files = generate_batched_slurm_files(
        total_tasks=n_gpu_tasks,
        script_generator=generate_gpu_slurm_script,
        prefix="Experiment2_GPU",
        scripts_dir=scripts_dir,
        max_concurrent=max_gpu_concurrent
    )

    if gpu_files:
        print(f"\nGenerated {len(gpu_files)} GPU batch script(s):")
        for filename, start, end in gpu_files:
            print(f"  - {filename} (tasks {start}-{end-1})")

    # Generate CPU scripts (batched)
    cpu_files = generate_batched_slurm_files(
        total_tasks=n_cpu_tasks,
        script_generator=generate_cpu_slurm_script,
        prefix="Experiment2_CPU",
        scripts_dir=scripts_dir,
        max_concurrent=max_cpu_concurrent
    )

    if cpu_files:
        print(f"\nGenerated {len(cpu_files)} CPU batch script(s):")
        for filename, start, end in cpu_files:
            print(f"  - {filename} (tasks {start}-{end-1})")

    print(f"\n{'='*70}")
    print("SETUP COMPLETE")
    print(f"{'='*70}\n")

    print("Next Steps:\n")

    # GPU submission instructions
    if gpu_files:
        print(f"  1. Submit GPU jobs sequentially:\n")
        for filename, _, _ in gpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        print()

    # CPU submission instructions
    if cpu_files:
        print(f"  2. Submit CPU jobs sequentially:\n")
        for filename, _, _ in cpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        print()

    # Submission script suggestion
    if len(gpu_files) + len(cpu_files) > 2:
        print(f"  TIP: Create a submission script:\n")
        print(f"     cat > submit_all_exp2.sh << 'EOF'")
        print(f"     #!/bin/bash")
        for filename, _, _ in gpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        for filename, _, _ in cpu_files:
            print(f"     sbatch scripts/Experiment2/{filename}")
        print(f"     EOF")
        print(f"     chmod +x submit_all_exp2.sh")
        print(f"     ./submit_all_exp2.sh\n")

    print(f"  3. Monitor:")
    print(f"     squeue -u $USER\n")

    print(f"  4. Check results:")
    print(f"     ls results/experiment2/pd/*.pkl | wc -l")
    print(f"     ls results/experiment2/lgd/*.pkl | wc -l\n")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
