#!/usr/bin/env python3
"""
Setup Script: Generate SLURM scripts with batching for VSC limits

This script:
1. Defines GPU vs CPU method categorization
2. Reads enabled methods and datasets from config
3. Counts GPU and CPU tasks
4. Generates BATCHED SLURM scripts (max 400 tasks per file)
5. Provides instructions for submission

GPU/CPU Categorization:
- GPU methods: Deep learning architectures requiring GPU acceleration
- CPU methods: Tree boosting + classical ML (efficient on CPU)
"""

import sys
import math
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.methods.method_config import GPU_METHODS, CPU_METHODS, NO_HPO_METHODS

# VSC-safe constants
MAX_TASKS_PER_SLURM = 400  # Safe buffer below 500 limit


# ======================================================================================
#                     SLURM SCRIPT TEMPLATES (BATCHED)
# ======================================================================================

def generate_gpu_slurm_script(batch_id, start_task, end_task, max_concurrent):
    """Generate GPU SLURM script for a batch of tasks."""
    
    n_tasks = end_task - start_task
    
    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"
    
    return f"""#!/bin/bash
#SBATCH --job-name=exp1_gpu{batch_id}
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --nodes="1" 
#SBATCH --output=results/experiment1/logs/slurm/gpu{batch_id}_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/gpu{batch_id}_%A_%a.err
#SBATCH --time=70:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=45G
#SBATCH --partition=gpu_p100
#SBATCH --array={array_range}

# ---------------------------------------------------------
# BATCH {batch_id}: Tasks {start_task}-{end_task-1}
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
echo "EXPERIMENT 1 - GPU - BATCH {batch_id}"
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
python -u scripts/Experiment1/Experiment1_GPU.py --array_id=$GLOBAL_TASK_ID --verbose
"""


def generate_cpu_slurm_script(batch_id, start_task, end_task, max_concurrent):
    """Generate CPU SLURM script for a batch of tasks."""
    
    n_tasks = end_task - start_task
    
    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"
    
    return f"""#!/bin/bash
#SBATCH --job-name=exp1_cpu{batch_id}
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --output=results/experiment1/logs/slurm/cpu{batch_id}_%A_%a.out
#SBATCH --error=results/experiment1/logs/slurm/cpu{batch_id}_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=batch
#SBATCH --array={array_range}

# ---------------------------------------------------------
# BATCH {batch_id}: Tasks {start_task}-{end_task-1}
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
echo "EXPERIMENT 1 - CPU - BATCH {batch_id}"
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
python -u scripts/Experiment1/Experiment1_CPU.py --array_id=$GLOBAL_TASK_ID --verbose
"""


# ======================================================================================
#                     TASK COUNTING AND BATCHING
# ======================================================================================

def count_tasks(methods, datasets):
    """Count total tasks for given methods and datasets."""
    count = 0
    for dataset in datasets:
        for method in methods:
            if method in NO_HPO_METHODS:
                count += 1  # Only NO_HPO
            else:
                count += 2  # NO_HPO + HPO
    return count


def generate_batched_slurm_files(total_tasks, script_generator, prefix, scripts_dir, max_concurrent):
    """
    Generate multiple SLURM files, each with max 400 tasks.
    
    Args:
        total_tasks: Total number of tasks
        script_generator: Function to generate script content
        prefix: Filename prefix ('Experiment1_GPU' or 'Experiment1_CPU')
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


def print_method_summary(gpu_methods, cpu_methods, pd_datasets, lgd_datasets):
    """Print summary of methods and datasets."""
    
    print(f"\n{'='*70}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nDatasets:")
    print(f"  PD ({len(pd_datasets)}):  {', '.join(sorted(pd_datasets))}")
    print(f"  LGD ({len(lgd_datasets)}): {', '.join(sorted(lgd_datasets))}")
    
    print(f"\nGPU Methods ({len(gpu_methods)}):")
    for i, method in enumerate(sorted(gpu_methods), 1):
        hpo_str = "NO_HPO only" if method in NO_HPO_METHODS else "NO_HPO + HPO"
        print(f"  {i:2d}. {method:<20} ({hpo_str})")
    
    print(f"\nCPU Methods ({len(cpu_methods)}):")
    for i, method in enumerate(sorted(cpu_methods), 1):
        hpo_str = "NO_HPO only" if method in NO_HPO_METHODS else "NO_HPO + HPO"
        print(f"  {i:2d}. {method:<20} ({hpo_str})")
    
    print(f"{'='*70}\n")


def main():
    experiment_name = "experiment1"
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 1 SETUP")
    print(f"{'='*70}\n")
    
    # Load config
    config = load_config("Experiment1")
    
    # Get enabled datasets
    pd_datasets = list(config['datasets']['pd'].keys())
    lgd_datasets = list(config['datasets']['lgd'].keys())
    all_datasets = pd_datasets + lgd_datasets
    
    if not all_datasets:
        print("❌ ERROR: No datasets enabled in config")
        sys.exit(1)
    
    # Get enabled methods (filtered by GPU/CPU)
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
        print("❌ ERROR: No methods enabled in config")
        sys.exit(1)
    
    # Print summary
    print_method_summary(gpu_methods, cpu_methods, pd_datasets, lgd_datasets)
    
    # Count tasks
    n_gpu_tasks = (
        count_tasks(gpu_pd_methods, pd_datasets) +
        count_tasks(gpu_lgd_methods, lgd_datasets)
    )
    
    n_cpu_tasks = (
        count_tasks(cpu_pd_methods, pd_datasets) +
        count_tasks(cpu_lgd_methods, lgd_datasets)
    )
    
    # Determine concurrency
    max_gpu_concurrent = min(16, n_gpu_tasks) if n_gpu_tasks > 0 else 1
    max_cpu_concurrent = min(32, n_cpu_tasks) if n_cpu_tasks > 0 else 1
    
    print(f"{'='*70}")
    print("TASK COUNTS")
    print(f"{'='*70}")
    print(f"GPU tasks:  {n_gpu_tasks:4d}")
    print(f"CPU tasks:  {n_cpu_tasks:4d}")
    print(f"Total:      {n_gpu_tasks + n_cpu_tasks:4d}")
    print(f"\nBatching strategy: {MAX_TASKS_PER_SLURM} tasks per SLURM file (VSC limit workaround)")
    print(f"{'='*70}\n")
    
    # Generate SLURM scripts
    print("Generating SLURM scripts...")
    scripts_dir = PROJECT_ROOT / "scripts" / "Experiment1"
    scripts_dir.mkdir(exist_ok=True)
    
    # Generate GPU scripts (batched)
    gpu_files = generate_batched_slurm_files(
        total_tasks=n_gpu_tasks,
        script_generator=generate_gpu_slurm_script,
        prefix="Experiment1_GPU",
        scripts_dir=scripts_dir,
        max_concurrent=max_gpu_concurrent
    )
    
    if gpu_files:
        print(f"\n✓ Generated {len(gpu_files)} GPU batch script(s):")
        for filename, start, end in gpu_files:
            print(f"  - {filename} (tasks {start}-{end-1})")
    
    # Generate CPU scripts (batched)
    cpu_files = generate_batched_slurm_files(
        total_tasks=n_cpu_tasks,
        script_generator=generate_cpu_slurm_script,
        prefix="Experiment1_CPU",
        scripts_dir=scripts_dir,
        max_concurrent=max_cpu_concurrent
    )
    
    if cpu_files:
        print(f"\n✓ Generated {len(cpu_files)} CPU batch script(s):")
        for filename, start, end in cpu_files:
            print(f"  - {filename} (tasks {start}-{end-1})")
    
    print(f"\n{'='*70}")
    print("✅ SETUP COMPLETE")
    print(f"{'='*70}\n")
    
    print("📋 Next Steps:\n")
    
    # GPU submission instructions
    if gpu_files:
        print(f"  1. Submit GPU jobs sequentially:\n")
        for filename, _, _ in gpu_files:
            print(f"     sbatch scripts/Experiment1/{filename}")
        print()
    
    # CPU submission instructions
    if cpu_files:
        print(f"  2. Submit CPU jobs sequentially:\n")
        for filename, _, _ in cpu_files:
            print(f"     sbatch scripts/Experiment1/{filename}")
        print()
    
    # Submission script suggestion
    if len(gpu_files) + len(cpu_files) > 3:
        print(f"  💡 TIP: Create a submission script:\n")
        print(f"     cat > submit_all_exp1.sh << 'EOF'")
        print(f"     #!/bin/bash")
        for filename, _, _ in gpu_files:
            print(f"     sbatch scripts/Experiment1/{filename}")
        for filename, _, _ in cpu_files:
            print(f"     sbatch scripts/Experiment1/{filename}")
        print(f"     EOF")
        print(f"     chmod +x submit_all_exp1.sh")
        print(f"     ./submit_all_exp1.sh\n")
    
    print(f"  3. Monitor:")
    print(f"     squeue -u $USER\n")
    
    print(f"  4. Check results:")
    print(f"     ls results/experiment1/pd/*.pkl | wc -l")
    print(f"     ls results/experiment1/lgd/*.pkl | wc -l\n")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()