#!/usr/bin/env python3
"""
Setup script: Generate task lists and configure SLURM array jobs.

This script prepares Experiment 1 for execution by:
1. Generating all task combinations (dataset × method × HPO mode)
2. Separating tasks into GPU and CPU groups
3. Updating SLURM scripts with correct array ranges
4. Providing execution instructions

The script respects NO_HPO_METHODS and only generates NO_HPO tasks
for methods that don't benefit from hyperparameter tuning.
"""

import sys
from pathlib import Path

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import experiment functions
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "Experiment1"))
from Experiment1 import generate_task_list, save_task_lists


def update_slurm_script(script_path, n_tasks, max_concurrent):
    """
    Update SLURM script with correct array range.
    
    Replaces the placeholder "#SBATCH --array=PLACEHOLDER" with actual
    array directive based on number of tasks.
    
    Args:
        script_path: Path to SLURM script file
        n_tasks: Number of tasks to run
        max_concurrent: Maximum concurrent array jobs
    """
    
    if not script_path.exists():
        print(f"⚠️  WARNING: {script_path} not found")
        return
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Create array directive
    if n_tasks == 0:
        array_directive = "#SBATCH --array=0"
        print(f"⚠️  WARNING: {script_path.name} has 0 tasks")
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"
        array_directive = f"#SBATCH --array={array_range}"
    
    # Replace placeholder
    if "#SBATCH --array=PLACEHOLDER" not in content:
        print(f"⚠️  WARNING: No placeholder found in {script_path.name}")
        return
    
    content = content.replace(
        "#SBATCH --array=PLACEHOLDER",
        array_directive
    )
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {script_path.name}: {array_directive}")


def print_method_summary(tasks_by_type):
    """Print summary of tasks by method and execution type."""
    
    from collections import defaultdict
    
    # Count tasks by method
    method_counts = defaultdict(lambda: {'NO_HPO': 0, 'HPO': 0, 'total': 0})
    
    for task in tasks_by_type['all']:
        method = task['method']
        hpo_mode = task['hpo_mode']
        method_counts[method][hpo_mode] += 1
        method_counts[method]['total'] += 1
    
    # Separate GPU and CPU methods
    gpu_methods = sorted([m for m in method_counts.keys() 
                         if any(t['method'] == m and t['is_gpu_method'] 
                               for t in tasks_by_type['all'])])
    cpu_methods = sorted([m for m in method_counts.keys() 
                         if any(t['method'] == m and not t['is_gpu_method'] 
                               for t in tasks_by_type['all'])])
    
    print(f"\n{'='*70}")
    print("METHOD BREAKDOWN")
    print(f"{'='*70}")
    
    if gpu_methods:
        print(f"\nGPU Methods ({len(gpu_methods)}):")
        print(f"  {'Method':<20} {'NO_HPO':>8} {'HPO':>8} {'Total':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
        for method in gpu_methods:
            counts = method_counts[method]
            print(f"  {method:<20} {counts['NO_HPO']:>8} {counts['HPO']:>8} {counts['total']:>8}")
    
    if cpu_methods:
        print(f"\nCPU Methods ({len(cpu_methods)}):")
        print(f"  {'Method':<20} {'NO_HPO':>8} {'HPO':>8} {'Total':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
        for method in cpu_methods:
            counts = method_counts[method]
            print(f"  {method:<20} {counts['NO_HPO']:>8} {counts['HPO']:>8} {counts['total']:>8}")
    
    print(f"{'='*70}\n")


def main():
    """Main setup routine."""
    
    experiment_name = "experiment1"
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 1 SETUP")
    print(f"{'='*70}\n")
    
    # Generate task lists
    print("Generating task lists...")
    all_tasks, tasks_by_type = generate_task_list(experiment_name)
    save_task_lists(experiment_name, tasks_by_type)
    
    # Get task counts
    n_gpu = len(tasks_by_type['gpu'])
    n_cpu = len(tasks_by_type['cpu'])
    n_total = len(tasks_by_type['all'])
    
    # Print method breakdown
    print_method_summary(tasks_by_type)
    
    # Determine concurrency limits
    # VSC has ~16-32 A100 GPUs per partition, but we limit to avoid overwhelming
    max_gpu_concurrent = min(16, n_gpu) if n_gpu > 0 else 1
    
    # CPU nodes are more abundant, but still limit concurrent jobs
    max_cpu_concurrent = min(64, n_cpu) if n_cpu > 0 else 1
    
    print(f"{'='*70}")
    print("SLURM CONFIGURATION")
    print(f"{'='*70}")
    print(f"Total tasks:      {n_total:4d}")
    print(f"  GPU tasks:      {n_gpu:4d} (max {max_gpu_concurrent} concurrent)")
    print(f"  CPU tasks:      {n_cpu:4d} (max {max_cpu_concurrent} concurrent)")
    print(f"{'='*70}\n")
    
    # Update SLURM scripts
    print("Updating SLURM scripts...")
    scripts_dir = PROJECT_ROOT / "scripts" / "Experiment1"
    
    update_slurm_script(
        scripts_dir / "Experiment1_GPU.slurm",
        n_gpu,
        max_gpu_concurrent
    )
    
    update_slurm_script(
        scripts_dir / "Experiment1_CPU.slurm",
        n_cpu,
        max_cpu_concurrent
    )
    
    print(f"\n{'='*70}")
    print("✅ SETUP COMPLETE")
    print(f"{'='*70}\n")
    
    # Provide execution instructions
    print("📋 Next Steps:\n")
    
    if n_gpu > 0:
        print(f"  1. Submit GPU jobs:")
        print(f"     sbatch scripts/Experiment1/Experiment1_GPU.slurm")
        print(f"     ({n_gpu} tasks, up to {max_gpu_concurrent} concurrent)\n")
    else:
        print(f"  1. No GPU tasks to run\n")
    
    if n_cpu > 0:
        print(f"  2. Submit CPU jobs:")
        print(f"     sbatch scripts/Experiment1/Experiment1_CPU.slurm")
        print(f"     ({n_cpu} tasks, up to {max_cpu_concurrent} concurrent)\n")
    else:
        print(f"  2. No CPU tasks to run\n")
    
    print(f"  3. Monitor execution:")
    print(f"     squeue -u $USER              # Check job status")
    print(f"     watch -n 5 'squeue -u $USER' # Auto-refresh every 5s")
    print(f"     ls results/experiment1/pd/*.pkl | wc -l  # Count results\n")
    
    print(f"  4. Check logs:")
    print(f"     tail -f results/experiment1/logs/slurm/gpu_*.out")
    print(f"     tail -f results/experiment1/logs/slurm/cpu_*.out")
    print(f"     cat results/experiment1/logs/errors.log  # All failures\n")
    
    print(f"{'='*70}\n")
    
    # Provide test command
    print("🧪 Test Single Task (before submitting all):\n")
    print(f"  python scripts/Experiment1/Experiment1.py --task_idx=0 --verbose\n")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()