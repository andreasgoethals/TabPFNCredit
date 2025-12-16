#!/usr/bin/env python3
"""Setup script: Generate task lists and configure SLURM arrays."""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "Experiment1"))
from Experiment1 import generate_task_list, save_task_lists


def update_slurm_script(script_path, n_tasks, max_concurrent):
    """Update SLURM script with correct array range."""
    
    if not script_path.exists():
        print(f"WARNING: {script_path} not found")
        return
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Create array directive
    if n_tasks == 0:
        array_directive = "#SBATCH --array=0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"
        array_directive = f"#SBATCH --array={array_range}"
    
    # Replace placeholder
    content = content.replace(
        "#SBATCH --array=PLACEHOLDER",
        array_directive
    )
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {script_path.name}: {array_directive}")


def main():
    experiment_name = "experiment1"
    
    print(f"\n{'='*70}")
    print("SETUP: Generating task lists")
    print(f"{'='*70}\n")
    
    # Generate tasks
    all_tasks, tasks_by_type = generate_task_list(experiment_name)
    save_task_lists(experiment_name, tasks_by_type)
    
    n_gpu = len(tasks_by_type['gpu'])
    n_cpu = len(tasks_by_type['cpu'])
    
    # Determine concurrency
    max_gpu_concurrent = min(16, n_gpu)  # VSC has ~16-32 A100 GPUs
    max_cpu_concurrent = min(64, n_cpu)  # CPUs more abundant
    
    print(f"\n{'='*70}")
    print("SLURM CONFIGURATION")
    print(f"{'='*70}")
    print(f"GPU tasks:  {n_gpu:4d} (max {max_gpu_concurrent} concurrent)")
    print(f"CPU tasks:  {n_cpu:4d} (max {max_cpu_concurrent} concurrent)")
    print(f"{'='*70}\n")
    
    # Update SLURM scripts
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
    print("Next steps:")
    print("  1. sbatch scripts/Experiment1/Experiment1_GPU.slurm")
    print("  2. sbatch scripts/Experiment1/Experiment1_CPU.slurm")
    print("  3. squeue -u $USER  # Monitor")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()