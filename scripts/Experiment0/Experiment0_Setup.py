#!/usr/bin/env python3
"""
Experiment0 Setup: Generate SLURM scripts for method validation

Simpler than Experiment1 - only NO_HPO mode, 2 datasets, all methods.
"""

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config

# Import GPU/CPU categorization from Experiment1
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "Experiment1"))
from Experiment1_Setup import GPU_METHODS, CPU_METHODS


def generate_gpu_slurm_script(n_tasks, max_concurrent):
    """Generate GPU SLURM script for Experiment0."""
    
    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"
    
    return f"""#!/usr/bin/bash
#SBATCH --job-name=exp0_gpu
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --nodes="1" 
#SBATCH --output=../../results/experiment0/logs/slurm/gpu_%A_%a.out
#SBATCH --error=../../results/experiment0/logs/slurm/gpu_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --partition=gpu_p100
#SBATCH --array={array_range}

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup conda from $VSC_DATA
export PATH="${{VSC_DATA}}/miniconda3/bin:${{PATH}}"
source activate TabPFNCredit

# USE CONDA'S C++ LIBRARIES (CRITICAL!)
export LD_LIBRARY_PATH="${{VSC_DATA}}/miniconda3/envs/TabPFNCredit/lib:${{LD_LIBRARY_PATH}}"

# Navigate to project
cd ${{VSC_DATA}}/TabPFNCredit

# Ensure result directories exist
mkdir -p results/experiment0/pd
mkdir -p results/experiment0/lgd
mkdir -p results/experiment0/logs/slurm

echo "=========================================="
echo "EXPERIMENT 0 - GPU VALIDATION"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Python:       $(which python)"
echo "Conda env:    $CONDA_DEFAULT_ENV"
echo "Working dir:  $(pwd)"
echo "=========================================="

# Run GPU orchestrator
python -u scripts/Experiment0/Experiment0_GPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
"""


def generate_cpu_slurm_script(n_tasks, max_concurrent):
    """Generate CPU SLURM script for Experiment0."""
    
    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"
    
    return f"""#!/usr/bin/bash
#SBATCH --job-name=exp0_cpu
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --output=../../results/experiment0/logs/slurm/cpu_%A_%a.out
#SBATCH --error=../../results/experiment0/logs/slurm/cpu_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --partition=batch
#SBATCH --array={array_range}

# Force unbuffered I/O
export PYTHONUNBUFFERED=1

# Setup conda from $VSC_DATA
export PATH="${{VSC_DATA}}/miniconda3/bin:${{PATH}}"
source activate TabPFNCredit

# USE CONDA'S C++ LIBRARIES (CRITICAL!)
export LD_LIBRARY_PATH="${{VSC_DATA}}/miniconda3/envs/TabPFNCredit/lib:${{LD_LIBRARY_PATH}}"

# Navigate to project
cd ${{VSC_DATA}}/TabPFNCredit

# Ensure result directories exist
mkdir -p results/experiment0/pd
mkdir -p results/experiment0/lgd
mkdir -p results/experiment0/logs/slurm

echo "=========================================="
echo "EXPERIMENT 0 - CPU VALIDATION"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Python:       $(which python)"
echo "Conda env:    $CONDA_DEFAULT_ENV"
echo "Working dir:  $(pwd)"
echo "=========================================="

# Run CPU orchestrator
python -u scripts/Experiment0/Experiment0_CPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
"""


def count_tasks(methods, datasets):
    """Count total tasks (NO_HPO only for all methods)."""
    return len(methods) * len(datasets)


def main():
    experiment_name = "experiment0"
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 0 SETUP - METHOD VALIDATION")
    print(f"{'='*70}\n")
    
    # Load config
    config = load_config("Experiment0")
    
    # Get enabled datasets
    pd_datasets = list(config['datasets']['pd'].keys())
    lgd_datasets = list(config['datasets']['lgd'].keys())
    all_datasets = pd_datasets + lgd_datasets
    
    if not all_datasets:
        print("❌ ERROR: No datasets enabled in config")
        sys.exit(1)
    
    print(f"Datasets: {', '.join(all_datasets)}")
    
    # Get enabled methods
    all_pd_methods = list(config['methods']['pd'].keys())
    all_lgd_methods = list(config['methods']['lgd'].keys())
    
    gpu_pd_methods = [m for m in all_pd_methods if m in GPU_METHODS]
    gpu_lgd_methods = [m for m in all_lgd_methods if m in GPU_METHODS]
    
    cpu_pd_methods = [m for m in all_pd_methods if m in CPU_METHODS]
    cpu_lgd_methods = [m for m in all_lgd_methods if m in CPU_METHODS]
    
    gpu_methods = set(gpu_pd_methods + gpu_lgd_methods)
    cpu_methods = set(cpu_pd_methods + cpu_lgd_methods)
    
    if not gpu_methods and not cpu_methods:
        print("❌ ERROR: No methods enabled in config")
        sys.exit(1)
    
    print(f"GPU Methods: {len(gpu_methods)}")
    print(f"CPU Methods: {len(cpu_methods)}")
    
    # Count tasks (NO_HPO only - 1 task per method per dataset)
    n_gpu_tasks = (
        count_tasks(gpu_pd_methods, pd_datasets) +
        count_tasks(gpu_lgd_methods, lgd_datasets)
    )
    
    n_cpu_tasks = (
        count_tasks(cpu_pd_methods, pd_datasets) +
        count_tasks(cpu_lgd_methods, lgd_datasets)
    )
    
    max_gpu_concurrent = min(16, n_gpu_tasks) if n_gpu_tasks > 0 else 1
    max_cpu_concurrent = min(32, n_cpu_tasks) if n_cpu_tasks > 0 else 1
    
    print(f"\n{'='*70}")
    print("TASK COUNTS (NO_HPO only)")
    print(f"{'='*70}")
    print(f"GPU tasks:  {n_gpu_tasks:4d} (max {max_gpu_concurrent} concurrent)")
    print(f"CPU tasks:  {n_cpu_tasks:4d} (max {max_cpu_concurrent} concurrent)")
    print(f"Total:      {n_gpu_tasks + n_cpu_tasks:4d}")
    print(f"{'='*70}\n")
    
    # Generate SLURM scripts
    print("Generating SLURM scripts...")
    scripts_dir = PROJECT_ROOT / "scripts" / "Experiment0"
    scripts_dir.mkdir(exist_ok=True)
    
    # GPU script
    gpu_content = generate_gpu_slurm_script(n_gpu_tasks, max_gpu_concurrent)
    with open(scripts_dir / "Experiment0_GPU.slurm", 'w', newline='\n') as f:
        f.write(gpu_content)
    (scripts_dir / "Experiment0_GPU.slurm").chmod(0o755)
    print("✓ Experiment0_GPU.slurm")
    
    # CPU script
    cpu_content = generate_cpu_slurm_script(n_cpu_tasks, max_cpu_concurrent)
    with open(scripts_dir / "Experiment0_CPU.slurm", 'w', newline='\n') as f:
        f.write(cpu_content)
    (scripts_dir / "Experiment0_CPU.slurm").chmod(0o755)
    print("✓ Experiment0_CPU.slurm")
    
    print(f"\n{'='*70}")
    print("✅ SETUP COMPLETE")
    print(f"{'='*70}\n")
    
    print("📋 Next Steps:\n")
    if n_gpu_tasks > 0:
        print(f"  1. Submit GPU jobs:")
        print(f"     sbatch scripts/Experiment0/Experiment0_GPU.slurm")
        print(f"     ({n_gpu_tasks} tasks)\n")
    
    if n_cpu_tasks > 0:
        print(f"  2. Submit CPU jobs:")
        print(f"     sbatch scripts/Experiment0/Experiment0_CPU.slurm")
        print(f"     ({n_cpu_tasks} tasks)\n")
    
    print(f"  3. Monitor:")
    print(f"     squeue -u $USER\n")
    
    print(f"  4. Check results:")
    print(f"     ls results/experiment0/pd/*.pkl")
    print(f"     ls results/experiment0/lgd/*.pkl\n")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()