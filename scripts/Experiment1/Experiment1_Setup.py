#!/usr/bin/env python3
"""
Setup Script: Generate SLURM scripts with correct array sizes

This script:
1. Defines GPU vs CPU method categorization
2. Reads enabled methods and datasets from config
3. Counts GPU and CPU tasks
4. COMPLETELY REWRITES SLURM scripts with correct array ranges
5. Provides instructions for submission

GPU/CPU Categorization:
- GPU methods: Deep learning architectures requiring GPU acceleration
- CPU methods: Tree boosting + classical ML (efficient on CPU)
"""

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.methods.method_config import NO_HPO_METHODS

# ======================================================================================
#                    GPU vs CPU METHOD CATEGORIZATION
# ======================================================================================

# GPU Methods - Neural architectures + tree boosting (run on GPU nodes)
GPU_METHODS = {
    # Basic neural architectures
    'mlp', 'resnet',
    
    # Attention-based transformers
    'ftt', 'saint', 'tabtransformer', 'tabptm', 'trompt',
    
    # Specialized deep learning
    'tabnet', 'node', 'tabr', 'grownet',
    
    # Advanced architectures
    'autoint', 'snn', 'danets', 'tabcaps', 'dcn2',
    'tangos', 'ptarl', 'switchtab', 'dnnr',
    
    # Modern architectures
    'modernNCA', 'hyperfast', 'bishop', 'realmlp',
    'protogate', 'mlp_plr', 'excelformer', 'grande',
    'amformer', 'tabm', 't2gformer', 'tabautopnpnet',
    'tabicl', 'limix', 'mitra',
    
    # Foundation models
    'tabpfn', 'tabpfn_v2', 'tabpfn_real',
    
    # Tree boosting (run on GPU nodes for compatibility)
    'xgboost', 'catboost', 'lightgbm',
}

# CPU Methods - Only classical ML (no tree boosting)
CPU_METHODS = {
    # Traditional ML models
    'RandomForest', 'LogReg', 'LinearRegression',
    'knn', 'svm', 'NaiveBayes', 'NCM',
    
    # Baseline models
    'dummy',
}


# ======================================================================================
#                    SLURM SCRIPT TEMPLATES
# ======================================================================================

def generate_gpu_slurm_script(n_tasks, max_concurrent):
    """Generate GPU SLURM script for Experiment1."""
    
    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"
    
    return f"""#!/bin/bash
#SBATCH --job-name=exp1_gpu
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --nodes="1" 
#SBATCH --output=../../results/experiment1/logs/slurm/gpu_%A_%a.out
#SBATCH --error=../../results/experiment1/logs/slurm/gpu_%A_%a.err
#SBATCH --time=04:00:00
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
mkdir -p results/experiment1/pd
mkdir -p results/experiment1/lgd
mkdir -p results/experiment1/logs/slurm

echo "=========================================="
echo "EXPERIMENT 1 - GPU JOB"
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
python -u scripts/Experiment1/Experiment1_GPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
"""


def generate_cpu_slurm_script(n_tasks, max_concurrent):
    """Generate CPU SLURM script for Experiment1."""
    
    if n_tasks == 0:
        array_range = "0"
    else:
        array_range = f"0-{n_tasks-1}%{max_concurrent}"
    
    return f"""#!/bin/bash
#SBATCH --job-name=exp1_cpu
#SBATCH --cluster="genius"
#SBATCH --account="lp_verbekelab" 
#SBATCH --output=../../results/experiment1/logs/slurm/cpu_%A_%a.out
#SBATCH --error=../../results/experiment1/logs/slurm/cpu_%A_%a.err
#SBATCH --time=24:00:00
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
mkdir -p results/experiment1/pd
mkdir -p results/experiment1/lgd
mkdir -p results/experiment1/logs/slurm

echo "=========================================="
echo "EXPERIMENT 1 - CPU JOB"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array ID:     $SLURM_ARRAY_TASK_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Python:       $(which python)"
echo "Conda env:    $CONDA_DEFAULT_ENV"
echo "Working dir:  $(pwd)"
echo "=========================================="

# Run CPU orchestrator
python -u scripts/Experiment1/Experiment1_CPU.py --array_id=$SLURM_ARRAY_TASK_ID --verbose
"""


# ======================================================================================
#                    TASK COUNTING AND SLURM GENERATION
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


def write_slurm_script(script_path, content, job_type):
    """
    Write SLURM script, completely overwriting any existing file.
    
    Args:
        script_path: Path to SLURM script
        content: Complete script content to write
        job_type: 'GPU' or 'CPU' for logging
    """
    
    try:
        # Ensure directory exists
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write complete script
        with open(script_path, 'w', newline='\n') as f:  # Force Unix line endings
            f.write(content)
        
        # Make executable
        script_path.chmod(0o755)
        
        print(f"✓ {script_path.name}: Written successfully")
        return True
        
    except Exception as e:
        print(f"❌ ERROR writing {script_path.name}: {e}")
        return False


def print_method_summary(gpu_methods, cpu_methods, pd_datasets, lgd_datasets):
    """Print summary of methods and datasets."""
    
    print(f"\n{'='*70}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nDatasets:")
    print(f"  PD:  {', '.join(pd_datasets)}")
    print(f"  LGD: {', '.join(lgd_datasets)}")
    
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
    max_cpu_concurrent = min(64, n_cpu_tasks) if n_cpu_tasks > 0 else 1
    
    print(f"{'='*70}")
    print("TASK COUNTS")
    print(f"{'='*70}")
    print(f"GPU tasks:  {n_gpu_tasks:4d} (max {max_gpu_concurrent} concurrent)")
    print(f"CPU tasks:  {n_cpu_tasks:4d} (max {max_cpu_concurrent} concurrent)")
    print(f"Total:      {n_gpu_tasks + n_cpu_tasks:4d}")
    print(f"{'='*70}\n")
    
    # Generate SLURM scripts
    print("Generating SLURM scripts...")
    scripts_dir = PROJECT_ROOT / "scripts" / "Experiment1"
    
    # Generate GPU script
    gpu_content = generate_gpu_slurm_script(n_gpu_tasks, max_gpu_concurrent)
    gpu_success = write_slurm_script(
        scripts_dir / "Experiment1_GPU.slurm",
        gpu_content,
        "GPU"
    )
    
    # Generate CPU script
    cpu_content = generate_cpu_slurm_script(n_cpu_tasks, max_cpu_concurrent)
    cpu_success = write_slurm_script(
        scripts_dir / "Experiment1_CPU.slurm",
        cpu_content,
        "CPU"
    )
    
    if not gpu_success or not cpu_success:
        print("\n❌ SETUP FAILED")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("✅ SETUP COMPLETE")
    print(f"{'='*70}\n")
    
    # Provide instructions
    print("📋 Next Steps:\n")
    
    if n_gpu_tasks > 0:
        print(f"  1. Submit GPU jobs:")
        print(f"     sbatch scripts/Experiment1/Experiment1_GPU.slurm")
        print(f"     ({n_gpu_tasks} tasks, up to {max_gpu_concurrent} concurrent)\n")
    else:
        print(f"  1. No GPU jobs to submit\n")
    
    if n_cpu_tasks > 0:
        print(f"  2. Submit CPU jobs:")
        print(f"     sbatch scripts/Experiment1/Experiment1_CPU.slurm")
        print(f"     ({n_cpu_tasks} tasks, up to {max_cpu_concurrent} concurrent)\n")
    else:
        print(f"  2. No CPU jobs to submit\n")
    
    print(f"  3. Monitor:")
    print(f"     squeue -u $USER")
    print(f"     watch -n 5 'squeue -u $USER'\n")
    
    print(f"  4. Check results:")
    print(f"     ls results/experiment1/pd/*.pkl | wc -l")
    print(f"     ls results/experiment1/lgd/*.pkl | wc -l\n")
    
    print(f"{'='*70}\n")
    
    # Test command
    print("🧪 Test Single Task (before submitting all):\n")
    if n_gpu_tasks > 0:
        print(f"  GPU: python scripts/Experiment1/Experiment1_GPU.py --array_id=0 --verbose")
    if n_cpu_tasks > 0:
        print(f"  CPU: python scripts/Experiment1/Experiment1_CPU.py --array_id=0 --verbose")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()