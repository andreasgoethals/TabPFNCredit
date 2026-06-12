#!/bin/bash -l
# ============================================================================
# preflight_v100.slurm -- one-shot Genius V100 compatibility test
# ----------------------------------------------------------------------------
# Verifies, on a real gpu_v100 node, that the project venv's torch 2.8 CUDA
# wheels + driver work (matmul) and that both Experiment-2/3 foundation
# models run end-to-end against the staged weights (offline).
#
#   sbatch scripts/preflight_v100.slurm
#   squeue -u $USER -M genius
#   cat preflight_v100.out
#
# Expected on success: "DEVICE: Tesla V100..." + "MATMUL OK" + two AUC lines
# + "PREFLIGHT DONE". Then enable the offload:
#   export TABPFN_GENIUS_GPUS=gpu_v100
#   tabpfncredit resubmit Experiment2 Experiment3
# ============================================================================
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=genius
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=preflight_v100.out

set -x
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "$VSC_DATA/TabPFNCredit/tabpfncreditvenv/bin/activate"
cd "$VSC_DATA/TabPFNCredit"

# Same weight-cache wiring the generated experiment jobs use (offline).
export TABPFN_STAGING_ROOT=/staging/leuven/stg_00211
export HF_HOME=$TABPFN_STAGING_ROOT/checkpoints/huggingface
export HUGGINGFACE_HUB_CACHE=$TABPFN_STAGING_ROOT/checkpoints/huggingface/hub
export TABPFN_MODEL_CACHE_DIR=$TABPFN_STAGING_ROOT/checkpoints/tabpfn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -c "import torch; print('DEVICE:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x=torch.randn(2048,2048,device='cuda'); print('MATMUL OK:', float((x@x).sum()))"
python -c "from src.methods.method_runner import run_talent_method; r=run_talent_method(task='pd', dataset='0008.german', method='tabpfn_v3', cv_splits=2); print('TABPFN_V3 AUC:', [round(f['metrics']['AUC'],3) for f in r.values()])"
python -c "from src.methods.method_runner import run_talent_method; r=run_talent_method(task='pd', dataset='0008.german', method='tabicl_v2', cv_splits=2); print('TABICL_V2 AUC:', [round(f['metrics']['AUC'],3) for f in r.values()])"
echo "PREFLIGHT DONE"
