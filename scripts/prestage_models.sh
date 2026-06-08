#!/bin/bash -l
# =============================================================================
# prestage_models.sh -- download foundation-model weights ONCE on the login node
# =============================================================================
# wICE COMPUTE nodes have NO outbound internet, so foundation models (TabPFN
# v2.5/v3, TabICL, TabDPT, Mitra, HyperFast) cannot download their weights or
# accept the TabPFN license at run time -- that is exactly why they failed in
# the last run (TabPFNLicenseError / ConnectionError / FileNotFoundError).
#
# Run this ONCE on a LOGIN node (which HAS internet). It populates a SHARED
# cache under $VSC_DATA that the SLURM jobs then read OFFLINE (the generated
# job scripts export the same HF_HOME / TABPFN_MODEL_CACHE_DIR and set
# HF_HUB_OFFLINE=1). Re-running is cheap (already-cached weights are skipped).
#
#   cd "$VSC_DATA/TabPFNCredit"
#   bash scripts/prestage_models.sh
#
# If a model needs an interactive licence prompt (TabPFN), answer it here on the
# login node -- the acceptance is written into the shared cache so the compute
# nodes inherit it.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- shared cache (MUST match the exports in src/utils/slurm_generator.py) ----
export HF_HOME="${VSC_DATA}/TabPFNCredit/.model_cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${VSC_DATA}/TabPFNCredit/.model_cache/torch"
export XDG_CACHE_HOME="${VSC_DATA}/TabPFNCredit/.model_cache/xdg"
export TABPFN_MODEL_CACHE_DIR="${VSC_DATA}/TabPFNCredit/.model_cache/tabpfn"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TABPFN_MODEL_CACHE_DIR"

# ONLINE here -- we WANT to download. (Do NOT set HF_HUB_OFFLINE in this script.)
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE 2>/dev/null || true

# Force CPU so this runs on a login node with no GPU; the DOWNLOAD happens
# regardless of device, which is all we need here.
export CUDA_VISIBLE_DEVICES=""

# ---- activate the project env (same logic as the generated job scripts) ----
module purge 2>/dev/null || true
: "${TABPFN_PYTHON_MODULE:=Python/3.12.3-GCCcore-13.3.0}"
module load "${TABPFN_PYTHON_MODULE}" 2>/dev/null || true
if [ -f "${REPO_ROOT}/tabpfncreditvenv/bin/activate" ]; then
    source "${REPO_ROOT}/tabpfncreditvenv/bin/activate"
elif [ -d "${VSC_DATA}/miniforge3" ]; then
    source "${VSC_DATA}/miniforge3/etc/profile.d/conda.sh"; conda activate tabpfncreditvenv 2>/dev/null || conda activate TabPFNCredit
fi

echo "=== Pre-staging foundation-model weights into ${VSC_DATA}/TabPFNCredit/.model_cache ==="
echo "(answer any TabPFN licence prompt with 'y' -- it is saved to the shared cache)"

# Drive each foundation method through the real pipeline on a tiny CPU run so
# that whatever each TALENT wrapper downloads is fetched WITH internet and lands
# in the shared cache. Each method is isolated (one failure doesn't stop others)
# and the result is thrown away (TABPFN_RESULTS_ROOT -> a temp dir).
export TABPFN_RESULTS_ROOT="$(mktemp -d)"
python - <<'PY'
import os, numpy as np
from src.methods.method_runner import run_talent_method

# Smallest PD dataset is fine; only the WEIGHT DOWNLOAD matters, not the score.
DATASET, TASK = "0008.german", "pd"
METHODS = ["tabpfn_v3", "tabpfn_v2_5", "tabicl_v2", "tabdpt", "mitra", "hyperfast"]

for m in METHODS:
    try:
        run_talent_method(task=TASK, dataset=DATASET, method=m,
                           cv_splits=1, row_limit=200, max_epoch=1, n_trials=1)
        print(f"  [ok]   prestaged {m}")
    except Exception as exc:  # noqa: BLE001 -- best-effort; download usually happens before any failure
        print(f"  [warn] {m}: {type(exc).__name__}: {exc}")
        print(f"         (if this is a download/licence error, fix it here on the "
              f"login node; if it's a CPU/GPU error the weights likely still cached.)")
PY

echo
echo "=== Done. Cache contents: ==="
du -sh "${VSC_DATA}/TabPFNCredit/.model_cache"/* 2>/dev/null || true
echo
echo "Now submit jobs as usual; the compute nodes read this cache OFFLINE."
