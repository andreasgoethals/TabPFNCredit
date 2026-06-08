#!/bin/bash -l
# ============================================================================
# setup_vsc_checkpoints.sh -- provision uploaded weights on the VSC (OFFLINE)
# ============================================================================
# Run this ONCE on the VSC after you have:
#   1. Downloaded weights locally with `python scripts/fetch_weights.py`, and
#   2. Uploaded the resulting `checkpoints/` folder to EITHER this repo's root
#      (`$VSC_DATA/TabPFNCredit/checkpoints/`) or the shared project storage
#      (`$TABPFN_STAGING_ROOT/checkpoints/`, default
#      `/staging/leuven/stg_00211/checkpoints/`). This script auto-detects
#      which one is populated (repo first, then project storage); override the
#      location with `TABPFN_CHECKPOINTS_DIR=/path/to/checkpoints`.
#
#   cd "$VSC_DATA/TabPFNCredit"
#   bash scripts/setup_vsc_checkpoints.sh
#
# What it does
# ------------
# MOST foundation models read their weights from a cache directory, so they
# work offline as soon as the generated SLURM scripts point HF_HOME /
# TABPFN_MODEL_CACHE_DIR at `checkpoints/` (they do, and set HF_HUB_OFFLINE=1).
# Nothing to do for those -- the upload is enough.
#
# A FEW models instead load from a fixed *path* (not a cache), so their weights
# must be physically placed where their loader looks:
#   * Mitra      -> <installed TALENT pkg>/model/models/models_mitra/{cls,reg}/
#                   (Tab2D.from_pretrained reads config.json + model.safetensors)
#   * HyperFast  -> <repo>/model/models/hyperfast/hyperfast.ckpt
#                   (its loader uses a CWD-relative path; jobs cd to the repo)
# This script copies those from checkpoints/talent_assets/ into place. It is
# idempotent (safe to re-run) and never touches the network.
#
# It does NOT modify the TALENT source -- it only drops data files into the
# already-installed package, exactly where the package's own loaders expect
# them (the upstream package ships some such weights itself; these two just
# are not bundled yet).
# ============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Locate the uploaded checkpoints: an explicit $TABPFN_CHECKPOINTS_DIR wins;
# otherwise prefer a populated repo-local checkpoints/, else fall back to the
# shared project storage -- the same repo-first-then-staging order the SLURM
# jobs use at run time.
STAGING_ROOT="${TABPFN_STAGING_ROOT:-/staging/leuven/stg_00211}"
if [ -n "${TABPFN_CHECKPOINTS_DIR:-}" ]; then
    CKPT="$TABPFN_CHECKPOINTS_DIR"
elif [ -d "$REPO_ROOT/checkpoints" ] && [ -n "$(ls -A "$REPO_ROOT/checkpoints" 2>/dev/null)" ]; then
    CKPT="$REPO_ROOT/checkpoints"
else
    CKPT="$STAGING_ROOT/checkpoints"
fi
ASSETS="$CKPT/talent_assets"

echo "=== TabPFNCredit checkpoint provisioning ==="
echo "repo:        $REPO_ROOT"
echo "checkpoints: $CKPT"

if [ ! -d "$CKPT" ]; then
    echo "ERROR: $CKPT not found. Upload the 'checkpoints/' folder produced by" >&2
    echo "       'python scripts/fetch_weights.py' to the repo root first." >&2
    exit 1
fi

# --- Activate the project env so we can locate the installed TALENT package --
module purge 2>/dev/null || true
: "${TABPFN_PYTHON_MODULE:=Python/3.12.3-GCCcore-13.3.0}"
module load "${TABPFN_PYTHON_MODULE}" 2>/dev/null || true
if [ -f "$REPO_ROOT/tabpfncreditvenv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/tabpfncreditvenv/bin/activate"
fi

# --- Locate the installed TALENT package directory ---------------------------
TALENT_DIR="$(python -c 'import os, TALENT; print(os.path.dirname(TALENT.__file__))' 2>/dev/null || true)"
if [ -z "$TALENT_DIR" ]; then
    echo "ERROR: could not import TALENT. Activate the project venv / install first." >&2
    exit 1
fi
echo "TALENT pkg:  $TALENT_DIR"
echo

status=0

# --- 1) Mitra: copy into the installed TALENT package ------------------------
provision_mitra() {
    local sub="$1"   # cls | reg
    local src="$ASSETS/models_mitra/$sub"
    local dst="$TALENT_DIR/model/models/models_mitra/$sub"
    if [ -f "$src/model.safetensors" ] && [ -f "$src/config.json" ]; then
        mkdir -p "$dst"
        cp -f "$src/config.json" "$src/model.safetensors" "$dst/"
        echo "  [ok]   Mitra ($sub) -> $dst"
    else
        echo "  [skip] Mitra ($sub): not in checkpoints (config.json+model.safetensors missing)."
        echo "         (fine if you did not fetch 'mitra'; otherwise re-run fetch_weights.py)"
    fi
}
echo "Mitra:"
provision_mitra cls
provision_mitra reg

# --- 2) HyperFast: copy to the repo-root CWD-relative path -------------------
echo "HyperFast:"
HF_SRC="$ASSETS/hyperfast/hyperfast.ckpt"
HF_DST="$REPO_ROOT/model/models/hyperfast/hyperfast.ckpt"
if [ -f "$HF_SRC" ]; then
    mkdir -p "$(dirname "$HF_DST")"
    cp -f "$HF_SRC" "$HF_DST"
    echo "  [ok]   HyperFast -> $HF_DST"
else
    echo "  [skip] HyperFast: $HF_SRC missing (fine if you did not fetch 'hyperfast')."
fi

# --- 3) Sanity: report the cache-based weights that need no copy -------------
echo
echo "Cache-based weights (no copy needed; read via HF_HOME / TABPFN_MODEL_CACHE_DIR):"
for d in "huggingface/hub" "tabpfn"; do
    if [ -d "$CKPT/$d" ] && [ -n "$(ls -A "$CKPT/$d" 2>/dev/null)" ]; then
        echo "  [ok]   $CKPT/$d  ($(du -sh "$CKPT/$d" 2>/dev/null | cut -f1))"
    else
        echo "  [warn] $CKPT/$d is empty -- TabPFN/TabICL/TabDPT may fail offline."
        status=1
    fi
done

echo
if [ "$status" -eq 0 ]; then
    echo "Done. Submit experiments as usual; compute nodes read everything offline."
else
    echo "Done with WARNINGS (see above). Re-run scripts/fetch_weights.py locally if needed."
fi
exit "$status"
