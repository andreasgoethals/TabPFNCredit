#!/bin/bash
# ============================================================================
# install.sh -- one-shot installer for TabPFNCredit (Linux / macOS / HPC)
# ============================================================================
# Installs the project + all dependencies into the *currently active*
# virtual environment, in the two steps required to work around TALENT's
# over-strict version pins (see pyproject.toml header for the full story):
#
#   1. pip install -e ".[<profile>]"   -- everything except TALENT
#   2. pip install --no-deps TALENT    -- TALENT, ignoring its lockfile
#
# Usage:
#     bash scripts/install.sh           # defaults to the "hpc" profile
#     bash scripts/install.sh local     # CPU-only workstation profile
#     bash scripts/install.sh hpc       # CUDA GPU / cluster profile
#
# Run this AFTER you have created and activated your venv, e.g.:
#     python -m venv tabpfncreditvenv
#     source tabpfncreditvenv/bin/activate
#     bash scripts/install.sh hpc
# ============================================================================

set -euo pipefail

PROFILE="${1:-hpc}"
if [[ "${PROFILE}" != "local" && "${PROFILE}" != "hpc" ]]; then
    echo "ERROR: profile must be 'local' or 'hpc' (got '${PROFILE}')." >&2
    exit 1
fi

# Land in the repo root regardless of where this was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Warn (don't fail) if no venv looks active -- installing into the system
# Python is almost never what you want.
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
    echo "WARNING: no virtual environment detected (\$VIRTUAL_ENV / \$CONDA_PREFIX unset)."
    echo "         Create + activate one first, e.g.:"
    echo "             python -m venv tabpfncreditvenv && source tabpfncreditvenv/bin/activate"
    echo "         Continuing in 3s (Ctrl-C to abort)..."
    sleep 3
fi

echo "==> [1/3] Upgrading pip"
python -m pip install --upgrade pip

echo "==> [2/3] Installing TabPFNCredit + deps  (profile: ${PROFILE})"
pip install -e ".[${PROFILE}]"

echo "==> [3/3] Installing TALENT with --no-deps (ignores its over-strict pins)"
pip install --no-deps "TALENT @ git+https://github.com/LAMDA-Tabular/TALENT@main"

echo
echo "Done. Verify with:  tabpfncredit doctor"
