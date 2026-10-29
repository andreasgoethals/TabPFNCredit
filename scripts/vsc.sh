#!/bin/bash -l
# Container entry point; Slurm RPCs stay on the host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v module >/dev/null 2>&1; then
    module load "${TABPFN_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
fi
exec python3 "$ROOT/scripts/vsc.py" "$@"
