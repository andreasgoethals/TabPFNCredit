# ============================================================================
# install.ps1 -- one-shot installer for TabPFNCredit (Windows PowerShell)
# ============================================================================
# Installs the project + all dependencies into the *currently active*
# virtual environment, in the two steps required to work around TALENT's
# over-strict version pins (see pyproject.toml header for the full story):
#
#   1. pip install -e ".[<profile>]"   -- everything except TALENT
#   2. pip install --no-deps TALENT    -- TALENT, ignoring its lockfile
#
# Usage (after creating + activating your venv):
#     py -3.12 -m venv tabpfncreditvenv
#     .\tabpfncreditvenv\Scripts\Activate.ps1
#     .\scripts\install.ps1            # defaults to the "local" profile
#     .\scripts\install.ps1 local      # CPU-only workstation profile
#     .\scripts\install.ps1 hpc        # CUDA GPU profile
# ============================================================================

param([string]$Profile = "local")

$ErrorActionPreference = "Stop"

if ($Profile -ne "local" -and $Profile -ne "hpc") {
    Write-Error "Profile must be 'local' or 'hpc' (got '$Profile')."
    exit 1
}

# Land in the repo root regardless of where this was invoked from.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not $env:VIRTUAL_ENV -and -not $env:CONDA_PREFIX) {
    Write-Warning "No virtual environment detected. Create + activate one first, e.g.:"
    Write-Warning "    py -3.12 -m venv tabpfncreditvenv; .\tabpfncreditvenv\Scripts\Activate.ps1"
    Start-Sleep -Seconds 3
}

Write-Host "==> [1/3] Upgrading pip"
python -m pip install --upgrade pip

Write-Host "==> [2/3] Installing TabPFNCredit + deps  (profile: $Profile)"
pip install -e ".[$Profile]"

Write-Host "==> [3/3] Installing TALENT with --no-deps (ignores its over-strict pins)"
pip install --no-deps "TALENT @ git+https://github.com/LAMDA-Tabular/TALENT@main"

Write-Host ""
Write-Host "Done. Verify with:  tabpfncredit doctor"
