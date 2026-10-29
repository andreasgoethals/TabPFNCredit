# TabPFNCredit on VSC

Read [TEMPLATE.md](TEMPLATE.md) and [AGENTS.md](../AGENTS.md). Preserve scientific
code, private preprocessing, observations, seeds/splits, settings, result names
and aggregation. Future methods require separate experiments; maintenance does
not regenerate paper outputs.

## Access

The user renews the existing Windows SSH-agent certificate manually in PowerShell:

```powershell
step ssh login --context VSC
```

Approve KU Leuven Authenticator yourself. Agents inspect validity; they never
initiate MFA, install another login, renew automatically or schedule reminders.
If expired/missing, run the command before remote work continues.

Use [TEMPLATE.md's probe](TEMPLATE.md#vsc-access-windows-and-agents): bounded
LF-stdin native `ssh.exe -F none -T`, batch mode and strict host-key checking.
Require stdout, stderr and deliberate exit 7. On 29 October the gateway dropped
raw stderr. The verified fallback sends a Python wrapper through the same channel,
captures Bash's two streams/status into JSON and exits with that status. Parse
both streams and require SSH-status agreement. Missing output is not success;
diagnose transport errors separately. Use native `scp.exe -F none` with the same
checks for transfers, then verify SHA-256. Computational checks run in Slurm.

## Active layout and commands

| Location | Contents |
|---|---|
| `$VSC_DATA/TabPFNCredit` | Actual Git checkout and `.git`, source/configuration, locks, compact plans/logs/provenance. All Git operations run here. |
| `/lustre1/project/stg_00211/TabPFNCredit` | Persistent data, checkpoints, results and immutable SIF runtime. |
| `$VSC_SCRATCH_LUSTRE1/tabpfncredit` | wICE job-owned temporary work and caches. |

The supported deployment is wICE on Lustre. The corresponding GPFS PROJECT
allocation is absent; Mindwell is not configured/validated. No new unpacked
cluster environments. PROJECT is not a backup: preserve independent copies and
check both byte and file-count quotas with `myquota`.

From the DATA checkout:

```bash
bash scripts/vsc.sh --help
# Preview: does not train or submit.
bash scripts/vsc.sh experiment Experiment0 --task pd --dataset 0008.german --method LogReg --no-submit
# Only for an authorized experiment:
bash scripts/vsc.sh experiment Experiment0
# Existing chained sweep interface, container-backed on VSC:
bash scripts/run_all_experiments.sh Experiment0 Experiment1
```

The shell wrapper loads host Python. The standard-library host launcher plans
inside Apptainer with forced `--no-submit`, then invokes host `sbatch` for the
recorded jobs. Atomic receipts record accepted/in-flight submissions; inspect an
uncertain submission before retrying. No Slurm installation inside the image.

The default PROJECT image is `runtimes/tabpfncredit-runtime-20261029-cli-fixed.sif`,
SHA-256 `cc479a249b6e808355e2cf468741713e1990fe8873e362185835dfa171f70d8a`.
[runtime.json](../config/vsc/runtime.json) and
[requirements.txt](../config/vsc/requirements.txt) record Python 3.12.3, 255
package versions, TALENT/TabPFN revisions and the library pin. Scientific packages
and installed patches are preserved. Only Click changed from 8.4.2 to compatible
8.1.8 to restore CLI startup. PyArrow/Hugging Face declaration bounds admit the
existing installed versions; those packages were not upgraded. Dependency ranges
and TALENT `main` in `pyproject.toml` are not a paper lockfile.

## Configuration and compatibility

| Variable | Meaning |
|---|---|
| `TABPFN_STAGING_ROOT` | Historical complete project directory; highest precedence. |
| `TABPFN_PROJECT_ROOT` / `VSC_PROJECT_ROOT` | Allocation root; appends `TabPFNCredit` when the historical variable is absent. |
| `TABPFN_RESULTS_ROOT` | Optional result directory; container mode requires PROJECT. |
| `TABPFN_CONTAINER_IMAGE` | Optional absolute versioned SIF under this PROJECT directory. |
| `TABPFN_CONTAINER_PYTHON` | Image interpreter; default `python`. |
| `TABPFN_APPTAINER_MODULE` | Optional site Apptainer module. |
| `TABPFN_VSC_PROFILE` | Wrapper selects `container`; explicit `legacy` retains historical full-sweep dispatch. |
| `TABPFN_CACHE_ROOT` | Legacy override; container workers isolate native scratch caches. |

Workstation defaults and legacy venv/Conda interfaces remain compatible. Lookup
order is unchanged. Container mode rejects a selected dataset outside PROJECT
instead of selecting different observations. Repo-local asset paths must be
absent or approved PROJECT links. Model loading stays offline. Mitra needs
package-internal weights; HyperFast needs its CWD-relative checkpoint via the
generated bind layout. `setup_vsc_checkpoints.sh` is for legacy writable installs.

Workers snapshot settings, validate DATA/Git/PROJECT/native scratch and forward
Slurm/GPU metadata into a clean container while preserving scientific plans and
packing. CPU requests use partition-default memory. Container mode rejects retired
Genius routes. Historical submission/array caps remain compatibility settings,
not certified current site limits. Check actual allocation before new campaigns.

## Publication and recovery

Historical JSON/NPZ and packed JSON-shard formats remain. Resume checks complete
folds but does not fingerprint scientific inputs: scientific changes need separate
result/cache identities. Generate summaries explicitly after jobs finish. Notebook
reruns regenerate paper figures/reports; do not run incidentally.

Workers verify durable result coverage and NPZ integrity, then write
`logs/manifests/publications/<job>_<array>.json` with plan hash, workspace, image
and result location. Only then remove their exact owned scratch workspace.
Failures preserve ownership markers and recovery work. Publication remains per
completed work item; no new mid-method checkpoint mechanism. Submission receipts
live in `logs/manifests/submissions/`.

Before migration/cleanup inspect running and pending jobs, logs, receipts and
`sacct`; freeze their source/plans/runtime. Capture scientific/private-source
hashes, packages/input/model provenance and backups; explain migrations and obtain
authorization. Never wipe shared scratch or alter another project's jobs. Retire
exact owned artifacts after consumer/publication checks and secure failed work.

## Verified alignment: 29 October 2026

Template matches canonical document commit `1c11bb1` byte-for-byte, SHA-256
`0ed244c16cab8ea86afab5ed90eea36eef3a9e218314c38538e227e4f009fd3c`.
VSC Library remains read-only at `0ed4ffa89d312b9bb6411681b6a14f81672439f3`.
The tested framework is recorded on the local `codex/vsc-alignment-20261029`
deployment branch, based on `980533a256d0fb4c349074eb451239d5220801cb`, and
transferred to the DATA checkout through a Git bundle. The exact deployment
revision is in the completion receipt. The workstation's working branch and
staging area are preserved; no public push is made. This private Git transport
replaces public publication for this authorized deployment. Private modules stay
ignored and are separately hash-verified; they are never added to the commit.
Previous remote history/build provenance is independently preserved locally under
`logs/maintenance/vsc-20261029/`. The obsolete sibling `TabPFNCredit-builds`
was removed after archival.
The login profile's stale automatic activation of the old project venv was
removed, preserving its Python module/account settings. Container execution uses
its independent `/opt/venv`; it does not import the retired DATA environment.

Runtime jobs `62468552`, `62468553`, `62468656`, `62468657` verified builds,
CLI, CPU probabilities/linear algebra/AUC/Brier and A100 MIG CUDA matrices:
tested numerical differences against the old environment were zero. Slurm job
`62471363` passed 149 targeted regressions. Worker `62471365` verified host
submission, existing-result resume, publication/image provenance and scratch
cleanup. Local regressions also cover legacy scripts/plans, paths, safe previews
and synthetic scientific contracts. These are infrastructure/runtime checks,
not full paper reproduction. No research campaign was launched.

All VSC result filenames existed locally. The 1,623 differing JSON/CSV versions
were downloaded to `results/history/vsc-20261029-results.tar.gz` with embedded
manifest; archive and every member were SHA-256 verified. Prediction NPZ files
were identical. Local paper results remain unchanged; VSC numerical files use
that canonical local set after verified synchronization.

Independent local SIF backups are in `runtimes/` and `runtimes/recovery/`.
Useful source/build/environment recovery records belong under
`logs/maintenance/vsc-20261029/`; large backups are ignored by Git. Final receipts
record synchronization, retired artifacts and measured storage. Consult current
`myquota` instead of treating a historic snapshot as current capacity.
