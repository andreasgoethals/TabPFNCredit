# TabPFNCredit on VSC

Read [project template](https://github.com/andreasgoethals/0.-Template/blob/5bfd08215091b1e03f6c78a8ddfbbb1390afd9bd/docs/TEMPLATE.md) and [AGENTS.md](AGENTS.md). Preserve scientific
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

Use [the template’s SSH probe](https://github.com/andreasgoethals/0.-Template/blob/5bfd08215091b1e03f6c78a8ddfbbb1390afd9bd/docs/TEMPLATE.md#vsc-access-windows-and-agents): bounded
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

The supported deployment is wICE on Lustre. New submissions explicitly select wICE. Mindwell uses a separate queue. No new unpacked
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
[runtime.json](config/vsc/runtime.json) and
[requirements.txt](config/vsc/requirements.txt) record Python 3.12.3, 255
package versions, TALENT/TabPFN revisions and the library pin. Scientific packages
and installed patches are preserved. Only Click changed from 8.4.2 to compatible
8.1.8 to restore CLI startup. PyArrow/Hugging Face declaration bounds admit the
existing installed versions; those packages were not upgraded. Dependency ranges
and the revision TALENT pin in `pyproject.toml` are not a paper lockfile.

## Revision runtime

Use the leading `--revision` option for the earlier TALENT revision.
[revision-runtime.json](config/vsc/revision-runtime.json) records the SHA-256 hashes of the paper image, fork wheel and deployed revision image.
The existing launcher without this option retains the historical image and
result paths. The revision uses `runtimes/tabpfncredit-talent-9fe7d9c.sif`
and writes to `output` on PROJECT by default. Its planning
and worker caches use separate scratch workspaces. Experiments 2 and 3 are
archived for this revision.

```bash
bash scripts/vsc.sh --revision list --show-profile
# First enable the desired methods in scripts/Experiment1/config/CONFIG_METHOD.yaml.
bash scripts/vsc.sh --revision experiment Experiment1 --no-submit
```

Experiment 1 starts with every method disabled. Check the selected adapters'
runtime versions and offline weights before submitting. Expand the same
revision run later by enabling additional methods with the same runtime and
scientific settings. The original method roster is in
[PAPER_METHODS.yaml](scripts/Experiment1/config/PAPER_METHODS.yaml).

VSC build job `62886778` produced the revision image from the paper image.
CPU job `62887627` exercised a two-trial logistic regression search, an untuned
linear regression, mixed-result loading and the Experiment 1 planning command.
GPU job `62887716` fitted FT-Transformer on an A100 MIG allocation and confirmed
that the two linear models still used CPU there. Job `62888096` supplied
conflicting device overrides and verified the resulting model states on the
same type of node. MLP and the linear models used CPU, and FT-Transformer used
CUDA. All three checks used synthetic data.

## New foundation models in Experiment 1

Use `--tfm-revision` for the models validated in
[tfm-runtime.json](config/vsc/tfm-runtime.json). Enable the selected methods
for PD and LGD in `scripts/Experiment1/config/CONFIG_METHOD.yaml`, then preview
their plans.

```bash
bash scripts/vsc.sh --tfm-revision experiment Experiment1 --no-submit
```

The runtime lock records the image, [installed packages](config/vsc/tfm-requirements.txt),
and PROJECT results and checkpoint directories. Use that results directory as
the revision input to the notebook assembly
command. Keep the same profile and scientific settings when extending the run.
Use the earlier `--revision` profile for the TabDPT 1.1 backend (`tabdpt`).
Use `tabdpt_v1_3` with the new profile.

## Revision preflight on wICE

Preview `bash scripts/vsc.sh --revision experiment Experiment0 --no-submit`.
The preflight uses both tasks, the archived Experiment 1 roster, two folds
and up to 256 rows per dataset. Each dataset/method gets its own array slot when
the slot budget allows. The HPO and untuned points stay together. Remove
`--no-submit` to run the pilot and check publication receipts and missing points.
The GPU preflight requests at least 40 minutes. The first H100 T2G-Former slot
exhausted the original ten-minute request during the PD test. CPU preflight
jobs retain the ten-minute minimum, and completed jobs release resources early.
Real-TabPFN reads its finetuned classifier from the Hugging Face Hub cache.
Stage it with `python -m src.utils.fetch_weights --only tabpfn_real` on a
connected workstation before copying the checkpoint tree to PROJECT.

Set `TABPFN_WICE_GPU_POOL=1` for jobs that fit either a full A100 or H100.
Slurm can schedule the same job on either partition. The common request uses
at most 16 host cores. A100 has the smaller default host memory per core.
Use a fixed GPU partition when comparing timings on identical hardware. Keep
the hardware and Slurm partition alongside elapsed time when comparing devices.
The wICE debug A100 partition admits one job per user at a time, for up to an
hour. It suits a single failing method; ordinary partitions support a parallel
pilot. [wICE GPU partitions](https://docs.vscentrum.be/leuven/wice_quick_start.html)
and [Slurm partition selection](https://slurm.schedmd.com/sbatch.html#OPT_partition).

## Planning work on wICE

The revision planner checks the selected dataset and method outputs before
requesting resources. A completed cell requires the expected folds and prediction
archive members. Keep using the same revision results root to resume. A changed
input fingerprint or training configuration requires a separate root.

| Setting | Effect |
|---|---|
| `TABPFN_SLOT_TARGET_MINUTES` | Revision packing target, default 240 minutes. A single longer method and its dependent HPO point stay together. The array cap can also produce longer slots. |
| `TABPFN_MAX_ARRAY_SLOTS` | Maximum planned slots per partition. A revision experiment defaults to 150 and a historical experiment to 40. Smaller values pack more work per slot. |
| `TABPFN_MAX_CONCURRENT` | Optional limit on simultaneous elements of each array. When set, the planner keeps one array per partition so splitting by duration cannot multiply that limit. |
| `TABPFN_CPU_CORES_PER_TASK` | CPU job cores. The historical default is 18. |
| `TABPFN_GPU_CPUS_PER_TASK` | Host cores for a GPU job. The default retains the existing host RAM budget. A100 supports up to 18 and H100 up to 16. |

Revision jobs run one Python task per slot. Independent slots provide parallelism.
Numerical thread pools inside the revision container use the allocated core count.
Alongside Slurm's 72-hour limit, revision arrays use estimated duration bands of
1, 4, 12, 24 and 72 hours when unthrottled. Each array requests its longest
slot estimate, with a ten-minute floor and the partition maximum. This lets
shorter work qualify for backfill. Time estimates cover a whole cell, including
folds and tuning. Changing the packing target does not shorten a fit.

A local preview of the full Experiment 1 configuration on 25 November 2026
contained 1,344 points, including HPO copies. At the 40-slot budget, the
longest estimated A100 slot exceeded 72 hours. With 150 slots per partition
the planner produced 406 slots in seven arrays. The longest request was
45 h 32 min on A100. These are estimates for the current configuration;
measure actual wait, elapsed time and memory before choosing a campaign size.

On 25 November 2026 wICE reported 2,500 MiB per allocated CPU on
`batch_sapphirerapids`, 7,000 on `gpu_a100` and 11,700 on `gpu_h100`.
The container requests partition-default memory. Reducing host cores also reduces
available RAM. Compare 4 and 8 cores on the same representative input first,
and retain headroom above measured peak memory. Inspect step-level CPU time
and MaxRSS, allocation elapsed time and TALENT's separate fold timings.
For GPU workloads compare rows, device and VRAM on the same input. A
shorter walltime only improves eligibility when the whole slot fits.

wICE and Mindwell have separate Slurm controllers and fair-share associations.
The same Leuven credit account can fund both. Inspect the appropriate queue
and account balance before a campaign. [Leuven Slurm](https://docs.vscentrum.be/leuven/slurm_specifics.html)
and [credit accounting](https://docs.vscentrum.be/leuven/credits.html).

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
packing. CPU requests use partition-default memory. New plans select only wICE partitions. Historical submission/array caps remain compatibility settings,
not certified current site limits. Check actual allocation before new campaigns.

## Publication and recovery

Historical JSON/NPZ and packed JSON-shard formats remain. Historical resume checks use fold counts. Revision workers also check dataset and configuration fingerprints. Generate summaries explicitly after jobs finish. Use a separate mixed snapshot for revision figures and reports.

Workers verify durable result coverage and NPZ integrity, then write
`output/<experiment>/manifests/publications/<job>_<array>.json` with plan hash, workspace, image
and result location. Only then remove their exact owned scratch workspace.
Failures preserve ownership markers and recovery work. Publication remains per
completed work item; no new mid-method checkpoint mechanism. Submission receipts
live in `output/manifests/submissions/`.

Before migration/cleanup inspect running and pending jobs, logs, receipts and
`sacct`; freeze their source/plans/runtime. Capture scientific/private-source
hashes, packages/input/model provenance and backups; explain migrations and obtain
authorization. Never wipe shared scratch or alter another project's jobs. Retire
exact owned artifacts after consumer/publication checks and secure failed work.

## Verified alignment: 29 October 2026

The template copy was rechecked against the canonical project during the
30 October storage review. Its revision and SHA-256 are recorded with that
dated review rather than presented here as a permanently current template pin.
The October deployment used VSC Library at `0ed4ffa89d312b9bb6411681b6a14f81672439f3`.
The tested framework snapshot is `a535554`, based on
`980533a256d0fb4c349074eb451239d5220801cb`. On 30 October, the user authorized
GitHub publication and consolidation onto one branch, `main`, on the workstation,
GitHub and VSC DATA checkout. Separate deployment branches are retired after
preserving unique history in verified local recovery bundles. For future
authorized releases, test/commit/push `main`, then use `git pull --ff-only` on
the clean DATA checkout after checking its job/recovery consumers. The current
revision and synchronization checks are recorded under
`archive/logs/maintenance/vsc-20261030/`. Private modules remain ignored and separately
hash-verified; they are never added to a public commit or recovery branch push.
Previous remote history/build provenance is independently preserved locally under
`archive/logs/maintenance/vsc-20261029/`. The obsolete sibling `TabPFNCredit-builds`
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
`archive/logs/maintenance/vsc-20261029/`; large backups are ignored by Git. Final receipts
record synchronization, retired artifacts and measured storage. Consult current
`myquota` instead of treating a historic snapshot as current capacity.

## Storage review: 30 October 2026

Measured allocation was 0.84 GiB for the DATA checkout (0.81 GiB of Git history)
and 37.61 GiB on PROJECT: checkpoints 26.24 GiB, runtime 5.16 GiB, datasets
3.96 GiB and results 2.26 GiB. These are dated project measurements, not shared
allocation quotas. Keep the history, scientific assets and independent recovery
copies; no layout migration or forced Git cleanup is needed.

The separate `$VSC_DATA/miniconda3/envs/tabpfncreditvenv` prefix was an unused
Python/packaging bootstrap, not the paper runtime. Its 31 Conda packages had no
scientific dependencies, and inspected jobs, gateway processes and interactive
settings had no consumers. After a full recovery archive was copied locally and
all 8,946 source files verified, Conda removed only that prefix offline. The
shared Conda installation/cache and legacy activation interfaces remain intact;
the removed bootstrap did not provide a working benchmark environment. Existing
workstation environments are unaffected. Do not infer reclaimed quota from its
0.23 GiB directory size: most files were hard-linked to the shared package cache.

All nine PROJECT asset symlinks resolved within this project; the single SIF
remained read-only. Scientific assets and private preprocessing were preserved.
Backup and retirement ran as Slurm maintenance jobs `62474727` and `62474770`;
no experiments were run. Receipts, the local recovery archive and compact
maintenance evidence are under `archive/logs/maintenance/vsc-20261030/storage-review/`.
The owned transfer workspace was removed after verified publication; neither
native scratch tier retains project work from this review.
