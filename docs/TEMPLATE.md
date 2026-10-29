# Research repository template

Andreas Goethals · KU Leuven · machine learning for tabular data
Contact: andreas.goethals@kuleuven.be · [Canonical template](https://github.com/andreasgoethals/0.-Template)

**A starting point, not a compliance contract.** Keep project-specific decisions in the project's
README/docs. Adapt deliberately and explain deviations. This file contains the portable policy;
copying it into an existing repository does not authorize moving data or changing active campaigns.

## Layout

```text
├── config/experiment0/example.yaml    replace with validation configs
├── data/
│   ├── raw/                          immutable inputs, outside Git
│   └── processed/                    reusable, rebuildable cache
├── docs/
│   ├── TEMPLATE.md                   this policy
│   ├── VSC.md                        project commands, storage and measured settings
│   ├── CHANGELOG.md                  short dated changes
│   └── AGENTS_MEMORY.md              run evidence and reusable lessons
├── notebooks/experiment0/example_analysis.ipynb
├── output/
│   ├── All_Results.md                all final notebook summaries
│   ├── CAPTIONS.md                   all paper captions
│   ├── manifests/submissions/        generated submission receipts; preserve
│   └── experiment0/                  add experiment1, experiment2, ... as needed
│       ├── figures/<notebook>/       PDF figures and report metadata
│       ├── logs/                     compact .log files
│       ├── manifests/                resolved configuration and provenance
│       └── results/                  retained numerical results
├── scripts/slurm/job.slurm           experiment launcher; adapt resource/runtime TODOs
├── src/
│   ├── data/loaders.py               project-defined loading/preprocessing
│   ├── utils/
│   │   ├── paths.py                  storage policy and path resolution
│   │   ├── atomic.py                 complete-file text replacement
│   │   ├── clean_run.py              preview/cleanup of disposable output
│   │   ├── run_notebooks.py          notebook subprocesses and aggregate reports
│   │   ├── submit_jobs.py            resumable submission of independent ready jobs
│   │   ├── logging_setup.py          console and .log output
│   │   ├── update_library.py         inspect/update one registered library pin
│   │   └── config.py                 project-defined YAML loader
│   └── visualize/
│       ├── style.py                  shared project style, A4 dimensions
│       └── figures.py                PDF FigureSaver
├── tests/                           behavioral tests, not template compliance
├── vsc-library/                     pinned read-only reference
├── _template/                       initializer and adoption guide; remove after setup
├── .github/dependabot.yml            VSC pin-update proposals
├── .vscode/settings.json             imports, notebooks, tests and library protection
├── .gitignore  .gitattributes  .gitmodules
└── AGENTS.md  CLAUDE.md  README.md  LICENSE  pyproject.toml
```

`__init__.py` files mark packages; `.gitkeep` retains empty directories. Create extra source
packages as needed. `checkpoints/`, `runtimes/` and `cache/` are generated asset namespaces,
not tracked scaffolding; their cluster locations are on PROJECT.

## `config/`

YAML files may describe runs or sweeps and may be flat or nested within an experiment.
Schema and loader are project-defined. Save the resolved configuration with each run.

### Experiments

Default to matching **`experiment0`, `experiment1`, `experiment2`, ...** folders under
**`config/`, `notebooks/` and `output/`**, with experiment first. Add only experiments that exist.
Shared `general/` folders, flat small projects and established layouts remain valid deviations.
Preserve existing results, run identities and resume paths when adopting this organization.

**Experiment 0 is VSC validation**, before expensive research: exercise relevant input/config,
CPU/GPU, numerical/output, resource and recovery behavior on the actual target environment.
Split into `experiment0/0a/`, `experiment0/0b/`, ... as useful, mirroring stages across all three
trees. Define coverage, pass criteria and commands in `docs/VSC.md`; keep evidence with outputs
and outcomes in agent memory. Revalidate affected behavior after changes. Local tests, a
synthetic example notebook or Slurm exit 0 alone do not establish VSC acceptance.

## `data/`

- `raw/`: immutable datasets; never commit, force-add, clean automatically or paste private rows into docs.
- `processed/`: reusable derived inputs; delete only when deliberately rebuilding them.
- Weights and resumable checkpoints belong in `checkpoints/`; reusable caches in `cache/`.
  Use `paths.py` so these resolve to PROJECT on VSC. Keep assets outside Git and retain licences.

## `docs/`

Use Markdown and uppercase filenames. `TEMPLATE.md` is generic; the other files describe this project.

- **`CHANGELOG.md`**: newest date first, `DD-MM-YYYY`; one concise bullet per substantive change.
  Be as short as possible while remaining complete; large changes may need more explanation.
- **`AGENTS_MEMORY.md`**: read before working. Keep compact run records (config/revision, job IDs,
  outcome, key result/evidence path), failed attempts and other reusable lessons. Record uncertainty;
  submission is not completion. Keep it organized, newest first, without losing useful evidence.
- **`VSC.md`**: fill in this project's paths, runtime specifications, quotas, cluster/partition,
  measured resources, validation gates, submissions, publication, resume, cleanup and backup steps.
  Shared infrastructure references remain in the read-only VSC Library.

### VSC access: Windows and agents

**The user renews access**, normally at the start of work, in Windows PowerShell:

```powershell
step ssh login --context VSC
```

Complete the browser sign-in and KU Leuven Authenticator approval. The tested lifetime is about
16 hours; inspect the certificate's actual validity below. Agents must not schedule reminders,
open login flows, enable automatic renewal hooks or bypass MFA. If renewal is needed during a
task, tell the user to run this command and wait for confirmation before retrying.

**One-time setup**, performed by the user or with explicit installation permission, following
[VSC Windows agent](https://docs.vscentrum.be/accounts/ssh_agent_windows.html) and
[Smallstep](https://docs.vscentrum.be/accounts/mfa_login_smallstep.html) instructions:

1. Use the native Windows OpenSSH **client**, not Git's bundled SSH or Pageant. If absent, install
   the client using [Microsoft's instructions](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse).
   The elevated PowerShell command is `Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0`.
   In an elevated PowerShell, enable the agent service once:

   ```powershell
   Set-Service -Name ssh-agent -StartupType Automatic
   Start-Service ssh-agent
   Get-Service ssh-agent
   ```

2. Install Smallstep, then bootstrap under the Windows account that will run the agents:

   ```powershell
   winget install Smallstep.step-ca
   step ca bootstrap --context VSC --team VSC --team-url=https://hpc.vub.be/_static/VSC-CA.json
   ```

3. Run the manual login command above. Native clients under the **same Windows account** can use
   that agent's valid certificate when permitted to reach its service. Other machines, cloud agents,
   WSL and separate SSH agents do not automatically inherit it. Never export private keys to bridge this.
   Verify the server host key against a trusted VSC source before accepting it on first connection;
   retain `known_hosts` verification. Do not install the optional Smallstep auto-login `ProxyCommand`.

Check certificates locally; compare the relevant VSC certificate's **Valid** interval with the
current time (including timezone). This does not renew access; keep identities/certificate output private:

```powershell
$openSsh = Join-Path $env:WINDIR 'System32\OpenSSH'
Get-Date -Format o
& "$openSsh\ssh-add.exe" -l
& "$openSsh\ssh-add.exe" -L | Select-String '\S+-cert-v01@openssh.com' | ForEach-Object {
    $_.Line | & "$openSsh\ssh-keygen.exe" -L -f -
}
```

**Direct inspection:** after a bounded read-only check, agents read relevant VSC files/logs/results,
quotas and Slurm state themselves instead of routinely requesting pasted terminal output. Keep reads
scoped and output bounded. Computational debugging belongs in Slurm within existing authorization
and resource limits; access grants no unrelated experiments, cancellations, downloads or deletions.

**Observed gateway behavior:** during setup, `ssh host command` returned no output with exit 0.
An LF-delimited script on stdin to native `ssh.exe -T`, **without a trailing remote command**, worked.
Treat this as an observed route-specific issue, not universal SSH behavior. A PowerShell example
using Python's standard library preserves LF bytes and bounds the whole check to 30 seconds:

```powershell
@'
import os, subprocess, sys
from pathlib import Path
ssh = Path(os.environ["WINDIR"]) / "System32/OpenSSH/ssh.exe"
target = "VSC_USER@login.hpc.kuleuven.be"  # Replace locally; never commit personal access details.
args = [str(ssh), "-F", "none", "-T", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=2", target]
# A deliberate failure checks stdout, stderr and remote exit propagation without changing files.
script = b"printf 'VSC_ACCESS_OK\\n'\nprintf 'VSC_ERROR_CHECK\\n' >&2\nexit 7\n"
p = subprocess.run(args, input=script, capture_output=True, timeout=30)
sys.stdout.buffer.write(p.stdout)
sys.stderr.buffer.write(p.stderr)
if p.returncode != 7 or b"VSC_ACCESS_OK" not in p.stdout or b"VSC_ERROR_CHECK" not in p.stderr:
    raise SystemExit(f"Unverified SSH command channel (status {p.returncode}); diagnose before work.")
'@ | python -
```

`-F none` avoids config hooks that could launch MFA; use an explicit hostname/user and a reviewed
noninteractive route if a proxy is required. Host-key checks remain enabled. The probe intentionally
expects **remote exit 7**; ordinary read-only scripts should use `set -eu`, explicit completion/status
markers and checked exit status. Keep stdout/stderr separate; never equate empty output/exit 0 with success.
Expired/missing certificates or `Permission denied` require authentication diagnosis (renew only when
needed); DNS/timeouts/refused connections indicate network/service issues, not proof of expiry.
Host-key failures require trusted verification. Missing markers or a remote command error require channel/
command diagnosis. Do not retry uncertain mutations. Check live VSC status/firewall guidance when relevant.

### Development and VSC validation loop

1. **Edit locally**, then run relevant local tests, lint and any affected notebooks. Fix failures
   before deployment; keep dependency/container specifications reproducible in Git.
2. **Review, commit and push** the tested revision when authorized. Record its commit and library
   pins. Do not use ad hoc remote edits or untracked file copies as the deployed implementation.
3. **Use SSH to update the DATA checkout** with `git pull --ff-only`; first verify its branch,
   clean state and that no active/queued/resumable job needs its current revision. Otherwise use
   a separate DATA checkout. Verify the remote commit equals the intended published revision and
   inspect pinned submodules before testing; never pull into a checkout still used by jobs.
4. **Test extensively on the actual VSC target through Slurm**, within the task's authorization.
   Local passes do not replace container/architecture, CPU/GPU, native I/O, numerical, integration,
   checkpoint/restart and publication tests relevant to the change. Define coverage and pass criteria;
   start bounded, then expand after gates pass. Parallelize independent checks under the speed policy.
   Keep dependencies in PROJECT containers, retained assets/results on PROJECT and temporary work
   on native SCRATCH. Consult the read-only VSC Library and current official guidance when unsure.
5. **Inspect the evidence directly**: read `.log` files, stderr, Slurm states/exit codes, resource
   use and required artifacts for every submitted test. Investigate warnings, incomplete coverage
   and failed/missing outputs; submission or exit 0 alone is insufficient. Record revision, image
   hash, configs, job IDs, results and unresolved limits in manifests and agent memory. Fix locally
   and repeat this loop, preserving successful evidence and compatible resume state.
6. **Finish with cleanup**, not abandoned files: track temporary checkouts, scripts, transfers,
   test artifacts and workspaces created on every tier. Publish/verify retained evidence first,
   confirm no active/queued/resumable consumer, then remove only the exact obsolete owned paths.
   Keep a compact ledger of cleanup completed or deferred, with the reason and next action.
   Preserve required DATA records, PROJECT assets/containers and backups; never wipe shared roots.

Use the task's existing compute/install/migration authorization; scoped cleanup of its own disposable
work is part of finishing it, after the checks above. A documentation-only change needs local document
checks rather than an unrelated VSC campaign. Report local and actual VSC
validation separately; if required cluster testing is blocked or unauthorized, state what remains.

### VSC storage policy

These are **our project choices**, informed by VSC guidance, not universal VSC storage defaults.

| Location | Project policy |
|---|---|
| **DATA** | Actual Git checkout including `.git`, code, configs, environment specifications and compact operational logs/manifests/reports. Clone, pull and submodule updates happen here. |
| **PROJECT** | Persistent datasets, weights, checkpoints, retained results, large reusable caches and container images containing installed venv/Conda dependencies. Outside Git; configurable allocated roots; no fallback to DATA. |
| **SCRATCH (GPFS and Lustre)** | Temporary isolated workspaces, intermediates, caches and offloading only. No obsolete workspaces; aim for empty project scratch between workloads when no job or recovery needs it. Never the sole copy of important files. |

- **Freeze the checkout used by active or queued jobs**, including submodule pins and environment
  versions. Use a separate DATA checkout for a new revision, or wait for all consumers to finish.
- Set `<PROJECT_UPPER>_PROJECT_ROOT` or `VSC_PROJECT_ROOT` to an allocated **shared root**;
  the resolver appends the project name. Legacy `*_STAGING_ROOT` names remain aliases.
  Set roots in the job or pass them explicitly; Leuven does not inherit arbitrary shell exports.
  Missing, invalid or unwritable large-asset destinations fail instead of redirecting writes.
- Use **GPFS for intensive Mindwell I/O, Lustre for wICE**. Suitable native PROJECT assets may
  be used directly; no mandatory full copy to scratch. If persistent assets are cross-mounted,
  stage only the required working set to native storage and verify it before compute.
  Check node accessibility, particularly GPFS access from older wICE nodes.
- Isolate writable databases/caches per job, or use a tested filesystem-compatible locking and
  publication protocol. Never copy a live SQLite database as an ordinary file. Compact formats
  reduce file counts but do not make concurrent writes safe automatically.
- **Always containerize project dependencies on PROJECT**: versioned immutable Apptainer/SIF images
  keep installed venv/Conda files inside the image because PROJECT has a strict file-count limit.
  No unpacked dependency trees or persistent container sandboxes there. Recreate from recorded
  definitions/locks for the destination; do not relocate populated environments. Validate architecture,
  host modules/drivers, dependencies and real CPU/GPU behavior before retiring originals; preserve
  versions needed by active/queued/resumable jobs. Build caches are temporary on SCRATCH. Local laptop
  venvs remain valid; this mandatory container rule is our cluster policy, not a universal VSC requirement.
- Check **byte and file-count quotas** on every relevant tier before building/staging/running.
  Personal DATA is backed up; verify the allocation's PROJECT protection separately and keep
  independent backups of irreplaceable outputs. Git and scratch are not those backups.
- Publish required results/checkpoints to persistent storage **periodically and at completion**;
  verify content and write the completion pointer last. A normal-exit trap is insufficient.
  Record scratch workspace ownership and publication evidence in DATA manifests. Remove only the
  exact owned workspace after verified publication **and** confirmation that no active, queued
  or resumable job needs it. Failed/incomplete work stays identified for recovery, not forgotten.
  Inspect both GPFS and Lustre project scratch after workloads; retire obsolete owned workspaces.
  Never interpret the empty-between-workloads aim as permission to delete active or recoverable work.
  Never wipe a shared scratch root or rely on
  site cleanup/periodic touching to manage it; node-local scratch can disappear at job end.

### Speed-first execution: Open OnDemand and Slurm

Optimize **elapsed time to verified results**, not token usage or minimum credits, within the
authorized budget. Open OnDemand is an interface to scheduled resources; an editor allocation
does not add queue priority. Submit campaigns as batch jobs, keep heavy work off login nodes,
and stop unused interactive allocations.

- **Submit independent ready jobs together, with no artificial concurrency limit by default.**
  Use `--array=0-9`, not a percentage throttle. No application pool, `--wait`, dependency or
  small batch chosen solely to serialize independent work. Optional `%N`/other throttles are
  for explicit user requests or a proposed, evidence-backed storage/service/correctness limit.
- Preserve genuine preparation/validation gates, training-before-evaluation and producer/consumer
  dependencies. Use same-cluster `afterok` where suitable; verify cross-cluster hand-offs explicitly.
- Especially split separable CPU tests/evaluations: ten balanced roughly two-hour jobs can replace
  one twenty-hour job. Preserve scientific splits/seeds and exact shard coverage; validate merges.
  Batch tiny items to amortize startup. Two hours is an example, not a walltime rule.
- **Slurm/site limits** govern available resources, priority, account/QoS running resources,
  submitted pending+running jobs and array size. Inspect live limits; do not hard-code a universal
  cap. When submission capacity is full, automatically refill as slots become available.
  Split plans into site-admissible arrays with stable work IDs; do not split by an arbitrary
  desired number of running jobs. `%N` does not reduce the submitted-job count.
- Request measured CPUs, RAM, GPUs and realistic walltime **per task**. Short suitable jobs can
  backfill; oversized allocations can queue longer. Worker processes × numerical threads must
  fit the allocation; more Slurm tasks/GPUs do not parallelize code that cannot use them.
  Leuven currently warns against combining explicit memory with `--cpus-per-task` (`srun` can fail).
  The launcher requests CPUs and partition-default memory: verify `DefMemPerCPU`/limits and actual
  allocation. An explicit-memory variant must remove the CPU directive and validate its thread budget.
- Give shards distinct outputs and temporary paths; use one validated merge writer. Measure
  queue/setup/run/merge time and I/O. For a demonstrated bottleneck, explain evidence and propose
  a targeted limit. Preserve successful work; never duplicate a pending or ambiguously submitted job.

The template's `submit_jobs.py` has no running-job cap. Its **explicit site submission ceiling**
is a queue-admission setting, verified for the supplied user/account/QoS/cluster scope, not a
project speed preference. It polls every 60 seconds while full/unavailable, records accepted IDs,
and resumes without resubmitting them. RPCs are serialized only to record acceptance safely;
jobs are not awaited. The plan contains ready `sbatch` argument lists, with no shell evaluation.
Preview is default; `--submit` requires authorization. Unthrottled arrays split to available
submission slots, preserving their original task IDs; IDs must fit the site's MaxArraySize.
Declare arrays in the plan, not hidden script/environment defaults. Select work from a stable
task-ID manifest, never a chunk's `SLURM_ARRAY_TASK_COUNT`/`MIN`/`MAX`, which change when split.
Explicitly throttled arrays wait for whole-array admission; oversized ones require a reviewed
split so their requested running limit is not silently multiplied.
Broader association/partition limits may still reject admission: definite capacity refusals retry;
generic QoS/resource errors or uncertain acceptance stop for reconciliation, never blind retries.
Keep the submitter alive while filling the queue; restart with the same plan/receipt after interruption.

`run_notebooks.py` uses available CPUs (and Slurm's per-task budget), with one numerical thread
per notebook by default; `--threads` and optional lower `--workers` share that budget. This is
**allocation safety**, not a cap on independent Slurm jobs. Its 1,800-second per-notebook timeout
is adjustable with `--timeout`. There is no four-worker default.

Access/storage/scheduling reviewed 29-10-2026. VSC Library pin
`0ed4ffa89d312b9bb6411681b6a14f81672439f3`: `guides/HPC_AND_ACCESS.md`, `STORAGE.md`,
`SOFTWARE.md`, `SLURM.md`; snapshot `source/accounts/{mfa_login_smallstep,ssh_agent_windows}.rst` and
`source/leuven/{slurm_specifics,tier2_hardware/parallel_filesystems}.rst`.
Live sources: [storage](https://docs.vscentrum.be/leuven/tier2_hardware/parallel_filesystems.html),
[Leuven Slurm](https://docs.vscentrum.be/leuven/slurm_specifics.html),
[arrays](https://slurm.schedmd.com/job_array.html),
[containers](https://docs.vscentrum.be/compute/software/installing_software/containers.html).
Library auto-login, DATA-environment and throttled-array examples are **not our project policy**;
adapt them in the project, never edit the reference submodule.

## `notebooks/`

Thin reports of finished computation: import logic from `src/`; no `def` or `class`.
Use the same experiment/stage folders as configs. Pass a relative path without `.ipynb`
to `FigureSaver`, e.g. `experiment0/0a/report`, so equal filenames cannot overwrite each other.
Display figures inline; save PDFs through the saver. The final code cell prints the complete
text report in notebook section order. Replace the synthetic example when real reports exist.

## `output/`

Use the resolver on every tier. Experiment/stage folders precede these categories:

- `logs/`: `.log` only; useful debugging context, IDs, paths, shapes, timings and failures.
  Keep compact and avoid secrets/raw rows; large detailed traces belong on PROJECT.
- `manifests/`: resolved config, code/library/runtime identities, progress and publication evidence.
- `results/`: retained scores, predictions and detailed numerical evidence on PROJECT; protected
  from routine cleanup. Archive/retire deliberately with backups, not as incidental "clean run" output.
- `figures/<notebook>/`: one PDF per figure, `_figures.json` captions, retained `_stdout.txt`
  final summary and `_failure.txt` when incomplete. Each saver clears only its own generated files.

Two project-wide files stay directly under `output/`: **`CAPTIONS.md`**, grouped alphabetically
by relative notebook path and then drawing order; **`All_Results.md`**, one verbatim final-summary
block per relative notebook path, alphabetically sorted, retaining notebook section order.
Captions are paper-ready descriptions of axes/data/conditions; interpretation belongs in the paper.
The global `manifests/submissions/` holds submission receipts/locks and is protected from cleanup.

## `scripts/`

Long experiment entrypoints, Bash and Slurm launchers belong here; unrelated maintenance utilities
belong in `src/utils/`. `slurm/job.slurm` takes explicit DATA checkout, shared PROJECT root,
PROJECT SIF image, expected Git commit and Python arguments. It rejects unpacked runtimes, checks revision/dirty state,
uses isolated scratch caches, appends `.log` output, and never installs, pulls, checkpoints,
requeues or deletes workspaces automatically. The project implements and tests publication/resume.
Fill resource/module TODOs; adapt paths and experiment arguments together. Slurm Bash is Linux;
local commands use PowerShell 5.1 (no `&&`, ternary or `??`).

## `src/`

Importable logic; add packages freely. Generic defaults are editable, not a mandatory framework.

### `utils/`

- `paths.py`: central resolver; DATA reports, PROJECT assets, SCRATCH workspaces. Off-cluster,
  unset persistent roots resolve inside the repo; local scratch uses `output/workspaces/`.
  `results_dir("scores.csv", experiment="experiment1")`, `logs_dir("experiment1")` and
  `manifests_dir("experiment1")` align outputs; stage IDs such as `experiment0/0a` also work.
  `config_path("experiment1/baseline")` locates YAML; `figures_dir` mirrors notebook parents.
  `raw_dir`, `processed_dir`, `checkpoints_dir`, `cache_dir`, `runtimes_dir` use PROJECT;
  `workspace_dir(run_id)` names disposable work. `resolve_writable` probes and fails without fallback.
- `atomic.py`: sibling-temp then replace for text/metadata; not locking or a multi-file transaction.
- `clean_run.py`: list by default, delete only with `--clean`; optional `--processed` rebuilds
  processed inputs. Across all experiments/tiers, preserve `results/`, `submissions/`, `workspaces/`,
  raw inputs, checkpoints, runtime images/environments, reusable `cache/`, libraries and markers.
  Reject linked paths/junctions and validate targets first. Stop writers and retain needed logs
  before cleaning; scratch cleanup needs its separate publication/ownership check.
- `run_notebooks.py`: recursively discover nonhidden Python notebooks, execute in subprocesses,
  capture final-cell stdout and rebuild reports. Partial reruns preserve other reports; failures
  are explicit. `--only experiment1/report` selects an ID. Only `%matplotlib inline` is ignored;
  kernel-specific features need a project Jupyter runner.
- `submit_jobs.py`: preview/submit a ready-job plan, maintain receipts and refill site capacity;
  see the VSC runbook for usage and failure recovery. No scientific workflow or gate is inferred.
- `logging_setup.py`: console and `.log`; call `setup(experiment="experiment1", run_id="...")`.
- `update_library.py`: inspect by default; explicit update/commit for one registered pin, never push.
- `config.py`: intentionally empty; implement the project's YAML schema/loader.

### `visualize/`

`style.py` centralizes sizes, typography, grid and colours **within this project**. The template
supplies A4 geometry/print settings; the project defines its own look/palette. Use `WIDTH_FULL`,
`WIDTH_HALF` or `WIDTH_THIRD` with `figsize`; make labels legible at final paper size and avoid
rescaling that shrinks text. Split dense panels; check grayscale/colour accessibility.
`figures.py` supplies `FigureSaver`: clear only the notebook's owned files before drawing,
number PDFs in drawing order, record captions atomically and leave figures open for inline display.

### `data/`

`loaders.py` is intentionally empty. Use the resolver; publish cache completion markers last,
so interrupted preparation cannot look complete.

## `tests/`

As many meaningful tests as needed, not one per module. Headers explain intent. Defaults cover
storage boundaries, cleanup, container-only launch/exit handling, concurrency budgets, mocked Slurm admission/recovery, notebook
reports, PDF/A4 behavior, atomic writes, initialization and library updates. `conftest.py` sets
imports/headless plotting and isolates storage in temporary directories. No real jobs are submitted.
There is no test CI/hook: agents run `python -m pytest -q` and `python -m ruff check .` before delivery,
plus `python -m src.utils.run_notebooks` after notebook/visualization changes. Local passes do not
certify VSC/runtime compatibility. Do not lint, format or test shared library contents.

## Shared library submodules

The template includes **VSC Library**; TFM/DM are independent optional references, not runtime packages.

| Library | Repository | Mountpoint / purpose |
|---|---|---|
| VSC | [VSC_Library](https://github.com/andreasgoethals/VSC_Library) | `vsc-library/`: infrastructure guides and Slurm examples |
| TFM | [TFM_Library](https://github.com/andreasgoethals/TFM_Library) | `tfm-library/`: tabular-model literature and upstream implementations |
| DM | [DM_Library](https://github.com/andreasgoethals/DM_Library) | `dm-library/`: decision-model literature and upstream implementations |

Initialize registered pins from the **DATA checkout** (locally, from the local repo):

```powershell
git submodule update --init -- vsc-library
git submodule status
```

To add a missing library, use only the needed command:

```powershell
git submodule add -b main https://github.com/andreasgoethals/VSC_Library.git vsc-library
git submodule add -b main https://github.com/andreasgoethals/TFM_Library.git tfm-library
git submodule add -b main https://github.com/andreasgoethals/DM_Library.git dm-library
```

`.gitmodules` records URLs; gitlinks pin exact revisions. Review and commit only the registration
and chosen mountpoint, then push when authorized. Never update a checkout used by active/queued jobs.
Read papers/snapshots in place, cite paths/symbols rather than snapshot line numbers, and record pins
with dependent evidence. Consumer submodules are read-only: no edits, refreshes, linting or tests.
Only TFM/DM's ignored `PROJECT_SPECIFIC.md`, made from their own template, is an exception;
durable project decisions belong in the parent's docs. VSC has no local-file exception.

Corrections belong in canonical `0. Literature` checkouts. That workspace's `scripts/maintain.py`
maintains libraries; after review/commit/push, `scripts/propagate_to_downstream.py --dry-run`
previews pin propagation and `--push` publishes it. It discovers existing registrations, never
adds unrequested libraries; pass `--root` for consumers outside the searched Projects tree.
For one consumer: `python -m src.utils.update_library --library vsc-library --update --commit`,
then an authorized `git push`. Alternatively update with `git submodule update --remote -- <mountpoint>`
and review `git diff --submodule=log`. Preserve local changes and read the library changelog first.

## Root files and tooling

- `README.md`: downstream project front page; retain only the short template-credit chapter at its end.
- `AGENTS.md`: shared agent instructions. `CLAUDE.md` imports it with `@AGENTS.md`; configure other
  agents explicitly if needed. Do not assume tests or instructions execute automatically.
- `LICENSE`: MIT, Andreas Goethals; third-party materials retain their own licences.
- `pyproject.toml`: Python 3.11–3.12, project metadata/dependencies and pytest/Ruff settings.
  Record reproducible environment specifications alongside it; package ranges are not a lockfile.
- `.gitignore`: exclude assets/environments/caches; keep only selected small reports. Root-specific
  rules start with `/`; unanchored `figures/` would also match nested output folders.
- `.gitattributes`: UTF-8 text with LF checkout for Linux scripts; avoid CRLF Slurm failures.
- `.gitmodules`: independent library registrations; exclude new mounts from packaging/lint/tests.
- `.vscode/settings.json`: source imports, root-relative notebooks, pytest/Ruff and read-only libraries.
- `.github/dependabot.yml`: proposes VSC pin updates; does not merge them or run project tests.
- `_template/README.md`: new/existing-project instructions. `init_project.py` validates and substitutes
  placeholders, skipping assets, submodules and linked directories; no install, Git mutation or deletion.
  Remove `_template/` after setup; verify template edits in an initialized throwaway copy.

Comments explain non-obvious **why**, not obvious mechanics. Keep docs short, structured and consistent.
All access/storage examples use placeholders for personal logins, allocations, paths and accounts;
never publish credentials, certificate contents or temporary authentication links.

### Receiving template updates

After the canonical update is reviewed and pushed, copy its `docs/TEMPLATE.md` byte-for-byte into
each explicitly selected downstream repository; compare hashes, review the one-file diff, commit
and push when authorized. New projects receive it with **Use this template**. Existing projects
do not update automatically: this document-copy workflow is separate from library-pin propagation.
It does not authorize changing downstream code, active campaigns or storage. Review adoption separately.
