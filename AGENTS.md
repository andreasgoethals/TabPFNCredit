# Agent instructions

## Published research compatibility

Read [docs/TEMPLATE.md](docs/TEMPLATE.md) and the project runbook
[docs/VSC.md](docs/VSC.md). Keep changes to execution infrastructure separate
from scientific changes. Preserve preprocessing (including private local
modules), observations, splits/seeds, methods, hyperparameters, metrics,
result schemas, filenames, aggregation and resume identities. Do not regenerate
paper results/figures or upgrade dependencies as incidental maintenance.
Keep workstation defaults and legacy deployment interfaces compatible. Add new
methods only through a separately authorized experiment/configuration.

The dependency ranges and TALENT `main` in `pyproject.toml` are not a frozen
paper runtime. Preserve existing environments and model/input versions;
capture their provenance before proposing a replacement. Test framework
changes with synthetic/local regressions and distinguish them from actual
Slurm validation or evidence of numerical reproduction.

## VSC access and execution

- Use one project branch, `main`, on the workstation, GitHub and VSC DATA
  checkout. Do not create separate deployment branches. For authorized releases,
  test locally, commit/push `main`, then use `git pull --ff-only` on VSC after
  checking its clean state and active/queued/resumable consumers. Preserve unique
  history in a verified local recovery bundle before retiring an old branch.
- The user renews access manually in Windows PowerShell with
  `step ssh login --context VSC`, including KU Leuven Authenticator approval.
  Reuse the existing native Windows SSH agent. Do not install/configure a
  per-project login, start MFA, enable automatic renewal or schedule reminders.
- Inspect certificate validity without renewing it. If missing/expired, ask
  the user to run that command and wait for confirmation. Network, host-key
  and command-channel failures require diagnosis, not an automatic login.
- Use TEMPLATE.md's bounded LF-stdin native `ssh.exe -F none -T` method with
  batch mode and strict host-key checks. Verify stdout, stderr and deliberate
  failure status; an empty exit-0 response does not prove execution.
  If the gateway drops raw stderr, use the independently verified remote
  Python capture envelope described in docs/VSC.md and check its returned
  streams/status against the SSH exit status.
- Directly inspect relevant remote files, logs, results, quotas and job state
  after verifying access. Keep reads scoped. Computational debugging belongs
  in Slurm within the user's authorization; access alone authorizes no new
  workloads, downloads, cancellations or campaign changes.

## Storage and migration

- DATA: actual Git checkout and `.git`, all Git operations, configuration,
  dependency specifications and compact logs/plans/provenance.
- PROJECT: persistent datasets, weights, checkpoints, retained results and
  versioned immutable SIF images containing installed dependencies. No new
  unpacked cluster environments or fallback of large writes to DATA.
- Native scratch: isolated temporary work/caches only; Lustre for wICE, GPFS
  for Mindwell. Preserve failed/recoverable work and publish durable outputs
  periodically. Check both byte and file-count quotas and independent backups.
- New VSC deployments use the explicit container profile in docs/VSC.md.
  Use `bash scripts/vsc.sh` for image planning and host-side submission;
  preserve the runtime lock in `config/vsc/`. Successful workers clean only
  their owned scratch workspace after verifying durable publication; retain
  failed workspaces and receipts for recovery.
  Legacy launchers remain for compatibility with existing paper campaigns;
  they are not approval to deploy another unpacked environment.
- Freeze checkouts, plans, runtime versions and library pins used by active,
  queued or resumable jobs. Explain any migration and obtain authorization
  before copying/moving assets, rebuilding environments, changing campaigns
  or deleting scratch. Never wipe a shared scratch root. Verify publication
  and all consumers before retiring an exact owned workspace.

## Shared VSC Library

`vsc-library/` is a pinned, entirely read-only infrastructure submodule. Read its
README, source snapshot, guides and Slurm examples; never edit, refresh, lint,
format or test inside it. There is no PROJECT_SPECIFIC exception. Keep this
project's VSC choices and operating instructions in `docs/VSC.md`, outside the
submodule. Shared corrections belong in the standalone VSC Library checkout.
Record the library pin when operational evidence depends on it.
