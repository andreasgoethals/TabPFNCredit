# Running on the KU Leuven VSC

A step-by-step guide to running the TabPFNCredit benchmark on the VSC
SLURM cluster (Genius / wICE). It covers where the project's files live
(general data storage vs. the large project storage), **staging model
weights for compute nodes that have no internet**, how large sweeps are
split across SLURM array jobs, and how to resume or migrate a partial run.

> New here? Read the **[README](README.md)** first for the CLI, the
> experiments, and how to run locally. This guide assumes the benchmark
> already runs on your machine and focuses only on the cluster.

---

## Prerequisites

- A VSC account with access to the Genius / wICE partitions.
- The project checked out under `$VSC_DATA/TabPFNCredit`.

### Configure your site (env vars)

Set your Slurm credit account on the login node before generating/submitting —
it is read from the environment and kept out of the repo:

```bash
export TABPFN_SLURM_ACCOUNT=lp_<your_project>     # your regular project credit account
# optional: override where results/caches/checkpoints live (auto-detected otherwise)
export TABPFN_STAGING_ROOT=/your/project/storage
```

Add `TABPFN_SLURM_ACCOUNT` to your `~/.bashrc` so every job picks it up — if it
is unset the generator emits a placeholder account and the job is rejected.
Use a **regular project account**: pilot projects are no longer valid now that
Tier-2 Mindwell is in production.

### Cluster routing

- **Default routing is wICE** (`batch_sapphirerapids`, `gpu_a100`, `gpu_h100`).
  **Genius GPU (V100)** and **Mindwell GPU** (`gpu_b200`, NVIDIA B200) are wired
  as cross-cluster **replica** targets: set `TABPFN_ALL_CLUSTERS=1` to fan the
  small-data Exp 2/3 sweeps across all of them at once (see the env table).
  Replicas never gate the primary dependency chain, so a reserved or
  unconfigured cluster simply drops harmlessly — wICE always completes the run.
- **7-day CPU walltime:** the `*_long` CPU partitions allow up to 168 h (vs 72 h
  on `batch_*`); the generator targets the 72 h partitions.

---

## Storage layout

The benchmark uses two filesystems and resolves them automatically — no flags
needed:

| Filesystem | Holds | Why |
|---|---|---|
| **General data storage** — `$VSC_DATA/TabPFNCredit` (the repo) | code, **logs** | small quota, backed up, persistent |
| **Project storage** — `$TABPFN_STAGING_ROOT` (auto-detected on the VSC; override via the env var) | **datasets, checkpoints, results, caches** | large, non-purged; keeps heavy I/O off the small `$VSC_DATA` quota |

- **Datasets & checkpoints** are read **repo first, then project storage** —
  put them in either place. On the cluster they live under
  `$TABPFN_STAGING_ROOT/{data,checkpoints}/`.
- `src/data/dataset_preprocessing.py` is private and gitignored because it
  contains proprietary raw-dataset schema and cleaning rules. Keep your local
  copy in the repo before preprocessing raw datasets on the VSC.
- **Results & caches** are written to project storage
  (`$TABPFN_STAGING_ROOT/{results,cache}/`).
- **Logs** always stay on `$VSC_DATA` (`<repo>/logs/<experiment>/`), never on
  project storage.

Override the root by exporting `TABPFN_STAGING_ROOT` before submitting
(results and cache also honour `TABPFN_RESULTS_ROOT` / `TABPFN_CACHE_ROOT`).
`tabpfncredit doctor` prints every resolved path.

---

## 1. One-time setup

On a **login node**:

```bash
cd "$VSC_DATA/TabPFNCredit"
module purge
module load cluster/genius/login
module load Python/3.12.3-GCCcore-13.3.0     # 'module spider Python/3.12' for the exact name

python -m venv tabpfncreditvenv
source tabpfncreditvenv/bin/activate
pip install -e ".[hpc]"
```

---

## 2. Stage foundation-model weights

VSC **compute nodes have no outbound internet**, so foundation models
(TabPFN v2/v2.5/v3, TabICL, TabDPT, MITRA, HyperFast) cannot download their
weights at run time. Download them once on a machine that *does* have
internet, upload the folder, and provision it on the cluster.

> Only running classical / deep methods (XGBoost, CatBoost, FT-Transformer,
> …)? They download nothing — skip this whole section.

**(a) On a machine with internet** (inside the project venv):

```bash
python -m src.utils.fetch_weights                 # -> ./checkpoints  (several GB)
# or a subset:
python -m src.utils.fetch_weights --only tabpfn_v3 tabicl_v2
```

**(b) Upload `checkpoints/` to the project storage on the VSC** (the repo
root works too — both are found automatically):

```bash
rsync -av checkpoints/ <vsc>:"$TABPFN_STAGING_ROOT"/checkpoints/   # or $VSC_DATA/TabPFNCredit/checkpoints/
```

**(c) On a VSC login node, provision once:**

```bash
cd "$VSC_DATA/TabPFNCredit"
bash scripts/setup_vsc_checkpoints.sh
```

The generated job scripts auto-detect the `checkpoints/` location (repo
first, then project storage), point `HF_HOME` / `TABPFN_MODEL_CACHE_DIR` at
it, and set `HF_HUB_OFFLINE=1`, so compute nodes read every weight offline.
A missing weight fails fast with a clear error instead of hanging on a
blocked network call.

---

## 3. Run an experiment

```bash
tabpfncredit experiment Experiment0          # auto: preprocess -> SLURM arrays
```

On the VSC this generates and submits per-partition SLURM array jobs. Summary
CSVs are rebuilt after the arrays finish with `tabpfncredit summarize` or by
running the notebooks. To submit the whole chained benchmark
(Experiments 0 → 1 → 2 → 3):

```bash
bash scripts/run_all_experiments.sh
```

Monitor:

```bash
squeue -u $USER
sacct -j <jobid>
```

---

## 4. How large sweeps are split across jobs

- The scheduling unit is a **single sweep point** (one curve point). For the
  big sweeps — Experiment 2 (training-set size) and Experiment 3 (minority
  proportion) — a `(dataset, method)` cell whose estimated cost exceeds one
  slot's wall-time budget is **split across multiple array tasks that run in
  parallel** ("multiple runs per dataset"), so no single job runs a whole sweep
  (e.g. a slow dataset like HackerEarth) serially. Each task writes its own
  packed shard file (`<method>__shard_<jobid>_<task>.json`); the skip-check,
  the `resubmit` gap-scan and the summariser all read the **union** across a
  cell's shards, so a point computed by any task — in this submission or a
  previous one — counts as done. Cheap cells stay in a single shard, keeping
  the file/inode count low.
- Each array task is packed to fit under the partition **wall-time limit**
  and capped at **`TABPFN_MAX_ARRAY_SLOTS` per partition**, because every array
  element counts toward the per-user submission limit (~500 on the `normal`
  QoS — query yours with
  `sacctmgr show qos normal format=Name,MaxSubmitJobsPerUser`). The default is
  **40** for `tabpfncredit experiment` (so the chained `run_all_experiments.sh`,
  which pre-submits all experiments at once, stays under the cap), and
  **auto-scaled** for `tabpfncredit resubmit` (a ~450 budget split across the
  experiments it submits → ≈150/partition for a single experiment).
- To parallelise a big sweep even harder, raise the cap for a standalone run —
  keep the **total** submitted array tasks under your QoS limit:

  ```bash
  TABPFN_MAX_ARRAY_SLOTS=150 tabpfncredit experiment Experiment3
  # both wICE GPU queues at once on the small-data Exp 2/3 sweeps:
  TABPFN_REPLICATE_PARTITIONS=gpu_h100,gpu_a100 tabpfncredit resubmit Experiment3
  ```

The per-point time estimates are deliberately conservative (they double as
the wall-time budget), so the generator may ask for more slots than the run
actually needs. Because runs are resumable, an over- or under-estimate is
harmless.

### Scheduler tuning knobs (env vars, all optional)

| Variable | Default | Effect |
|---|---|---|
| `TABPFN_MAX_ARRAY_SLOTS` | 40 (`experiment`) / auto (`resubmit`) | Array tasks per partition. Every element counts toward the per-user submit limit (~500 on `normal` QoS). `resubmit` auto-raises this to split a ~450 budget across the experiments it submits (≈150/partition for a single experiment), so a standalone resubmit parallelises hard out of the box; an explicit value always wins. |
| `TABPFN_ALL_CLUSTERS` | unset | **Use every cluster at once.** Replicates the small-data GPU sweeps (Exp 2/3) onto **all** GPU partitions across wICE (`gpu_a100`, `gpu_h100`) + Genius (`gpu_v100`) + Mindwell (`gpu_b200`) in one submission. wICE stays the afterok primary; the others are pure accelerators raced via skip-if-done, so a cluster you can't reach — or where torch lacks that GPU's kernels — drops **harmlessly** (the submit is caught and skipped; wICE still completes the run). Equivalent to `TABPFN_REPLICATE_PARTITIONS=gpu_h100,gpu_a100,gpu_v100,gpu_b200`. Mindwell B200 needs a torch with Blackwell `sm_100` kernels + a 2025a Python module (export `TABPFN_PYTHON_MODULE`) and Mindwell credits on the account; if any are missing, those replicas just fail and wICE/Genius carry the work. |
| `TABPFN_MAX_CONCURRENT` | unset | Re-adds a `%N` throttle on how many array elements run at once (none by default — SLURM fairshare governs). |
| `TABPFN_CPU_CORES_PER_TASK` | 18 | Cores per CPU array task (half a node → two tasks pack per node). Set 36 for a whole node. |
| `TABPFN_GPU_SPREAD` | 1 | Spill whole cells between `gpu_a100` ↔ `gpu_h100` when one queue is overloaded. Set 0 to pin work to its home partition (H100 costs ~4× the credits of A100). |
| `TABPFN_GENIUS_GPUS` | unset | Offload (MOVE) **small-data** GPU work (Experiment 2's row-capped and Experiment 3's subsampled sweeps) to the idle Genius fleet: set to `gpu_v100`. Only points with a row cap ≤ `TABPFN_GENIUS_ROW_CAP` (default 60000) or a sampling target move; full-dataset foundation fits never do. **P100 is not usable** — the project's torch 2.8 CUDA wheels ship `sm_70+` kernels only, and Pascal is `sm_60` (a `gpu_p100` request is ignored with a warning). Cross-cluster arrays do not gate the primary dependency chain — re-run `tabpfncredit summarize` after they finish. |
| `TABPFN_REPLICATE_PARTITIONS` | unset | **Aggressive mode**: COPY the small-data GPU work to every listed partition (e.g. `gpu_v100,cpu`) *in addition to* its wICE home, racing all queues at once. Every point is skipped at run time if its result already exists, and replicas traverse the work from different ends, so duplicate compute stays small. CPU replicas only take points with a row cap ≤ `TABPFN_CPU_FOUNDATION_ROW_CAP` (default 10000; in-context fits on CPU are slow) with cost estimates scaled by `TABPFN_CPU_FOUNDATION_SLOWDOWN` (default 10×). Run `tabpfncredit resubmit` once more after everything finishes to mop up any points lost to cross-cluster write races on the packed Exp 2/3 files. Takes precedence over `TABPFN_GENIUS_GPUS`. |

---

## 5. Resume — and reuse results you already have

Every Experiment 0/1 result is a single file
(`<results>/<experiment>/<task>/<dataset>/<method>.json`); Experiment 2/3 pack a
cell's sweep points into one packed file per array task — `<method>.json`, or
`<method>__shard_<jobid>_<task>.json` when a slow cell is split across several
tasks. The runner **skips any point whose result already exists** with the full
fold count ("skip-if-done"), reading the **union** across a cell's shards. To
resume after a time-out or cancellation, just run the same command again —
finished points are read from disk and skipped, so nothing is lost between
submissions:

```bash
scancel -M all -u $USER                # (optional) clear anything still queued
tabpfncredit experiment Experiment0    # re-runs only the missing points
```

When most of an experiment is already done, prefer **`resubmit`** — it scans
the results, prints an `expected / done / missing` report, and packs ONLY the
missing points into dense fresh arrays (instead of re-sharding everything and
queueing slots with nothing left to do). It also **auto-scales the per-partition
array cap** to spend the submit budget aggressively (≈150 slots/partition for a
single experiment), so the missing points fan out across as many parallel tasks
as the QoS limit allows:

```bash
tabpfncredit resubmit Experiment1      # one experiment
tabpfncredit resubmit --all            # all four at once
```

It wipes `scripts/<Exp>/_generated/` before writing new scripts, so cancel
any still-pending arrays first (`squeue -u $USER`).

### Cancel everything, refresh dependencies, and resubmit across all clusters

The full "start clean and go maximally aggressive" recipe — scans **every**
experiment for missing points and fans them out across wICE + Genius + Mindwell:

```bash
# 1. Cancel everything still queued/running (all clusters).
scancel -M all -u "$USER"

# 2. Activate the environment on the login node.
cd "$VSC_DATA/TabPFNCredit"
module purge
module load Python/3.12.3-GCCcore-13.3.0          # 'module spider Python/3.12' for the exact name
source tabpfncreditvenv/bin/activate

# 3. (optional) override the Slurm account; the repo default is already correct.
# export TABPFN_SLURM_ACCOUNT=lp_<your_project>

# 4. Pull the latest benchmark code (editable install -> the pull is live).
git pull

# 5. Refresh TALENT in the venv without touching torch/etc.
pip install --force-reinstall --no-deps --no-cache-dir TALENT

# 6. Wipe any old summaries on PROJECT STORAGE so nothing stale lingers.
rm -f "${TABPFN_STAGING_ROOT:-/lustre1/project/stg_00211/TabPFNCredit}"/results/summaries/*.csv

# 7. Resubmit ONLY the missing points of ALL experiments, across every cluster.
TABPFN_ALL_CLUSTERS=1 tabpfncredit resubmit --all
```

Summaries are **not** produced automatically anymore. Once the run finishes,
rebuild the CSVs by **opening the notebooks** (each one refreshes its
experiment's summary on run, deleting the old one first) or, on the VSC,
`tabpfncredit summarize --experiment <name>`.

`resubmit --all` first prints an `expected / done / missing` report per
experiment, then submits only the gaps; `TABPFN_ALL_CLUSTERS=1` replicates the
small-data Exp 2/3 GPU sweeps onto every GPU partition (wICE A100/H100, Genius
V100, Mindwell B200) so a slow dataset like HackerEarth is chewed through by all
clusters at once.

**Already have results from an earlier run?** Copy them into the new results
root once and they're skipped automatically. Results now live on project
storage, so move any you kept under the old `$VSC_DATA` location across,
preserving the `<experiment>/<task>/<dataset>/` layout:

```bash
rsync -av --exclude='*/logs/' \
    "$VSC_DATA/TabPFNCredit/results/" \
    "$TABPFN_STAGING_ROOT/results/"
```

`--exclude='*/logs/'` leaves logs on `$VSC_DATA` (they don't affect
skip-if-done). A point is skipped only if its `<method>.json` has all
`cv_splits` folds; a partially-finished point re-runs and overwrites, so a
half-complete copy is safe.

---

## 6. After the run finishes — consolidate & download

Once every array has completed (`squeue -M all -u "$USER"` is empty):

```bash
cd "$VSC_DATA/TabPFNCredit"
module purge && module load Python/3.12.3-GCCcore-13.3.0
source tabpfncreditvenv/bin/activate

# 1. Collapse the Experiment 2/3 shard files into one <method>.json per cell.
#    Results are unchanged (the summariser already reads the union of a cell's
#    shards) -- this is pure housekeeping. Preview with --dry-run first.
python -m src.utils.consolidate_shards --dry-run
python -m src.utils.consolidate_shards

# 2. (optional) Build the summary CSVs on the cluster, so you can download just
#    those (a few MB) instead of the full JSON tree. The notebooks also rebuild
#    them locally, so this is only a download-size optimisation.
for e in Experiment0 Experiment1 Experiment2 Experiment3; do
  tabpfncredit summarize --experiment "$e"
done
```

Then pull the results to your laptop. `rsync` only transfers what changed, so
repeat syncs are fast (the firewall MFA prompt is the same as `scp`):

```bash
# Run this on your LOCAL machine. Whole results tree:
rsync -avz --info=progress2 \
  vsc<id>@login.hpc.kuleuven.be:/lustre1/project/stg_00211/TabPFNCredit/results/ \
  "<local path>/TabPFNCredit/results/"

# ...or, if you only run notebooks locally, just the summary CSVs (tiny):
rsync -avz --info=progress2 \
  vsc<id>@login.hpc.kuleuven.be:/lustre1/project/stg_00211/TabPFNCredit/results/summaries/ \
  "<local path>/TabPFNCredit/results/summaries/"
```

Finally, locally, regenerate every figure + the copy-pasteable results dump in
one command (clears, restart-runs all notebooks, writes `results/All_Results.md`
and the figure `CAPTIONS.md`):

```bash
python -m src.utils.run_notebooks
```
