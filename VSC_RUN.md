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

### Cluster status & compute accounts (June 2026)

- **Submit with a regular project account.** Jobs use
  `#SBATCH --account=lp_verbekelab` (set in `slurm_generator.py`). Tier-2
  **Mindwell** is now in production, so the old pilot project is no longer
  valid — all jobs (Genius / wICE / Mindwell) must use a regular project
  account; `lp_verbekelab` already is one. Credit rates are in the VSC docs.
- **Default routing is wICE** (`batch_sapphirerapids`, `gpu_a100`, `gpu_h100`),
  which is fully released. **Genius GPU (V100)** and the new **Mindwell GPU**
  (`gpu_b200`, NVIDIA B200) are wired as cross-cluster **replica** targets:
  set `TABPFN_ALL_CLUSTERS=1` to fan the small-data Exp 2/3 sweeps across all of
  them at once (see the env table). Replicas never gate the summarize job, so a
  still-reserved Genius GPU or an unconfigured Mindwell environment simply drops
  harmlessly — wICE always completes the run.
- **7-day CPU walltime:** wICE/Mindwell `*_long` CPU partitions allow up to
  168 h (vs 72 h on the regular `batch_*`). The generator targets the 72 h
  partitions; long CPU baselines (rare) can be sent to `batch_sapphirerapids_long`
  manually if ever needed.

---

## Storage layout

The benchmark uses two filesystems and resolves them automatically — no flags
needed:

| Filesystem | Holds | Why |
|---|---|---|
| **General data storage** — `$VSC_DATA/TabPFNCredit` (the repo) | code, **logs** | small quota, backed up, persistent |
| **Project storage** — `$TABPFN_STAGING_ROOT` (default `/staging/leuven/stg_00211`) | **datasets, checkpoints, results, caches** | large, non-purged; keeps heavy I/O off the small `$VSC_DATA` quota |

- **Datasets & checkpoints** are read **repo first, then project storage** —
  put them in either place. On the cluster they live under
  `$TABPFN_STAGING_ROOT/{data,checkpoints}/`.
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
python scripts/fetch_weights.py                 # -> ./checkpoints  (several GB)
# or a subset:
python scripts/fetch_weights.py --only tabpfn_v3 tabicl_v2
```

**(b) Upload `checkpoints/` to the project storage on the VSC** (the repo
root works too — both are found automatically):

```bash
rsync -av checkpoints/ <vsc>:/staging/leuven/stg_00211/checkpoints/
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
tabpfncredit experiment Experiment0          # auto: preprocess -> SLURM arrays -> summarize
```

On the VSC this generates per-partition SLURM array jobs plus a dependent
`summarize` job and submits them. To submit the whole chained benchmark
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
| `TABPFN_GENIUS_GPUS` | unset | Offload (MOVE) **small-data** GPU work (Experiment 2's row-capped and Experiment 3's subsampled sweeps) to the idle Genius fleet: set to `gpu_v100`. Only points with a row cap ≤ `TABPFN_GENIUS_ROW_CAP` (default 60000) or a sampling target move; full-dataset foundation fits never do. **P100 is not usable** — the project's torch 2.8 CUDA wheels ship `sm_70+` kernels only, and Pascal is `sm_60` (a `gpu_p100` request is ignored with a warning). The summarize job cannot wait on cross-cluster arrays — re-run `tabpfncredit summarize` after they finish. |
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

### Cancel everything, update TALENT, and resubmit across all clusters

The full "start clean and go maximally aggressive" recipe — scans **every**
experiment for missing points and fans them out across wICE + Genius + Mindwell:

```bash
# 1. Cancel everything still queued/running (all clusters).
scancel -M all -u "$USER"

# 2. Pull your latest TALENT fork + the benchmark, and refresh the install.
cd "$VSC_DATA/TabPFNCredit" && git pull
cd "$VSC_DATA/TALENT"       && git pull          # your personal TALENT fork
cd "$VSC_DATA/TabPFNCredit"
source tabpfncreditvenv/bin/activate
pip install -e "$VSC_DATA/TALENT"                # editable -> future fork pulls are live, no reinstall
pip install -e ".[hpc]"                          # refresh the benchmark CLI

# 3. Resubmit ONLY the missing points of ALL experiments, across every cluster.
TABPFN_ALL_CLUSTERS=1 tabpfncredit resubmit --all

# 4. After the cross-cluster (Genius/Mindwell) arrays finish, refresh the CSVs
#    (the wICE-gated summarize ran automatically; this folds in cross-cluster results).
for E in Experiment0 Experiment1 Experiment2 Experiment3; do
    tabpfncredit summarize --experiment "$E"
done
```

`resubmit --all` first prints an `expected / done / missing` report per
experiment, then submits only the gaps; `TABPFN_ALL_CLUSTERS=1` replicates the
small-data Exp 2/3 GPU sweeps onto every GPU partition (wICE A100/H100, Genius
V100, Mindwell B200) so a slow dataset like HackerEarth is chewed through by all
clusters at once. Adjust the `git pull` paths to wherever your TALENT fork and
the repo are checked out on the VSC.

**Already have results from an earlier run?** Copy them into the new results
root once and they're skipped automatically. Results now live on project
storage, so move any you kept under the old `$VSC_DATA` location across,
preserving the `<experiment>/<task>/<dataset>/` layout:

```bash
rsync -av --exclude='*/logs/' \
    "$VSC_DATA/TabPFNCredit/results/" \
    "/staging/leuven/stg_00211/results/"
```

`--exclude='*/logs/'` leaves logs on `$VSC_DATA` (they don't affect
skip-if-done). A point is skipped only if its `<method>.json` has all
`cv_splits` folds; a partially-finished point re-runs and overwrites, so a
half-complete copy is safe.
