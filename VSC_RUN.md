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

- The scheduling unit is a **single sweep point** (one result file), not a
  whole `(dataset, method)` cell. The big sweeps — Experiment 2 (training-set
  size) and Experiment 3 (minority proportion) — are sharded across array
  tasks, so no single job runs a whole sweep serially.
- Each array task is packed to fit under the partition **wall-time limit**
  and capped at **`TABPFN_MAX_ARRAY_SLOTS` (default 40) per partition**,
  because every array element counts toward the per-user submission limit
  (~500 on the `normal` QOS) and the chained `run_all_experiments.sh`
  pre-submits all experiments at once.
- To parallelise a big sweep harder, raise the cap for a standalone run —
  keep the **total** submitted array tasks under 500:

  ```bash
  TABPFN_MAX_ARRAY_SLOTS=150 tabpfncredit experiment Experiment2
  ```

The per-point time estimates are deliberately conservative (they double as
the wall-time budget), so the generator may ask for more slots than the run
actually needs. Because runs are resumable, an over- or under-estimate is
harmless.

### Scheduler tuning knobs (env vars, all optional)

| Variable | Default | Effect |
|---|---|---|
| `TABPFN_MAX_ARRAY_SLOTS` | 40 | Array tasks per partition. Every element counts toward the ~500-job submit limit. |
| `TABPFN_MAX_CONCURRENT` | unset | Re-adds a `%N` throttle on how many array elements run at once (none by default — SLURM fairshare governs). |
| `TABPFN_CPU_CORES_PER_TASK` | 18 | Cores per CPU array task (half a node → two tasks pack per node). Set 36 for a whole node. |
| `TABPFN_GPU_SPREAD` | 1 | Spill whole cells between `gpu_a100` ↔ `gpu_h100` when one queue is overloaded. Set 0 to pin work to its home partition (H100 costs ~4× the credits of A100). |
| `TABPFN_GENIUS_GPUS` | unset | Offload (MOVE) **small-data** GPU work (Experiment 2's row-capped and Experiment 3's subsampled sweeps) to the idle Genius fleet: set to `gpu_v100`. Only points with a row cap ≤ `TABPFN_GENIUS_ROW_CAP` (default 60000) or a sampling target move; full-dataset foundation fits never do. **P100 is not usable** — the project's torch 2.8 CUDA wheels ship `sm_70+` kernels only, and Pascal is `sm_60` (a `gpu_p100` request is ignored with a warning). The summarize job cannot wait on cross-cluster arrays — re-run `tabpfncredit summarize` after they finish. |
| `TABPFN_REPLICATE_PARTITIONS` | unset | **Aggressive mode**: COPY the small-data GPU work to every listed partition (e.g. `gpu_v100,cpu`) *in addition to* its wICE home, racing all queues at once. Every point is skipped at run time if its result already exists, and replicas traverse the work from different ends, so duplicate compute stays small. CPU replicas only take points with a row cap ≤ `TABPFN_CPU_FOUNDATION_ROW_CAP` (default 10000; in-context fits on CPU are slow) with cost estimates scaled by `TABPFN_CPU_FOUNDATION_SLOWDOWN` (default 10×). Run `tabpfncredit resubmit` once more after everything finishes to mop up any points lost to cross-cluster write races on the packed Exp 2/3 files. Takes precedence over `TABPFN_GENIUS_GPUS`. |

---

## 5. Resume — and reuse results you already have

Every result is a single file
(`<results>/<experiment>/<task>/<dataset>/<method>.json`), and the runner
**skips any point whose result already exists** with the full fold count
("skip-if-done"). To resume after a time-out or cancellation, just run the
same command again — finished points are read from disk and skipped, so
nothing is lost between submissions:

```bash
scancel -M all -u $USER                # (optional) clear anything still queued
tabpfncredit experiment Experiment0    # re-runs only the missing points
```

When most of an experiment is already done, prefer **`resubmit`** — it scans
the results, prints an `expected / done / missing` report, and packs ONLY the
missing points into dense fresh arrays (instead of re-sharding everything and
queueing slots with nothing left to do):

```bash
tabpfncredit resubmit Experiment1      # one experiment
tabpfncredit resubmit --all            # all four at once
```

It wipes `scripts/<Exp>/_generated/` before writing new scripts, so cancel
any still-pending arrays first (`squeue -u $USER`).

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
