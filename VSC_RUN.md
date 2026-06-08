# Running on the KU Leuven VSC

A step-by-step guide to running the TabPFNCredit benchmark on the VSC
SLURM cluster (Genius / wICE). It covers the one thing the cluster needs
special handling for — **staging model weights for compute nodes that have
no internet** — plus how large sweeps are split across SLURM array jobs and
how to resume a partial run.

> New here? Read the **[README](README.md)** first for the CLI, the
> experiments, and how to run locally. This guide assumes the benchmark
> already runs on your machine and focuses only on the cluster.

---

## Prerequisites

- A VSC account with access to the Genius / wICE partitions.
- The project checked out under `$VSC_DATA/TabPFNCredit`.

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

**(b) Upload `checkpoints/` to the repo root on the VSC:**

```bash
rsync -av checkpoints/ <vsc>:$VSC_DATA/TabPFNCredit/checkpoints/
```

**(c) On a VSC login node, provision once:**

```bash
cd "$VSC_DATA/TabPFNCredit"
bash scripts/setup_vsc_checkpoints.sh
```

The generated job scripts point `HF_HOME` / `TABPFN_MODEL_CACHE_DIR` at
`checkpoints/` and set `HF_HUB_OFFLINE=1`, so compute nodes read every
weight offline. A missing weight fails fast with a clear error instead of
hanging on a blocked network call.

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

---

## 5. Resume a partial run

Every result is a single file, and the runner skips any cell whose result
already exists ("skip-if-done"). To resume after a time-out or
cancellation, just run the same command again — completed points are read
from disk and skipped, so nothing is lost between submissions:

```bash
scancel -M all -u $USER                # (optional) clear anything still queued
tabpfncredit experiment Experiment0    # re-runs only the missing points
```
