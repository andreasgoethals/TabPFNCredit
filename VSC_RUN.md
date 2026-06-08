# Running on the VSC (KU Leuven) — checklist

A short, ordered checklist for a clean run on **wICE**. Covers the two things
that broke the last run (no compute-node internet; jobs over 72 h) and how the
sweeps now shard across SLURM jobs.

## 0. One-time setup per checkout

```bash
cd "$VSC_DATA/TabPFNCredit"
git pull
python -m venv tabpfncreditvenv && source tabpfncreditvenv/bin/activate
pip install -e ".[hpc]"        # single step; pulls TALENT from the fork
```

> TALENT-side bugs (regression MSE assertion, torch `weights_only`, tabicl_v2,
> catboost GPU-on-CPU, amformer, …) are **already fixed** in the fork the
> install pulls from (`andreasgoethals/TALENT`). No manual patching needed.

## 1. Provision foundation-model weights (download LOCALLY, upload, provision)

wICE **compute nodes have no outbound internet**, so TabPFN (v2/v2.5/v3),
TabICL, TabDPT, Mitra and HyperFast cannot download their weights at run time.
We therefore download them **on your local machine** (which has internet) and
upload the resulting `checkpoints/` folder.

**(a) On your LOCAL machine** (inside the project venv, with internet):

```bash
python scripts/fetch_weights.py            # -> ./checkpoints  (several GB)
# or a subset:  python scripts/fetch_weights.py --only tabpfn_v3 tabicl_v2
```

**(b) Upload `checkpoints/` to the repo root on the VSC:**

```bash
rsync -av checkpoints/ <vsc>:$VSC_DATA/TabPFNCredit/checkpoints/
```

**(c) On a VSC LOGIN node, provision it ONCE** (no network; just places the few
models that load from a package-internal path — Mitra, HyperFast):

```bash
cd "$VSC_DATA/TabPFNCredit"
bash scripts/setup_vsc_checkpoints.sh
```

The generated job scripts export `HF_HOME` / `TABPFN_MODEL_CACHE_DIR` pointing
at `checkpoints/` and set `HF_HUB_OFFLINE=1`, so the compute nodes read every
weight offline. A missing weight fails fast with a clear error rather than
hanging on a blocked network call.

## 2. Run an experiment

```bash
# clears nothing; skip-if-done means re-runs only do what's missing
tabpfncredit experiment Experiment0      # auto: preprocess -> SLURM arrays -> summarize
```

On the VSC this auto-generates per-partition SLURM arrays + a dependent
`summarize` job and submits them. Locally it runs in-process.

### Parallelism, the 72 h wall, and the 500-job limit

* The scheduling unit is a **sweep point** (one result file), not a whole
  `(dataset, method)` cell. Experiment 2's row sweep and Experiment 3's minority
  sweep (thousands of points per cell) are **sharded across array tasks**, so no
  single job runs the whole sweep serially past 72 h.
* Each array task is packed to fit under the **72 h** wICE wall and capped at
  **`MAX_ARRAY_SLOTS` (default 40) per partition**, because every array element
  counts toward the VSC per-user submit limit (~500 on the `normal` QOS) and the
  `run_all_experiments.sh` chain pre-submits all experiments at once
  (4 experiments × ≤3 partitions × 40 = 480 < 500).
* **If a slot still hits 72 h** (the generator warns when the estimated work
  needs more slots than the cap allows), just **re-submit** — `skip-if-done`
  resumes from the points already saved. Nothing is lost.
* **To parallelise a big sweep harder in one shot**, run it standalone with a
  higher cap (keep the *total* submitted array tasks < 500):

  ```bash
  TABPFN_MAX_ARRAY_SLOTS=150 tabpfncredit experiment Experiment2
  ```

  e.g. 150 across the CPU + GPU partitions Experiment 2 uses stays under 500.

> The per-point time estimates are deliberately **pessimistic** (they double as
> a walltime budget), so the generator may report needing far more slots than
> the real run takes. In practice a sweep completes in 1–few submissions; the
> resumable design makes over-/under-estimates harmless.

## 3. The nested sweeps (why the curves are clean signal)

* **Experiment 2 (learning curve):** lowering `row_limit` keeps a **strict
  subset** of the larger cap's rows (fixed-seed, class-stratified for PD so the
  minority survives small caps). The metric change reflects *fewer rows*, not a
  different random draw.
* **Experiment 3 (imbalance):** lowering the minority proportion **deletes more
  of the same minority rows** (nested permutation prefix). The change reflects
  *fewer minority cases*, not a lucky/unlucky draw.

## 4. Recover from a partial run

```bash
git pull                            # get the latest fixes
scancel -M all -u $USER             # cancel anything still queued
bash scripts/setup_vsc_checkpoints.sh  # only if step 1(c) wasn't done
tabpfncredit experiment Experiment0    # resumes; only missing points re-run
```

`0014.algorithmwatch` (and any dataset too large to preprocess on the login
node) is **no longer dropped** — it is preprocessed on the compute node (256 GB)
at run time, atomically, so concurrent tasks are safe.
