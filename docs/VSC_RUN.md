# Running on the VSC (KU Leuven) — checklist

A short, ordered checklist for a clean run on **wICE**. Covers the two things
that broke the last run (no compute-node internet; jobs over 72 h) and how the
sweeps now shard across SLURM jobs.

## 0. One-time setup per checkout

```bash
cd "$VSC_DATA/TabPFNCredit"
git pull
# install (see pyproject.toml header for the two-step --no-deps TALENT install)
```

## 1. Pre-stage foundation-model weights (ONCE, on a login node)

wICE **compute nodes have no outbound internet**, so TabPFN (v2.5/v3), TabICL,
TabDPT, Mitra and HyperFast cannot download weights or accept the TabPFN licence
at run time. Last run they all failed for exactly this reason. Do it once on a
login node (which has internet):

```bash
bash scripts/prestage_models.sh      # answer the TabPFN licence prompt with 'y'
```

This fills a **shared** cache under `$VSC_DATA/TabPFNCredit/.model_cache/`. The
generated job scripts export the same `HF_HOME` / `TABPFN_MODEL_CACHE_DIR` and
set `HF_HUB_OFFLINE=1`, so the compute nodes read it offline. (If a model warns
during pre-staging, fix it there — a missing weight will fail fast on the
compute node rather than hang.)

## 2. Apply the TALENT-side fixes

Several failures live in TALENT's own code (regression MSE assertion, torch-2.6
`weights_only`, protogate, **tabicl_v2** — which Experiments 2 & 3 need —,
catboost GPU-on-CPU, amformer, mitra). Apply the patches in
[`docs/TALENT_FIXES.md`](TALENT_FIXES.md) to your TALENT fork and re-install.
Until then those methods will keep failing (visibly, per-cell — they no longer
abort the whole sweep).

## 3. Run an experiment

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

## 4. The nested sweeps (why the curves are clean signal)

* **Experiment 2 (learning curve):** lowering `row_limit` keeps a **strict
  subset** of the larger cap's rows (fixed-seed, class-stratified for PD so the
  minority survives small caps). The metric change reflects *fewer rows*, not a
  different random draw.
* **Experiment 3 (imbalance):** lowering the minority proportion **deletes more
  of the same minority rows** (nested permutation prefix). The change reflects
  *fewer minority cases*, not a lucky/unlucky draw.

## 5. Recover from a partial run

```bash
git pull                       # get the latest fixes
scancel -M all -u $USER        # cancel anything still queued
bash scripts/prestage_models.sh    # only if step 1 wasn't done
tabpfncredit experiment Experiment0    # resumes; only missing points re-run
```

`0014.algorithmwatch` (and any dataset too large to preprocess on the login
node) is **no longer dropped** — it is preprocessed on the compute node (256 GB)
at run time, atomically, so concurrent tasks are safe.
