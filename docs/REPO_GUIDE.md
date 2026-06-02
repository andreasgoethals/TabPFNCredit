# TabPFNCredit — repo guide

This document is the short tour: **what the repo does, how to use it, what each experiment runs, how to launch it on the VSC, and how results are stored.** Keep it open when you onboard. The long-form README is for marketing; this file is for operators.

---

## 1. What the repo is

TabPFNCredit is a **credit-risk benchmarking harness on top of [TALENT](https://github.com/LAMDA-Tabular/TALENT)**. It takes raw credit datasets (≈14 PD + ≈7 LGD), preprocesses them once into TALENT's `(N, C, y) + info.json` format, splits them into deterministic CV folds, then runs every TALENT method (≈55 of them, from classical baselines through the foundation-model family: TabPFN v1/v2/v2.5/v3/Real, TabICL v1/v2, TabDPT, MITRA, LimiX, …) under both `NO_HPO` and `HPO` modes, and writes per-fold metrics + predictions to disk.

Everything model-related is delegated to TALENT. This repo only owns:

* **Data preprocessing** (per-dataset cleaning, leakage scrubbing, log transforms).
* **CV split assembly** (stratified for PD, random for LGD; cached on disk via `joblib.Memory`).
* **Per-method orchestration** (foundation-model val/test downsampling for OOM safety, per-fold HPO with SLURM-safe merged JSON, resumable checkpoint dirs).
* **Credit-risk metric enrichment** (Gini, KS, MAPE-with-zero-exclusion, cost-sensitive expected loss / profit curves) on top of TALENT's built-in `AUC / F1 / Brier / ECE / RMSE / R²`.
* **Result storage** (per-fold JSON + npz, per-method `summary.json`).
* **Visualisation utilities** (heatmaps, ranking bars, learning curves, calibration plots).

The recently-rewritten layer ([`src/methods/method_runner.py`](../src/methods/method_runner.py)) sits on top of TALENT's new typed Python API (`TALENT.run()` returning a `RunResult` with `predict_proba` / `predict_labels` / `threshold` / `metrics` / `metric_names`) — so this repo no longer manipulates `sys.argv`, no longer has custom softmax/sigmoid helpers, and no longer maintains its own method registry.

---

## 2. Pipeline at a glance

```
                       ┌────────────────────────────────────────┐
                       │ data/raw/{pd,lgd}/<dataset>/*.csv      │
                       │ (one folder per credit dataset)        │
                       └────────────────────────────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────────────┐
                      │  src/data/dataset_preprocessing.py       │
                      │  - dataset-specific cleaning             │
                      │  - leakage column drop + log transforms  │
                      └─────────────────────────────────────────┘
                                       │
                                       ▼   (cached on first use)
                  ┌───────────────────────────────────────────────┐
                  │ data/processed/{pd,lgd}/<dataset>/*.npy + info.json │
                  │ (TALENT-format)                                     │
                  └───────────────────────────────────────────────┘
                                       │
                                       ▼
            ┌─────────────────────────────────────────────────────┐
            │ src/data/data_feeder.py                              │
            │  - StratifiedKFold (PD) / KFold (LGD)                │
            │  - winsorize / drop near-constants / optional PCA    │
            │  - cached across SLURM workers via joblib.Memory     │
            └─────────────────────────────────────────────────────┘
                                       │
                                       ▼
       ┌────────────────────────────────────────────────────────────┐
       │ src/methods/method_runner.py :: run_talent_method()        │
       │  - foundation-model val/test downsampling                  │
       │  - calls TALENT.run() once per fold                        │
       │    (returns RunResult with predict_proba, threshold, …)    │
       │  - enrich_pd_metrics / enrich_lgd_metrics                  │
       │  - cost_sensitive_summary (expected loss / profit curves)  │
       │  - per-fold HPO JSON merged under FileLock                 │
       │  - resumable checkpoint dir hashed on (config, fold, seed) │
       └────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
          ┌──────────────────────────────────────────────────────────┐
          │ src/utils/result_io.py :: save_fold + update_method_summary │
          │                                                          │
          │ results/<experiment>/<task>/<dataset>/<method>/<HPO|NO_HPO>/ │
          │   ├── fold_<id>.json     (metrics, threshold, times)     │
          │   ├── fold_<id>.npz      (y_true, y_proba, y_pred, …)    │
          │   └── summary.json       (aggregated across all folds)   │
          │                                                          │
          │ Old pickle layout (results/<experiment>/<task>/<dataset>.pkl) │
          │ still written by Experiment*.py for backward-compat.     │
          └──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                ┌────────────────────────────────────────────┐
                │ src/utils/summarize_results_polars.py       │
                │  - scans the JSON layout                    │
                │  - writes results/summaries/                │
                │       <experiment>_per_fold.csv             │
                │       <experiment>_per_method.csv           │
                └────────────────────────────────────────────┘
                                       │
                                       ▼
                ┌────────────────────────────────────────────┐
                │ notebooks/Experiment*.ipynb                 │
                │  - thin: load CSV, call src.visualizations  │
                │  - figures saved to figures/<experiment>/   │
                └────────────────────────────────────────────┘
```

---

## 3. The four experiments

Each experiment lives under `scripts/Experiment<N>/` with three YAML configs in `config/`:

* `CONFIG_DATA.yaml` — split sizes, seed, dataset toggles, paths.
* `CONFIG_METHOD.yaml` — `{method: true/false}` for every TALENT method, per task.
* `CONFIG_EXPERIMENT.yaml` — training knobs (`max_epochs`, `batch_size`, …).

| # | Question it answers | Folds | HPO | Datasets |
|---|---|---|---|---|
| **0** | "Does each method run end-to-end on real credit data?" — pilot screening to drop broken methods before the full sweep. | 2 | NO_HPO | 2 (cheap) |
| **1** | The headline benchmark. Every method on every dataset, both `NO_HPO` and `HPO` modes. Produces the figures that go in the paper. | 5 | NO_HPO + HPO | 14 PD + 7 LGD |
| **2** | Learning-curve analysis: how does each method's metric track with training-set size? Loops over `row_limit` from `row_min` to `row_max` in `row_step` increments. | 5 | NO_HPO | smaller subset |
| **3** | Class-imbalance stress test: down-sample positives so the minority rate sweeps from `minority_proportion_min` to `minority_proportion_max`. PD only. | 5 | NO_HPO | PD-only subset |

The CV split is **stratified** on PD targets and **random** on LGD targets (LGD is continuous). The `row_limit` for an experiment caps the total dataset before splitting; the method-intrinsic `train_row_limit` (e.g. 50 000 for TabPFN v2.5) caps only the training set after splitting so test/val coverage stays consistent across methods.

---

## 4. Running it

### 4.1 Local quick check

```bash
# install + smoke test
pip install -r requirements_local.txt   # or requirements.txt for the lightweight version
pip install -e .                         # makes the `tabpfncredit` CLI available
pytest tests/ -m "not slow and not gpu"   # ~30 seconds; 34 tests should pass

# run one (dataset, method, task) cell
tabpfncredit run \
    --experiment Experiment0 \
    --dataset   0001.gmsc \
    --method    tabpfn_v3 \
    --task      pd \
    --verbose

# results land at:
#   results/experiment0/pd/0001.gmsc/tabpfn_v3/NO_HPO/fold_*.{json,npz}
#   results/experiment0/pd/0001.gmsc/tabpfn_v3/NO_HPO/summary.json
```

List every method TALENT exposes:

```bash
tabpfncredit list                                # all 55
tabpfncredit list --architecture deep --hardware gpu
```

Aggregate fold results into CSV summaries (replaces the legacy
`summarize_results.py`):

```bash
tabpfncredit summarize --experiment Experiment0
# writes results/summaries/experiment0_per_fold.csv and per_method.csv
```

Pretty-print the merged YAML config for an experiment:

```bash
tabpfncredit config --experiment Experiment1
```

### 4.2 VSC (Genius / wICE)

Copy / sync the repo into `$VSC_DATA/TabPFNCredit/` and create the conda env:

```bash
ssh genius.hpc.kuleuven.be
cd $VSC_DATA
module purge
source $VSC_DATA/miniconda3/bin/activate
conda create -y -n TabPFNCredit python=3.10
conda activate TabPFNCredit
cd TabPFNCredit
pip install -r requirements_vsc.txt        # default: cu121 wheel for wICE H100
pip install -e .
```

To target the legacy Genius P100 nodes, uncomment the `cu118` block in `requirements_vsc.txt` first.

A SLURM submission looks like (the templates in `scripts/_slurm_templates.py` generate these):

```bash
sbatch scripts/Experiment0/Experiment0_GPU0.slurm   # GPU foundation models
sbatch scripts/Experiment0/Experiment0_CPU0.slurm   # CPU classical methods
```

The new CLI also supports the SLURM array layout directly — each array task picks one (dataset, method) cell deterministically:

```bash
# inside a SLURM array script:
tabpfncredit slurm-task \
    --experiment Experiment0 \
    --partition gpu_foundation \
    --array-id $SLURM_ARRAY_TASK_ID
```

`--partition` is one of `cpu`, `gpu_standard`, `gpu_foundation`, or `all`. The method list for each partition is **derived from TALENT's registry** filtered by `hardware` and `FOUNDATION_METHODS` — adding a new method to TALENT auto-includes it.

Job staggering is now deterministic (`sleep $((SLURM_ARRAY_TASK_ID % 30))`), so 100 array slots that start at the same time don't all hit the filesystem in the same second.

### 4.3 Notebooks (figures only)

After a run finishes, fire up the notebooks for visualisation:

```bash
jupyter notebook notebooks/Experiment0.ipynb
```

Every `notebooks/Experiment*.ipynb` is now a **thin caller** of `src.visualizations.experiment_plots`. To change a figure, edit the helper module — never the notebook. Re-running `python notebooks/_rewrite_notebooks.py` regenerates the notebook stubs.

---

## 5. Result storage — what's saved, where, and why

There are **three** layers, in increasing aggregation order:

### Layer 1 — per-fold raw outputs

```
results/<experiment_lower>/<task>/<dataset>/<method>/<HPO|NO_HPO>/
    fold_<id>.json
    fold_<id>.npz
```

* **`fold_<id>.json`** (<1 KB, `jq`-friendly): `metrics` dict, `threshold` chosen on validation, `train_time`, `predict_time`, `hpo_config`, `used_hpo`.
* **`fold_<id>.npz`** (compressed numpy bundle): `y_true`, `y_prob`, `y_pred`, `val_y_true`, `val_y_prob` — everything you need to recompute any metric without re-running the model.

These are **append-only**: a fold is written exactly once, so there's no concurrent-write contention at this layer.

### Layer 2 — per-method summary

```
results/<experiment_lower>/<task>/<dataset>/<method>/<HPO|NO_HPO>/summary.json
```

Aggregated mean / std / n_folds for each metric across folds, plus per-fold training/prediction time. Updated incrementally after every `run_talent_method()` call under an exclusive `FileLock`, so multiple SLURM array slots writing to the same `(dataset, method)` summary cannot lose updates (this was a real bug pre-refactor — see the regression test in `tests/test_file_lock.py`).

### Layer 3 — global CSVs

```
results/summaries/
    <experiment>_per_fold.csv      # one row per (dataset, method, fold)
    <experiment>_per_method.csv    # one row per (dataset, method) — aggregated
```

Built by `tabpfncredit summarize --experiment <name>`, backed by polars (≈10× faster than the previous pandas chain). These are the CSVs the notebooks consume.

### Backward-compat pickles

The legacy pickle path used by the original experiment drivers — `results/<experiment>/<task>/<dataset>.pkl` keyed by `{HPO_mode: {method: {fold_id: ...}}}` — is **still written** by `scripts/Experiment*/Experiment*.py` so existing analyses keep working. The new CLI writes the JSON/npz layout alongside.

### HPO config persistence

```
results/<experiment_lower>/<task>/<dataset>/<method>/HPO_PER_FOLD/
    <method>-all-folds.json
```

When `tune=True`, every fold's selected hyperparameters are merged into a single JSON under exclusive lock. This is the authoritative reproducibility record for HPO runs.

### Checkpoints (resumable)

```
results/<experiment_lower>/<task>/<dataset>/<method>/<HPO|NO_HPO>/
    <dataset>_<method>_fold<id>_seed<n>_<config_hash>/
        epoch-last-*.pth
        best-val-*.pth
        trlog
```

The `<config_hash>` is a stable SHA1 of `(dataset, method, fold, seed, config)`. Re-running the same configuration **resumes** from the existing checkpoint instead of retraining — crucial for SLURM job recovery.

### Folds cache

```
.cache/folds/
    joblib/...
```

`joblib.Memory` caches every prepared `(dataset, split params)` tuple across processes, so SLURM workers preparing the same dataset reuse the cached CV split instead of re-running PCA / winsorization / outlier removal per worker. Safe to delete; will rebuild on next run.

### Figures

```
figures/<experiment_lower>/
    pd/<plot_name>.{pdf,png}
    lgd/<plot_name>.{pdf,png}
```

Written by `src.visualizations.experiment_plots` (heatmaps, ranking bars, learning / imbalance curves) and `src.visualizations.calibration_plots` (reliability diagrams). PDFs for papers, PNGs for slides.

---

## 6. Frequently asked operational questions

**Q. A SLURM job ran out of wall-clock time. Do I lose its work?**
No. Each fold checkpoints into a hashed directory; restarting picks up where it died. The merged HPO JSON only gets a fold-id update when that fold *finishes* — partial runs don't corrupt it.

**Q. How do I add a brand-new TALENT method?**
TALENT's registry is the single source of truth. Once the method is registered there, this repo picks it up automatically (`CPU_METHODS` / `GPU_METHODS` / `FOUNDATION_METHODS` are all derived). Just flip its toggle in `scripts/Experiment*/config/CONFIG_METHOD.yaml`.

**Q. How do I add a brand-new credit dataset?**
Drop raw files into `data/raw/{pd|lgd}/<dataset>/`, add a dataset-specific block to `src/data/dataset_preprocessing.py` if it needs special cleaning, then toggle it on in `scripts/Experiment*/config/CONFIG_DATA.yaml`. The first run will populate `data/processed/...`.

**Q. How do I bump the TALENT version?**
Edit `requirements.txt` line 41 (`TALENT @ git+https://...@<branch-or-sha>`) and `pip install -r requirements.txt --upgrade`. Run `pytest tests/test_method_config.py::TestNewMethodsRegistered -v` to confirm the four new foundation models are still registered.

**Q. Where are the actual model checkpoints?**
TALENT downloads them lazily on first use (the new bundled-path fix uses `importlib.resources` so they live inside the TALENT package's install dir, not in this repo). For air-gapped VSC nodes, pre-fetch with `python -c "import TALENT; TALENT.run('tabpfn_v3', ...)"` on a login node first.

**Q. The tests pass but I can't repro a result number from a previous run.**
Possible causes: (1) TALENT version drift — pin the SHA in `requirements.txt`. (2) numpy/pytorch float determinism on different GPUs — record the GPU in the SLURM `.out` log; we don't promise bit-identical reproducibility across hardware. (3) Threshold tuning is now on by default; runs from before that landed used `argmax` thresholds.
