# TabPFNCredit

**Benchmarking tabular foundation models for credit-risk prediction.**

A credit-risk evaluation harness on top of [TALENT](https://github.com/LAMDA-Tabular/TALENT). It preprocesses ≈14 PD + ≈7 LGD datasets into TALENT-format arrays, assembles deterministic CV folds, runs every TALENT method (≈55 of them, including the full TabPFN family v1/v2/v2.5/v3/Real, TabICL v1/v2, TabDPT, MITRA, LimiX, MLPs, transformers, tree ensembles) under both `NO_HPO` and `HPO` modes, and stores per-fold metrics + predictions in a per-(dataset, method) JSON + npz layout that is easy to inspect, easy to migrate, and SLURM-array-safe.

---

## Contents

- [Quick start](#quick-start)
- [Tasks, datasets, methods](#tasks-datasets-methods)
- [The four experiments](#the-four-experiments)
- [Pipeline](#pipeline)
- [The CLI](#the-cli)
- [Running on the VSC (Genius / wICE)](#running-on-the-vsc-genius--wice)
- [Result storage](#result-storage)
- [Logging](#logging)
- [Migrating legacy `.pkl` results](#migrating-legacy-pkl-results)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Developer tooling](#developer-tooling)
- [FAQ](#faq)

---

## Quick start

```bash
git clone https://github.com/andreasgoethals/tabpfncredit.git
cd tabpfncredit
pip install -r requirements_local.txt    # or requirements.txt for the lightweight set
pip install -e .                         # exposes the `tabpfncredit` CLI

# Smoke-test: ~30 seconds, 34 tests should pass
pytest tests/ -m "not slow and not gpu"

# Run one (dataset, method, task) cell of Experiment 0
tabpfncredit run --experiment Experiment0 \
    --dataset 0001.gmsc --method tabpfn_v3 --task pd --verbose
```

Results land at `results/experiment0/pd/0001.gmsc/tabpfn_v3.json` + `.npz`.

---

## Tasks, datasets, methods

| Task | Type | Datasets | Headline metrics |
|---|---|---|---|
| **PD** (Probability of Default) | Binary classification | 14 | AUC, Gini, KS, F1, Brier, ECE, Expected_Loss_Normalized |
| **LGD** (Loss Given Default)    | Regression on `[0, 1]`  | 7  | R², RMSE, MAE, Spearman_Corr |

The 55 methods cover **foundation models** (TabPFN family, TabICL v1/v2, TabDPT, MITRA, LimiX, HyperFast, TabPTM), **transformer-based** (FT-Transformer, SAINT, AutoInt, AMFormer, T2G-Former, TROMPT, …), **MLP/ResNet** variants (RealMLP, MLP_PLR, TabM, …), **tree-mimic** networks (TabNet, NODE, GrowNet, DCN-v2, GRANDE), and **classical ML** (XGBoost, CatBoost, LightGBM, RandomForest, LogReg, KNN, SVM, NaiveBayes, NCM, dummy).

The single source of truth for what each method *needs* — its `cat_policy`, `normalization`, output type, GPU/CPU placement, in-context row limit, HPO support — lives inside TALENT's `MethodSpec` registry. This wrapper queries it via [`src/methods/method_config.py`](src/methods/method_config.py); no method metadata is duplicated here.

---

## The four experiments

| # | Question it answers | Folds | HPO | Datasets | Methods |
|---|---|---|---|---|---|
| **0** | "Does each method run end-to-end on real credit data?" — pilot screening. | 1 | NO_HPO | all 14 PD + 7 LGD | all enabled methods |
| **1** | The headline benchmark. Every (dataset, method) pair in both `NO_HPO` and `HPO`. Drives the paper figures. | 5 | NO_HPO + HPO | all 14 PD + 7 LGD | all enabled methods |
| **2** | Learning-curve analysis: metric vs training-set size. Configurable `row_max` / `row_min` / `row_step` **per task**. | 5 | NO_HPO | PD: datasets with ≥20 000 rows. LGD: datasets with ≥5 000 rows. | `tabpfn_v3`, `tabicl_v2`, `xgboost`, `LogReg` (PD) / `LinearRegression` (LGD) |
| **3** | Class-imbalance stress test: minority proportion swept **0.15 → 0.0025 in steps of 0.0005**. PD only. | 5 | NO_HPO | PD subset | `tabpfn_v3`, `tabicl_v2`, `xgboost`, `LogReg` |

Each experiment is configured by three YAML files under `scripts/Experiment<N>/config/`:

- `CONFIG_DATA.yaml` — split sizes, seed, dataset toggles, paths.
- `CONFIG_METHOD.yaml` — `{method: true/false}` per task.
- `CONFIG_EXPERIMENT.yaml` — training knobs and experiment-specific sweep parameters (e.g. `learning_curve.pd` / `learning_curve.lgd`; `minority_proportion_*`).

CV splits are **stratified** for PD (preserves class proportions across folds) and **K-fold** for LGD (continuous target). The `row_limit` field in `CONFIG_DATA.yaml` caps the *total* dataset before splitting; the method-intrinsic `train_row_limit` (e.g. 50 000 for TabPFN v2.5, ~1M for TabPFN v3 with `ignore_pretraining_limits=true`) caps only the *training* set after splitting so test/val coverage stays comparable across methods.

---

## Pipeline

```
data/raw/{pd,lgd}/<dataset>/*.csv
        │
        ▼
src/data/dataset_preprocessing.py   # per-dataset cleaning + leakage scrubbing
        │
        ▼
data/processed/{pd,lgd}/<dataset>/  # TALENT-format (N.npy, C.npy, y.npy, info.json)
        │
        ▼
src/data/data_feeder.py             # StratifiedKFold (PD) / KFold (LGD)
                                    # + post-split winsorize / drop near-constant / PCA
                                    # cached across SLURM workers via joblib.Memory
        │
        ▼
src/methods/method_runner.py :: run_talent_method()
        │   • foundation-model val/test downsampling (OOM safety)
        │   • TALENT.run() once per fold (returns RunResult with
        │     predict_proba, predict_labels, threshold, metrics)
        │   • enrich_pd_metrics / enrich_lgd_metrics  (Gini, KS, MAPE, …)
        │   • cost_sensitive_summary  (Expected loss, profit curve, H-measure)
        │   • resumable checkpoint dir hashed on (config, fold, seed)
        ▼
src/utils/result_io.py :: save_method()
        │   results/<exp>/<task>/<dataset>/<method>.json
        │   results/<exp>/<task>/<dataset>/<method>.npz
        ▼
src/utils/summarize_results_polars.py :: summarize_to_csv()
        │   results/summaries/<exp>_per_fold.csv
        │   results/summaries/<exp>_per_method.csv
        ▼
notebooks/Experiment*.ipynb         # thin: load CSV, call src.visualizations
                                    # figures saved to figures/<exp>/
```

---

## The CLI

The `tabpfncredit` console script (registered in `pyproject.toml`) is the recommended entry point for everything. It exposes five commands; each derives its method list and partition filter from TALENT's registry so adding a new method requires no CLI code changes.

```bash
# List every registered method (optional filters)
tabpfncredit list
tabpfncredit list --architecture deep --hardware gpu

# Run one (dataset, method, task) cell
tabpfncredit run --experiment Experiment0 \
    --dataset 0001.gmsc --method tabpfn_v3 --task pd

# Run the (dataset, method, task) cell assigned to a SLURM array slot
tabpfncredit slurm-task --experiment Experiment0 \
    --partition gpu_foundation --array-id $SLURM_ARRAY_TASK_ID

# Aggregate fold results into per-fold and per-method CSVs
tabpfncredit summarize --experiment Experiment0

# Pretty-print the merged YAML config for an experiment
tabpfncredit config --experiment Experiment1
```

The `--partition` filter for `slurm-task` is one of `cpu`, `gpu_foundation`, `gpu_standard`, or `all`. The method list for each is computed by filtering TALENT's registry by `hardware` and the `FOUNDATION_METHODS` set in `src/methods/method_config.py`.

---

## Running on the VSC (Genius / wICE)

Copy / sync the repo to `$VSC_DATA/TabPFNCredit/` and provision the conda env:

```bash
ssh genius.hpc.kuleuven.be
cd $VSC_DATA
module purge
source $VSC_DATA/miniconda3/bin/activate
conda create -y -n TabPFNCredit python=3.10
conda activate TabPFNCredit
cd TabPFNCredit
pip install -r requirements_vsc.txt        # default targets wICE H100 (cu121)
pip install -e .
```

To target the legacy Genius P100 nodes, uncomment the cu118 block in `requirements_vsc.txt` first.

A SLURM submission looks like:

```bash
sbatch scripts/Experiment0/Experiment0_GPU0.slurm   # GPU methods
sbatch scripts/Experiment0/Experiment0_CPU0.slurm   # CPU methods
```

The SLURM scripts are templated by `scripts/_slurm_templates.py`. They each:

1. Stagger their start with `sleep $((SLURM_ARRAY_TASK_ID % 30))` — deterministic, avoids the thundering-herd I/O storm that a `RANDOM%60` stagger would cause.
2. Activate the `TabPFNCredit` conda env from `$VSC_DATA/miniconda3`.
3. Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` to keep foundation models from fragmenting the GPU's allocator.
4. Invoke `python scripts/Experiment<N>/Experiment<N>_{CPU,GPU,GPU_Foundation,GPU_Standard}.py --array_id=$SLURM_ARRAY_TASK_ID` which deterministically picks one `(dataset, method, task)` cell from the experiment's enabled list.

The new way to drive a SLURM array (which also derives the method list from TALENT's registry instead of hand-maintained Python lists) is the unified Typer CLI:

```bash
# inside a SLURM array script:
tabpfncredit slurm-task \
    --experiment Experiment0 \
    --partition gpu_foundation \
    --array-id $SLURM_ARRAY_TASK_ID
```

Both paths work; the legacy `Experiment<N>_*.py` drivers are kept for backwards compatibility.

---

## Result storage

Each `(experiment, task, dataset, method)` tuple gets **one JSON + one npz** file. Different SLURM array slots never write to the same file, so no locks are needed for the result writes.

```
results/<experiment>/<task>/<dataset>/<method>.json    ─ scalars: metrics, threshold, fit_time, predict_time, hpo flag, dataset info, aggregated mean/std across folds
results/<experiment>/<task>/<dataset>/<method>.npz     ─ arrays: y_true, y_prob, y_pred, val_y_true, val_y_prob (keyed `fold_<id>_<name>`)
```

A typical Experiment-1 sweep produces ≈770 JSON + ≈770 npz files (55 methods × 14 datasets). `jq` queries individual JSONs cheaply; the polars summariser scans all of them in seconds.

### Aggregated CSVs

```
results/summaries/
    <experiment>_per_fold.csv      # one row per (dataset, method, fold)
    <experiment>_per_method.csv    # one row per (dataset, method), with mean / std / median per metric
```

Built by `tabpfncredit summarize --experiment <name>`, backed by [polars](https://pola.rs) (≈10× faster than the previous pandas chain on a full benchmark sweep). These are what the notebooks consume.

### Resumable checkpoints

```
results/.checkpoints/<task>/<dataset>/<method>/<dataset>_<method>_fold<id>_seed<n>_<hash>/
    best-val-*.pth
    epoch-last-*.pth
    trlog
```

The `<hash>` is a stable SHA1 of `(dataset, method, fold, seed, config)`. Re-running the same configuration *resumes* from the existing checkpoint instead of retraining — crucial for SLURM job recovery.

### Folds cache

```
.cache/folds/
    joblib/...
```

`joblib.Memory` caches every prepared `(dataset, split params)` tuple across processes, so SLURM workers preparing the same dataset reuse the cached CV split instead of re-running PCA / winsorization / outlier removal per worker. Safe to delete; will rebuild on next run.

### Figures

```
figures/<experiment>/
    pd/<plot_name>.{pdf,png}
    lgd/<plot_name>.{pdf,png}
```

Written by `src.visualizations.experiment_plots` (heatmaps, ranking bars, learning / imbalance curves) and `src.visualizations.calibration_plots` (reliability diagrams). PDFs for papers, PNGs for slides.

---

## Logging

Per experiment we keep three log files under `results/<experiment>/logs/`:

- **`<dataset>_<method>.log`** — one per (dataset, method); detailed DEBUG-level trace useful for postmortems.
- **`summary.log`** — one shared file; INFO line per task START and DONE/FAIL (with wall-clock and the headline metric). The single source for "what ran when".
- **`errors.log`** — only ERROR lines (full tracebacks). Single grep target after a failed sweep.

SLURM's own `--output` / `--error` streams (the `*.out` / `*.err` files SLURM writes itself) live in the same `logs/` directory — there is **no separate `slurm/` subfolder**. The verbosity policy is intentionally minimal: each task logs its start, its end (with wall-clock and the headline metric), and any errors. Everything more granular sits at DEBUG and only appears in the per-task `.log`.

---

## Migrating legacy `.pkl` results

We re-run Experiments 0, 2, and 3 from scratch but want to **keep the existing Experiment 1 results** so we don't pay the GPU-hour cost twice. The migration script converts the old layout (`results/experiment1/<task>/<dataset>.pkl` keyed by `{HPO_mode: {method: {fold_id: ...}}}`) into the new per-(dataset, method) JSON + npz layout:

```bash
# 1. Inspect what would happen
python scripts/migrate_pkl_to_json.py --experiment experiment1 --dry-run

# 2. Run the migration for real
python scripts/migrate_pkl_to_json.py --experiment experiment1

# 3. Optional: delete the old pickles once you're happy
python scripts/migrate_pkl_to_json.py --experiment experiment1 --delete-old
```

NO_HPO results land as `<method>.json`; HPO results land as `<method>__HPO.json` so both modes coexist. Newly-added foundation models (`tabpfn_v3`, `tabicl_v2`, `tabpfn_v2_5`, `tabdpt`) are then run with the regular Experiment 1 driver and end up in the same layout alongside the migrated entries.

---

## Repository layout

```
src/
├── data/
│   ├── preprocessing.py            # cached TALENT-format conversion (preprocess_dataset)
│   ├── dataset_preprocessing.py    # per-dataset cleaning rules
│   └── data_feeder.py              # CV-fold assembly + post-split anti-leakage
├── methods/
│   ├── method_config.py            # thin layer over TALENT registry (PreprocessingConfig + derived sets)
│   ├── method_runner.py            # run_talent_method() -- TALENT.run() per fold + credit-risk metric enrichment
│   ├── method_metrics.py           # PD / LGD metric helpers (Gini, KS, MAPE-with-zeros, ...)
│   └── cost_metrics.py             # expected loss + profit curves + H-measure
├── utils/
│   ├── config_reader.py            # YAML loader with per-experiment validators
│   ├── result_io.py                # save_method / load_method / scan_results
│   ├── file_lock.py                # cross-platform FileLock (fcntl / portalocker)
│   ├── logging_setup.py            # hybrid per-task + summary + errors logging
│   ├── summarize_results_polars.py # polars-backed CSV aggregator
│   ├── storage_handler.py          # experiment-path helper used by SLURM drivers
│   └── remove_results.py           # selective method removal
├── visualizations/
│   ├── experiment_plots.py         # heatmaps, ranking bars, learning / imbalance curves
│   └── calibration_plots.py        # reliability diagrams
└── cli.py                          # Typer entry point: list / run / slurm-task / summarize / config

scripts/
├── Experiment{0,1,2,3}/
│   ├── config/CONFIG_{DATA,METHOD,EXPERIMENT}.yaml
│   ├── Experiment<N>.py            # shared core (per-task run)
│   ├── Experiment<N>_CPU.py        # CPU-method orchestrator (legacy)
│   ├── Experiment<N>_GPU*.py       # GPU-method orchestrator(s) (legacy)
│   └── Experiment<N>_*.slurm       # SLURM array submissions
├── _slurm_templates.py             # SLURM-script generator
└── migrate_pkl_to_json.py          # legacy results migration

notebooks/                          # thin viewers: load CSV, call src.visualizations
tests/                              # pytest suite
data/                               # raw + processed datasets
results/                            # per-(dataset, method) JSON+npz + logs + figures
```

---

## Tests

```bash
pytest tests/                 # 34 fast tests, ~10 seconds
pytest tests/ -m smoke        # end-to-end runs of the cheapest methods on synthetic data
pytest tests/ -m "not gpu"    # CI invocation -- auto-skips GPU-only tests
```

The suite covers (a) every bug we've fixed (regression tests), (b) the registry-derived sets in `method_config.py`, (c) `method_metrics`, `cost_metrics`, `calibration_plots`, `file_lock`, and (d) the registry import surface.

---

## Developer tooling

`pre-commit` runs hooks on every staged file at commit time (`pip install -r requirements.txt && pre-commit install`):

- **ruff** — lint + autofix + format.
- **black** + **isort** — fallback formatters for files ruff doesn't touch.
- **check-yaml / check-toml / check-json** — config-file sanity.
- **nbstripout** — strips notebook outputs so PRs don't churn on cell outputs.
- **trailing-whitespace**, **end-of-file-fixer**, **check-added-large-files** (>2 MB), **mixed-line-ending** — house-keeping.

If a hook **modifies** a file, the commit fails and asks you to re-stage; if it just **reports** an error, you fix it manually. Delete `.pre-commit-config.yaml` if you don't want any of this — nothing else depends on it.

`pyproject.toml` carries the corresponding `[tool.ruff]`, `[tool.black]`, `[tool.isort]`, `[tool.mypy]`, `[tool.pytest.ini_options]` blocks.

---

## FAQ

**Q. A SLURM job ran out of wall-clock time. Do I lose its work?**
No. Each fold checkpoints into a hashed directory under `results/.checkpoints/`; restarting picks up where it died. The per-(dataset, method) JSON is only written once *every fold* of that method finishes, so partial runs don't corrupt published results.

**Q. How do I add a brand-new TALENT method?**
TALENT's registry is the single source of truth. Once the method is registered there, this repo picks it up automatically (`CPU_METHODS` / `GPU_METHODS` / `FOUNDATION_METHODS` are all derived). Just flip its toggle in `scripts/Experiment*/config/CONFIG_METHOD.yaml`.

**Q. How do I add a brand-new credit dataset?**
Drop raw files into `data/raw/{pd|lgd}/<dataset>/`, add a dataset-specific block to `src/data/dataset_preprocessing.py` if it needs special cleaning, then toggle it on in `scripts/Experiment*/config/CONFIG_DATA.yaml`. The first run will populate `data/processed/...`.

**Q. How do I bump the TALENT version?**
Edit `requirements.txt`'s `TALENT @ git+...` line and `pip install -r requirements.txt --upgrade`. Run `pytest tests/test_method_config.py::TestNewMethodsRegistered -v` to confirm the four newest foundation models are still registered.

**Q. Where are the actual model checkpoints?**
TALENT downloads them lazily on first use (the new bundled-path resolution uses `importlib.resources` so they live inside the TALENT package's install dir, not in this repo). For air-gapped VSC nodes, pre-fetch with `python -c "import TALENT; TALENT.run('tabpfn_v3', ...)"` on a login node first.

**Q. I want to re-summarise without re-running anything.**
`tabpfncredit summarize --experiment Experiment1` scans all `<method>.json` files and rebuilds the per-fold and per-method CSVs in seconds.

**Q. Reproducibility caveats?**
TALENT pins the random seed via `set_seeds(seed)` before each fold. Bit-identical reproducibility across hardware is **not** promised: numpy / pytorch float ops differ slightly between GPU generations. Record the GPU in the SLURM `.out` log and pin the TALENT commit SHA in `requirements.txt` before archiving a release.
