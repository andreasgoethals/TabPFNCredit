# TabPFNCredit

**Benchmarking tabular foundation models for credit-risk prediction.**

A credit-risk evaluation harness on top of [TALENT](https://github.com/LAMDA-Tabular/TALENT). It preprocesses ≈14 PD + ≈7 LGD datasets into TALENT-format arrays, assembles deterministic CV folds, runs every TALENT method (≈55 of them — the full TabPFN family v1/v2/v2.5/v3/Real, TabICL v1/v2, TabDPT, MITRA, LimiX, MLPs, transformers, tree ensembles) under both `NO_HPO` and `HPO` modes, and stores per-fold metrics + predictions in a per-(dataset, method) JSON + npz layout that is easy to inspect and SLURM-array-safe.

The repo is **VSC-first**: the CLI auto-generates SLURM array scripts right-sized to each KU Leuven partition (Genius P100 / wICE A100 / wICE H100), packs cheap methods together to respect the scheduler, and writes results to `$VSC_DATA` (small, permanent, backed up — result files are ≈40 MB per sweep so they fit the 75 GB quota easily).

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
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Developer tooling and `pyproject.toml`](#developer-tooling-and-pyprojecttoml)
- [FAQ](#faq)

---

## Quick start

```bash
git clone https://github.com/andreasgoethals/tabpfncredit.git
cd tabpfncredit
pip install -e ".[dev,local]"     # local dev + tests + lint
pytest tests/ -m "not slow and not gpu"        # ~10 s; 49 tests pass

# Run one (dataset, method) cell of Experiment 0
tabpfncredit run --experiment Experiment0 \
    --dataset 0001.gmsc --method tabpfn_v3 --task pd --verbose
```

Results land at `./results/experiment0/pd/0001.gmsc/tabpfn_v3.{json,npz}`.

---

## Tasks, datasets, methods

| Task | Type | Datasets | Headline metrics |
|---|---|---|---|
| **PD** (Probability of Default) | Binary classification | 14 | AUC, Gini, KS, F1, Brier, ECE, Expected_Loss_Normalized |
| **LGD** (Loss Given Default) | Regression on `[0, 1]` | 7 | R², RMSE, MAE, Spearman_Corr |

The 55 methods cover **foundation models** (TabPFN family, TabICL v1/v2, TabDPT, MITRA, LimiX, HyperFast, TabPTM), **transformer-based** (FT-Transformer, SAINT, AutoInt, AMFormer, T2G-Former, TROMPT, …), **MLP/ResNet** variants (RealMLP, MLP_PLR, TabM, …), **tree-mimic** networks (TabNet, NODE, GrowNet, DCN-v2, GRANDE), and **classical ML** (XGBoost, CatBoost, LightGBM, RandomForest, LogReg, KNN, SVM, NaiveBayes, NCM, dummy).

`MethodSpec` from TALENT's registry is the single source of truth for what each method *needs* (`cat_policy`, `normalization`, GPU/CPU placement, in-context row limit, HPO support). The wrapper queries it via [`src/methods/method_config.py`](src/methods/method_config.py).

Each method also has a **runtime profile** in [`src/methods/runtime_profile.py`](src/methods/runtime_profile.py) — Tier (FAST / MEDIUM / SLOW / FOUNDATION) + an estimated seconds-per-fold — used by the SLURM generator to pick partitions, walltimes, and packing.

---

## The four experiments

| # | Question it answers | Folds | HPO | Datasets | Methods |
|---|---|---|---|---|---|
| **0** | "Does each method run end-to-end?" — pilot screening. | 1 | NO_HPO | all 14 PD + 7 LGD | all enabled |
| **1** | Headline benchmark — every (dataset, method) in `NO_HPO` and `HPO`. Drives the paper. | 5 | NO_HPO + HPO | all 14 PD + 7 LGD | all enabled |
| **2** | Learning-curve sweep: metric vs training-set size. Configurable `row_max` / `row_min` / `row_step` **per task**. | 5 | NO_HPO | PD: ≥20 000 rows. LGD: ≥5 000 rows. | `tabpfn_v3`, `tabicl_v2`, `xgboost`, `LogReg` (PD) / `LinearRegression` (LGD) |
| **3** | Class-imbalance sweep: minority proportion **0.15 → 0.0025** step `0.0005`. PD only. | 5 | NO_HPO | PD subset | `tabpfn_v3`, `tabicl_v2`, `xgboost`, `LogReg` |

Each experiment is driven by three YAMLs under `scripts/Experiment<N>/config/`:

- `CONFIG_DATA.yaml` — split sizes, seed, dataset toggles, paths.
- `CONFIG_METHOD.yaml` — `{method: true/false}` per task.
- `CONFIG_EXPERIMENT.yaml` — training knobs + experiment-specific sweep parameters (`learning_curve.pd/lgd`, `minority_proportion_*`).

CV splits: **stratified** for PD, **K-fold** for LGD. The `row_limit` in `CONFIG_DATA.yaml` caps the *total* dataset before splitting; the method-intrinsic `train_row_limit` (e.g. 50 000 for TabPFN v2.5) caps only the *training* set after splitting so test/val coverage stays comparable across methods.

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
        │   • foundation-model val/test downsampling
        │   • TALENT.run() per fold -> RunResult (predict_proba, threshold, metrics)
        │   • enrich_pd_metrics / enrich_lgd_metrics (Gini, KS, MAPE, ...)
        │   • cost_sensitive_summary (expected loss + profit curves)
        │   • resumable checkpoint dir hashed on (config, fold, seed)
        ▼
src/utils/result_io.py :: save_method()
        │   ${TABPFN_RESULTS_ROOT|./results}/<exp>/<task>/<dataset>/<method>.json + .npz
        ▼
src/utils/summarize_results_polars.py :: summarize_to_csv()
        │   <results>/summaries/<exp>_per_{fold,method}.csv
        ▼
notebooks/Experiment*.ipynb         # thin: load CSV, call src.visualizations
                                    # figures saved to figures/<exp>/
```

---

## The CLI

The `tabpfncredit` console script (registered in `pyproject.toml`) is the recommended entry point for everything. Seven commands:

```bash
# Enumerate registered methods (optionally with runtime tier + chosen partition)
tabpfncredit list --show-profile

# Run one (dataset, method, task) cell -- skips if already complete
tabpfncredit run --experiment Experiment1 --dataset 0001.gmsc \
    --method tabpfn_v3 --task pd --verbose

# Generate VSC-optimised SLURM scripts for one experiment
tabpfncredit slurm-generate --experiment Experiment1

# (Called by the generated SLURM scripts) run the cell assigned to this array slot
tabpfncredit slurm-task --experiment Experiment1 --partition gpu_h100 \
    --array-id $SLURM_ARRAY_TASK_ID

# Aggregate fold results into per-fold and per-method CSVs
tabpfncredit summarize --experiment Experiment1

# Quick environment / VSC sanity check
tabpfncredit doctor
```

---

## Running on the VSC (Genius / wICE)

### One-time setup

```bash
ssh genius.hpc.kuleuven.be   # or wice
module purge
source $VSC_DATA/miniconda3/bin/activate    # or miniforge3 -- both work
conda create -y -n TabPFNCredit python=3.10
conda activate TabPFNCredit

cd $VSC_DATA/TabPFNCredit
pip install -e ".[vsc]"     # cu121 PyTorch wheels + portalocker
```

### Per-sweep workflow

```bash
# 1) Generate the SLURM array scripts for one experiment
tabpfncredit slurm-generate --experiment Experiment1

# Output table tells you which scripts were emitted, e.g.:
#   experiment1_cpu.slurm      cpu        28   slots   00:21:33
#   experiment1_gpu_p100.slurm gpu_p100   469  slots   00:27:00
#   experiment1_gpu_h100.slurm gpu_h100   168  slots   00:40:00

# 2) Submit. Each script is fully self-contained (right shebang, modules, conda, ...).
sbatch scripts/Experiment1/_generated/experiment1_cpu.slurm
sbatch scripts/Experiment1/_generated/experiment1_gpu_p100.slurm
sbatch scripts/Experiment1/_generated/experiment1_gpu_h100.slurm

# 3) Monitor
slurm_jobinfo <jobid>
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

### What the generated SLURM scripts do for you (per VSC docs)

The generator follows every documented VSC best practice:

- **Shebang `#!/bin/bash -l`** so `~/.bashrc` and the cluster module load. The lowercase L is critical — without it modules do not initialise.
- **`--clusters=wice` / `--clusters=genius`** (plural, the canonical form) plus the mandatory `--account=lp_verbekelab`.
- **Per-partition CPU + memory right-sized** to the documented per-GPU caps:
  | Partition | CPUs / GPU | Memory / GPU | Walltime cap | Where |
  |---|---|---|---|---|
  | `batch` (CPU) | 36 | ~140 GB | 72 h | Genius |
  | `gpu_p100` | 9 | 44 GB | 72 h | Genius |
  | `gpu_a100` | 18 | 125 GB | 72 h | wICE |
  | `gpu_h100` | 16 | 186 GB | 72 h | wICE |
- **`--gpu_cmode=shared`** so multiple short methods can share one GPU when packed together (default at VSC; we set it explicitly).
- **Results land on `$VSC_DATA`** (small, permanent, 75 GB quota — plenty for ~40 MB of result files per sweep). For unusually heavy I/O you can override the location by exporting `TABPFN_RESULTS_ROOT=$VSC_SCRATCH/TabPFNCredit/results` inside the SLURM script before the `tabpfncredit slurm-task` call; we keep this off by default because the result writes are tiny and `$VSC_SCRATCH` files are auto-deleted after 30 days of inactivity.
- **`module --force purge`** in the job script (modules in `~/.bashrc` are documented to be fragile).
- **Cheap methods get packed**: any FAST / MEDIUM tier cell gets bundled with sibling cells so each array slot has roughly 10 minutes of work. Foundation models always go solo (one array slot each) so a slow run does not block a fast one.
- **Deterministic stagger** `sleep $((SLURM_ARRAY_TASK_ID % 30))` avoids the metadata-server thundering-herd that simultaneous job starts would cause.
- **Worker-NG hint**: when a sweep has more than ~500 array slots the CLI prints a notice — VSC's docs recommend the Worker framework over plain job arrays for that scale, and you can switch by following the printed link.

### Cluster pick

The generator picks Genius P100 for "standard" GPU methods and wICE H100 for foundation models (configurable with `--prefer-h100=false` to fall back to A100 if H100 is contended). The TRES billing weights (per the docs) are roughly:

| Resource | Weight | Relative cost |
|---|---|---|
| P100-min | 41.67 | 1× |
| A100-min | 141.67 | ~3.4× |
| H100-min | 569.44 | ~14× |

So budget-watchers should use `tabpfncredit list --show-profile` to verify which partition each method lands on before submitting a big sweep, and `sam-quote sbatch <script>` to estimate credit cost beforehand.

---

## Result storage

Each `(experiment, task, dataset, method[, sweep_point])` tuple gets **one JSON + one npz** file. Different SLURM array slots never write to the same file, so no locks are needed for result writes.

```
${TABPFN_RESULTS_ROOT|./results}/<experiment>/<task>/<dataset>/<method>.json   # scalars
${TABPFN_RESULTS_ROOT|./results}/<experiment>/<task>/<dataset>/<method>.npz    # arrays
```

For sweep experiments the method name gets a suffix encoding the sweep point so each point still gets its own file:

```
results/experiment1/pd/0001.gmsc/xgboost.json            # NO_HPO
results/experiment1/pd/0001.gmsc/xgboost__HPO.json       # HPO
results/experiment2/pd/<dataset>/tabpfn_v3__row20000.json
results/experiment3/pd/<dataset>/tabicl_v2__min0p0025.json
```

On VSC, `TABPFN_RESULTS_ROOT` is set by the generated SLURM scripts to `$VSC_DATA/TabPFNCredit/results` (permanent + backed up). Locally it is unset and defaults to `./results`.

### Aggregated CSVs

```
<results>/summaries/<experiment>_per_fold.csv      # one row per (dataset, method, fold)
<results>/summaries/<experiment>_per_method.csv    # one row per (dataset, method) -- aggregated
```

Built by `tabpfncredit summarize --experiment <name>`, backed by [polars](https://pola.rs).

### Resumable checkpoints

```
<results>/.checkpoints/<task>/<dataset>/<method>/<dataset>_<method>_fold<id>_seed<n>_<hash>/
```

`<hash>` is a stable SHA1 of `(dataset, method, fold, seed, config)`. Re-running the same configuration *resumes* from the existing checkpoint instead of retraining — crucial for SLURM job recovery.

### Folds cache

```
.cache/folds/joblib/...
```

`joblib.Memory` caches every prepared `(dataset, split params)` tuple across processes, so SLURM workers share the cached CV split instead of re-running PCA / winsorization / outlier removal per worker.

### Figures

```
figures/<experiment>/
    pd/<plot_name>.{pdf,png}
    lgd/<plot_name>.{pdf,png}
figures/data_exploration/<plot_name>.{pdf,png}
```

Notebooks call `reset_figure_dir(...)` first so re-running a notebook gives a clean output set.

---

## Logging

Per experiment we keep three log files under `<results>/<experiment>/logs/`:

- **`<dataset>_<method>.log`** — one per (dataset, method); detailed DEBUG-level trace.
- **`summary.log`** — INFO line per task START and DONE/FAIL (with wall-clock and headline metric). Single grep target for sweep status.
- **`errors.log`** — only ERROR lines (full tracebacks).

SLURM's `*.out` / `*.err` files land in the same `logs/` directory (no separate `slurm/` subfolder). Verbosity policy: minimal. INFO = start, finish, headline metric, downsampling. WARNING = LGD clipping > 5%, val/test caps. ERROR = traceback.

---

## Repository layout

```
src/
├── cli.py                          # `tabpfncredit` Typer CLI
├── data/
│   ├── preprocessing.py            # cached TALENT-format conversion
│   ├── dataset_preprocessing.py    # per-dataset cleaning
│   └── data_feeder.py              # CV-fold assembly + post-split anti-leakage
├── methods/
│   ├── method_config.py            # thin layer over TALENT registry
│   ├── method_runner.py            # TALENT.run() per fold + metric enrichment
│   ├── method_metrics.py           # PD / LGD metric helpers
│   ├── cost_metrics.py             # expected loss + profit curves
│   └── runtime_profile.py          # tier + sec/fold per method (drives SLURM)
├── slurm/
│   └── generator.py                # VSC-optimised SLURM script generator
├── utils/
│   ├── config_reader.py            # YAML loader with per-experiment validators
│   ├── result_io.py                # save_method / load_method / scan_results
│   ├── file_lock.py                # cross-platform FileLock
│   ├── logging_setup.py            # hybrid per-task + summary + errors
│   ├── summarize_results_polars.py # polars-backed CSV aggregator
│   ├── storage_handler.py          # experiment-path helper
│   └── remove_results.py           # selective method removal
└── visualizations/
    ├── experiment_plots.py         # heatmaps, ranking bars, learning curves
    ├── calibration_plots.py        # reliability diagrams
    └── data_exploration.py         # backs the Data_Exploration notebook

scripts/
├── Experiment{0,1,2,3}/
│   ├── config/CONFIG_{DATA,METHOD,EXPERIMENT}.yaml
│   ├── Experiment<N>.py            # shared core (per-cell run)
│   └── _generated/                 # emitted by `tabpfncredit slurm-generate`
└── (no more hand-written .slurm or _Setup.py files)

notebooks/                          # thin viewers calling src.visualizations
tests/                              # pytest suite
data/                               # raw + processed datasets
results/                            # local results (overridden by $TABPFN_RESULTS_ROOT)
figures/                            # all generated plots
```

---

## Tests

```bash
pytest tests/                 # 49 fast tests, ~10 seconds
pytest tests/ -m smoke        # end-to-end runs of cheap methods on synthetic data
pytest tests/ -m "not gpu"    # CI invocation -- auto-skips GPU-only tests
```

Coverage includes the registry-derived sets in `method_config`, the `method_metrics` / `cost_metrics` helpers, file locking, calibration plots, sweep-suffix round-trips, and every previously-fixed bug as a regression test.

---

## Developer tooling and `pyproject.toml`

### What is `pyproject.toml`?

`pyproject.toml` is the modern (PEP 517 / 518 / 621) **single source of truth** for everything about a Python project:

1. **Build configuration** — tells `pip` how to install the project. Without it, `pip install -e .` would not work.
2. **Project metadata** — name, version, dependencies, entry points (replaces the old `setup.py` + `setup.cfg`).
3. **Tool configuration** — `[tool.ruff]`, `[tool.black]`, `[tool.mypy]`, `[tool.pytest.ini_options]` all live here so all linters and formatters read one file.

In this repo it does three concrete things:

| Block | Purpose |
|---|---|
| `[project]` + `dependencies` | The runtime stack (TALENT, sklearn, polars, typer, …) |
| `[project.optional-dependencies]` | `dev` / `local` / `vsc` extras — install via `pip install -e ".[vsc]"` |
| `[project.scripts]` | Registers the `tabpfncredit` console command pointing at `src/cli.py:app` |
| `[tool.ruff]` / `[tool.black]` / `[tool.isort]` / `[tool.mypy]` / `[tool.pytest.ini_options]` | One-stop configuration |
| `[tool.setuptools.packages.find]` | What to include in the wheel |

**Is it necessary?** Yes — for the editable install, the `tabpfncredit` CLI, and the centralised tool configs. Removing it would break `pip install -e .` and the `tabpfncredit` entry point.

### `.pre-commit-config.yaml`

Optional Git-hook framework (`pip install pre-commit && pre-commit install`). Every commit then auto-runs ruff (lint + autofix + format), black, isort, JSON/YAML syntax checks, trailing-whitespace strip, big-file guard, and `nbstripout` (strips notebook outputs). Delete this file if you do not want any of this — nothing else depends on it.

---

## FAQ

**Q. A SLURM job ran out of wall-clock. Do I lose its work?**
No. Each fold checkpoints into a hashed directory under `<results>/.checkpoints/`. Restarting picks up where it died. Per-(dataset, method) JSON is only written once every fold of that method finishes, so partial runs do not corrupt published results.

**Q. How do I add a brand-new TALENT method?**
TALENT's registry is the single source of truth. Once registered there, this repo picks it up automatically (`CPU_METHODS` / `GPU_METHODS` / `FOUNDATION_METHODS` all derive from it). Flip its toggle in `scripts/Experiment*/config/CONFIG_METHOD.yaml`, optionally add a `runtime_profile.py` entry so the SLURM generator picks the right partition.

**Q. How do I add a brand-new credit dataset?**
Drop raw files into `data/raw/{pd|lgd}/<dataset>/`, add a dataset-specific block to `src/data/dataset_preprocessing.py` if it needs cleaning, then toggle it on in `scripts/Experiment*/config/CONFIG_DATA.yaml`.

**Q. How do I bump TALENT?**
Edit `pyproject.toml`'s `TALENT @ git+...` line and `pip install -e . --upgrade`. Run `pytest tests/test_method_config.py::TestNewMethodsRegistered -v` to confirm registered methods still resolve.

**Q. The migrated Experiment 1 results — do they have the new cost / calibration metrics?**
Yes — we already backfilled `Expected_Loss_Normalized`, `Optimal_Profit`, `Optimal_Profit_Threshold`, `H_Measure`, `Brier`, `ECE` from the stored `y_true` / `y_prob` arrays in the `.npz` files (840 result files, 4 200 fold entries).

**Q. Reproducibility caveats?**
TALENT pins the random seed before every fold. Bit-identical reproducibility across GPU generations is **not** promised: numpy / pytorch float ops differ slightly between P100, A100, H100. Record the GPU in the SLURM `.out` log and pin the TALENT commit SHA in `pyproject.toml`'s `TALENT @ git+...` line before archiving a release.

**Q. I want to re-summarise without re-running anything.**
`tabpfncredit summarize --experiment Experiment1` scans every `<method>.json` and rebuilds the CSVs in seconds.

**Q. Why is the figure folder cleared at the top of each notebook?**
So re-running a notebook gives you a clean output set instead of mixing old + new figures. Each notebook calls `reset_figure_dir(FIGURES_DIR)` before anything else.
