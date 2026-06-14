# TabPFNCredit

**Benchmarking tabular foundation models for credit-risk prediction.**

TabPFNCredit evaluates modern tabular foundation models (the TabPFN
family, TabICL, TabDPT, MITRA, LimiX, HyperFast, TabPTM) against deep
tabular networks (FT-Transformer, SAINT, RealMLP, TabM, …) and classical
baselines (XGBoost, CatBoost, LightGBM, RandomForest, LogisticRegression,
…) on **14 PD (Probability of Default)** and **7 LGD (Loss Given Default)**
datasets. It is built on top of
[TALENT](https://github.com/LAMDA-Tabular/TALENT).

The same `tabpfncredit` CLI runs on a laptop **or** on a SLURM-based HPC
cluster: it auto-detects the environment and either runs in-process or
generates and submits SLURM jobs. The SLURM generator targets the KU
Leuven VSC partitions by default but adapts to any cluster by editing one
table.

---

## Contents

- [1. Quick start](#1-quick-start)
- [2. Command-line interface](#2-command-line-interface)
- [3. Tasks, datasets, and methods](#3-tasks-datasets-and-methods)
- [4. The four experiments](#4-the-four-experiments)
- [5. Pipeline](#5-pipeline)
- [6. Repository layout](#6-repository-layout)
- [7. Result storage and logging](#7-result-storage-and-logging)
- [8. Tests](#8-tests)
- [9. Citation](#9-citation)
- [10. License](#10-license)

---

## 1. Quick start

One command does everything: `tabpfncredit experiment <name>`
auto-preprocesses any missing data, runs locally or submits to SLURM, and
summarizes the results when done. Installation is a single `pip install`
that pulls TALENT and every dependency in one pass.

**Requirements:** Python 3.10–3.12 (3.12 recommended; 3.13+ not yet
supported). Install Python 3.12 from <https://www.python.org/downloads/>
if your launcher can't find it.

### 1.1 Local install and first run

**Windows (PowerShell):**

```powershell
git clone https://github.com/andreasgoethals/tabpfncredit.git
cd tabpfncredit

py -3.12 -m venv tabpfncreditvenv
.\tabpfncreditvenv\Scripts\Activate.ps1

pip install -e ".[local]"          # project + all dependencies (incl. TALENT)
tabpfncredit experiment Experiment0
```

**macOS / Linux:**

```bash
git clone https://github.com/andreasgoethals/tabpfncredit.git
cd tabpfncredit

python3.12 -m venv tabpfncreditvenv
source tabpfncreditvenv/bin/activate

pip install -e ".[local]"
tabpfncredit experiment Experiment0
```

### 1.2 HPC install (SLURM cluster)

```bash
ssh <your.cluster>
module purge
module load cluster/genius/login                   # KU Leuven VSC: unlocks the Python module
module load Python/3.12.3-GCCcore-13.3.0            # 'module spider Python/3.12' to find the exact name

cd <your_repo_clone>
python -m venv tabpfncreditvenv
source tabpfncreditvenv/bin/activate
pip install -e ".[hpc]"
```

Submit the full benchmark — Experiments 0 → 1 → 2 → 3, chained with SLURM
`--dependency=afterok`:

```bash
bash scripts/run_all_experiments.sh
```

Pass one or more experiment names to run only those (list order is the
chain order):

```bash
bash scripts/run_all_experiments.sh Experiment0
bash scripts/run_all_experiments.sh Experiment2 Experiment3
```

> **Running on the KU Leuven VSC?** Read **[VSC_RUN.md](VSC_RUN.md)**
> first. Compute nodes there have no internet, so foundation-model weights
> must be staged once before running; that guide also covers how sweeps
> shard across array tasks under the wall-time and submission limits, and
> how to resume a partial run.

### 1.3 Install profiles

| Profile | For | What it gives you |
|---|---|---|
| `local` | Laptop / workstation | CPU PyTorch + faiss-cpu + dev tools |
| `hpc` | SLURM cluster | CUDA-12 PyTorch + annoy + dev tools. For CUDA 11.8 GPUs append `--index-url https://download.pytorch.org/whl/cu118`. |

A single `pip install -e ".[local]"` (or `".[hpc]"`) installs the project,
TALENT, and all dependencies. The build backend is
[hatchling](https://hatch.pypa.io/latest/), so no `*.egg-info/` is dropped
into the source tree.

### 1.4 Adapting to a non-VSC cluster

The generated SLURM scripts target the VSC partitions (Genius P100, wICE
A100, wICE H100) by default. On a different SLURM cluster, edit the
`PARTITIONS` table in `src/utils/slurm_generator.py` to match your
cluster's CPU / memory / GPU caps and wall-time limits.

---

## 2. Command-line interface

```bash
tabpfncredit experiment <ExperimentName>
  [--task pd|lgd]          # filter to one task
  [--dataset 0001.gmsc]    # filter to one dataset
  [--method tabpfn_v3]     # filter to one method
  [--no-submit]            # generate SLURM scripts but don't sbatch
  [--after <SLURM_JOB_ID>] # chain after another experiment
```

What `experiment` does:

1. Auto-preprocess any dataset not yet cached under
   `data/processed/<task>/<dataset>/`.
2. **Locally**: run every selected cell in-process, then summarize.
3. **On an HPC cluster**: regenerate fresh per-partition SLURM scripts
   under `scripts/<Experiment>/_generated/`, `sbatch` them, and submit a
   summarize job with `--dependency=afterok:<arrays>`.

Helper commands:

| Command | Purpose |
|---|---|
| `tabpfncredit resubmit <names...> \| --all` | Scan results for every missing (task, dataset, method, sweep/HPO) point and submit ONLY those, packed into dense fresh arrays. Wipes previously generated scripts first. Works locally (report + scripts) and on a cluster (also submits). |
| `tabpfncredit summarize --experiment <name>` | Rebuild the per-fold + per-method CSVs. |
| `tabpfncredit list [--show-profile]` | Print registered methods + runtime tier + target partition. |
| `tabpfncredit doctor` | Print environment variables, torch / CUDA info, and the results root. |

Re-running `tabpfncredit experiment <name>` is always safe: completed points
are skipped (one result file per point is the resume unit). `resubmit` does
the same thing more efficiently when most of an experiment is already done —
it queues only the missing work instead of re-sharding everything.

---

## 3. Tasks, datasets, and methods

| Task | Type | Datasets | Headline metrics |
|---|---|---|---|
| **PD** (Probability of Default) | Binary classification | 14 | AUC, Gini, KS, F1, Brier, ECE, AP / AP_normalized, Expected_Loss_Normalized |
| **LGD** (Loss Given Default) | Regression on `[0, 1]` | 7 | R², RMSE, MAE, Spearman_Corr |

The ~55 enabled methods cover **foundation models** (TabPFN family,
TabICL v1/v2, TabDPT, MITRA, LimiX, HyperFast, TabPTM),
**transformers** (FT-Transformer, SAINT, AutoInt, T2G-Former, TROMPT, …),
**MLP / ResNet** (RealMLP, MLP_PLR, TabM, …), **tree-mimic** (TabNet,
NODE, GrowNet, GRANDE), and **classical** baselines (XGBoost, CatBoost,
LightGBM, RandomForest, LogisticRegression, KNN, SVM).

Each method's behaviour (categorical policy, normalisation, GPU/CPU
placement, in-context row limit, HPO support) comes from TALENT's
`MethodSpec` registry. The runtime tier in
`src/methods/runtime_profile.py` drives the partition, wall-time, and
packing decisions for SLURM. Toggle methods on or off per experiment in
`scripts/Experiment*/config/CONFIG_METHOD.yaml`.

### Average Precision, baseline-corrected

Average Precision (area under the precision–recall curve) has a no-skill
baseline equal to the positive prevalence π — so a raw AP of 0.30 is
excellent on a 1%-default dataset but worthless on a 30%-default one, and
raw AP cannot be compared across datasets. Every PD fold therefore also
records the **normalised deviation**

```
AP_normalized = (AP − π) / (1 − π)
```

which is 0 for a no-skill ranker and 1 for a perfect one regardless of
prevalence (Flach & Kull, *Precision-Recall-Gain Curves*, NeurIPS 2015).
This — not the raw absolute gap `AP − π` (which ignores how much headroom
`1 − π` is even available) nor the unbounded ratio `AP / π` — is the
quantity to compare across datasets. All four (`AP`, `AP_baseline` = π,
`AP_minus_baseline`, `AP_normalized`) are stored so you can recompute
either gap yourself.

### LGD targets are clipped to `[0, 1]`

LGD predictions and targets are both clipped to `[0, 1]`: preprocessing
clips the raw `y`, and inference clips every prediction. Each fold result
records how many predictions were clipped via `n_clipped_below` /
`n_clipped_above`.

---

## 4. The four experiments

| # | Question it answers | Folds | HPO | Datasets | Methods |
|---|---|---|---|---|---|
| **0** | Pilot: does each method run end-to-end? Use the outcomes to curate Experiment 1. | 1 | NO_HPO | All 14 PD + 7 LGD. | All toggled in `CONFIG_METHOD.yaml`. |
| **1** | Headline benchmark — drives the paper. | 5 | NO_HPO + HPO | All 14 PD + 7 LGD. | **Curated by hand** in `CONFIG_METHOD.yaml`. |
| **2** | Learning-curve sweep: metric vs training-set size. | 5 | NO_HPO | Auto from `learning_curve.<task>.row_max`: PD ≥ 30 000, LGD ≥ 4 600. | `tabpfn_v3`, `tabicl_v2`, `xgboost`, `LogReg` / `LinearRegression`. |
| **3** | Class-imbalance sweep: minority proportion **0.15 → 0.0025**, step **0.0005**. PD only. | 5 | NO_HPO | PD with ≥ 30 000 rows **and** natural minority rate > `minority_proportion_max`. | `tabpfn_v3`, `tabicl_v2`, `xgboost`, `LogReg`. |

Each experiment is driven by three YAMLs under
`scripts/Experiment<N>/config/`:

- `CONFIG_DATA.yaml` — splits and dataset selection,
- `CONFIG_METHOD.yaml` — per-task method toggles,
- `CONFIG_EXPERIMENT.yaml` — training knobs and sweep parameters.

**Experiment 1's method set is curated by hand.** After Experiment 0
finishes, inspect
`results/experiment0/summaries/experiment0_per_method.csv` and edit
`scripts/Experiment1/config/CONFIG_METHOD.yaml` to enable only the methods
that ran successfully.

**Dataset selection for the sweep experiments is automatic:**

- **Experiment 2** selects every dataset with **≥ `row_max`** rows
  (`learning_curve.<task>.row_max` in `CONFIG_EXPERIMENT.yaml`). `row_max`
  is also the top of the training-size sweep, so the ceiling and the
  dataset set are a single knob (the `CONFIG_DATA.yaml` dataset blocks are
  intentionally empty).
- **Experiment 3** keeps a PD dataset only if it passes **both** filters:
  (1) `min_rows: N` in its `CONFIG_DATA.yaml` (≥ N rows), and (2) its
  natural minority rate **exceeds** `minority_proportion_max` (from
  `CONFIG_EXPERIMENT.yaml`). The sweep subsamples the minority class
  *down* from that ceiling, so a dataset already more imbalanced than the
  ceiling can't reach the top and is dropped (logged at INFO level).

### Why the sweep curves are clean signal

- **Experiment 2 (learning curve):** lowering `row_limit` keeps a **strict
  subset** of the larger cap's rows (fixed-seed, class-stratified for PD so
  the minority survives small caps). The metric change reflects *fewer
  rows*, not a different random draw.
- **Experiment 3 (imbalance):** the subsampling is **nested / cumulative**.
  A single fixed-seed permutation of the minority rows is taken once, and
  each lower target keeps a shorter **prefix** of it — so stepping
  `0.15 → 0.1495` only ever *deletes more* of the same minority rows, never
  re-draws a fresh subset. The trend is therefore attributable purely to
  *fewer minority cases*, with no lucky/unlucky-draw variance between
  adjacent points. Train, validation, and test are all subsampled to the
  target rate; the majority class is never touched, so size shrinks only
  because minority rows leave.

---

## 5. Pipeline

```
data/raw/{pd,lgd}/<dataset>.{csv,parquet}
        |
        v
src/data/dataset_preprocessing.py   # per-dataset cleaning + leakage scrubbing
                                    # + LGD target clipping to [0, 1]
        |
        v
data/processed/{pd,lgd}/<dataset>/  # TALENT-format (N.npy, C.npy, y.npy, info.json)
        |
        v
src/data/data_feeder.py             # StratifiedKFold (PD) / KFold (LGD)
                                    # + post-split winsorize / drop near-constant / PCA
                                    # cached across SLURM workers via joblib.Memory
        |
        v
src/methods/method_runner.py        # TALENT.run() per fold -> RunResult
                                    # + foundation-model val/test downsampling
                                    # + LGD predictions clipped to [0, 1]
                                    # + enrich_{pd,lgd}_metrics + cost_sensitive_summary
        |
        v
src/utils/result_io.py              # one JSON + one npz per (dataset, method[, sweep])
        |
        v
src/utils/result_summary.py         # per-fold + per-method CSVs
        |
        v
notebooks/Experiment*.ipynb         # plots from src.visualizations -> figures/<exp>/*.pdf
```

**Skip-if-done is implicit:** the runner checks for the `<method>.json`
result file and skips that cell if it already exists. There are no
separate checkpoint files, so a re-run only does the work that is missing.

---

## 6. Repository layout

```
README.md                           # this file
VSC_RUN.md                          # KU Leuven VSC run guide (weights, sharding, recovery)
pyproject.toml                      # single-step install + tooling config

src/
  cli.py                            # `tabpfncredit` Typer CLI
  data/
    preprocessing.py                # cached TALENT-format conversion
    dataset_preprocessing.py        # per-dataset cleaning + LGD target clipping
    dataset_inventory.py            # row counts -> min_rows filter
    data_feeder.py                  # CV-fold assembly + post-split anti-leakage
  methods/
    method_config.py                # thin layer over the TALENT registry
    method_runner.py                # TALENT.run() per fold + metric enrichment
    method_metrics.py               # PD / LGD metric helpers
    cost_metrics.py                 # expected loss + profit curves
    runtime_profile.py              # tier + sec/fold per method (drives SLURM)
  utils/
    config_reader.py                # YAML loader (min_rows + validators)
    result_io.py                    # save_method / load_method / scan_results
    result_summary.py               # polars-backed per-fold + per-method CSVs
    slurm_generator.py              # SLURM script generator
    file_lock.py                    # cross-platform FileLock
  visualizations/
    experiment_plots.py             # heatmaps, ranking bars, learning curves
    calibration_plots.py            # reliability diagrams
    data_exploration.py             # backs the Data_Exploration notebook

scripts/
  Experiment{0,1,2,3}/
    config/CONFIG_{DATA,METHOD,EXPERIMENT}.yaml
    _generated/                     # SLURM scripts (auto-emitted, gitignored)
  run_all_experiments.sh            # submit the full chained HPC sweep
  fetch_weights.py                  # download foundation-model weights -> checkpoints/ (run LOCALLY)
  setup_vsc_checkpoints.sh          # provision the uploaded checkpoints/ on the VSC (offline)

notebooks/                          # thin viewers calling src.visualizations
tests/                              # pytest suite

data/                               # raw + processed datasets       (gitignored)
results/                            # per-(dataset, method) JSON+npz  (gitignored)
figures/                            # generated PDF plots             (gitignored)
checkpoints/                        # downloaded model weights        (gitignored)
```

---

## 7. Result storage and logging

Each `(experiment, task, dataset, method[, sweep_point])` tuple produces
one JSON + one npz under `$TABPFN_RESULTS_ROOT` (on an HPC cluster this points
at the large project storage; locally it defaults to `./results`):

```
${TABPFN_RESULTS_ROOT|./results}/<experiment>/<task>/<dataset>/<method>.json
${TABPFN_RESULTS_ROOT|./results}/<experiment>/<task>/<dataset>/<method>.npz
```

Sweep points get a suffix on the method name:

```
results/experiment1/pd/0001.gmsc/xgboost.json            # NO_HPO
results/experiment1/pd/0001.gmsc/xgboost__HPO.json       # HPO
results/experiment2/pd/<dataset>/tabpfn_v3__row20000.json
results/experiment3/pd/<dataset>/tabicl_v2__min0p0025.json
```

Two CSVs are aggregated automatically (locally at the end of an
`experiment` call; on SLURM as a dependency job):

```
<results>/summaries/<experiment>_per_fold.csv
<results>/summaries/<experiment>_per_method.csv
```

Logging uses Python's standard `logging` (run with `--verbose` for
INFO-level: start, finish, headline metric, foundation-model downsampling,
per-fold minority counts). On an HPC cluster each array slot's
stdout/stderr is captured by SLURM under `logs/<experiment>/` on the general
data storage (`$VSC_DATA`), named per job + array id — kept off the project
storage so logs always persist.

Figures are saved as **PDF only** to `figures/<experiment>/<plot>.pdf` and
rendered inline in the notebook outputs. Every notebook wipes its own figure
folder at the top of a run (`reset_figure_dir`), so a rerun never mixes old
and new figures.

### Analysis notebooks

All plotting / statistics code lives under `src/` — the notebooks are thin
viewers:

| Notebook | What it shows |
|---|---|
| `Data_Exploration` | Dataset inventory, class balance, LGD target shapes, per-dataset structure. |
| `Experiment0` | Pilot coverage + quick performance / cost overview. |
| `Experiment1.1-PD` / `1.2-LGD` | Headline benchmark: metric heatmaps, rankings, boxplots, win/loss, PAMA, HPO effect, cost/quality frontier. |
| `Experiment1.3-Stat` | Full statistical methodology of Demšar (2006) + García & Herrera (2008): Friedman + Iman–Davenport, Nemenyi **critical-difference diagrams**, Bonferroni–Dunn and step procedures, all-pairwise APVs (Holm, **Shaffer**, **Bergmann–Hommel**), Wilcoxon / sign tests (backed by `src/utils/statistical_testing.py`). |
| `Experiment2` / `Experiment3` | Learning curves / imbalance curves — one line per method, averaged over datasets. |
| `Results_Checking` | Completeness / sanity audit of the result files. |

---

## 8. Tests

```bash
pytest tests/                 # fast tests, ~10 s
pytest tests/ -m smoke        # end-to-end runs of cheap methods on synthetic data
pytest tests/ -m "not gpu"    # CI invocation -- auto-skips GPU-only tests
```

Coverage includes the registry-derived method sets, the PD / LGD metric
helpers, file locking, calibration plots, sweep-suffix round-trips, and
the JSON+npz `save_method` round-trip.

## 9. Citation

If you use this benchmark, please cite it. GitHub's "Cite this repository"
button reads [`CITATION.cff`](CITATION.cff); a BibTeX entry for the
accompanying paper will be added here on publication.

## 10. License

Released under the MIT License — see [`LICENSE.txt`](LICENSE.txt).

This repository builds on the [TALENT](https://github.com/LAMDA-Tabular/TALENT)
tabular-learning toolkit and the foundation-model packages it wraps
(`tabpfn`, `tabicl`, `tabdpt`, …); each retains its own license and should be
cited separately when used.
