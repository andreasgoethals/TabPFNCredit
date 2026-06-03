# TabPFNCredit

**Benchmarking tabular foundation models for credit-risk prediction.**

TabPFNCredit evaluates modern tabular foundation models (the TabPFN
family, TabICL, TabDPT, MITRA, LimiX, HyperFast, TabPTM) against deep
tabular networks (FT-Transformer, SAINT, RealMLP, TabM, …) and classical
baselines (XGBoost, CatBoost, LogisticRegression, …) on **14 PD
(Probability of Default)** and **7 LGD (Loss Given Default)** datasets,
on top of [TALENT](https://github.com/LAMDA-Tabular/TALENT).

You can run the benchmark on a laptop or on a SLURM-based **HPC
cluster**; the CLI auto-detects which environment it is in. The
published benchmark was run on the KU Leuven VSC (Genius P100, wICE
A100, wICE H100) — the SLURM generator is tuned to that cluster but is
trivially adaptable to others.

---

## Contents

- [1. Quick start](#1-quick-start)
- [2. Command-line interface](#2-command-line-interface)
- [3. Repository layout](#3-repository-layout)
- [4. Pipeline](#4-pipeline)
- [5. Tasks, datasets, and methods](#5-tasks-datasets-and-methods)
- [6. The four experiments](#6-the-four-experiments)
- [7. Result storage and logging](#7-result-storage-and-logging)
- [8. Tests](#8-tests)

---

## 1. Quick start

`tabpfncredit experiment <name>` is the only command you need. It
auto-preprocesses missing data, auto-runs locally or auto-submits to
SLURM, and auto-summarizes once done.

### 1.1 Local install (Windows PowerShell)

Requires **Python 3.12** (3.10 and 3.11 also work; 3.13 / 3.14 don't).
Install Python 3.12 from <https://www.python.org/downloads/> if `py
-3.12` complains it isn't installed.

```powershell
git clone https://github.com/andreasgoethals/tabpfncredit.git
cd tabpfncredit

py -3.12 -m venv tabpfncreditvenv
.\tabpfncreditvenv\Scripts\Activate.ps1

pip install -e ".[local]"

tabpfncredit experiment Experiment0
```

macOS / Linux substitute (for the two PowerShell-only lines):

```bash
# python3.12 -m venv tabpfncreditvenv
# source tabpfncreditvenv/bin/activate
```

### 1.2 HPC install (SLURM cluster)

```bash
ssh <your.cluster>
module purge
conda create -y -n tabpfncreditvenv python=3.12
conda activate tabpfncreditvenv

cd <your_repo_clone>
pip install -e ".[hpc]"
```

To submit the full benchmark — Experiment 0 → 1 → 2 → 3 chained with
SLURM `--dependency=afterok`:

```bash
bash scripts/run_all_experiments.sh
```

Pass one or more experiment names to run only those:

```bash
bash scripts/run_all_experiments.sh Experiment0
bash scripts/run_all_experiments.sh Experiment2 Experiment3
```

The generated SLURM scripts target the KU Leuven VSC partitions
(Genius P100, wICE A100, wICE H100) by default. On a different SLURM
cluster, edit `src/slurm/generator.py`'s `PARTITIONS` table to match
your cluster's CPU / memory / GPU caps.

### Install profiles

Just two:

| Command | What it gives you |
|---|---|
| `pip install -e ".[local]"` | Local workstation — CPU PyTorch + faiss-cpu + dev tools. |
| `pip install -e ".[hpc]"` | HPC cluster — CUDA 12.1 PyTorch + annoy + dev tools. For CUDA 11.8 add `--index-url https://download.pytorch.org/whl/cu118`. |

Both profiles are self-contained. The build backend is
[hatchling](https://hatch.pypa.io/latest/), so no `tabpfncredit.egg-info/`
is dropped into the source tree.

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

What it does:

1. Auto-preprocess any dataset not yet cached under
   `data/processed/<task>/<dataset>/`.
2. **Locally**: run every selected cell in-process, then summarize.
3. **On an HPC cluster**: wipe stale scripts under
   `scripts/<Experiment>/_generated/`, regenerate fresh per-partition
   SLURM scripts, `sbatch` them, and submit a summarize job with
   `--dependency=afterok:<arrays>`.

Helper commands:

| Command | Purpose |
|---|---|
| `tabpfncredit summarize --experiment <name>` | Rebuild per-fold + per-method CSVs. |
| `tabpfncredit list [--show-profile]` | Print registered methods + runtime tier + target partition. |
| `tabpfncredit doctor` | Print env vars, torch info, results-root. |

---

## 3. Repository layout

```
src/
  cli.py                          # `tabpfncredit` Typer CLI
  data/
    preprocessing.py              # cached TALENT-format conversion
    dataset_preprocessing.py      # per-dataset cleaning + LGD target clipping
    dataset_inventory.py          # row counts -> min_rows filter
    data_feeder.py                # CV-fold assembly + post-split anti-leakage
  methods/
    method_config.py              # thin layer over TALENT registry
    method_runner.py              # TALENT.run() per fold + metric enrichment
    method_metrics.py             # PD / LGD metric helpers
    cost_metrics.py               # expected loss + profit curves
    runtime_profile.py            # tier + sec/fold per method (drives SLURM)
  slurm/
    generator.py                  # SLURM script generator
  utils/
    config_reader.py              # YAML loader (min_rows + validators)
    result_io.py                  # save_method / load_method / scan_results
    file_lock.py                  # cross-platform FileLock
    logging_setup.py              # hybrid per-task + summary + errors
    summarize_results_polars.py   # polars-backed CSV aggregator
  visualizations/
    experiment_plots.py           # heatmaps, ranking bars, learning curves
    calibration_plots.py          # reliability diagrams
    data_exploration.py           # backs the Data_Exploration notebook

scripts/
  Experiment{0,1,2,3}/
    config/CONFIG_{DATA,METHOD,EXPERIMENT}.yaml
    _generated/                   # SLURM scripts (auto-emitted, gitignored)
  run_all_experiments.sh          # master script for the full HPC sweep

notebooks/                        # thin viewers calling src.visualizations
tests/                            # pytest suite

data/                             # raw + processed datasets    (gitignored)
results/                          # per-(dataset, method) JSON+npz (gitignored)
figures/                          # generated PDF plots          (gitignored)
```

---

## 4. Pipeline

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
src/utils/summarize_results_polars  # per-fold + per-method CSVs
        |
        v
notebooks/Experiment*.ipynb         # plots from src.visualizations -> figures/<exp>/*.pdf
```

Skip-if-done is implicit: the runner checks the `<method>.json` file
and skips that cell if it already exists. There are no separate
checkpoint files.

---

## 5. Tasks, datasets, and methods

| Task | Type | Datasets | Headline metrics |
|---|---|---|---|
| **PD** (Probability of Default) | Binary classification | 14 | AUC, Gini, KS, F1, Brier, ECE, Expected_Loss_Normalized |
| **LGD** (Loss Given Default) | Regression on `[0, 1]` | 7 | R², RMSE, MAE, Spearman_Corr |

LGD predictions and LGD targets are both clipped to `[0, 1]`:
preprocessing clips the raw `y`, inference clips every prediction. The
fold result records `n_clipped_below` / `n_clipped_above`.

The ~55 enabled methods cover **foundation models** (TabPFN family,
TabICL v1/v2, TabDPT, MITRA, LimiX, HyperFast, TabPTM),
**transformers** (FT-Transformer, SAINT, AutoInt, T2G-Former, TROMPT,
…), **MLP / ResNet** (RealMLP, MLP_PLR, TabM, …), **tree-mimic**
(TabNet, NODE, GrowNet, GRANDE), and **classical** (XGBoost, CatBoost,
LightGBM, RandomForest, LogisticRegression, KNN, SVM).

Method behaviour (categorical policy, normalisation, GPU/CPU
placement, in-context row limit, HPO support) comes from TALENT's
`MethodSpec` registry. The runtime tier in
`src/methods/runtime_profile.py` drives partition + walltime + packing.
Toggle methods on or off per experiment in
`scripts/Experiment*/config/CONFIG_METHOD.yaml`.

---

## 6. The four experiments

| # | Question it answers | Folds | HPO | Datasets | Methods |
|---|---|---|---|---|---|
| **0** | Pilot: does each method run end-to-end? Use the outcomes to curate Experiment 1. | 1 | NO_HPO | All 14 PD + 7 LGD. | All toggled in `CONFIG_METHOD.yaml`. |
| **1** | Headline benchmark — drives the paper. | 5 | NO_HPO + HPO | All 14 PD + 7 LGD. | **Curated by hand** in `CONFIG_METHOD.yaml` based on Exp 0 results. |
| **2** | Learning-curve sweep: metric vs training-set size. | 5 | NO_HPO | `min_rows`: PD ≥ 30 000, LGD ≥ 4 600. | `tabpfn_v3`, `tabicl_v2`, `xgboost`, `LogReg` / `LinearRegression`. |
| **3** | Class-imbalance sweep: minority proportion **0.15 → 0.0025**, step **0.0005**. PD only. | 5 | NO_HPO | `min_rows`: PD ≥ 30 000. | `tabpfn_v3`, `tabicl_v2`, `xgboost`, `LogReg`. |

Each experiment is driven by three YAMLs under
`scripts/Experiment<N>/config/`: `CONFIG_DATA.yaml` (splits + dataset
selection), `CONFIG_METHOD.yaml` (per-task method toggles),
`CONFIG_EXPERIMENT.yaml` (training knobs + sweep parameters).

For Experiment 1 the method set is curated by hand based on Experiment
0's outcomes — not derived automatically. Edit
`scripts/Experiment1/config/CONFIG_METHOD.yaml` after inspecting
`results/experiment0/summaries/experiment0_per_method.csv`.

For Experiments 2 and 3, dataset selection uses a `min_rows: N`
shorthand in `CONFIG_DATA.yaml` — every dataset with **≥ N** rows is
included automatically.

---

## 7. Result storage and logging

Each `(experiment, task, dataset, method[, sweep_point])` tuple
produces one JSON + one npz:

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

Three log files per experiment under `<results>/<experiment>/logs/`:

- `<dataset>_<method>.log` — per-cell DEBUG trace (incl. minority counts).
- `summary.log` — one INFO line per task start / done / fail.
- `errors.log` — ERROR tracebacks only.

Figures are saved as **PDF only** to `figures/<experiment>/<plot>.pdf`
and rendered inline in the notebook outputs.

---

## 8. Tests

```bash
pytest tests/                 # fast tests, ~10 s
pytest tests/ -m smoke        # end-to-end runs of cheap methods on synthetic data
pytest tests/ -m "not gpu"    # CI invocation -- auto-skips GPU-only tests
```

Coverage includes the registry-derived method sets, the PD / LGD
metric helpers, file locking, calibration plots, sweep-suffix
round-trips, the JSON+npz `save_method` round-trip, and every
previously-fixed bug as a regression test.
