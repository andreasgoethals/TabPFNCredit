<div align="center">

# TabPFNCredit

### Benchmarking tabular foundation models for credit-risk prediction

![Python](https://img.shields.io/badge/python-3.10–3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Built on TALENT](https://img.shields.io/badge/built%20on-TALENT-orange)
![Runs: laptop · SLURM HPC](https://img.shields.io/badge/runs-laptop%20·%20SLURM%20HPC-8A2BE2)

</div>

TabPFNCredit evaluates modern tabular foundation models (the TabPFN
family, TabICL v1/v2, TabDPT, MITRA, Google's TabFM — with LimiX,
HyperFast and TabPTM also wired in but toggled off by default) against deep
tabular networks (FT-Transformer, RealMLP, TabM, T2G-Former, …) and classical
baselines (XGBoost, CatBoost, LightGBM, RandomForest, LogisticRegression,
…) on **14 PD (Probability of Default)** and **7 LGD (Loss Given Default)**
datasets. It is built on top of
[TALENT](https://github.com/LAMDA-Tabular/TALENT).

The same `tabpfncredit` CLI runs on a laptop **or** on a SLURM-based HPC
cluster: it auto-detects the environment and either runs in-process or
generates and submits SLURM jobs. The SLURM generator targets the KU
Leuven VSC partitions by default but adapts to any cluster by editing one
table.

### At a glance

|  |  |
|---|---|
| **Tasks** | PD — probability of default (classification, 14 datasets) · LGD — loss given default (regression, 7 datasets) |
| **Methods** | 10 tabular foundation models · gradient boosting (CatBoost / XGBoost / LightGBM) · ~20 deep-tabular networks · classical baselines (LogReg, RF, SVM, …) |
| **Experiments** | Headline benchmark · data-efficiency sweep · imbalance-robustness sweep · pilot coverage |
| **Statistics** | PAMA · Friedman + Nemenyi · Wilcoxon / Holm · Bayesian ROPE · champion-level control tests |
| **Runs on** | a laptop (in-process) **or** a SLURM HPC cluster (auto-generated jobs) |
| **Built on** | [TALENT](https://github.com/LAMDA-Tabular/TALENT) |

---

## Contents

- [1. Quick start](#1-quick-start)
  - [1.1 Local install and first run](#11-local-install-and-first-run)
  - [1.2 HPC install (SLURM cluster)](#12-hpc-install-slurm-cluster)
  - [1.3 Install profiles](#13-install-profiles)
  - [1.4 Adapting to a non-VSC cluster](#14-adapting-to-a-non-vsc-cluster)
- [2. Command-line interface](#2-command-line-interface)
  - [Maintenance and analysis utilities](#maintenance-and-analysis-utilities-run-manually)
- [3. Tasks, datasets, and methods](#3-tasks-datasets-and-methods)
  - [Average Precision, baseline-corrected](#average-precision-baseline-corrected)
  - [Dataset display names and paper ordering](#dataset-display-names-and-paper-ordering)
  - [LGD targets are clipped to `[0, 1]`](#lgd-targets-are-clipped-to-0-1)
- [4. The four experiments](#4-the-four-experiments)
  - [Why the sweep curves are clean signal](#why-the-sweep-curves-are-clean-signal)
- [5. Pipeline](#5-pipeline)
- [6. Repository layout](#6-repository-layout)
- [7. Result storage and logging](#7-result-storage-and-logging)
  - [Analysis notebooks](#analysis-notebooks)
- [8. Tests](#8-tests)
- [9. Citation](#9-citation)
- [10. License](#10-license)

---

## 1. Quick start

One command starts any experiment: `tabpfncredit experiment <name>`
auto-preprocesses any missing data and either runs locally or submits to
SLURM. Local runs summarize immediately; on the VSC, rebuild summaries after
the arrays finish with `tabpfncredit summarize --experiment <name>` or by
running the notebooks. Installation is a single `pip install` that pulls
TALENT and every dependency in one pass.

**Requirements:** Python 3.10–3.12 (3.12 recommended; 3.13+ not yet
supported). Install Python 3.12 from <https://www.python.org/downloads/>
if your launcher can't find it.

### 1.1 Local install and first run

**Windows (PowerShell):**

```powershell
git clone https://github.com/andreasgoethals/TabPFNCredit.git
cd TabPFNCredit

py -3.12 -m venv tabpfncreditvenv
.\tabpfncreditvenv\Scripts\Activate.ps1

pip install -e ".[local]"          # project + all dependencies (incl. TALENT)
tabpfncredit experiment Experiment0
```

**macOS / Linux:**

```bash
git clone https://github.com/andreasgoethals/TabPFNCredit.git
cd TabPFNCredit

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
TALENT, and all dependencies. Method-specific optional extras (currently
only `tabfm`; needs Python ≥ 3.11) are listed in `pyproject.toml` and
install the same way, e.g. `pip install -e ".[local,tabfm]"`. The build
backend is
[hatchling](https://hatch.pypa.io/latest/), so no `*.egg-info/` is dropped
into the source tree.

### 1.4 Adapting to a non-VSC cluster

The generated SLURM scripts target the VSC wICE partitions
(`batch_sapphirerapids` CPU, A100, H100) by default; the Genius specs stay
in the table for manual use but are never picked automatically (torch 2.8
dropped Pascal/P100 support). On a different SLURM cluster, edit the
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
3. **On an HPC cluster**: write run-specific per-partition SLURM scripts
   under `scripts/<Experiment>/_generated/`, `sbatch` them, and print the
   dependency ID string used to chain later experiments.

Helper commands:

| Command | Purpose |
|---|---|
| `tabpfncredit resubmit <names...> \| --all` | Scan results for every missing (task, dataset, method, sweep/HPO) point and submit ONLY those, packed into dense fresh arrays. Generated plans are run-specific, so pending arrays cannot read a later submission's plan. Works locally (report + scripts) and on a cluster (also submits). |
| `tabpfncredit summarize --experiment <name>` | Rebuild the per-fold + per-method CSVs. |
| `tabpfncredit list [--architecture deep\|classical] [--hardware cpu\|gpu] [--show-profile]` | Print registered methods, optionally filtered; `--show-profile` adds runtime tier + target partition. |
| `tabpfncredit doctor` | Print environment variables, torch / CUDA info, and the results root. |

Re-running `tabpfncredit experiment <name>` is always safe: completed points
are skipped (one result file per point is the resume unit). `resubmit` does
the same thing more efficiently when most of an experiment is already done —
it queues only the missing work instead of re-sharding everything.

### Maintenance and analysis utilities (run manually)

These are run **by hand** (locally or on the cluster) — none of them run
automatically during an experiment. Each accepts `--help`, and the destructive
ones accept `--dry-run` to preview first.

| Utility | What it does | When to run it |
|---|---|---|
| `python -m src.utils.consolidate_shards` | Experiment 2/3 split one `(dataset, method)` cell's sweep across many SLURM array tasks, each writing its own `<method>__shard_<jobid>_<task>.json`. This merges all shards for a cell back into one tidy `<method>.json` and deletes the shards. Results are **unchanged** — the summariser already reads the union of a cell's shards; this is purely housekeeping. `--experiment`, `--dry-run`. | **Once, after an Exp 2/3 run finishes**, to collapse the many small shard files. |
| `python -m src.utils.run_notebooks` | Clears, restart-runs every analysis notebook with the project venv kernel — **in parallel** (`-j N`, default min(4, CPUs); each notebook is its own kernel process) with the per-experiment summary CSVs rebuilt once up front — collects each one's printed output into `results/All_Results.md`, and regenerates the figure captions once after a successful run. `--list`, `-v` (implies `-j 1`), `--md-only`. | After downloading results, to refresh every figure + the results dump in one command. |
| `python -m src.utils.generate_captions` | Writes a single `figures/CAPTIONS.md` with a paper-ready caption per figure, split into one chapter per notebook (in notebook order) and titled by figure file name. `--dry-run`. | Standalone caption refresh. Direct VS Code/Jupyter notebook runs also refresh this file after saving project figures. |
| `python -m src.utils.remove_results` | Prunes result files by `--experiment` / `--task` / `--dataset` / `--method` / `--hpo` / `--no-hpo` / `--folds`, then drops (or, with `--resummarize`, rebuilds) the affected summary CSVs. `--dry-run`. | To delete an obsolete or buggy method/dataset's results before re-running. |
| `python -m src.utils.fetch_weights` | Downloads the foundation-model checkpoints into `checkpoints/`. `--list`, `--only <models>`, `--skip <models>`. | **Once, locally**, before staging weights to the cluster (see `VSC_RUN.md`). |

(The pipeline commands — `tabpfncredit experiment / resubmit / summarize / list / doctor` — are in the table above.)

---

## 3. Tasks, datasets, and methods

| Task | Type | Datasets | Headline metrics |
|---|---|---|---|
| **PD** (Probability of Default) | Binary classification | 14 | AUC, Gini, KS, F1, Brier, ECE, AP / AP_normalized, Expected_Loss_Normalized |
| **LGD** (Loss Given Default) | Regression on `[0, 1]` | 7 | R², RMSE, MAE, Pearson_Corr, Spearman_Corr |

The headline benchmark enables 38 distinct methods (62 (task, method)
combinations across PD + LGD): **foundation models** (TabPFN family,
TabICL v1/v2, TabDPT, MITRA, Google's TabFM), **transformers**
(FT-Transformer, AutoInt, ExcelFormer, T2G-Former, …), **MLP / ResNet**
(RealMLP, MLP-PLR, TabM, …), **tree-mimic** (TabNet, DCN2, …), and
**classical** baselines (XGBoost, CatBoost, LightGBM, RandomForest,
LogisticRegression, KNN, SVM). Further registered methods (LimiX,
HyperFast, TabPTM, SAINT, TROMPT, NODE, GrowNet, …) ship toggled off and
can be enabled per experiment.

Enabling a disabled method needs no special treatment: flip its toggle in
the experiment's `CONFIG_METHOD.yaml` and re-send the experiment —
completed (dataset, method) points already have their result file and are
skipped, so `tabpfncredit resubmit <ExperimentName>` queues only the newly
enabled method's missing points. Foundation models additionally need their
weights staged once (`python -m src.utils.fetch_weights --only <method>`;
see `VSC_RUN.md`); any extra install requirement is noted next to the
method's toggle in `CONFIG_METHOD.yaml` and in `pyproject.toml`.

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

### Dataset display names and paper ordering

On disk a dataset is a slug like `0001.gmsc`; that slug is the join key
for every result file, cached prediction and figure path, and it is **never
renamed**. What a *reader* sees comes from
`src/data/dataset_registry.py` (gitignored -- see below), the single
source of truth for three things:

- **Display names** — proprietary datasets are anonymised (`PropPD1`, `PropPD2`,
  `PropLGD1`…`PropLGD5`); public ones use their real names.
- **Paper IDs** — `PD1..PD14` / `LGD1..LGD7`.
- **Ordering** — every dataset axis, legend, table and caption is sorted by
  `(is_proprietary, display_name)`, i.e. public datasets first in alphabetical
  order, then the proprietary ones. Sorting by slug (the old numbering) is a bug.

No display name may be hard-coded anywhere else. Plotting code calls
`display_name()` / `sort_key()`; per-dataset figures of proprietary datasets are
additionally written under a neutral filename (e.g. `pd_row_limit_proppd1_auc.pdf`)
so the real name never appears in a path used by the paper. Print the current
old → new mapping with:

```bash
python -m src.data.dataset_registry
```

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
| **2** | Learning-curve sweep: metric vs dataset size (`row_limit` caps the rows **before** the CV split, so train and test shrink together). | 5 | NO_HPO | Auto from `learning_curve.<task>.row_max`: PD ≥ 30 000, LGD ≥ 4 600. | `tabpfn_v3`, `tabicl_v2`, `xgboost`, `catboost`, `LogReg` / `LinearRegression`. |
| **3** | Class-imbalance sweep: minority proportion **0.15 → 0.0025**, step **0.0005**. PD only. | 5 | NO_HPO | PD with ≥ 30 000 rows **and** natural minority rate > `minority_proportion_max`. | `tabpfn_v3`, `tabicl_v2`, `xgboost`, `catboost`, `LogReg`. |

Each experiment is driven by three YAMLs under
`scripts/Experiment<N>/config/`:

- `CONFIG_DATA.yaml` — splits and dataset selection,
- `CONFIG_METHOD.yaml` — per-task method toggles,
- `CONFIG_EXPERIMENT.yaml` — training knobs and sweep parameters.

**One model fit per fold.** `seed_num` in `CONFIG_EXPERIMENT.yaml` is the
number of model-seed repeats per fold, and it is pinned to `1`. Every metric,
probability and prediction this repo records comes from a *single* fit
(TALENT's `RunResult` carries the last repeat's predictions), so repeats above
1 multiply the compute without changing anything that is reported.
Comparability across methods rests on the fixed **split** seed in
`CONFIG_DATA.yaml` — every method sees byte-identical folds — not on the model
seed. This must be set explicitly: left unset, TALENT's own packaged defaults
supply `seed_num: 15`.

**Every method is scored on the identical observations.**
`METHOD_TEST_VAL_LIMITS` in `src/methods/method_config.py` is deliberately
empty, so no method's validation or test fold is ever subsampled: a per-method
cap would mean comparing methods on different rows. Only the *training* side may
be capped (`METHOD_ROW_LIMITS`, from TALENT's registry, plus TabFM's
`max_num_rows` context cap), which changes what a model learns from, not what it
is measured on. One cap is ours rather than TALENT's: **TANGOS trains on at most
50,000 rows** (`_CAPACITY_ROW_CAPS` in `src/methods/method_config.py`). At 20 HPO
trials it is 21 fits per fold, and on the largest PD dataset a single fold did not
finish in 37 h, so a tuned run is out of reach of any wall time. The cap binds on
the 8 PD datasets whose training split exceeds it and on no LGD dataset, applies
to the tuned and untuned variants alike so their comparison stays clean, and — being
a training cap — leaves every method scored on the same rows. A consequence worth knowing when results are reused: because the
evaluation set is a property of the dataset alone, the observed target mean of a
dataset must be *byte-identical across methods*. If it is not, some result files
predate a preprocessing change and are stale —
`notebooks/Results_Checking.ipynb` is the place that surfaces it.

**Experiment 1's method set is curated by hand.** After Experiment 0
finishes, inspect
`results/experiment0/summaries/experiment0_per_method.csv` and edit
`scripts/Experiment1/config/CONFIG_METHOD.yaml` to enable only the methods
that ran successfully.

**Dataset selection for the sweep experiments is automatic:**

- **Experiment 2** selects every dataset with **≥ `row_max`** rows
  (`learning_curve.<task>.row_max` in `CONFIG_EXPERIMENT.yaml`). `row_max`
  is also the top of the dataset-size sweep, so the ceiling and the
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
src/data/dataset_preprocessing.py   # private, gitignored per-dataset cleaning
                                    # + leakage scrubbing / LGD target clipping
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
                                    # + every method scored on the SAME full
                                    #   val/test fold (no downsampling)
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

```text
TabPFNCredit/
├── README.md                           # this file
├── VSC_RUN.md                          # KU Leuven VSC run guide (weights, sharding, recovery)
├── pyproject.toml                      # single-step install + tooling config
├── CITATION.cff                        # citation metadata
├── LICENSE.txt                         # MIT licence
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py            # cached TALENT-format conversion
│   │   ├── dataset_preprocessing.py    # private, gitignored proprietary cleaning rules
│   │   ├── dataset_inventory.py        # dataset row counts -> min_rows filter
│   │   ├── dataset_registry.py         # private, gitignored: real -> anonymised
│   │   │                               #   dataset names + paper ordering
│   │   ├── dataset_names.py            # public accessor for the above
│   │   │                               #   (degrades to raw slugs if absent)
│   │   └── data_feeder.py              # CV-fold assembly + post-split anti-leakage
│   ├── methods/
│   │   ├── method_config.py            # thin layer over the TALENT registry
│   │   ├── method_names.py             # canonical figure display names (TALENT-free)
│   │   ├── method_runner.py            # TALENT.run() per fold + metric enrichment
│   │   ├── method_metrics.py           # PD / LGD metric helpers
│   │   ├── cost_metrics.py             # expected loss + profit curves
│   │   ├── runtime_profile.py          # tier + sec/fold per method (drives SLURM)
│   │   └── tabfm_chunked.py            # memory-safe TabFM inference (chunk + OOM retry)
│   ├── utils/
│   │   ├── cli.py                      # `tabpfncredit` Typer CLI (the entry point)
│   │   ├── paths.py                    # central path resolution (repo / project storage)
│   │   ├── config_reader.py            # YAML loader (min_rows + validators)
│   │   ├── result_io.py                # save_method / load_method / scan_results
│   │   ├── result_summary.py           # polars-backed per-fold + per-method CSVs
│   │   ├── results_checking.py         # completeness / integrity audit
│   │   ├── statistical_testing.py      # Friedman/Nemenyi/Bayesian/PAMA + CD diagrams
│   │   ├── slurm_generator.py          # SLURM script generator
│   │   ├── resubmit_planner.py         # gap scan -> only the missing points
│   │   ├── consolidate_shards.py       # merge Exp 2/3 per-task shard files
│   │   ├── remove_results.py           # prune results by exp/task/dataset/method/HPO/fold
│   │   ├── run_notebooks.py            # clear + restart-run all notebooks -> All_Results.md
│   │   ├── generate_captions.py        # auto-write the single figures/CAPTIONS.md
│   │   ├── fetch_weights.py            # download foundation-model weights (run LOCALLY)
│   │   ├── runtime_quiet.py            # quieten noisy library logging in notebooks
│   │   └── verify_inference_chunking.py  # check chunked == single-pass inference
│   └── visualizations/
│       ├── experiment_plots.py         # heatmaps, ranking bars, learning/imbalance curves
│       └── data_exploration.py         # backs the Data_Exploration notebook
│
├── scripts/
│   ├── Experiment{0,1,2,3}/
│   │   ├── config/CONFIG_{DATA,METHOD,EXPERIMENT}.yaml
│   │   └── _generated/                 # SLURM scripts (auto-emitted, gitignored)
│   ├── run_all_experiments.sh          # submit the full chained HPC sweep
│   └── setup_vsc_checkpoints.sh        # provision uploaded checkpoints/ on the VSC
│
├── notebooks/                          # thin viewers calling src.visualizations
│   ├── CONFIG_NOTEBOOKS.yaml           # per-task method filters for the notebooks
│   │                                   #   (exclude lists + champion-stat inclusion lists)
│   ├── Data_Exploration.ipynb
│   ├── Experiment0.ipynb               # pilot coverage + quick overview
│   ├── Experiment1.1-PD.ipynb          # PD headline benchmark
│   ├── Experiment1.2-PD-Stat.ipynb     # PD all-learner statistics
│   ├── Experiment1.3-PD-FamilyStat.ipynb   # PD champion-level statistics
│   ├── Experiment1.4-LGD.ipynb         # LGD headline benchmark
│   ├── Experiment1.5-LGD-Stat.ipynb    # LGD all-learner statistics
│   ├── Experiment1.6-LGD-FamilyStat.ipynb  # LGD champion-level statistics
│   ├── Experiment2.1-PD.ipynb          # PD data-efficiency sweep
│   ├── Experiment2.2-LGD.ipynb         # LGD data-efficiency sweep
│   ├── Experiment3.ipynb               # imbalance-robustness sweep
│   ├── Individual_Method_Runner.ipynb  # ad-hoc single-method tool
│   └── Results_Checking.ipynb          # results completeness audit
│
├── tests/                              # pytest suite
│
├── data/                               # raw + processed datasets               (gitignored)
├── results/                            # per-(dataset, method) JSON + summaries  (gitignored,
│                                       #   except the committed All_Results.md text dump)
├── figures/                            # generated PDFs + CAPTIONS.md            (gitignored)
└── checkpoints/                        # downloaded model weights               (gitignored)
```

`src/data/dataset_preprocessing.py` is intentionally omitted from git. It
contains raw proprietary dataset column names, leakage filters, and
dataset-specific cleaning rules. Keep your local copy at that path when you
need to preprocess raw data; fresh clones can still import the package and use
already processed `data/processed/` arrays.

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

Two CSVs are aggregated locally at the end of an `experiment` call. On the VSC,
run `tabpfncredit summarize --experiment <name>` after the arrays finish, or
run the notebooks, which refresh their own summaries:

```
<results>/summaries/<experiment>_per_fold.csv
<results>/summaries/<experiment>_per_method.csv
```

Logging uses Python's standard `logging` (run with `--verbose` for
INFO-level: start, finish, headline metric, per-fold minority counts). On
an HPC cluster each array slot's
stdout/stderr is captured by SLURM under `logs/<experiment>/` on the general
data storage (`$VSC_DATA`), named per job + array id — kept off the project
storage so logs always persist.

Figures are saved as **PDF only** to `figures/<experiment>/<plot>.pdf` and
rendered inline in the notebook outputs. Every notebook wipes its own figure
folder at the top of a run (`reset_figure_dir`), so a rerun never mixes old
and new figures.

### Analysis notebooks

All plotting / statistics code lives under `src/` — the notebooks are thin
viewers. Which methods they **show** is governed by one file,
[`notebooks/CONFIG_NOTEBOOKS.yaml`](notebooks/CONFIG_NOTEBOOKS.yaml): a per-task
`exclude` list hides those methods' results from every analysis notebook
except `Experiment0`'s pilot (applied centrally in `load_summary`, and
announced in the notebook output whenever it drops anything), and a per-task
`champions` list selects the methods the champion-level notebooks (1.3 / 1.6)
include. It is display-only — it never changes what the experiments run or
what is stored on disk.

| Notebook | What it shows |
|---|---|
| `Data_Exploration` | Dataset inventory, class balance, LGD target shapes, per-dataset structure. |
| `Experiment0` | Pilot coverage + quick performance / cost overview. |
| `Experiment1.1-PD` | Headline PD benchmark. Each matrix metric (**AUC**, **Brier**, **F1**) in three views — heatmap, per-method bar, across-dataset box (box = per-fold spread, dot = per-dataset mean) — with the **AUC rank** kept next to AUC; then the **HPO effect**, a **time analysis** (train + predict + HPO; tunable methods' train time × `n_trials`) with the cost/quality frontier, a **TabPFN v3 vs baselines** head-to-head — against **CatBoost** and **log. reg** (relative AUC-gain-vs-size trend + per-dataset `y = x` scatter) — a **prediction-calibration analysis** (observed-vs-predicted summary, decile reliability curves, and a per-dataset predicted-vs-actual grid), and a summary table. |
| `Experiment1.2-PD-Stat` | Full **all-learner** statistical analysis (PD): PAMA (two charts — all winners, and only methods winning ≥ 2 folds), Friedman + Iman–Davenport, the more powerful Friedman-Aligned-Ranks & Quade omnibus tests, Nemenyi **critical-difference diagrams** (compact, paper-ready), Win/Loss matrix, Holm-corrected significant pairs (Wilcoxon **and** paired t-test), all-pairwise adjusted-p-value matrix (Shaffer / Bergmann–Hommel), and a **Bayesian signed-rank ROPE** analysis (Benavoli et al., 2017). Backed by `src/utils/statistical_testing.py`. |
| `Experiment1.3-PD-FamilyStat` | **Champion-level** statistical analysis (PD): one champion per family (the `champions` list in `notebooks/CONFIG_NOTEBOOKS.yaml`; default **TabPFN-3**, **CatBoost**, **T2G-Former**, **Logistic Regression**) — with a **TabPFN-3-as-control** test (Bonferroni–Dunn) plus the same omnibus / CD / Win-Loss / Holm / **Bayesian ROPE** battery and a copy-paste report. Higher power and a cleaner answer to "is the foundation model competitive with each established family?"; the complete all-learner version stays in 1.2. |
| `Experiment1.4-LGD` | Headline LGD benchmark, same structure as 1.1: **R²** and **Pearson correlation** in three views, **R² rank**, **HPO effect**, **time analysis**, the **TabPFN v3 vs baselines** head-to-head — relative TabPFN-3 R² improvement vs dataset size against **CatBoost** and **lin. reg**, plus absolute per-dataset `y = x` scatters for both baselines — a **prediction-calibration analysis** (observed-vs-predicted summary, decile reliability curves, bias vs mean LGD), and a summary table. |
| `Experiment1.5-LGD-Stat` | Full all-learner statistical analysis (LGD): the same battery as 1.2, on **R²**. |
| `Experiment1.6-LGD-FamilyStat` | **Champion-level** statistical analysis (LGD): **TabPFN-3**, **CatBoost**, **TabM**, **Linear Regression** (the `champions` list in `notebooks/CONFIG_NOTEBOOKS.yaml`). Same structure as 1.3; with only 7 LGD datasets the **Bayesian ROPE** result is emphasised over the (low-power) frequentist tests. |
| `Experiment2.1-PD` / `Experiment2.2-LGD` | Learning curves (split by task): AUC (PD) / R² (LGD) vs **dataset size** (`row_limit` caps the rows before the CV split), in four pooled views (raw curve · raw curve with a lower-right inset zooming the shaded `rows <= 1000` region · moving average · moving average over transparent raw points), **per-dataset** raw-point plots, a data-efficiency table, and a summary of the metric's **evolution** across the whole sweep. |
| `Experiment3` | Imbalance-robustness curves (PD): AUC and prevalence-corrected **AP_normalized** vs minority-class proportion, in the same four pooled views, with the lower-right inset zooming the shaded `minority proportion <= 0.025` region, plus per-dataset raw-point plots, a degradation table, and an **evolution** summary across the sweep. |
| `Results_Checking` | Completeness / sanity audit of the result files: missing, incomplete, malformed, anomalous and not-in-config results, plus an **evaluation-set mismatch** check that fails loudly if a dataset's methods were not all scored on the same observations (the signature of results carried over from before a preprocessing change). |

**Run them all at once.** `python -m src.utils.run_notebooks` clears, restarts and re-runs every analysis notebook with the project venv kernel, collects each one's printed output into `results/All_Results.md`, and regenerates the consolidated `figures/CAPTIONS.md` once after a successful run. Notebooks execute **in parallel** (`-j N`, default min(4, CPUs)) — each is an independent kernel process with its own figure directory and its own `All_Results.md` section, and the shared per-experiment summary CSVs are rebuilt once up front instead of once per kernel (which also speeds up `-j 1` runs). Direct VS Code/Jupyter notebook runs refresh `figures/CAPTIONS.md` whenever project figures are saved, so the file is current after the notebook finishes. `Individual_Method_Runner` is skipped entirely; `Results_Checking` is re-run as a QA pass but its output is not collected into `All_Results.md`. Use `--list` to preview the order, `-v` for live output, `--md-only` to only refresh `All_Results.md`.

---

## 8. Tests

```bash
pytest tests/                 # whole suite, well under a minute
pytest tests/ -m smoke        # end-to-end runs of cheap methods on synthetic data
pytest tests/ -m "not gpu"    # CI invocation -- auto-skips GPU-only tests
```

Coverage includes the registry-derived method sets, the PD / LGD metric helpers,
sweep-suffix round-trips, the JSON+npz `save_method` round-trip, the notebook
runner's contracts, A4 figure geometry, and chunked-vs-single-pass inference
equivalence. Two act as publish gates: no tracked file may name a proprietary
dataset, and no notebook cell may reference a name nothing imports.

## 9. Citation

If you use this benchmark, please cite the accompanying paper — *Foundation
Models for Credit Risk Prediction: A Game Changer?*
([arXiv:2605.18147](https://arxiv.org/abs/2605.18147)) — rather than the
software entry. GitHub's "Cite this repository" button reads
[`CITATION.cff`](CITATION.cff), which lists the paper as the preferred
citation; the journal reference will replace the preprint on publication.

## 10. License

Released under the MIT License — see [`LICENSE.txt`](LICENSE.txt).

This repository builds on the [TALENT](https://github.com/LAMDA-Tabular/TALENT)
tabular-learning toolkit and the foundation-model packages it wraps
(`tabpfn`, `tabicl`, `tabdpt`, …); each retains its own license and should be
cited separately when used.
