# TabPFNCredit

**Benchmarking Tabular Foundation Models for Credit Risk Prediction.**

A rigorous evaluation framework comparing tabular foundation models (TabPFN, TabPFN v2, TabPFN Real, MITRA, TabICL) against classical ML and deep-learning baselines on credit-risk prediction tasks. Built on the [TALENT](https://github.com/LAMDA-Tabular/TALENT) framework.

---

## Tasks, datasets, methods

| Task | Type | Datasets | Key metrics |
|---|---|---|---|
| **PD** (Probability of Default) | Binary classification | 14 | AUC, Gini, KS, F1, Brier |
| **LGD** (Loss Given Default)    | Regression on `[0, 1]`  | 7  | R², RMSE, MAE, Spearman |

49 methods across **foundation models** (TabPFN family, MITRA, TabICL, TabPTM, HyperFast), **transformer-based** (FT-Transformer, SAINT, AutoInt, AMFormer, T2G-Former, TROMPT, …), **deep tabular** (MLP, ResNet, RealMLP, TabNet, NODE, TabR, ModernNCA, TabM, …), **tree boosting** (XGBoost, CatBoost, LightGBM), and **classical ML** (LogReg, RandomForest, KNN, SVM, NaiveBayes, NCM). The full registry — including per-method `cat_policy`, normalisation, output type, and method-intrinsic train/eval row limits — lives in [`src/methods/method_config.py`](src/methods/method_config.py).

### Experiments

| # | Description | Tasks |
|---|---|---|
| **0** | Pilot study to select methods for the full benchmark | PD + LGD |
| **1** | Full benchmark with `NO_HPO` and `HPO` modes                | PD + LGD |
| **2** | Learning-curve analysis (performance vs train-set size)     | PD + LGD |
| **3** | Class-imbalance analysis (performance vs minority proportion) | PD only |

---

## Pipeline at a glance

```
Stage 1 (once per dataset)
  Raw CSV -> dataset_preprocessing.py -> preprocessing.py -> {N.npy, C.npy, y.npy, info.json}

Stage 2 (per experiment run)
  data_feeder.py
   - optional global row cap / Exp3 imbalance resampling (pre-split, by design)
   - StratifiedKFold (PD) / KFold (LGD)
   - per-fold, training-only:
       * method-specific train cap (e.g. TabPFN v1: 5k, v2: 50k, MITRA: 5k)
       * drop near-constant columns        (training stats)
       * outlier removal                   (training stats)
       * PCA to 99 components if >99 feats (training stats)
       * winsorize to [0.1%, 99.9%]        (training stats)
  method_runner.py
   - process-local folds cache (one DataFeeder.prepare() per dataset per slot)
   - TALENT preprocessing + method.fit + predict
   - per-fold HPO via TALENT (config_hpo/{task}/{dataset}/{method}/HPO_PER_FOLD/fold_N/)
   - probability extraction + LGD clipping to [0, 1] + metric calculation
   - atomic pickle write (FileLock + .tmp + os.replace)
```

**Anti-leakage invariants** (enforced in code):
- Every stats-fitted preprocessor (outliers, winsorize, PCA, near-constant) is fitted on TRAINING data only and applied to val/test.
- Method-intrinsic train caps subsample training rows only — val/test are never thinned.
- F1-threshold optimisation uses the VALIDATION set; the chosen threshold is applied to TEST.
- Per-fold HPO writes go to a per-fold sub-directory so concurrent SLURM array slots cannot race on TALENT's internal `*-tuned.json` cache.

**Note on Experiment 3 resampling.** The class-imbalance resampling in [`data_feeder.py`](src/data/data_feeder.py) is applied to the full dataset **before** CV splitting — *intentional design choice*, so that all CV folds (train, val, and test) share the artificial minority ratio. The imbalance curves therefore measure "performance when the deployed distribution is also imbalanced at this ratio," not "performance on a naturally-distributed test set."

Configurable parameters live in `scripts/Experiment{0-3}/config/CONFIG_{DATA,METHOD,EXPERIMENT}.yaml`. Defaults for Experiment 1 are `cv_splits=5`, `val_size=0.2`, `seed=42`, `max_epochs=50`, `batch_size=255`, `n_trials=20`.

---

## Metrics

**PD (classification):** AUC, Gini, KS, Brier, LogLoss, Average Precision, Accuracy, Balanced Accuracy, F1, Precision, Recall, MCC. Threshold-based metrics use the optimal F1 threshold found on the validation set (no test-set leakage).

**LGD (regression):** R², MSE, RMSE, MAE, MedAE, Max Error, Pearson, Spearman, MAPE, Explained Variance. All predictions are clipped to `[0, 1]` before metric calculation.

---

## Quick start (local)

```bash
git clone https://github.com/andreasgoethals/tabpfncredit.git
cd tabpfncredit
python -m venv venv
source venv/bin/activate                   # Linux/macOS
# venv\Scripts\activate                    # Windows
pip install -e .                           # editable install removes the sys.path boilerplate
pip install -r requirements_local.txt
```

Sanity-check a single cell locally before submitting to the cluster:

```bash
python scripts/Experiment1/Experiment1_GPU.py \
    --dataset 0008.german --method xgboost --task_type pd --hpo_mode NO_HPO --verbose
```

Jupyter dev server:

```bash
venv/Scripts/jupyter lab --port 8889 --notebook-dir notebooks
```

---

## Running on VSC (Vlaams Supercomputer Centrum)

This is the canonical workflow for running the benchmark on the KU Leuven VSC clusters (`genius` for P100 GPU + CPU batch, `wice` for H100 GPU).

### 1. First-time setup (login node)

```bash
# 1) Activate conda
module purge
source "$VSC_DATA/miniconda3/bin/activate"

# 2) Create the environment from scratch
conda create -y -n TabPFNCredit python=3.10
conda activate TabPFNCredit

# 3) Clone (or sync) the repo into $VSC_DATA
cd "$VSC_DATA"
git clone https://github.com/andreasgoethals/tabpfncredit.git TabPFNCredit
cd TabPFNCredit

# 4) Install
pip install -e .
pip install -r requirements_vsc.txt       # CUDA 11.8 PyTorch wheels + Annoy (Faiss alternative)
```

> **Reproducibility note.** TALENT is currently installed from `main`. Before archiving a release, pin it to a commit SHA:
> `TALENT @ git+https://github.com/LAMDA-Tabular/TALENT@<40-char-sha>`.

### 2. Configure the run

Each experiment has its own `scripts/Experiment{N}/config/` directory with three YAML files:

| File | Knobs |
|---|---|
| `CONFIG_DATA.yaml` | Dataset on/off toggles, `cv_splits`, `val_size`, `test_size`, `seed`, `row_limit` |
| `CONFIG_METHOD.yaml` | Method on/off toggles per task (PD / LGD) |
| `CONFIG_EXPERIMENT.yaml` | `max_epochs`, `batch_size`, `n_trials`, `early_stopping` |

Toggle methods/datasets on/off with `true`/`false`. The task type (PD/LGD/BOTH) is auto-inferred from the enabled datasets.

### 3. Generate the SLURM scripts

```bash
# Optional: failure-notification email is auto-added to the generated #SBATCH block
export TABPFN_SLURM_NOTIFY_EMAIL="you@kuleuven.be"

# Standard recommended invocation (foundation models go to wICE H100, others to genius P100):
python scripts/Experiment1/Experiment1_Setup.py --foundation-on-wice
```

Each Setup script prints a configuration summary (datasets, methods per hardware tier, task counts) and writes batched `.slurm` files of ≤400 array elements each (VSC caps `--array` at 500). The generated scripts:

- use absolute `${VSC_DATA}/TabPFNCredit/...` paths for `--output` / `--error` (so they work regardless of where `sbatch` was invoked from),
- carry `#SBATCH --requeue` so a node failure re-queues the slot automatically,
- run under `set -euo pipefail` with `mkdir -p` of the log dir,
- stagger array-slot starts deterministically by `SLURM_ARRAY_TASK_ID % 30` seconds (avoids the I/O thundering-herd without RANDOM's worst-case 60 s penalty).

#### Task model (Experiment 1)

* **GPU task** = `(dataset, method, task_type)`. Each slot runs *both* `NO_HPO` and `HPO`; the HPO call hits the **process-local folds cache** so the data prep happens once.
* **CPU task** = `(dataset, task_type)`. Each slot runs *all enabled CPU methods × both HPO modes*, all sharing the cached folds. This collapses the CPU array length from `Ndatasets × Nmethods × 2` to `Ndatasets`.
* Already-completed `(dataset, method, hpo_mode)` cells are skipped automatically inside each slot (per-method idempotence check in `Experiment1.py`).

#### Submitting

```bash
sbatch scripts/Experiment1/Experiment1_GPU0.slurm
sbatch scripts/Experiment1/Experiment1_GPU_Foundation0.slurm
sbatch scripts/Experiment1/Experiment1_CPU0.slurm
# (and any additional batch files printed by Setup, e.g. _GPU1.slurm)
```

The Setup script prints the full list of `sbatch ...` lines at the end so you can copy-paste.

### 4. Monitoring

```bash
# Queue
squeue --me                                       # all clusters
squeue --me -M wice                               # wICE only

# Per-job logs (paths printed by the Setup banner)
tail -f $VSC_DATA/TabPFNCredit/results/experiment1/logs/slurm/gpu0_<JOBID>_<ARRAYID>.out

# Per-(dataset, method) logs
ls $VSC_DATA/TabPFNCredit/results/experiment1/logs/pd/
tail -f $VSC_DATA/TabPFNCredit/results/experiment1/logs/pd/0008.german_xgboost_NO_HPO.log

# Aggregated failure log (one entry per failed cell)
cat $VSC_DATA/TabPFNCredit/results/experiment1/logs/errors.log
```

### 5. Partial reruns

The intended workflow when adding methods, fixing config bugs, or invalidating subsets of results:

```bash
# Wipe specific results
python src/utils/remove_results.py --experiment experiment1 --method realmlp     # one method, all datasets
python src/utils/remove_results.py --experiment experiment1 --dataset 0008.german  # one dataset, all methods

# Build a focused SLURM array that runs ONLY the missing (dataset, method, hpo_mode) cells
python scripts/retry_failed.py --experiment experiment1 --dry-run                  # preview
python scripts/retry_failed.py --experiment experiment1                            # writes _Retry_GPU.slurm + _Retry_CPU.slurm
python scripts/retry_failed.py --experiment experiment1 --tasks gpu                # GPU-only retry

sbatch scripts/Experiment1/Experiment1_Retry_GPU.slurm
sbatch scripts/Experiment1/Experiment1_Retry_CPU.slurm
```

`retry_failed.py` also handles the **new-method case**: if you enable a previously-disabled method in `CONFIG_METHOD.yaml`, every `(dataset, that_method, hpo_mode)` cell is reported as missing and queued. The retry script uses each orchestrator's single-cell mode (`--dataset … --method … --task_type … --hpo_mode …`) so the array runs exactly the missing cells, no over-execution.

### 6. Focused Setup (skip-the-config-edit shortcut)

Instead of editing `CONFIG_METHOD.yaml` for a small focused rerun, narrow the Setup with allow-lists:

```bash
# Only emit SLURM scripts for two specific methods on two specific datasets
python scripts/Experiment1/Experiment1_Setup.py \
    --foundation-on-wice \
    --methods-only "xgboost,catboost" \
    --datasets-only "0008.german,0013.hmeq"
```

The filters propagate into the generated SLURM scripts so the array-runtime task list matches the Setup-time count.

### 7. Aggregating results

When the SLURM jobs are done, aggregate the per-dataset pickles into summary CSVs:

```bash
python src/utils/summarize_results.py --experiment experiment1
# outputs:
#   results/experiment1/summary/summary_{pd,lgd}_raw.csv          (one row per fold)
#   results/experiment1/summary/summary_{pd,lgd}_aggregated.csv   (mean +/- std across folds)
#   results/experiment1/summary/pivot_{pd,lgd}_{AUC,F1,R2}_{no_hpo,hpo}.csv
```

Then open the analysis notebooks (see below) to generate the figures.

### 8. VSC troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `sbatch: error: Batch script contains DOS line breaks (\r\n)` | File was edited on Windows | `dos2unix scripts/Experiment1/Experiment1_GPU0.slurm` |
| `OOM kill` on `home_credit` foundation runs | P100 too small | Re-run with `--foundation-on-wice` and submit the `_Foundation*.slurm` |
| Job sits in `PD` (pending) for hours | Cluster busy | `sprio --me` to see priority; consider `--partition=interactive` for sanity tests |
| `ModuleNotFoundError: No module named 'TALENT'` inside the job | Wrong conda env | First line of the job logs prints the env; verify `source activate TabPFNCredit` resolved |
| Pickle reads `[]` or 0-byte mid-write | Pre-hardening bug | Should not happen anymore — writes are atomic via `FileLock` + `os.replace`. Re-run with `retry_failed.py` |
| Two folds racing on `{method}-tuned.json` | Pre-hardening bug | Should not happen anymore — TALENT writes go to `HPO_PER_FOLD/fold_{N}/`. Re-run if stale config files predate the fix |
| `srun: error: --gpus-per-node not allowed` | Submitting wICE-bound script to genius (or vice versa) | The Setup script tags the right cluster in `#SBATCH --cluster=`; submit unmodified files |

---

## Analysing results (notebooks)

Aggregated CSVs feed the analysis notebooks under [`notebooks/`](notebooks). Figures are saved as **vector PDF at dpi=150** into [`figures/`](figures), bucketed per experiment:

| Notebook | Output |
|---|---|
| `Experiment0.ipynb`            | Pilot-study method selection |
| `Experiment1.1-PD.ipynb`       | PD benchmark: heatmaps, distributions, ranks, training time |
| `Experiment1.2-LGD.ipynb`      | LGD benchmark: heatmaps, distributions, ranks, training time |
| `Experiment1.3-Stat.ipynb`     | Friedman/Iman-Davenport, Nemenyi CD, Wilcoxon+Holm, PAMA, bootstrap CIs, win/loss matrix |
| `Experiment2.ipynb`            | Learning curves vs training set size |
| `Experiment3.ipynb`            | Class-imbalance curves vs minority proportion |
| `Data_Exploration.ipynb`       | Dataset characteristic plots |
| `Individual_Method_Runner.ipynb` | Debug / test individual methods one-off |

---

## Repository layout

```
TabPFNCredit/
├── notebooks/                        # Analysis notebooks (PDF figures, dpi=150)
├── scripts/
│   ├── Experiment{0-3}/
│   │   ├── config/                   # CONFIG_DATA / CONFIG_METHOD / CONFIG_EXPERIMENT YAML
│   │   ├── ExperimentN.py            # Per-cell runner (shared by GPU and CPU orchestrators)
│   │   ├── ExperimentN_{GPU,CPU}.py  # Orchestrators -- accept --array_id or single-cell args
│   │   ├── ExperimentN_Setup.py      # Generates batched .slurm files
│   │   └── ExperimentN_*.slurm       # Generated, do not edit by hand
│   ├── _slurm_templates.py           # Shared SLURM header / prologue / epilogue
│   └── retry_failed.py               # Scans pickles, emits a retry SLURM array
├── src/
│   ├── data/
│   │   ├── preprocessing.py          # Load/cache TALENT-format arrays
│   │   ├── dataset_preprocessing.py  # Per-dataset raw cleaning
│   │   └── data_feeder.py            # CV split + per-fold post-split preprocessing
│   ├── methods/
│   │   ├── method_config.py          # Single-source-of-truth method registry
│   │   ├── method_runner.py          # TALENT wrapper + process-local folds cache
│   │   ├── method_metrics.py         # PD and LGD metrics
│   │   └── method_debugger.py        # Quick method testing
│   └── utils/
│       ├── config_reader.py          # YAML configuration loader
│       ├── file_lock.py              # FileLock + atomic pickle write
│       ├── storage_handler.py        # Pickle I/O paths
│       ├── summarize_results.py      # Aggregate pickles to CSV
│       └── remove_results.py         # Selective result removal
├── data/
│   ├── raw/{pd,lgd}/                 # Raw CSV datasets
│   └── processed/                    # Cached .npy arrays (gitignored)
├── results/experiment{0-3}/          # Generated at runtime, gitignored
└── figures/                          # All generated figures (vector PDF, dpi=150)
```

---

## Hardening already in place

**Correctness / reproducibility.**
- Per-fold HPO config directories (`config_hpo/.../fold_{id}/`) prevent the race where two concurrent array slots on different folds collide on TALENT's `{method}-tuned.json` cache.
- Pickle writes go through `FileLock` + `os.replace()` so a mid-write crash leaves either the old file or the new file, never a zero-byte stub.
- `find_optimal_threshold_f1` adds a fine grid in `[1e-4, 1e-2]` and `[0.99, 1-1e-4]` — necessary for Experiment 3's `minority_proportion=0.01` regime where the optimum threshold can be sub-1%.

**Throughput / VSC ergonomics.**
- **Folds cache** in `method_runner` (process-local, LRU capacity 4). NO_HPO + HPO for the same method share folds; all CPU methods on one dataset share folds.
- **Bundled tasks** cut the GPU array length ~2× and the CPU array length by `|cpu_methods| × 2`.
- **Deterministic sleep stagger** (`SLURM_ARRAY_TASK_ID % 30`) replaces the previous `RANDOM%60` thundering-herd while halving the worst-case wallclock penalty.
- **Retry helper** (`scripts/retry_failed.py`) so partial reruns submit only the missing cells.
- **Filter CLI** on `Experiment1_Setup.py` (`--methods-only`, `--datasets-only`) for focused regeneration without editing the YAML; filters are threaded through to the orchestrators so array indices stay in sync.
- All SLURM scripts use absolute `${VSC_DATA}/TabPFNCredit/...` paths, carry `#SBATCH --requeue`, run under `set -euo pipefail`, and optionally add `--mail-type=FAIL` if `TABPFN_SLURM_NOTIFY_EMAIL` is set at generation time.
- Foundation models run on wICE H100 (soft isolation via `--mem=100G`) while standard GPU methods stay on genius P100; toggle with `Experiment1_Setup.py --foundation-on-wice`.

**Structure / packaging.**
- One `FileLock` (`src/utils/file_lock.py`) replaces three duplicated locking implementations.
- One `config_reader` factory replaces the three per-experiment loaders.
- `scripts/_slurm_templates.py` centralises SLURM header / prologue / epilogue.
- `pyproject.toml` makes the project `pip install -e .`-able (no more `sys.path.insert(...)` at the top of every script).
- `requirements.txt` bounds previously-unpinned `pytorch-lightning`, `dill`, `msgpack`, `safetensors`, etc., and flags the TALENT git URL as must-pin-before-release.

---

## Acknowledgments

- **[TALENT Framework](https://github.com/LAMDA-Tabular/TALENT)** — Unified interface for tabular methods.
- **[TabPFN](https://github.com/automl/TabPFN)** — Tabular foundation model.
- **VSC (Vlaams Supercomputer Centrum)** — Computational resources.

## License

MIT — see [LICENSE.txt](LICENSE.txt).

## Contact

- **Author**: Andreas Goethals
- **GitHub**: [andreasgoethals](https://github.com/andreasgoethals)
