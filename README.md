# TabPFNCredit

**Benchmarking Tabular Foundation Models for Credit Risk Prediction**

A rigorous evaluation framework comparing Tabular Foundation Models (TabPFN, TabPFN v2, TabPFN Real, MITRA, TabICL) against classical ML and deep learning baselines on credit risk prediction tasks. Built on the [TALENT](https://github.com/LAMDA-Tabular/TALENT) framework.

---

## Research Overview

### Tasks & Datasets

| Task | Type | # Datasets | Key Metrics |
|------|------|-----------|-------------|
| **PD** (Probability of Default) | Binary Classification | 15 | AUC, Gini, KS, F1, Brier |
| **LGD** (Loss Given Default) | Regression (0-1) | 7 | R2, RMSE, MAE, Spearman |

**PD datasets**: `gmsc`, `taiwan_creditcard`, `vehicle_loan`, `lendingclub`, `case_study`, `myhom`, `hackerearth`, `cobranded`, `german`, `bank_status`, `thomas`, `loan_default`, `home_credit`, `hmeq`, `algorithmwatch`

**LGD datasets**: `heloc`, `loss2`, `axa`, `base_model`, `base_modelisation`, `lgd_freddie`, `lgd_lendingclub`

### Supported Methods (49)

| Category | Methods | Count |
|----------|---------|-------|
| **Foundation Models** | TabPFN, TabPFN v2, TabPFN Real, MITRA, TabICL, TabPTM, HyperFast | 7 |
| **Transformer-Based** | FT-Transformer, SAINT, TabTransformer, AutoInt, ExcelFormer, AMFormer, T2G-Former, TROMPT | 8 |
| **Deep Tabular** | MLP, ResNet, SNN, RealMLP, MLP-PLR, TabNet, NODE, TabR, GrowNet, DANets, TabCaps, DCN2, TANGOS, PTARL, SwitchTab, DNNR, ModernNCA, BISHOP, ProtoGate, GRANDE, TabAutoPNPNet, TabM, LiMiX | 23 |
| **Tree Boosting** | XGBoost, CatBoost, LightGBM | 3 |
| **Classical ML** | LogisticRegression, RandomForest, KNN, SVM, NaiveBayes, NCM, LinearRegression, Dummy | 8 |

All methods are configured in a single registry at `src/methods/method_config.py`.

### Experiments

| Experiment | Description | Tasks |
|-----------|-------------|-------|
| **Experiment 0** | Pilot study to select methods for the full benchmark | PD + LGD |
| **Experiment 1** | Full benchmark with NO_HPO and HPO modes | PD + LGD |
| **Experiment 2** | Learning curve analysis: performance vs training set size | PD + LGD |
| **Experiment 3** | Class imbalance analysis: performance vs minority proportion | PD only |

---

## Data Pipeline

The pipeline is designed to prevent data leakage at every stage. Preprocessing steps that compute statistics (outlier bounds, PCA components, winsorization percentiles) are always fitted on training data only, then applied to validation and test sets.

```
                           STAGE 1: Dataset Preparation (once per dataset)
                          ================================================
  Raw CSV ──> dataset_preprocessing.py ──> Clean DataFrame
                                               |
                                         preprocessing.py
                                               |
                                    Separate features by type:
                                      N.npy (numerical, float32)
                                      C.npy (categorical, int64, -1=missing)
                                      y.npy (target)
                                      info.json (metadata)
                                               |
                                         [Cached to disk]

                           STAGE 2: Per-Fold Processing (per experiment run)
                          ====================================================
                                    data_feeder.py
                                         |
                          +----- Optional pre-split operations -----+
                          |  Global row limit (debugging)           |
                          |  Resampling (Exp3: class imbalance)     |
                          +-----------------------------------------+
                                         |
                              K-Fold Cross-Validation Split
                        (StratifiedKFold for PD, KFold for LGD)
                                         |
                         +------+------+------+------+
                         |      |      |      |      |
                       Fold1  Fold2  Fold3  ... ... FoldK
                         |
              +--- Per-fold post-split preprocessing ---+
              |  (all fitted on TRAINING data only)     |
              |                                         |
              |  1. Method train limit (train only)     |
              |     TabPFN v1/MITRA: 5k                 |
              |     TabPFN v2/TabICL: 50k               |
              |     AMFormer/TANGOS: 100k               |
              |     Stratified subsampling for PD       |
              |                                         |
              |  2. Drop near-constant columns          |
              |     >99% same value in training set     |
              |     Same columns dropped from val/test  |
              |                                         |
              |  3. Remove outliers (training only)     |
              |     Hybrid: percentile rarity (0.1%)    |
              |     + IQR magnitude (5x from median)    |
              |     Val/test rows are never removed     |
              |                                         |
              |  4. PCA (if features > 99)              |
              |     Fit on training, transform val/test |
              |     Reduces to 99 principal components  |
              |     Absorbs categoricals into PCA space |
              |                                         |
              |  5. Winsorization                       |
              |     Compute 0.1%-99.9% bounds on train  |
              |     Clip val/test to those bounds       |
              +-----------------------------------------+
                         |
                   method_runner.py
                         |
              +--- TALENT preprocessing ---+
              |  NaN imputation            |
              |  Categorical encoding      |
              |  Normalization             |
              +----------------------------+
                         |
              method.fit(train) + predict(test)
                         |
                   method_metrics.py
                         |
                  Results (.pkl per dataset)
```

### Configurable Parameters

All parameters are configurable via YAML files in `scripts/ExperimentN/config/`:

| Parameter | Config File | Default (Exp1) | Description |
|-----------|-------------|---------------:|-------------|
| `cv_splits` | CONFIG_DATA | 5 | Number of cross-validation folds |
| `val_size` | CONFIG_DATA | 0.2 | Fraction of training data for validation |
| `test_size` | CONFIG_DATA | 0.2 | Test fraction (only used if cv_splits=1) |
| `seed` | CONFIG_DATA | 42 | Random seed for reproducibility |
| `row_limit` | CONFIG_DATA | null | Global row cap (null = use all rows) |
| `max_epochs` | CONFIG_EXPERIMENT | 50 | Maximum training epochs (deep methods) |
| `batch_size` | CONFIG_EXPERIMENT | 255 | Training batch size |
| `n_trials` | CONFIG_EXPERIMENT | 20 | HPO trials per fold |
| `early_stopping` | CONFIG_EXPERIMENT | true | Early stopping for deep methods |
| `early_stopping_patience` | CONFIG_EXPERIMENT | 10 | Patience epochs |

---

## Method Preprocessing Policies

Every method's preprocessing requirements are enforced by hard `assert` statements in the TALENT source code. These constraints are not configurable -- they are architectural requirements of each method. The table below documents the complete policy for each method, verified against the TALENT source.

**Global defaults** (applied when a method has no specific override):

| Policy | Default Value | Description |
|--------|--------------|-------------|
| `cat_policy` | Per method (see table) | Categorical encoding -- TALENT-enforced |
| `normalization` | `standard` | StandardScaler (zero mean, unit variance) |
| `num_policy` | `none` | No numerical feature transformation |
| `num_nan_policy` | `median` | Impute numerical NaN with column median |
| `cat_nan_policy` | `new` | Treat categorical NaN as a new category |

### Per-Method Configuration

| Method | Cat Policy | Normalization | Num Policy | Output Type | Tasks | Train Limit | Eval Limit |
|--------|-----------|---------------|------------|-------------|-------|-------------|------------|
| **Foundation Models** | | | | | | | |
| tabpfn | indices | none | none | PROBABILITIES | PD | 5,000 | 50,000 ¹ |
| tabpfn_v2 | indices | none | none | PROBABILITIES | PD+LGD | 50,000 | 50,000 |
| tabpfn_real | indices | none | none | PROBABILITIES | PD | 50,000 | 50,000 |
| mitra | indices | none | none | LOGITS | PD+LGD | 5,000 | 5,000 |
| tabicl | indices | none | none | PROBABILITIES | PD | 50,000 | 50,000 |
| tabptm | ohe | standard | none | LOGITS | PD+LGD | — | — |
| hyperfast | indices | none | none | PROBABILITIES | PD | — | — |
| **Transformer-Based** | | | | | | | |
| ftt | indices | standard | none | LOGITS | PD+LGD | — | — |
| saint | indices | standard | none | LOGITS | PD+LGD | — | — |
| tabtransformer | indices | standard | none | LOGITS | PD+LGD | — | — |
| autoint | indices | standard | none | LOGITS | PD+LGD | — | — |
| excelformer | ohe | standard | none | LOGITS | PD+LGD | — | — |
| amformer | indices | standard | none | LOGITS | PD+LGD | 100,000 | — |
| trompt | indices | standard | none | LOGITS | PD+LGD | — | — |
| t2gformer | indices | standard | none | LOGITS | PD+LGD | — | — |
| **Deep Tabular** | | | | | | | |
| mlp | ohe | standard | none | LOGITS | PD+LGD | — | — |
| resnet | ohe | standard | none | LOGITS | PD+LGD | — | — |
| snn | indices | standard | none | LOGITS | PD+LGD | — | — |
| realmlp | indices | standard | none | PROBABILITIES | PD+LGD | — | — |
| mlp_plr | tabr_ohe | standard | none | LOGITS | PD+LGD | — | — |
| tabnet | ohe | standard | none | PROBABILITIES | PD+LGD | — | — |
| node | ohe | standard | none | LOGITS | PD+LGD | — | — |
| tabr | tabr_ohe | standard | none | LOGITS | PD | — | — |
| grownet | indices | standard | none | LOGITS | PD+LGD | — | — |
| danets | ohe | standard | none | LOGITS | PD+LGD | — | — |
| tabcaps | ohe | standard | none | LOGITS | PD | — | — |
| dcn2 | indices | standard | none | LOGITS | PD+LGD | — | — |
| tangos | ohe | standard | none | LOGITS | PD+LGD | 100,000 | — |
| ptarl | indices | standard | none | LOGITS | PD+LGD | — | — |
| switchtab | ohe | standard | none | LOGITS | PD+LGD | — | — |
| dnnr | ohe | standard | none | LOGITS | LGD | — | — |
| modernNCA | tabr_ohe | standard | none | LOGITS | PD+LGD | — | — |
| bishop | indices | standard | none | LOGITS | PD+LGD | — | — |
| protogate | ohe | standard | none | LOGITS | PD | — | — |
| grande | indices | standard | none | LOGITS | PD | — | — |
| tabautopnpnet | tabr_ohe | standard | none | LOGITS | PD+LGD | — | — |
| tabm | indices | standard | none | LOGITS | PD+LGD | — | — |
| limix | indices | none | none | LOGITS | PD+LGD | — | — |
| **Tree Boosting** | | | | | | | |
| xgboost | ohe | standard | none | PROBABILITIES | PD+LGD | — | — |
| catboost | indices | standard | none | PROBABILITIES | PD+LGD | — | — |
| lightgbm | ohe | standard | none | PROBABILITIES | PD+LGD | — | — |
| **Classical ML** | | | | | | | |
| LogReg | ohe | standard | none | PROBABILITIES | PD | — | — |
| LinearRegression | ohe | standard | none | CLASS_LABELS | LGD | — | — |
| RandomForest | ohe | standard | none | PROBABILITIES | PD+LGD | — | — |
| knn | ohe | standard | none | PROBABILITIES | PD+LGD | — | — |
| svm | ohe | standard | none | PROBABILITIES | PD+LGD | — | — |
| NaiveBayes | ohe | standard | none | PROBABILITIES | PD | — | — |
| NCM | ohe | standard | none | PROBABILITIES | PD | — | — |
| dummy | ohe | standard | none | PROBABILITIES | PD | — | — |

### Column Definitions

**Cat Policy** -- Categorical feature encoding, enforced by TALENT `assert` statements:
- **`indices`**: Integer codes passed to learned embedding layers (`nn.Embedding`). Required by methods with embedding-based architectures (transformers, foundation models, CatBoost).
- **`ohe`**: One-hot encoding, concatenated into the numerical feature matrix. Required by methods without embedding layers (MLP, ResNet, all classical ML except CatBoost). TALENT enforces `assert(cat_policy != 'indices')`.
- **`tabr_ohe`**: TabR-specific one-hot encoding that keeps numerical and categorical arrays separate. Required by TabR-family methods. TALENT enforces `assert(cat_policy == 'tabr_ohe')`.

Note: Ordinal encoding is never used. Categorical features in credit risk datasets are nominal (e.g., loan purpose, employment type), so ordinal encoding would impose false ordinal relationships.

**Normalization**:
- **`standard`**: StandardScaler (zero mean, unit variance). Default for most methods.
- **`none`**: No normalization. Required by foundation models (TabPFN, Mitra, etc.) and LiMiX, which handle normalization internally. Enforced by `assert(args.normalization == 'none')`.

**Num Policy**: Numerical feature encoding (e.g., binning). Set to `none` for all methods -- no numerical transformations are applied.

**Output Type** -- What the method's `predict()` returns for classification:
- **LOGITS**: Raw network output (unbounded values). Converted to probabilities via softmax (2D) or sigmoid (1D) by `method_runner.py`.
- **PROBABILITIES**: Calibrated probabilities from `predict_proba()`. Used directly.
- **CLASS_LABELS**: Continuous predictions (LinearRegression only, regression-only method).

All classification methods return probabilities. In the current TALENT version:
- **SVM**: `LinearSVC` wrapped in `CalibratedClassifierCV(method='sigmoid', cv=3)` to enable `predict_proba()`
- **NCM**: Custom `_predict_proba()` computes softmax over negative Euclidean distances to class centroids
- **NaiveBayes**: Uses `GaussianNB.predict_proba()` directly
- **Dummy**: Uses `DummyClassifier.predict_proba()`
- **RealMLP**: Classifier's `predict()` modified to return `predict_proba()` output

**Train Limit / Eval Limit**: Method-intrinsic dataset size caps, applied after CV splitting and independently from the global `row_limit` debug parameter.

- **Train Limit**: Maximum training rows. For ICL foundation models (TabPFN, MITRA, TabICL), this reflects the GPU memory constraint of loading the full training set as a transformer context. For standard deep learning methods with O(N²) attention (AMFormer, TANGOS), it is a practical compute cap to bound training time. Subsampling is stratified for PD tasks to preserve class distribution.
- **Eval Limit**: Maximum validation/test rows. Only ICL foundation models require this, because their inference cost scales with both N_train and N_test simultaneously (cross-attention between context and query). Standard deep learning and classical methods predict in mini-batches, so they impose no constraint on evaluation set size (shown as —).
- **tabpfn v1 asymmetry** (5k train / 50k eval): TabPFN v1 processes each test query independently against the fixed training context, so N_test does not add to GPU memory. The larger eval limit maximises evaluation reliability without any architectural cost.
- **MITRA symmetry** (5k / 5k): Full cross-attention between training context and all test queries means GPU memory scales as O(N_train × N_test); both limits must be equal.
- All other ICL models (tabpfn_v2, tabpfn_real, tabicl) use equal train and eval limits.

¹ tabpfn eval limit is intentionally 10× the train limit; see explanation above.

---

## Metrics

### PD (Classification)

| Metric | Type | Description |
|--------|------|-------------|
| AUC | Probability-based | Area Under ROC Curve |
| Gini | Probability-based | 2 * AUC - 1 |
| KS | Probability-based | Kolmogorov-Smirnov statistic |
| Brier | Probability-based | Brier score loss |
| LogLoss | Probability-based | Log loss (cross-entropy) |
| Avg_Precision | Probability-based | Average precision (PR-AUC) |
| Accuracy | Threshold-based | Overall accuracy |
| Balanced_Accuracy | Threshold-based | Mean per-class accuracy |
| F1 | Threshold-based | F1 score (binary) |
| Precision | Threshold-based | Precision |
| Recall | Threshold-based | Recall (sensitivity) |
| MCC | Threshold-based | Matthews correlation coefficient |

Threshold-based metrics use an optimal threshold determined by maximizing F1 on the **validation set** (no data leakage). The threshold found on the validation set is then applied to the test set.

All classification methods return probabilities, so all probability-based metrics are available for every PD method.

### LGD (Regression)

| Metric | Description |
|--------|-------------|
| R2 | Coefficient of determination (R-squared) |
| MSE | Mean squared error |
| RMSE | Root mean squared error |
| MAE | Mean absolute error |
| MedAE | Median absolute error |
| Max_Error | Maximum absolute error |
| Pearson | Pearson correlation coefficient |
| Spearman | Spearman rank correlation coefficient |
| MAPE | Mean absolute percentage error |
| Explained_Variance | Explained variance score |

All LGD predictions are clipped to [0, 1] before metric calculation.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/andreasgoethals/tabpfncredit.git
cd tabpfncredit
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements_local.txt
```

For VSC supercomputer, use `requirements_vsc.txt` instead.

### 2. Configuration

Each experiment has its own config directory under `scripts/ExperimentN/config/` with three YAML files:

- **`CONFIG_DATA.yaml`**: Dataset toggles, split settings (`cv_splits`, `val_size`, `seed`, `row_limit`)
- **`CONFIG_METHOD.yaml`**: Method selection per task (PD/LGD)
- **`CONFIG_EXPERIMENT.yaml`**: Training parameters (`max_epochs`, `batch_size`, `n_trials`, `early_stopping`)

### 3. Running Experiments

```bash
# Generate SLURM scripts
python scripts/Experiment1/Experiment1_Setup.py

# Submit to cluster
sbatch scripts/Experiment1/Experiment1_GPU.slurm
sbatch scripts/Experiment1/Experiment1_CPU.slurm
```

### 4. Analyzing Results

Results are stored as pickle files in `results/experimentN/{pd,lgd}/`. Generate summary CSVs:

```bash
python src/utils/summarize_results.py --experiment experiment1
```

Analysis notebooks in `notebooks/`:

| Notebook | Purpose |
|----------|---------|
| `Experiment0.ipynb` | Method selection from pilot study |
| `Experiment1.ipynb` | Full benchmark: heatmaps, rank analysis, PAMA, correlation, training time |
| `Experiment2.ipynb` | Learning curves: performance vs training set size |
| `Experiment3.ipynb` | Class imbalance: performance vs minority proportion |
| `Data_Exploration.ipynb` | Dataset characteristics analysis |
| `Individual_Method_Runner.ipynb` | Debug and test individual methods |

---

## Repository Structure

```
TabPFNCredit/
├── notebooks/                         # Analysis & visualization
│   ├── Experiment{0-3}.ipynb          # Per-experiment analysis
│   ├── Data_Exploration.ipynb         # Dataset characteristics
│   └── Individual_Method_Runner.ipynb # Method debugging
│
├── scripts/
│   └── Experiment{0-3}/
│       ├── config/
│       │   ├── CONFIG_DATA.yaml       # Dataset + split settings
│       │   ├── CONFIG_METHOD.yaml     # Method selection (PD/LGD)
│       │   └── CONFIG_EXPERIMENT.yaml # Training parameters
│       ├── ExperimentN.py             # Main experiment runner
│       ├── ExperimentN_Setup.py       # SLURM script generator
│       └── ExperimentN_*.slurm        # Generated SLURM scripts
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py           # Load/cache TALENT-format arrays
│   │   ├── dataset_preprocessing.py   # Per-dataset raw data cleaning
│   │   └── data_feeder.py            # CV splitting + post-split preprocessing
│   ├── methods/
│   │   ├── method_config.py          # Single-source-of-truth method registry
│   │   ├── method_runner.py          # TALENT method wrapper
│   │   ├── method_metrics.py         # PD and LGD metric calculation
│   │   └── method_debugger.py        # Quick method testing
│   └── utils/
│       ├── config_reader.py          # YAML configuration parser
│       ├── storage_handler.py        # Pickle file I/O with locking
│       ├── summarize_results.py      # Result aggregation to CSV
│       └── remove_results.py         # Selective result removal
│
├── data/
│   ├── raw/{pd,lgd}/                 # Raw CSV datasets
│   └── processed/                    # Cached .npy arrays
│
├── results/experiment{0-3}/
│   ├── {pd,lgd}/                     # Pickle result files
│   ├── summary/                      # Aggregated CSVs
│   └── figures/                      # Generated plots
│
├── config_hpo/                       # Per-fold HPO configs (auto-generated)
├── requirements_local.txt
└── requirements_vsc.txt
```

---

## Acknowledgments

- **[TALENT Framework](https://github.com/LAMDA-Tabular/TALENT)** -- Unified interface for tabular methods
- **[TabPFN](https://github.com/automl/TabPFN)** -- Tabular foundation model
- **VSC (Vlaams Supercomputer Centrum)** -- Computational resources

## License

MIT License -- see [LICENSE.txt](LICENSE.txt) for details.

## Contact

- **Author**: Andreas Goethals
- **GitHub**: [andreasgoethals](https://github.com/andreasgoethals)
