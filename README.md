# TabPFNCredit

**Benchmarking Tabular Foundation Models for Credit Risk Prediction**

A rigorous evaluation framework comparing Tabular Foundation Models (TabPFN, TabPFN v2, TabPFN Real, MITRA, TabICL) against classical ML and deep learning baselines on credit risk prediction tasks. Built on the [TALENT](https://github.com/LAMDA-Tabular/TALENT) framework.

---

## Key Features

- **Rigorous Cross-Validation**: 5-fold CV with proper fold isolation and no data leakage
- **Per-Fold Hyperparameter Optimization**: Independent HPO for each fold using TALENT's built-in optimization
- **Concurrent Execution**: File locking support for safe parallel execution on SLURM clusters
- **Comprehensive Metrics**: Internally calculated metrics for consistency across all method types
- **Centralized Method Registry**: Single-source-of-truth configuration for all 49 supported methods
- **Smart Preprocessing**: Method-specific categorical encoding, normalization, and NaN policies automatically enforced
- **Row Limit Enforcement**: Method-intrinsic training limits (e.g., TabPFN: 5k) applied after splitting to preserve full test/val sets
- **Post-Split Preprocessing**: Near-constant column removal, outlier detection, PCA, and winsorization -- all fitted on training data only
- **Advanced Analysis**: Rank correlation, PAMA analysis, dataset characteristic studies, learning curves

---

## Research Overview

### Tasks & Datasets

| Task | Type | # Datasets | Key Metrics |
|------|------|-----------|-------------|
| **PD** (Probability of Default) | Binary Classification | 15 | AUC, Gini, KS, F1, Brier |
| **LGD** (Loss Given Default) | Regression (0-1) | 8 | R2, RMSE, MAE, Spearman |

**PD datasets**: `gmsc`, `taiwan_creditcard`, `vehicle_loan`, `lendingclub`, `case_study`, `myhom`, `hackerearth`, `cobranded`, `german`, `bank_status`, `thomas`, `loan_default`, `home_credit`, `hmeq`, `algorithmwatch`

**LGD datasets**: `heloc`, `loss2`, `taiwan_creditcard`, `axa`, `base_model`, `base_modelisation`, `lgd_freddie`, `lgd_lendingclub`

### Supported Methods

| Category | Methods | Count |
|----------|---------|-------|
| **Foundation Models** | TabPFN, TabPFN v2, TabPFN Real, MITRA, TabICL, TabPTM, HyperFast | 7 |
| **Transformer-Based** | FT-Transformer, SAINT, TabTransformer, AutoInt, ExcelFormer, AMFormer, T2G-Former, TROMPT | 8 |
| **Deep Tabular** | MLP, ResNet, SNN, RealMLP, MLP-PLR, TabNet, NODE, TabR, GrowNet, DANets, TabCaps, DCN2, TANGOS, PTARL, SwitchTab, DNNR, ModernNCA, BISHOP, ProtoGate, GRANDE, TabAutoPNPNet, TabM, LiMiX | 23 |
| **Tree Boosting** | XGBoost, CatBoost, LightGBM | 3 |
| **Classical ML** | LogisticRegression, RandomForest, KNN, SVM, NaiveBayes, NCM, LinearRegression, Dummy | 8 |

**Total: 49 methods** (all configured in a single registry at `src/methods/method_config.py`)

### Experiments

| Experiment | Description | Tasks |
|-----------|-------------|-------|
| **Experiment 0** | Pilot study to select methods for the full benchmark | PD + LGD |
| **Experiment 1** | Full benchmark with NO_HPO and HPO modes | PD + LGD |
| **Experiment 2** | Learning curve analysis: performance vs training set size | PD + LGD |
| **Experiment 3** | Class imbalance analysis: performance vs minority proportion | PD only |

---

## Architecture

### Data Pipeline

```
Raw CSV  -->  preprocessing.py  -->  Cached .npy arrays
                                          |
                                    data_feeder.py
                                          |
                              +-----+-----+-----+
                              |     |     |     |
                            Fold1 Fold2 Fold3 ... FoldK
                              |
                    Post-split preprocessing
                    (per fold, no leakage):
                      1. Drop near-constant columns
                      2. Remove outliers (train only)
                      3. PCA / Winsorization
                              |
                        method_runner.py
                              |
                    TALENT method.fit() + predict()
                              |
                        method_metrics.py
                              |
                      Results (.pkl per dataset)
```

### Method Configuration

All method properties are defined in a **single registry** (`src/methods/method_config.py`):

- **Output type**: `LOGITS` (needs softmax), `PROBABILITIES` (calibrated), or `CLASS_LABELS` (continuous regression output)
- **Categorical encoding**: `indices` (embedding-based), `ohe` (one-hot), or `tabr_ohe` (TabR-specific)
- **Hardware**: GPU or CPU
- **Row limits**: Method-intrinsic training caps (applied after splitting)
- **HPO support**: Whether hyperparameter optimization is meaningful

Categorical features in these datasets are **never considered ordinal**, so ordinal encoding is never used. Each method's required encoding is verified against TALENT's `assert` statements.

### Method Task Availability

Not all methods support both PD (classification) and LGD (regression). The table below shows which methods are configured for each task (from `CONFIG_METHOD.yaml`):

| Method | PD (Classification) | LGD (Regression) |
|--------|:-------------------:|:-----------------:|
| **Foundation Models** | | |
| tabpfn | Yes | - |
| tabpfn_v2 | Yes | Yes |
| tabpfn_real | Yes | - |
| mitra | Yes | Yes |
| tabicl | Yes | - |
| tabptm | Yes | Yes |
| hyperfast | Yes | - |
| **Transformer-Based** | | |
| ftt | Yes | Yes |
| saint | Yes | Yes |
| tabtransformer | Yes | Yes |
| autoint | Yes | Yes |
| excelformer | Yes | Yes |
| amformer | Yes | Yes |
| trompt | Yes | Yes |
| t2gformer | Yes | Yes |
| **Deep Tabular** | | |
| mlp | Yes | Yes |
| resnet | Yes | Yes |
| snn | Yes | Yes |
| realmlp | Yes | Yes |
| mlp_plr | Yes | Yes |
| tabnet | Yes | Yes |
| node | Yes | Yes |
| tabr | Yes | - |
| grownet | Yes | Yes |
| danets | Yes | Yes |
| tabcaps | Yes | - |
| dcn2 | Yes | Yes |
| tangos | Yes | Yes |
| ptarl | Yes | Yes |
| switchtab | Yes | Yes |
| dnnr | - | Yes |
| modernNCA | Yes | Yes |
| bishop | Yes | Yes |
| protogate | Yes | - |
| grande | Yes | - |
| tabautopnpnet | Yes | Yes |
| tabm | Yes | Yes |
| limix | Yes | Yes |
| **Tree Boosting** | | |
| xgboost | Yes | Yes |
| catboost | Yes | Yes |
| lightgbm | Yes | Yes |
| **Classical ML** | | |
| LogReg | Yes | - |
| LinearRegression | - | Yes |
| RandomForest | Yes | Yes |
| knn | Yes | Yes |
| svm | Yes | Yes |
| NaiveBayes | Yes | - |
| NCM | Yes | - |
| dummy | Yes | - |

**PD-only** (12): tabpfn, tabpfn_real, tabicl, hyperfast, tabr, tabcaps, protogate, grande, LogReg, NaiveBayes, NCM, dummy

**LGD-only** (2): dnnr, LinearRegression

**Both PD + LGD** (35): All remaining methods

### Preprocessing Strategy

| Step | When | Scope | Details |
|------|------|-------|---------|
| Dataset cleaning | Before caching | Full dataset | Dataset-specific (see `dataset_preprocessing.py`) |
| Categorical encoding | Before caching | Full dataset | `.cat.codes` (integer indices); -1 for missing |
| Resampling | Before splitting | Full dataset | Optional minority class adjustment (Exp3) |
| Row limit (global) | Before splitting | Full dataset | User/debug cap |
| CV splitting | - | Creates folds | Stratified for PD, random for LGD |
| Row limit (method) | After splitting | Training only | Stratified subsampling preserves class distribution |
| Near-constant columns | After splitting | Fit on train | Drops columns where >99% of all rows have same value |
| Outlier removal | After splitting | Train only | Hybrid: percentile rarity + IQR magnitude |
| PCA | After splitting | Fit on train | Triggered when features > 100; absorbs categoricals |
| Winsorization | After splitting | Fit on train | Clips val/test to training percentile bounds |
| TALENT preprocessing | During fit() | Per TALENT | NaN imputation, encoding, normalization |

### Output Types and Probability Extraction

| Output Type | Methods | Probability Handling |
|-------------|---------|---------------------|
| **LOGITS** | Most deep learning methods (MLP, ResNet, FTT, SAINT, etc.) | Softmax (2D) or sigmoid (1D) applied |
| **PROBABILITIES** | TabPFN*, TabNet, RealMLP, XGBoost, CatBoost, LightGBM, LogReg, KNN, RF, SVM, NCM, NaiveBayes, Dummy | Used directly |
| **CLASS_LABELS** | LinearRegression (regression-only) | Continuous predictions; not used for classification |

### Method Preprocessing Policies

Each method has specific preprocessing requirements enforced by TALENT's `assert` statements. These are configured in the method registry (`src/methods/method_config.py`) and applied automatically.

**Default policies** (applied when a method has no override): `normalization=standard`, `num_nan_policy=median`, `cat_nan_policy=new`, `num_policy=none`

| Method | Cat Policy | Normalization | Num Policy | Output Type |
|--------|-----------|---------------|------------|-------------|
| **Foundation Models** | | | | |
| tabpfn | indices | none | none | PROBABILITIES |
| tabpfn_v2 | indices | none | none | PROBABILITIES |
| tabpfn_real | indices | none | none | PROBABILITIES |
| mitra | indices | none | none | LOGITS |
| tabicl | indices | none | none | PROBABILITIES |
| tabptm | ohe | standard | none | LOGITS |
| hyperfast | indices | none | none | PROBABILITIES |
| **Transformer-Based** | | | | |
| ftt | indices | standard | default | LOGITS |
| saint | indices | standard | default | LOGITS |
| tabtransformer | indices | standard | default | LOGITS |
| autoint | indices | standard | default | LOGITS |
| excelformer | ohe | standard | default | LOGITS |
| amformer | indices | standard | default | LOGITS |
| trompt | indices | standard | default | LOGITS |
| t2gformer | indices | standard | default | LOGITS |
| **Deep Tabular** | | | | |
| mlp | ohe | standard | default | LOGITS |
| resnet | ohe | standard | default | LOGITS |
| snn | indices | standard | default | LOGITS |
| realmlp | indices | standard | default | PROBABILITIES |
| mlp_plr | tabr_ohe | standard | none | LOGITS |
| tabnet | ohe | standard | default | PROBABILITIES |
| node | ohe | standard | default | LOGITS |
| tabr | tabr_ohe | standard | none | LOGITS |
| grownet | indices | standard | default | LOGITS |
| danets | ohe | standard | default | LOGITS |
| tabcaps | ohe | standard | default | LOGITS |
| dcn2 | indices | standard | default | LOGITS |
| tangos | ohe | standard | default | LOGITS |
| ptarl | indices | standard | default | LOGITS |
| switchtab | ohe | standard | default | LOGITS |
| dnnr | ohe | standard | default | LOGITS |
| modernNCA | tabr_ohe | standard | none | LOGITS |
| bishop | indices | standard | default | LOGITS |
| protogate | ohe | standard | default | LOGITS |
| grande | indices | standard | default | LOGITS |
| tabautopnpnet | tabr_ohe | standard | none | LOGITS |
| tabm | indices | standard | default | LOGITS |
| limix | indices | none | default | LOGITS |
| **Tree Boosting** | | | | |
| xgboost | ohe | standard | default | PROBABILITIES |
| catboost | indices | standard | default | PROBABILITIES |
| lightgbm | ohe | standard | default | PROBABILITIES |
| **Classical ML** | | | | |
| LogReg | ohe | standard | default | PROBABILITIES |
| LinearRegression | ohe | standard | default | CLASS_LABELS |
| RandomForest | ohe | standard | default | PROBABILITIES |
| knn | ohe | standard | default | PROBABILITIES |
| svm | ohe | standard | default | PROBABILITIES |
| NaiveBayes | ohe | standard | default | PROBABILITIES |
| NCM | ohe | standard | default | PROBABILITIES |
| dummy | ohe | standard | default | PROBABILITIES |

**Cat Policy legend**: `indices` = integer codes for embedding layers; `ohe` = one-hot encoding; `tabr_ohe` = TabR-specific OHE (separate N/C arrays)

**Normalization**: `standard` = StandardScaler (zero mean, unit variance); `none` = no normalization applied

**Num Policy**: `default` = uses the global default; `none` = no numerical encoding applied

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

- **`CONFIG_DATA.yaml`**: Dataset toggles, split settings (cv_splits, val_size, seed, row_limit)
- **`CONFIG_METHOD.yaml`**: Method selection per task (PD/LGD)
- **`CONFIG_EXPERIMENT.yaml`**: Training parameters (max_epochs, batch_size, n_trials), plus experiment-specific parameters (row limits for Exp2, minority proportions for Exp3)

### 3. Running Experiments

```bash
# Generate SLURM scripts
python scripts/Experiment1/Experiment1_Setup.py

# Submit to cluster
sbatch scripts/Experiment1/Experiment1_GPU.slurm
sbatch scripts/Experiment1/Experiment1_CPU.slurm
```

### 4. Analyzing Results

Results are stored as pickle files in `results/experimentN/{pd,lgd}/`. Generate summary CSVs and use the analysis notebooks:

```bash
python src/utils/summarize_results.py --experiment experiment1
```

Analysis notebooks in `notebooks/`:

| Notebook | Purpose |
|----------|---------|
| `Experiment0.ipynb` | Method selection analysis from pilot study |
| `Experiment1.ipynb` | Full benchmark analysis: heatmaps, rank analysis, PAMA, correlation studies, training time |
| `Experiment2.ipynb` | Learning curve analysis: degradation, rank evolution, correlation with training size |
| `Experiment3.ipynb` | Class imbalance analysis: degradation, rank evolution, correlation with minority proportion |
| `Data_Exploration.ipynb` | Dataset characteristics analysis |
| `Individual_Method_Runner.ipynb` | Debug and test individual methods |

---

## Repository Structure

```
TabPFNCredit/
├── notebooks/
│   ├── Experiment0.ipynb           # Pilot study analysis
│   ├── Experiment1.ipynb           # Full benchmark analysis
│   ├── Experiment2.ipynb           # Learning curve analysis
│   ├── Experiment3.ipynb           # Class imbalance analysis
│   ├── Data_Exploration.ipynb      # Dataset characteristics
│   └── Individual_Method_Runner.ipynb
│
├── scripts/
│   ├── Experiment0/                # Pilot experiment
│   ├── Experiment1/                # Full benchmark
│   ├── Experiment2/                # Learning curves
│   └── Experiment3/                # Class imbalance
│       ├── config/                 # Per-experiment YAML configs
│       ├── ExperimentN.py          # Main experiment runner
│       ├── ExperimentN_Setup.py    # SLURM script generator
│       └── ExperimentN_*.slurm     # Generated SLURM scripts
│
├── src/
│   ├── data/
│   │   ├── data_feeder.py          # CV splitting + post-split preprocessing
│   │   ├── dataset_preprocessing.py # Per-dataset raw data cleaning
│   │   └── preprocessing.py        # Load/cache TALENT-format arrays
│   ├── methods/
│   │   ├── method_config.py        # Single-source-of-truth method registry
│   │   ├── method_runner.py        # TALENT method wrapper + probability extraction
│   │   ├── method_metrics.py       # PD and LGD metric calculation
│   │   └── method_debugger.py      # Quick method testing utility
│   └── utils/
│       ├── config_reader.py        # YAML configuration parser
│       ├── storage_handler.py      # Pickle file I/O with locking
│       ├── summarize_results.py    # Result aggregation to CSV
│       └── remove_results.py       # Selective result removal
│
├── data/
│   ├── raw/                        # Raw CSV datasets (pd/ and lgd/)
│   └── processed/                  # Cached TALENT-formatted .npy arrays
│
├── results/
│   ├── experiment0/
│   ├── experiment1/
│   ├── experiment2/
│   └── experiment3/
│       ├── {pd,lgd}/              # Pickle result files
│       ├── summary/               # Aggregated CSVs & pivot tables
│       └── figures/               # Generated plots
│
├── config_hpo/                     # Per-fold HPO configs (auto-generated)
│
├── requirements.txt
├── requirements_local.txt
└── requirements_vsc.txt
```

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

Optimal threshold determined by maximizing F1 on the **validation set** (no data leakage). For test evaluation, the threshold found on the validation set is applied.

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

## Acknowledgments

- **[TALENT Framework](https://github.com/LAMDA-Tabular/TALENT)** - Unified interface for tabular methods
- **[TabPFN](https://github.com/automl/TabPFN)** - Tabular foundation model
- **VSC (Vlaams Supercomputer Centrum)** - Computational resources

## License

MIT License - see [LICENSE.txt](LICENSE.txt) for details.

## Contact

- **Author**: Andreas Goethals
- **GitHub**: [andreasgoethals](https://github.com/andreasgoethals)
