# TabPFNCredit

**Benchmarking Tabular Foundation Models for Credit Risk Prediction**

A rigorous evaluation framework comparing Tabular Foundation Models (TabPFN, TabPFN v2) against classical machine learning (XGBoost, CatBoost, LightGBM) and deep learning baselines (TabNet, ResNet, FT-Transformer, TabR, etc.) on credit risk prediction tasks. Built on the [TALENT](https://github.com/LAMDA-Tabular/TALENT) framework with enhancements for production-grade benchmarking.

---

## 🎯 Key Features

- **Rigorous Cross-Validation**: 5-fold CV with proper fold isolation and no data leakage
- **Per-Fold Hyperparameter Optimization**: Independent HPO for each fold using TALENT's built-in optimization
- **Concurrent Execution**: File locking support for safe parallel execution on SLURM clusters
- **Comprehensive Metrics**: Internally calculated metrics for consistency across all method types.
- **Smart Preprocessing**: Method-specific preprocessing policies automatically enforced
- **Row Limit Enforcement**: Auto-capping for TabPFN (10k rows) and TabPFN v2 (50k rows)
- **Advanced Analysis**: Rank correlation, PAMA analysis, and dataset characteristic studies
- **Production-Ready**: Handles CUDA tensors, logit extraction, threshold optimization, and prediction clipping

---

## 📊 Research Overview

This framework evaluates methods on two credit risk tasks:

### Tasks & Datasets

| Task | Type | Datasets | Key Metrics |
|------|------|----------|-------------|
| **PD** (Probability of Default) | Binary Classification | 15 datasets: `gmsc`, `home_credit`, `lendingclub`, `german`, `taiwan`, `give_me_some_credit`, `fico_heloc`, `polish`, `pakdd`, `kdd2009`, `south_german`, `credit_approval`, `australian`, `japanese`, `thomas` | AUC, Gini, KS, F1, Brier |
| **LGD** (Loss Given Default) | Regression (0-1) | 7 datasets: `heloc`, `axa`, `loss2`, `base_model`, `lgd_synthetic`, `mortgage_default`, `recovery_rate` | R², RMSE, MAE, Spearman |

### Supported Methods

Methods are categorized by hardware requirements for efficient SLURM scheduling:

| Category | Hardware | Methods |
|----------|----------|---------|
| **Foundation Models** | GPU | TabPFN, TabPFN v2, tabicl |
| **Deep Learning** | GPU | TabNet, ResNet, FT-Transformer, TabR, NODE, SAINT, MLP, AutoInt, DCN2, ModernNCA, TabTransformer |
| **Tree Boosting** | GPU* | XGBoost, CatBoost, LightGBM |
| **Classical ML** | CPU | RandomForest, SVM, KNN, LogisticRegression, NaiveBayes, LinearRegression, NCM |

*\*Tree boosting methods run on GPU nodes for consistency with deep learning benchmarks.*

Each method runs in two modes:
- **NO_HPO**: Default hyperparameters (TALENT's built-in defaults)
- **HPO**: Per-fold hyperparameter optimization (independent tuning for each CV fold)

---

## 🚀 Quick Start

### 1. Installation

#### Local Machine
```bash
git clone https://github.com/andreasgoethals/tabpfncredit.git
cd tabpfncredit
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements_local.txt
```

#### VSC Supercomputer (Genius/wICE Clusters)
```bash
module purge
module load Python/3.10.8-GCCcore-12.2.0
conda create -n TabPFNCredit python=3.10 -y
conda activate TabPFNCredit
pip install -r requirements_vsc.txt
```

### 2. Data Preparation

Place your raw datasets in `data/raw/`. The framework expects:
- CSV format with features and target column
- Target column named appropriately for task type
- No missing target values

The `DataFeeder` will automatically:
- Cache processed datasets in `data/processed/`
- Apply appropriate preprocessing based on method requirements
- Split data into train/val/test with proper stratification (PD) or standard splitting (LGD)

### 3. Configuration

Control experiments via three YAML files in `config/`:

#### `CONFIG_DATA.yaml`
```yaml
# Enable/disable datasets
datasets:
  pd:
    gmsc: true
    home_credit: true
    german: false  # Disable specific datasets
  lgd:
    heloc: true
    axa: true

# Data splitting
cv_splits: 5          # Number of cross-validation folds
test_size: 0.2        # Test set proportion (only applicable if cv_splits = 1)
val_size: 0.15        # Validation set proportion (from training data)
seed: 42              # Random seed for reproducibility
row_limit: null       # Optional row limit for all datasets 
```

#### `CONFIG_METHOD.yaml`
```yaml
# Enable/disable methods per task
methods:
  pd:
    tabpfn: true
    tabpfn_v2: true
    xgboost: true
    catboost: true
    resnet: false  # Disable specific methods
  lgd:
    tabpfn: true
    catboost: true
```

#### `CONFIG_EXPERIMENT.yaml`
```yaml
# Training parameters
max_epochs: 100
batch_size: 1024
early_stopping: true
early_stopping_patience: 10
n_trials: 50          # Number of HPO trials
```

### 4. Running Experiments

The framework uses an automated setup script to categorize methods and generate SLURM scripts.

#### Step A: Generate SLURM Scripts (Required for Cluster)
```bash
python scripts/Experiment1/Experiment1_Setup.py
```

This reads your config and generates:
- `Experiment1_GPU.slurm` with correct array size for GPU methods
- `Experiment1_CPU.slurm` with correct array size for CPU methods

#### Step B: Submit Jobs to SLURM
```bash
# Submit GPU-accelerated methods
sbatch scripts/Experiment1/Experiment1_GPU.slurm

# Submit CPU-only methods
sbatch scripts/Experiment1/Experiment1_CPU.slurm
```

### 5. Monitoring Progress

Results are saved incrementally as pickle files and each dataset has its own results file:
```
results/experiment1/
├── pd/
│   ├── {dataset}.pkl
└── lgd/
    ├── {dataset}.pkl
```

Each pickle contains per-mode (HPO or NO_HPO), per-method and per-fold results with:
- Predictions (probabilities, binary classes, or regression values)
- Ground truth labels
- Comprehensive metrics
- Training time
- HPO configuration (if used)

### 6. Analyzing Results

#### Generate Summary CSVs
```bash
python src/utils/summarize_results.py --experiment experiment1
```

Creates in `results/experiment1/summary/`:
- `summary_{task}_aggregated.csv`: Mean ± Std across folds
- `pivot_{task}_{metric}_{hpo}.csv`: Methods × Datasets comparison tables
- `summary_{task}_raw.csv`: Individual fold-level results

#### Advanced Analysis (Jupyter Notebooks)

Use `notebooks/Experiment1.ipynb` for:
- **Performance Heatmaps**: Methods × Datasets with color-coded rankings
- **Rank Analysis**: Average and median ranks across datasets
- **Critical Difference Diagrams**: Statistical significance testing (Friedman + Nemenyi)
- **PAMA Analysis**: Probability of achieving maximal accuracy
- **Correlation Studies**: Method ranks vs dataset characteristics
  - Dataset size (n_rows)
  - Feature count (n_features)
  - Dimensionality (n_rows × n_features)
  - Class imbalance (minority class proportion for PD)
- **Scatter Plots**: Method performance vs dataset properties
- **Training Time Analysis**: Total time across folds with rankings

---

## 📂 Repository Structure
```
TabPFNCredit/
├── README.md
├── LICENSE.txt
├── requirements.txt            # Core dependencies
├── requirements_local.txt      # Local development
├── requirements_vsc.txt        # VSC supercomputer
│
├── config/                     # Experiment configuration
│   ├── CONFIG_DATA.yaml        # Dataset toggles & split settings
│   ├── CONFIG_METHOD.yaml      # Method selection per task
│   └── CONFIG_EXPERIMENT.yaml  # Training & HPO parameters
│
├── config_hpo/                 # Tuned hyperparameters (auto-generated)
│   ├── pd/
│   │   └── {dataset}/{method}/HPO_PER_FOLD/{method}-all-folds.json
│   └── lgd/
│       └── {dataset}/{method}/HPO_PER_FOLD/{method}-all-folds.json
│
├── data/
│   ├── raw/                    # Place raw CSV datasets here
│   └── processed/              # Cached TALENT-formatted datasets
│
├── notebooks/
│   ├── Experiment1.ipynb       # Main analysis notebook
│   ├── Data_Exploration.ipynb  # Dataset characteristics analysis
│   └── Individual_Method_Runner.ipynb  # Debug single method
│
├── results/
│   └── experiment1/
│       ├── pd/                 # PD task results (pickle files)
│       ├── lgd/                # LGD task results (pickle files)
│       ├── summary/            # Aggregated CSVs & pivot tables
│       └── figures/            # Generated plots (auto-created)
│
├── scripts/
│   └── Experiment1/
│       ├── Experiment1.py          # Main experiment runner
│       ├── Experiment1_Setup.py    # SLURM script generator
│       ├── Experiment1_GPU.slurm   # GPU job submission (auto-generated)
│       └── Experiment1_CPU.slurm   # CPU job submission (auto-generated)
│
└── src/
    ├── data/
    │   ├── data_feeder.py          # Cross-validation data preparation
    │   └── preprocessing.py        # Method-specific preprocessing
    │
    ├── methods/
    │   ├── method_runner.py        # TALENT method wrapper
    │   ├── method_config.py        # Method categorization & policies
    │   ├── method_metrics.py       # Metric calculation
    │   └── method_debugger.py      # Quick method testing
    │
    └── utils/
        ├── config_reader.py        # YAML configuration parser
        ├── storage_handler.py      # Pickle file I/O
        └── summarize_results.py    # Result aggregation
```

---

## 📈 Metrics & Evaluation

All metrics are calculated internally for consistency across method types.

### PD (Classification) Metrics

**Probability-based** (require `y_prob`):
- **AUC** (Area Under ROC Curve)
- **Gini** (2 × AUC - 1)
- **KS** (Kolmogorov-Smirnov statistic)
- **Brier Score** (Mean squared error of probabilities)
- **Log Loss** (Cross-entropy loss)
- **Average Precision** (Area under precision-recall curve)

**Prediction-based** (require `y_pred`):
- **Accuracy**, **Balanced Accuracy**
- **F1**, **Precision**, **Recall**
- **Matthews Correlation Coefficient (MCC)**

**Threshold Optimization**:
- Optimal threshold determined by maximizing F1 on **validation set** (no data leakage)
- If validation predictions unavailable, falls back to test set (logged as warning)

### LGD (Regression) Metrics

**Error Metrics**:
- **R²** (Coefficient of determination)
- **MSE**, **RMSE** (Mean squared error)
- **MAE** (Mean absolute error)
- **MedAE** (Median absolute error)
- **Max Error** (Worst-case error)

**Correlation Metrics**:
- **Pearson Correlation**
- **Spearman Correlation**

**Other**:
- **MAPE** (Mean absolute percentage error)
- **Explained Variance**

**Prediction Clipping**:
- All LGD predictions are clipped to [0, 1] range
- Clipping statistics (count below 0, count above 1) are logged

---

## 🔧 Technical Details

### Per-Fold Hyperparameter Optimization

When `tune: true` in `CONFIG_EXPERIMENT.yaml`:

1. **Independent Optimization**: Each fold gets its own HPO run
2. **No Data Leakage**: Fold N's hyperparameters optimized only on Fold N's train+val data
3. **Validation-Based**: TALENT optimizes on validation loss (built-in default)
4. **Persistent Storage**: Configs saved to `config_hpo/{task}/{dataset}/{method}/HPO_PER_FOLD/{method}-all-folds.json`
5. **File Locking**: Safe concurrent writes for SLURM array jobs

Example config structure:
```json
{
  "fold_0": {
    "hyperparameters": {"learning_rate": 0.001, "n_estimators": 500},
    "n_trials": 50,
    "timestamp": "2025-01-07T14:32:15"
  },
  "fold_1": {...},
  ...
}
```

### Method-Specific Preprocessing Policies

The framework automatically enforces method requirements:

| Method | Categorical Encoding | Numerical Encoding | Normalization | Notes |
|--------|---------------------|-------------------|---------------|-------|
| TabPFN | None | None | None | Pre-trained, handles raw data |
| TabPFN v2 | None | None | None | Pre-trained, handles raw data |
| XGBoost | Label/Target | None | None | Tree-based, handles categoricals |
| CatBoost | None | None | None | Native categorical support |
| TabNet | One-Hot | None | Standard | Requires OHE for categoricals |
| ResNet | One-Hot | Quantile | Standard | Full preprocessing |
| TabR | TabR-specific | Quantile | Standard | Custom categorical handling |

User-specified preprocessing is validated against method requirements. Conflicts raise errors.


---

## 🎨 Visualization & Analysis

The analysis notebook (`notebooks/Experiment1.ipynb`) provides:

### 1. Performance Heatmaps

### 2. Rank Analysis

### 3. PAMA (Probability of Achieving Maximal Accuracy)

### 4. Correlation Analysis ('Dataset Characteristics & Method Performance)

### 5. Training Time Analysis

---
### For Computational Efficiency

1. **Run setup script before submission**: Ensures correct SLURM array sizes
2. **Use GPU methods on GPU nodes**: Even for tree boosting (consistency)
3. **Monitor intermediate results**: Check pickle files during long runs
4. **Enable early stopping**: Prevents wasted compute on converged models

### For Reproducibility

1. **Fix random seed** in `CONFIG_DATA.yaml`
2. **Document hyperparameter configs**: Saved automatically in `config_hpo/`
3. **Version control configs**: Track which datasets/methods were enabled

---

## 🙏 Acknowledgments

This work builds upon:
- **[TALENT Framework](https://github.com/LAMDA-Tabular/TALENT)** - Unified interface for tabular methods
- **[TabPFN](https://github.com/automl/TabPFN)** - Tabular foundation model
- **VSC (Vlaams Supercomputer Centrum)** - Computational resources

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE.txt](LICENSE.txt) for details.

---

## 📧 Contact

For questions, issues, or contributions:
- **Author**: Andreas Goethals
- **GitHub**: [andreasgoethals](https://github.com/andreasgoethals)
- **Issues**: [GitHub Issues](https://github.com/andreasgoethals/tabpfncredit/issues)

---
