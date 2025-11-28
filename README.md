# TabPFNCredit

**Benchmarking Tabular Foundation Models for Credit Risk Prediction**

A comprehensive benchmarking framework that systematically compares Tabular Foundation Models (TabPFN, TabPFN v2) against classical machine learning methods and deep learning baselines for credit risk prediction tasks. Built on the [TALENT](https://github.com/LAMDA-Tabular/TALENT) framework.

## Overview

TabPFNCredit provides a rigorous empirical evaluation of tabular machine learning methods in the credit risk domain. The framework supports:

- **Probability of Default (PD)** — Binary classification tasks
- **Loss Given Default (LGD)** — Regression tasks

Each method is evaluated with both default hyperparameters and Optuna-optimized configurations, enabling fair comparison of out-of-the-box performance versus tuned performance.

## Repository Structure

```
TabPFNCredit/
├── LICENSE.txt
├── requirements.txt                 # Base dependencies
├── requirements_local.txt           # Local machine (includes PyTorch CPU)
├── requirements_vsc.txt             # VSC supercomputer (CUDA 11.8)
│
├── config/                          # Configuration files
│   ├── CONFIG_DATA.yaml             # Dataset paths, splits, and toggles
│   ├── CONFIG_EXPERIMENT.yaml       # Training parameters (epochs, batch size, HPO trials)
│   └── CONFIG_METHOD.yaml           # Method toggles for PD and LGD tasks
│
├── data/                            # Data directory (not included in repo)
│   └── raw/
│       ├── pd/                      # PD classification datasets
│       └── lgd/                     # LGD regression datasets
│
├── notebooks/
│   ├── experiment_runner.ipynb      # Run complete experiments from notebook
│   └── Individual_Method_Runner.ipynb  # Quick testing of single methods
│
├── scripts/
│   ├── __init__.py
│   └── Experiment1/                 # Main experiment scripts
│       ├── __init__.py
│       ├── Experiment1.py           # Python entry point for experiments
│       ├── Experiment1_Array.slurm  # SLURM array job template
│       ├── Experiment1_Fast.slurm   # Fast datasets (16 datasets, ~4h each)
│       ├── Experiment1_Single.slurm # Single dataset for debugging
│       └── Experiment1_Slow.slurm   # Large datasets (4 datasets, ~72h each)
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_feeder.py           # DataFeeder: unified data loading and CV splitting
│   │   ├── dataset_preprocessing.py # Dataset-specific preprocessing logic
│   │   └── preprocessing.py         # Preprocessing entry point
│   │
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── method_runner.py         # Core method execution via TALENT
│   │   ├── all_methods_runner.py    # Run all enabled methods on a dataset
│   │   ├── HPO_runner.py            # Run NO_HPO vs HPO comparison
│   │   └── method_debugger.py       # Debug and validate all methods
│   │
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   └── Summarize_Results.py     # Aggregate results into CSV summaries
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config_reader.py         # Load YAML configurations
│       └── storage_handler.py       # Save/load experiment results
│
└── results/                         # Experiment outputs (generated at runtime)
    └── {experiment_name}/
        ├── pd/                      # PD results (.pkl files per dataset)
        ├── lgd/                     # LGD results (.pkl files per dataset)
        ├── logs/                    # Experiment logs
        └── summary/                 # Aggregated CSV files
```

## Datasets

### Probability of Default (PD) — 15 Datasets

| ID | Dataset | Rows | Description |
|----|---------|------|-------------|
| 0001 | gmsc | 150K | Give Me Some Credit (Kaggle) |
| 0002 | taiwan_creditcard | 30K | Taiwan Credit Card Default |
| 0003 | vehicle_loan | 233K | Vehicle Loan Default |
| 0004 | lendingclub | 42K | Lending Club Loans |
| 0005 | case_study | 32K | Credit Case Study |
| 0006 | myhom | 28K | Home Loan Default |
| 0007 | hackerearth | 252K | HackerEarth ML Challenge |
| 0008 | cobranded | 26K | Co-branded Credit Cards |
| 0009 | german | 1K | German Credit (UCI) |
| 0010 | bank_status | 100K | Bank Loan Status |
| 0011 | thomas | 1K | Thomas Credit Scoring |
| 0012 | loan_default | 255K | Loan Default Prediction |
| 0013 | home_credit | 307K | Home Credit Default Risk |
| 0014 | hmeq | 6K | Home Equity Loans |
| 0015 | algorithmwatch | 159K | AlgorithmWatch (2987 features → PCA) |

### Loss Given Default (LGD) — 5 Datasets

| ID | Dataset | Description |
|----|---------|-------------|
| 0001 | heloc | Home Equity Line of Credit |
| 0002 | loss2 | Loss Severity Dataset |
| 0003 | axa | AXA Insurance LGD |
| 0004 | base_model | Base LGD Model |
| 0005 | base_modelisation | Base Modelisation |

## Methods

### Classical Machine Learning

| Method | PD | LGD | HPO Support |
|--------|:--:|:---:|:-----------:|
| XGBoost | ✓ | ✓ | ✓ |
| LightGBM | ✓ | ✓ | ✓ |
| CatBoost | ✓ | ✓ | ✓ |
| Random Forest | ✓ | ✓ | ✓ |
| Logistic Regression | ✓ | — | ✓ |
| Linear Regression | — | ✓ | ✓ |
| K-Nearest Neighbors | ✓ | ✓ | ✓ |
| Support Vector Machine | ✓ | ✓ | ✓ |
| Naive Bayes | ✓ | — | ✓ |
| Nearest Class Mean (NCM) | ✓ | — | — |
| Dummy Baseline | ✓ | — | — |

### Deep Learning / Transformers

| Method | PD | LGD | HPO Support | Notes |
|--------|:--:|:---:|:-----------:|-------|
| TabPFN | ✓ | — | — | Pre-trained, classification only |
| TabPFN v2 | ✓ | ✓ | — | Pre-trained, supports regression |
| MLP | ✓ | ✓ | ✓ | Multilayer Perceptron |
| TabNet | ✓ | ✓ | ✓ | Attention-based |
| ResNet | ✓ | ✓ | ✓ | Residual networks for tabular |
| FT-Transformer | ✓ | ✓ | ✓ | Feature Tokenizer Transformer |
| SAINT | ✓ | ✓ | ✓ | Self-Attention + Intersample |
| NODE | ✓ | ✓ | ✓ | Neural Oblivious Decision Ensembles |
| TabTransformer | ✓ | ✓ | ✓ | Transformer for categorical |
| AutoInt | ✓ | ✓ | ✓ | Automatic Feature Interaction |
| DCN2 | ✓ | ✓ | ✓ | Deep & Cross Network v2 |
| + more... | ✓ | ✓ | ✓ | See CONFIG_METHOD.yaml |

## Installation

### Local Machine

```bash
# Clone repository
git clone https://github.com/andreasgoethals/tabpfncredit.git
cd tabpfncredit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements_local.txt
```

### VSC Supercomputer

```bash
# Load modules
module purge
module load Python/3.10.8-GCCcore-12.2.0

# Create conda environment
conda create -n TabPFNCredit python=3.10 -y
conda activate TabPFNCredit

# Install dependencies (CUDA 11.8)
pip install -r requirements_vsc.txt
```

## Configuration

All experiments are controlled through three YAML configuration files in the `config/` directory.

### CONFIG_DATA.yaml

Controls dataset selection and cross-validation settings:

```yaml
split:
  test_size: 0.2      # Test set fraction (only used if cv_splits=1)
  val_size: 0.2       # Validation fraction of training data
  cv_splits: 5        # Number of CV folds
  seed: 42            # Random seed
  row_limit: null     # Optional row limit for debugging

paths:
  pd_dir: "data/raw/pd"
  lgd_dir: "data/raw/lgd"

dataset_pd:
  0001.gmsc: true
  0002.taiwan_creditcard: true
  # ... toggle datasets on/off
  
dataset_lgd:
  0001.heloc: true
  # ...
```

### CONFIG_EXPERIMENT.yaml

Controls training parameters:

```yaml
max_epochs: 200              # Maximum training epochs for deep models
batch_size: 128              # Batch size for training
n_trials: 50                 # Number of HPO trials (Optuna)
early_stopping: true         # Enable early stopping
early_stopping_patience: 25  # Patience for early stopping
```

### CONFIG_METHOD.yaml

Controls which methods are included in experiments:

```yaml
methods:
  pd:                        # Classification methods
    xgboost: true
    lightgbm: true
    catboost: true
    tabpfn: true
    tabpfn_v2: true
    mlp: true
    # ...
    
  lgd:                       # Regression methods
    xgboost: true
    lightgbm: true
    tabpfn_v2: true
    mlp: true
    # ...
```

## Usage

### Notebooks

#### Individual_Method_Runner.ipynb

Quick testing of a single method on a single dataset:

```python
# Configuration at top of notebook
METHOD = "xgboost"           # Method to test
DATASET = "0001.gmsc"        # Dataset name
TASK = "pd"                  # 'pd' or 'lgd'
ROW_LIMIT = 10000            # Limit rows for fast testing
MAX_EPOCHS = 15              # Limit epochs
CV_SPLITS = 1                # Single fold for debugging
TUNE = False                 # No HPO for quick test
```

#### experiment_runner.ipynb

Run the complete experiment from a notebook environment:

```python
from scripts.Experiment1 import run_experiment1

run_experiment1(
    experiment_name="experiment1",
    skip_completed=True,   # Skip datasets with existing results
    verbose=True
)
```

### Command Line

```bash
# Run all datasets sequentially
python scripts/Experiment1/Experiment1.py

# Run specific dataset by index (for parallel execution)
python scripts/Experiment1/Experiment1.py --dataset_idx=0

# Debug all methods on sample data
python src/methods/method_debugger.py
```

### SLURM (HPC Cluster)

```bash
# Submit fast datasets (~4 hours each, 16 datasets)
sbatch scripts/Experiment1/Experiment1_Fast.slurm

# Submit large datasets (~72 hours each, 4 datasets)
sbatch scripts/Experiment1/Experiment1_Slow.slurm

# Debug single dataset
sbatch scripts/Experiment1/Experiment1_Single.slurm
```

Dataset index mapping for SLURM array jobs:
- **0-14**: PD datasets (0001.gmsc through 0015.algorithmwatch)
- **15-19**: LGD datasets (0001.heloc through 0005.base_modelisation)

## Results

### Output Structure

Results are saved as pickle files in `results/{experiment_name}/`:

```
results/experiment1/
├── pd/
│   ├── 0001.gmsc.pkl
│   ├── 0002.taiwan_creditcard.pkl
│   └── ...
├── lgd/
│   ├── 0001.heloc.pkl
│   └── ...
├── logs/
│   └── *.log
└── experiment_metadata.json
```

### Result Format

Each pickle file contains:

```python
{
    'NO_HPO': {                    # Default hyperparameters
        'xgboost': {
            1: {                   # Fold ID
                'y_true': np.array([...]),
                'y_pred': np.array([...]),
                'y_prob': np.array([...]),   # Probabilities (PD only)
                'metrics': {
                    'AUC': 0.85,
                    'Gini': 0.70,
                    'KS': 0.45,
                    # ...
                },
                'train_time': 12.5,
                'info': {'n_num_features': 10, 'n_cat_features': 5}
            },
            2: {...},              # Fold 2
            # ...
        },
        'tabpfn': {...},
        # ...
    },
    'HPO': {                       # Optuna-optimized
        'xgboost': {...},
        # ...
    }
}
```

### Evaluation Metrics

**PD (Classification):**
- AUC, Gini, KS Statistic
- Brier Score, Log Loss
- Accuracy, Balanced Accuracy
- F1, Precision, Recall, MCC
- Average Precision, Average Recall

**LGD (Regression):**
- R², RMSE, MAE, MSE
- MAPE, Correlation, Spearman

### Postprocessing

Aggregate all results into summary CSV files:

```bash
python src/postprocessing/Summarize_Results.py --experiment experiment1
```

This creates:

```
results/experiment1/summary/
├── summary_pd_raw.csv           # All folds, all methods, all metrics
├── summary_lgd_raw.csv
├── summary_pd_aggregated.csv    # Mean ± std per method/dataset/hpo_mode
├── summary_lgd_aggregated.csv
├── pivot_pd_AUC_no_hpo.csv      # Methods × Datasets (default params)
├── pivot_pd_AUC_hpo.csv         # Methods × Datasets (tuned params)
├── pivot_lgd_R2_no_hpo.csv
└── pivot_lgd_R2_hpo.csv
```

## Key Components

### DataFeeder (`src/data/data_feeder.py`)

Unified data loading and cross-validation splitting:

```python
from src.data.data_feeder import DataFeeder

feeder = DataFeeder(
    task="pd",
    dataset="0001.gmsc",
    test_size=0.2,
    val_size=0.2,
    cv_splits=5,
    seed=42,
    row_limit=None
)

# Returns dict: {fold_id: ((N, C, y), info)}
folds = feeder.prepare()
```

### Method Runner (`src/methods/method_runner.py`)

Core interface to TALENT methods:

```python
from src.methods.method_runner import run_talent_method, get_available_methods, supports_hpo

# Check available methods
methods = get_available_methods()
# {'classical': [...], 'deep': [...]}

# Check if method supports HPO
supports_hpo('xgboost')  # True
supports_hpo('tabpfn')   # False (pre-trained)

# Run a method
results = run_talent_method(
    task='pd',
    dataset='0001.gmsc',
    method='xgboost',
    tune=True,
    # ...
)
```

### HPO Runner (`src/methods/HPO_runner.py`)

Runs all methods with both default and optimized hyperparameters:

```python
from src.methods.HPO_runner import run_hpo_comparison

results = run_hpo_comparison(
    task='pd',
    dataset='0001.gmsc',
    test_size=0.2,
    val_size=0.2,
    cv_splits=5,
    seed=42,
    n_trials=50,
    verbose=True
)
# Returns: {'NO_HPO': {...}, 'HPO': {...}}
```

## TALENT Integration

This project builds on [TALENT](https://github.com/LAMDA-Tabular/TALENT). Key integration details:

### Method-Specific Requirements

| Method | cat_policy | normalization | Row Limit | Feature Limit |
|--------|------------|---------------|-----------|---------------|
| TabPFN | indices | none | 10,000 | 100 |
| TabPFN v2 | indices | none | 50,000 | — |
| Classical | ordinal | standard | — | — |

### Preprocessing Policies

- **cat_nan_policy**: All classical methods require `'new'` (creates new category for NaN)
- **num_nan_policy**: Default is `'mean'` (impute with mean)

### Contributions to TALENT

This project contributed [PR #87](https://github.com/LAMDA-Tabular/TALENT/pull/87) to TALENT, adding probability prediction support for classical methods (SVM, NCM, NaiveBayes, Dummy), enabling proper AUC calculation.

## Development

### Debug All Methods

Test all enabled methods quickly:

```bash
python src/methods/method_debugger.py
python src/methods/method_debugger.py --quiet  # Less verbose
```

### Add a New Dataset

1. Place raw data file in `data/raw/pd/` or `data/raw/lgd/`
2. Add preprocessing logic to `src/data/dataset_preprocessing.py`
3. Enable in `config/CONFIG_DATA.yaml`

### Add a New Method

1. Verify method exists in TALENT
2. Enable in `config/CONFIG_METHOD.yaml` under `pd` and/or `lgd`
3. If method doesn't support HPO, add to `NO_HPO_METHODS` list in `method_runner.py`

## License

MIT License — see [LICENSE.txt](LICENSE.txt)

## Citation

```bibtex
@software{tabpfncredit2025,
  author = {Goethals, Andreas},
  title = {TabPFNCredit: Benchmarking Tabular Foundation Models for Credit Risk},
  year = {2025},
  url = {https://github.com/andreasgoethals/tabpfncredit}
}
```

## Acknowledgments

- [TALENT](https://github.com/LAMDA-Tabular/TALENT) — Tabular Analytics and Learning Toolbox
- [TabPFN](https://github.com/automl/TabPFN) — Prior-Data Fitted Networks for Tabular Data
- Prof. Stefan Lessmann, KU Leuven — Supervision
