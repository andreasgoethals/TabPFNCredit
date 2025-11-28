# TabPFNCredit: Foundation Models for Credit Risk Prediction

**TabPFNCredit** is a comprehensive benchmarking framework designed to evaluate the performance of Tabular Foundation Models (specifically **TabPFN** and **TabPFN v2**) against state-of-the-art Gradient Boosting methods and Deep Learning baselines in the context of Credit Risk Modeling.

This repository supports both **Probability of Default (PD)** classification tasks and **Loss Given Default (LGD)** regression tasks, utilizing the [TALENT](https://github.com/LAMDA-Tabular/TALENT) framework for standardized model execution.

## Key Features

- **Dual-Task Support**: Seamlessly handles Classification (PD) and Regression (LGD)
- **20 Datasets**: 15 PD classification + 5 LGD regression datasets from diverse sources
- **40+ Methods**: Classical ML, Gradient Boosting, and Deep Learning methods via TALENT
- **Zero-Shot vs. Fine-Tuned**: Benchmarks TabPFN's zero-shot capabilities against tuned baselines
- **Automated HPO**: Integrated Hyperparameter Optimization using Optuna
- **Cluster Ready**: Optimized SLURM scripts for VSC/HPC clusters (Fast/Slow job splits)
- **Robust Data Pipeline**: Automated preprocessing, PCA for wide datasets, Parquet support
- **Result Aggregation**: Automated summarization with pivot tables and cross-validation statistics

## Repository Structure

```
TabPFNCredit/
├── config/                          # Centralized YAML configuration
│   ├── CONFIG_DATA.yaml             # Dataset selection, paths, split settings
│   ├── CONFIG_EXPERIMENT.yaml       # Training epochs, batch sizes, HPO trials
│   └── CONFIG_METHOD.yaml           # Enable/disable specific algorithms
│
├── data/                            # Data storage (gitignored)
│   ├── raw/                         # Raw datasets (.csv or .parquet)
│   │   ├── pd/                      # PD classification datasets
│   │   └── lgd/                     # LGD regression datasets
│   └── processed/                   # Cached preprocessed numpy arrays
│
├── notebooks/                       # Interactive development
│   ├── Data_Exploration.ipynb       # Dataset statistics and visualization
│   ├── Individual_Method_Runner.ipynb  # Test single method/dataset pairs
│   └── experiment_runner.ipynb      # Run experiments from Jupyter
│
├── results/                         # Experiment outputs
│   └── experiment1/                 # Results organized by experiment
│       ├── pd/                      # PD task results (.pkl per dataset)
│       ├── lgd/                     # LGD task results (.pkl per dataset)
│       └── summary/                 # Aggregated CSV summaries
│
├── scripts/                         # Execution scripts
│   └── Experiment1/                 # Main experiment scripts
│       ├── Experiment1.py           # Main entry point
│       ├── Experiment1_Fast.slurm   # HPC: 16 standard datasets (40h)
│       └── Experiment1_Slow.slurm   # HPC: 4 large datasets (72h)
│
├── src/                             # Core source code
│   ├── data/                        # Data loading and preprocessing
│   │   ├── dataset_preprocessing.py # Dataset-specific cleaning logic
│   │   ├── data_feeder.py           # Data loading utilities
│   │   └── preprocessing.py         # General preprocessing functions
│   ├── methods/                     # Model execution
│   │   ├── method_runner.py         # TALENT wrapper with metrics
│   │   ├── all_methods_runner.py    # Run all methods on all datasets
│   │   ├── HPO_runner.py            # Hyperparameter optimization
│   │   └── method_debugger.py       # Quick debugging utility
│   ├── postprocessing/              # Result analysis
│   │   └── Summarize_Results.py     # Aggregate results into CSVs
│   └── utils/                       # Utilities
│       └── config.py                # Configuration loading
│
├── requirements.txt                 # Main dependencies
├── requirements_local.txt           # Local development (CPU)
└── requirements_vsc.txt             # VSC cluster (GPU)
```

## Datasets

### PD (Classification) - 15 Datasets

| ID | Dataset | Samples | Features | Source |
|----|---------|---------|----------|--------|
| 0001 | gmsc | 150,000 | 10 | Give Me Some Credit (Kaggle) |
| 0002 | taiwan_creditcard | 30,000 | 23 | UCI Repository |
| 0003 | vehicle_loan | 233,154 | 41 | Analytics Vidhya |
| 0004 | lendingclub | 9,578 | 14 | LendingClub |
| 0005 | case_study | ~10,000 | 20 | Academic case study |
| 0006 | myhom | ~5,000 | 15 | Financial institution |
| 0007 | hackerearth | 252,000 | 36 | HackerEarth competition |
| 0008 | cobranded | ~20,000 | 30 | Co-branded credit cards |
| 0009 | german | 1,000 | 20 | UCI German Credit |
| 0010 | bank_status | ~12,000 | 15 | Bank account status |
| 0011 | thomas | ~5,000 | 12 | Thomas et al. textbook |
| 0012 | loan_default | ~35,000 | 25 | Loan default prediction |
| 0013 | home_credit | 307,511 | 121 | Home Credit (Kaggle) |
| 0014 | hmeq | 5,960 | 12 | Home Equity dataset |
| 0015 | algorithmwatch | 158,700 | 2,987 | AlgorithmWatch (PCA reduced) |

### LGD (Regression) - 5 Datasets

| ID | Dataset | Samples | Features | Source |
|----|---------|---------|----------|--------|
| 0001 | heloc | ~10,000 | 23 | Home Equity Line of Credit |
| 0002 | loss2 | ~5,000 | 15 | Loss severity data |
| 0003 | axa | ~8,000 | 20 | Insurance LGD |
| 0004 | base_model | ~10,000 | 18 | Base LGD model |
| 0005 | base_modelisation | ~10,000 | 18 | LGD modelisation |

## Methods

### Classical Machine Learning
- **XGBoost**, **LightGBM**, **CatBoost** - Gradient Boosting
- **Random Forest**, **KNN**, **SVM**, **Logistic/Linear Regression**
- **Naive Bayes**, **NCM** (Nearest Class Mean), **Dummy** (baseline)

### Deep Learning / Transformers
- **TabPFN v1** - Zero-shot tabular foundation model (classification only)
- **TabPFN v2** - Improved version with regression support
- **TabNet**, **MLP**, **ResNet**, **SAINT**, **FT-Transformer**
- **NODE**, **TabTransformer**, **AutoInt**, **DCN2**, and 20+ more via TALENT

## Installation

### Prerequisites
- Python 3.10 or 3.11 (3.13 not yet supported)
- CUDA 11.8+ (optional, for GPU acceleration)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/andreasgoethals/TabPFNCredit.git
cd TabPFNCredit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### VSC/HPC Setup

```bash
# Load required modules
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2023.07-gfbf-2023a

# Create venv with system packages
python -m venv venv --system-site-packages
source venv/bin/activate

# Install dependencies
pip install -r requirements_vsc.txt
```

## Configuration

### 1. Dataset Selection (`config/CONFIG_DATA.yaml`)

```yaml
split:
  cv_splits: 5        # 5-fold cross-validation
  test_size: 0.2      # Test split (if cv_splits=1)
  val_size: 0.2       # Validation split from training
  seed: 42
  row_limit: null     # Set integer for debugging (e.g., 1000)

dataset_pd:
  0001.gmsc: true
  0002.taiwan_creditcard: true
  # ... enable/disable datasets

dataset_lgd:
  0001.heloc: true
  # ...
```

### 2. Method Selection (`config/CONFIG_METHOD.yaml`)

```yaml
methods:
  pd:
    xgboost: true
    tabpfn_v2: true
    catboost: true
    # ... enable/disable methods
  lgd:
    xgboost: true
    tabpfn_v2: true
    # ...
```

### 3. Experiment Settings (`config/CONFIG_EXPERIMENT.yaml`)

```yaml
max_epochs: 200      # Deep learning training epochs
batch_size: 128
n_trials: 50         # Optuna HPO trials
early_stopping: true
```

## Usage

### Running Locally

```bash
# Run full benchmark
python scripts/Experiment1/Experiment1.py

# Options
python scripts/Experiment1/Experiment1.py --no_skip    # Force re-run
python scripts/Experiment1/Experiment1.py --quiet      # Less output
```

### Running on HPC (SLURM)

```bash
# Fast job: 16 standard datasets (~40 hours)
sbatch scripts/Experiment1/Experiment1_Fast.slurm

# Slow job: 4 large datasets (~72 hours)
sbatch scripts/Experiment1/Experiment1_Slow.slurm

# Monitor jobs
squeue -u $USER
```

### Interactive Testing

Use the Jupyter notebooks for development:

```bash
# Test single method/dataset
jupyter notebook notebooks/Individual_Method_Runner.ipynb

# Explore dataset statistics
jupyter notebook notebooks/Data_Exploration.ipynb
```

## Results

### Output Structure

```
results/experiment1/
├── pd/
│   ├── 0001.gmsc.pkl           # Raw results (all folds, metrics, predictions)
│   └── ...
├── lgd/
│   └── ...
└── summary/                     # Generated by Summarize_Results.py
    ├── summary_pd_raw.csv       # All fold results
    ├── summary_pd_aggregated.csv # Mean ± std per method/dataset
    ├── pivot_pd_AUC_no_hpo.csv  # Methods × Datasets comparison
    └── ...
```

### Result File Format (`.pkl`)

```python
{
    'NO_HPO': {
        'xgboost': {
            1: {'metrics': {'AUC': 0.85, 'Gini': 0.70, ...}, 'train_time': 2.3, ...},
            2: {...},  # Fold 2
            # ...
        },
        'tabpfn_v2': {...},
    },
    'HPO': {
        'xgboost': {...},  # With hyperparameter optimization
    }
}
```

### Summarizing Results

```bash
# Generate summary CSVs
python src/postprocessing/Summarize_Results.py

# For specific experiment
python src/postprocessing/Summarize_Results.py --experiment experiment2
```

**Output files:**
- `summary_pd_raw.csv` - Every fold: method, dataset, fold_id, AUC, Gini, KS, Brier, train_time, ...
- `summary_pd_aggregated.csv` - Mean ± std across folds
- `pivot_pd_AUC_no_hpo.csv` - Quick comparison table (methods × datasets)

## Metrics

### PD (Classification)
- **AUC** (Area Under ROC Curve) - Primary metric
- **Gini** coefficient
- **KS** (Kolmogorov-Smirnov) statistic
- **Brier** score
- **LogLoss**, **Accuracy**, **F1**, **Precision**, **Recall**, **MCC**

### LGD (Regression)
- **R²** (Coefficient of Determination) - Primary metric
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Correlation**, **Spearman** correlation

## Contributing

### Adding a New Dataset

1. Add CSV/Parquet to `data/raw/pd/` or `data/raw/lgd/`
2. Add preprocessing logic in `src/data/dataset_preprocessing.py`
3. Enable in `config/CONFIG_DATA.yaml`

### Adding a New Method

1. Ensure it's supported by [TALENT](https://github.com/LAMDA-Tabular/TALENT)
2. Enable in `config/CONFIG_METHOD.yaml`

## TALENT Contributions

This project contributed probability support for classical methods to TALENT ([PR #87](https://github.com/LAMDA-Tabular/TALENT/pull/87)):

- **SVM**: Wrapped with `CalibratedClassifierCV` for `predict_proba()`
- **NCM**: Added softmax over centroid distances for probabilities
- **NaiveBayes**: Enabled native `predict_proba()`
- **Dummy**: Uses `strategy='prior'` for probability support

This enables proper AUC calculation for all classical methods.

## License

This project is licensed under the MIT License - see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgments

- **[TALENT](https://github.com/LAMDA-Tabular/TALENT)** - Tabular Analytics and Learning Toolbox
- **[TabPFN](https://github.com/automl/TabPFN)** - Tabular Prior-Data Fitted Networks
- **KU Leuven** - Research support
- **VSC** (Vlaams Supercomputer Centrum) - Compute resources

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{tabpfncredit2024,
  author = {Goethals, Andreas},
  title = {TabPFNCredit: Foundation Models for Credit Risk Prediction},
  year = {2024},
  url = {https://github.com/andreasgoethals/TabPFNCredit}
}
```

## Contact

- **Author**: Andreas Goethals
- **Affiliation**: KU Leuven, PhD Researcher
- **Promotor**: Prof. Wouter Verbeke
