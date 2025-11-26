# TabPFNCredit

A comprehensive benchmarking framework for credit risk modeling, comparing classical machine learning methods and deep learning approaches across probability of default (PD) and loss given default (LGD) tasks.

## Overview

TabPFNCredit provides a systematic evaluation of machine learning methods for credit scoring using the TALENT framework. The project compares:
- **Classical Methods**: XGBoost, LightGBM, CatBoost, RandomForest, LogisticRegression, KNN, SVM
- **Deep Learning Methods**: MLP, TabNet, TabPFN, PFN-v2, and more

Each method is evaluated with and without hyperparameter optimization (HPO) across multiple credit risk datasets.

## Key Features

- **Unified TALENT Integration**: Seamless interface to TALENT's extensive collection of tabular ML methods
- **HPO Benchmarking**: Automatic comparison of default vs. optimized hyperparameters using Optuna
- **Cross-Validation**: Robust evaluation with proper fold isolation and config persistence
- **Dual Task Support**: Handles both classification (PD) and regression (LGD) tasks
- **Method-Specific Preprocessing**: Automatic enforcement of method requirements (encoding, normalization, etc.)
- **Persistent Configurations**: HPO configs saved and reused across runs for reproducibility

## Project Structure
```
TabPFNCredit/
├── config/                  # Configuration files
│   ├── CONFIG_DATA.yaml         # Dataset selection & row limit, sampling, ...
│   ├── CONFIG_METHOD.yaml       # Method selection 
│   └── CONFIG_EXPERIMENT.yaml   # HPO and deep learning Training parameters
│
├── data/                    # Dataset storage (gitignored)
│   ├── raw/                     # Original datasets
│   └── processed/               # Preprocessed datasets (cached)
│
├── notebooks/               # Jupyter notebooks for testing
│   └── individual_method_tester.ipynb
│
├── results/                 # Experiment outputs (gitignored)
│   └── experiment1/
│       ├── pd/                  # PD task results
│       ├── lgd/                 # LGD task results
│       └── config_hpo/          # Saved HPO configurations
│
├── scripts/                 # Experiment runners
│   └── Experiment1.py           # HPO benchmark experiment
│
└── src/                     # Core implementation
    ├── data/                    # Data loading and preprocessing
    │   ├── data_feeder.py
    │   ├── preprocessing.py
    │   └── dataset_preprocessing.py
    │
    ├── methods/                 # Method execution orchestrators
    │   ├── method_runner.py         # TALENT method interface
    │   ├── all_methods_runner.py    # Multi-method runner
    │   └── HPO_runner.py            # HPO comparison runner
    │
    └── utils/                   # Utilities
        ├── config_reader.py
        ├── storage_handler.py
```

## Setup

### Requirements

- Python 3.10+
- CUDA 12.8+ (for GPU support)

### Installation
```bash
# Clone repository
git clone https://github.com/andreasgoethals/TabPFNCredit.git
cd TabPFNCredit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### 1. Data Configuration (`config/CONFIG_DATA.yaml`)

Control which datasets are used for experiments:
```yaml
dataset_pd:
  0001.gmsc: true      # Enable this dataset
  0002.heloc: false    # Disable this dataset
  # ... more datasets

dataset_lgd:
  0001.heloc: true
  # ... more datasets

split:
  test_size: 0.2
  val_size: 0.2
  cv_splits: 5
  seed: 42
  row_limit: None
```

**Important**: Set only the datasets you want to run to `true`.

### 2. Method Configuration (`config/CONFIG_METHOD.yaml`)

Select which methods to benchmark:
```yaml
methods:
  pd:
    xgboost: true
    lightgbm: true
    catboost: true
    RandomForest: true
    LogReg: true
    knn: false
    svm: false
    mlp: true
    tabnet: true
    tabpfn: true
    PFN-v2: true
    # ... more methods
  
  lgd:
    xgboost: true
    lightgbm: true
    # ... (no LogReg/NaiveBayes for regression)
```

### 3. Experiment Configuration (`config/CONFIG_EXPERIMENT.yaml`)

Control training parameters:
```yaml
max_epochs: 200
batch_size: 1024
early_stopping: true
early_stopping_patience: 10
n_trials: 100        # HPO trials
```

## Running Experiments

### Experiment 1: HPO Benchmark

Compares all enabled methods with and without hyperparameter optimization:
```bash
python scripts/Experiment1.py
```

Or use the provided notebook:
```bash
jupyter notebook notebooks/individual_method_tester.ipynb
```

### Results Structure

Results are saved in `results/experiment1/`:
```
results/experiment1/
├── pd/
│   ├── 0001.gmsc.pkl              # Results: NO_HPO vs HPO
│   └── 0001.gmsc_metadata.json    # Dataset metadata
├── lgd/
│   └── 0001.heloc.pkl
└── config_hpo/
    ├── pd/
    │   └── 0001.gmsc/
    │       ├── xgboost-tuned.json    # Saved HPO configs
    │       ├── lightgbm-tuned.json
    │       └── ...
    └── lgd/
        └── 0001.heloc/
            └── ...
```

### Result Format

Each `.pkl` file contains:
```python
{
    'NO_HPO': {
        'xgboost': {
            1: {  # Fold 1
                'y_true': array([...]),
                'y_pred': array([...]),
                'y_prob': array([...]),  # For classification
                'metrics': [...],
                'train_time': 2.5,
                # ... more metadata
            },
            # ... more folds
        },
        # ... more methods
    },
    'HPO': {
        # Same structure as NO_HPO
    }
}
```

## Key Features Explained

### HPO Configuration Persistence

- **First fold**: Runs HPO and saves optimal config to `config_hpo/{task}/{dataset}/{method}-tuned.json`
- **Subsequent folds**: Automatically loads and reuses the saved config
- **Across runs**: Saved configs persist for reproducibility (only used when `tune=True`)

### Method-Specific Requirements

The framework automatically handles method-specific preprocessing:
- **TabPFN/PFN-v2**: Requires `cat_policy='indices'`, `normalization='none'`
- **TabPTM**: Requires `cat_policy='ohe'`, `normalization='standard'`
- **CatBoost**: Requires `cat_policy='indices'`
- And more...

### Automatic Row Limits

Methods with architectural constraints are automatically capped:
- **TabPFN**: 10,000 rows max (in-context learning limit)
- **PFN-v2**: 50,000 rows max

## Development

### Testing Individual Methods

Use the notebook for quick testing:
```python
from src.methods.method_runner import run_talent_method

results = run_talent_method(
    task='pd',
    dataset='0001.gmsc',
    method='xgboost',
    test_size=0.2,
    val_size=0.2,
    cv_splits=5,
    seed=42,
    tune=True,
    n_trials=100,
    verbose=True
)
```

### Adding New Datasets

1. Place dataset in `data/raw/{task}/` (e.g., `data/raw/pd/mynewdata.csv`)
2. Add to `config/CONFIG_DATA.yaml`:
```yaml
   dataset_pd:
     mynewdata: true
```
3. Run experiment

### Archiving Old Results

Results are automatically archived with timestamps when re-running experiments with existing results.

## Troubleshooting

### CUDA Errors
- Ensure CUDA 12.8+ is installed
- Check GPU availability: `torch.cuda.is_available()`

### Method-Specific Issues
- **LightGBM warnings**: Automatically suppressed
- **CatBoost verbosity**: Automatically silenced
- **TabPFN memory**: Automatically capped at 10k rows

### Config Conflicts
- Check that method names match TALENT's exact format (case-sensitive)
- Verify preprocessing policies don't conflict with method requirements

## Citation

If you use this framework, please cite:
```bibtex
@software{tabpfncredit2024,
  title={TabPFNCredit: A Benchmarking Framework for Credit Risk Modeling},
  author={Your Name},
  year={2024},
  url={https://github.com/andreasgoethals/TabPFNCredit}
}
```

## License

[Your License Here]

## Acknowledgments

This project builds upon the [TALENT](https://github.com/qile2000/LAMDA-TALENT) framework for tabular learning.