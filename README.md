# TabPFNCredit: Foundation Models for Credit Risk Prediction

**TabPFNCredit** is a comprehensive benchmarking framework designed to evaluate the performance of Tabular Foundation Models (specifically **TabPFN** and **TabPFN v2**) against state-of-the-art Gradient Boosting methods and Deep Learning baselines in the context of Credit Risk Modeling.

This repository supports both **Probability of Default (PD)** classification tasks and **Loss Given Default (LGD)** regression tasks, utilizing the [TALENT](https://github.com/LAMDA-Tabular/TALENT) framework for standardized model execution.

## Key Features

- **Dual-Task Support**: Seamlessly handles Classification (PD) and Regression (LGD).
- **Zero-Shot vs. Fine-Tuned**: Benchmarks TabPFN's zero-shot capabilities against tuned baselines.
- **Automated HPO**: Integrated Hyperparameter Optimization using Optuna.
- **Cluster Ready**: Includes SLURM scripts for parallel execution on VSC/HPC clusters.
- **TALENT Integration**: Wraps the TALENT library to provide access to 20+ deep tabular methods.
- **Robust Data Pipeline**: Automated preprocessing, dimensionality reduction (PCA) for wide datasets, and caching.

## Repository Structure
```
TabPFNCredit/
├── config/                     # Centralized configuration
│   ├── CONFIG_DATA.yaml        # Dataset selection, paths, and split logic
│   ├── CONFIG_EXPERIMENT.yaml  # Training epochs, batch sizes, HPO trials
│   └── CONFIG_METHOD.yaml      # Enable/Disable specific algorithms
├── data/                       # Data storage (gitignored)
│   ├── raw/                    # Place raw CSVs here (inside /pd and /lgd subfolders)
│   └── processed/              # Cached numpy arrays (generated automatically)
├── notebooks/                  # Interactive testing
│   ├── experiment_runner.ipynb # Run full experiments from Jupyter
│   └── individual_method_tester.ipynb # Debug single method/dataset pairs
├── results/                    # Experiment outputs
│   └── experiment1/            # Structured results (pickles and metadata)
├── scripts/                    # Execution scripts
│   ├── Experiment1.py          # Main entry point for benchmarking
│   ├── Experiment1_Array.slurm # HPC job script (Array mode)
│   └── Experiment1_Single.slurm# HPC job script (Single mode)
└── src/                        # Core source code
    ├── data/                   # Data loading, specific dataset cleaning, & preprocessing
    ├── methods/                # Wrappers for TALENT, HPO runners, and debugging
    └── utils/                  # Config reading and result storage handlers
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8 or 12.x (if using GPU acceleration)

### Setup

1. **Clone the repository:**
```bash
   git clone https://github.com/andreasgoethals/TabPFNCredit.git
   cd TabPFNCredit
```

2. **Create a virtual environment:**
```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows
```

3. **Install dependencies:**

   - **Local Machine (CPU/Nightly):**
```bash
     pip install -r requirements_local.txt
```
   - **HPC/VSC (CUDA 11.8):**
```bash
     pip install -r requirements_vsc.txt
```

   *Note: The installation includes the `TALENT` framework directly from GitHub.*

## Configuration

Control the benchmark via YAML files in the `config/` directory.

### 1. Select Datasets (`config/CONFIG_DATA.yaml`)

Define which datasets to include in the run. Ensure raw CSVs match the naming convention in `data/raw/pd/` or `data/raw/lgd/`.
```yaml
paths:
  pd_dir: "data/raw/pd"
  lgd_dir: "data/raw/lgd"

dataset_pd:
  0001.gmsc: true
  0002.taiwan_creditcard: true
  # ...

split:
  cv_splits: 5      # 5-fold Cross Validation
  test_size: 0.2    # Used if cv_splits = 1
  row_limit: null   # Set to integer (e.g., 1000) for debugging
```

### 2. Select Methods (`config/CONFIG_METHOD.yaml`)

Enable or disable specific algorithms.
```yaml
methods:
  pd:
    xgboost: true
    tabpfn_v2: true
    catboost: true
    LogReg: true
    # ...
  lgd:
    xgboost: true
    LinearRegression: true
    # ...
```

### 3. Experiment Settings (`config/CONFIG_EXPERIMENT.yaml`)

Control the intensity of the benchmark.
```yaml
max_epochs: 200      # For Deep Learning methods
batch_size: 128
n_trials: 50         # Number of Optuna trials for HPO
early_stopping: true
```

## Usage

### 1. Data Preparation

Ensure your raw CSV files are placed in:

- `data/raw/pd/{dataset_name}.csv`
- `data/raw/lgd/{dataset_name}.csv`

*Refer to `src/data/dataset_preprocessing.py` to see the expected cleaning logic for specific dataset names (e.g., `0001.gmsc`, `0014.hmeq`).*

### 2. Running Locally

To run the full benchmark (all enabled datasets and methods) sequentially:
```bash
python scripts/Experiment1.py
```

Options:

- `--no_skip`: Force re-run even if results exist.
- `--quiet`: Reduce console output.

### 3. Running on Cluster (SLURM)

The repository is optimized for VSC (Flemish Supercomputer Center).

**Submit an Array Job (Parallel Datasets):**
This runs different datasets in parallel across different nodes/GPUs.
```bash
sbatch scripts/Experiment1_Array.slurm
```

**Submit a Debug Job (Single):**
```bash
sbatch scripts/Experiment1_Single.slurm
```

### 4. Debugging

To quickly test **all** methods on the first dataset with reduced epochs and rows:
```bash
python src/methods/method_debugger.py
```

## Results

Results are automatically saved to `results/experiment1/`. The storage handler organizes files by task and dataset.

**File Structure:**
```
results/experiment1/
├── pd/
│   ├── 0001.gmsc.pkl               # Pickle containing results
│   └── 0001.gmsc_metadata.json     # Metadata
└── lgd/
    └── 0001.heloc.pkl
```

**Result Content (`.pkl`):**
The pickle file contains a dictionary with two main keys: `NO_HPO` (Default params) and `HPO` (Tuned params).
```python
results = {
    'NO_HPO': {
        'xgboost': { 1: {'metrics': {...}, 'y_pred': ...}, ... },
        'tabpfn':  { 1: {'metrics': {...}, 'y_pred': ...}, ... } 
    },
    'HPO': {
        'xgboost': { 1: {'metrics': {...}, 'y_pred': ...}, ... },
        # TabPFN results are shared if HPO is not applicable
    }
}
```

## Contributing

**Adding a new dataset:**

1. Add the CSV to `data/raw/pd/` or `data/raw/lgd/`.
2. Add a preprocessing routine in `src/data/dataset_preprocessing.py`.
3. Enable it in `config/CONFIG_DATA.yaml`.

**Adding a new method:**

1. Ensure it is supported by TALENT.
2. Enable it in `config/CONFIG_METHOD.yaml`.

## License

This project is licensed under the MIT License - see the `LICENSE.txt` file for details.

## Acknowledgments

- **TALENT**: This framework is built upon [LAMDA-TALENT](https://github.com/LAMDA-Tabular/TALENT).
- **TabPFN**: Integrates the official [TabPFN](https://github.com/automl/TabPFN) implementations.