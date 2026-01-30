# TabPFNCredit

**Benchmarking Tabular Foundation Models for Credit Risk Prediction**

A rigorous evaluation framework comparing Tabular Foundation Models (TabPFN, TabPFN v2, TabPFN Real, MITRA, TabICL) against classical ML and deep learning baselines on credit risk prediction tasks. Built on the [TALENT](https://github.com/LAMDA-Tabular/TALENT) framework.

---

## Key Features

- **Rigorous Cross-Validation**: 5-fold CV with proper fold isolation and no data leakage
- **Per-Fold Hyperparameter Optimization**: Independent HPO for each fold using TALENT's built-in optimization
- **Concurrent Execution**: File locking support for safe parallel execution on SLURM clusters
- **Comprehensive Metrics**: Internally calculated metrics for consistency across all method types
- **Smart Preprocessing**: Method-specific preprocessing policies automatically enforced
- **Row Limit Enforcement**: Auto-capping for TabPFN (10k rows) and TabPFN v2 (50k rows)
- **Advanced Analysis**: Rank correlation, PAMA analysis, dataset characteristic studies, learning curves

---

## Research Overview

### Tasks & Datasets

| Task | Type | # Datasets | Key Metrics |
|------|------|-----------|-------------|
| **PD** (Probability of Default) | Binary Classification | 15 | AUC, Gini, KS, F1, Brier |
| **LGD** (Loss Given Default) | Regression (0-1) | 8 | R², RMSE, MAE, Spearman |

**PD datasets**: `gmsc`, `taiwan_creditcard`, `vehicle_loan`, `lendingclub`, `case_study`, `myhom`, `hackerearth`, `cobranded`, `german`, `bank_status`, `thomas`, `loan_default`, `home_credit`, `hmeq`, `algorithmwatch`

**LGD datasets**: `heloc`, `loss2`, `taiwan_creditcard`, `axa`, `base_model`, `base_modelisation`, `lgd_freddie`, `lgd_lendingclub`

### Supported Methods

| Category | Methods |
|----------|---------|
| **Foundation Models** | TabPFN, TabPFN v2, TabPFN Real, MITRA, TabICL |
| **Deep Learning** | FT-Transformer, ResNet, MLP, TabNet, TabR, ModernNCA, AutoInt, DCN2, SAINT, TabTransformer, NODE |
| **Tree Boosting** | XGBoost, CatBoost, LightGBM |
| **Classical ML** | LogisticRegression, LinearRegression, RandomForest, SVM, KNN, NaiveBayes, NCM |

### Experiments

| Experiment | Description | Tasks |
|-----------|-------------|-------|
| **Experiment 0** | Pilot study to select methods for the full benchmark | PD + LGD |
| **Experiment 1** | Full benchmark with NO_HPO and HPO modes | PD + LGD |
| **Experiment 2** | Learning curve analysis: performance vs training set size | PD + LGD |
| **Experiment 3** | Class imbalance analysis: performance vs minority proportion | PD only |

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
│   │   ├── data_feeder.py          # Cross-validation data preparation
│   │   ├── dataset_preprocessing.py # Raw data preprocessing
│   │   └── preprocessing.py        # Method-specific preprocessing
│   ├── methods/
│   │   ├── method_runner.py        # TALENT method wrapper
│   │   ├── method_config.py        # Method categorization & policies
│   │   ├── method_metrics.py       # Metric calculation
│   │   └── method_debugger.py      # Quick method testing
│   └── utils/
│       ├── config_reader.py        # YAML configuration parser
│       ├── storage_handler.py      # Pickle file I/O with locking
│       └── summarize_results.py    # Result aggregation
│
├── data/
│   ├── raw/                        # Raw CSV datasets (pd/ and lgd/)
│   └── processed/                  # Cached TALENT-formatted datasets
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
├── requirements.txt
├── requirements_local.txt
└── requirements_vsc.txt
```

---

## Metrics

### PD (Classification)
AUC, Gini, KS, Brier, Log Loss, Average Precision, Accuracy, Balanced Accuracy, F1, Precision, Recall, MCC

Optimal threshold determined by maximizing F1 on the validation set (no data leakage).

### LGD (Regression)
R², MSE, RMSE, MAE, MedAE, Max Error, Pearson, Spearman, MAPE, Explained Variance

All LGD predictions are clipped to [0, 1].

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
