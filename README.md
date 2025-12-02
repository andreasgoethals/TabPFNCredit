# TabPFNCredit

**Benchmarking Tabular Foundation Models for Credit Risk Prediction**

A rigorous framework comparing Tabular Foundation Models (TabPFN, TabPFN v2) against classical machine learning (XGBoost, CatBoost) and deep learning baselines (TabNet, ResNet) on credit risk tasks. Built on the [TALENT](https://github.com/LAMDA-Tabular/TALENT) framework.

## 🚀 Quick Start

### 1. Installation

**Local Machine**

```bash
git clone https://github.com/andreasgoethals/tabpfncredit.git
cd tabpfncredit
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements_local.txt
```

**VSC Supercomputer (Genius/wICE)**

```bash
module purge && module load Python/3.10.8-GCCcore-12.2.0
conda create -n TabPFNCredit python=3.10 -y && conda activate TabPFNCredit
pip install -r requirements_vsc.txt
```

### 2. Running Experiments

**Run locally (Sequential):**

```bash
# Run the full benchmark sequentially
python scripts/Experiment1/Experiment1.py

# Run a specific dataset by index (e.g., 0)
python scripts/Experiment1/Experiment1.py --dataset_idx=0
```

**Run on SLURM (Parallel):**

```bash
sbatch scripts/Experiment1/Experiment1_Fast.slurm  # Small datasets
sbatch scripts/Experiment1/Experiment1_Slow.slurm  # Large datasets
```

## ⚙️ Configuration

Control the benchmark via YAML files in `config/`:

| File | Purpose |
|------|---------|
| `CONFIG_DATA.yaml` | Toggle datasets, set paths, and define split ratios (default: 5-fold CV). |
| `CONFIG_METHOD.yaml` | Enable/disable specific models for PD (classification) or LGD (regression). |
| `CONFIG_EXPERIMENT.yaml` | Set training parameters: `n_trials` (HPO), `max_epochs`, `batch_size`. |

## 📊 Supported Assets

### Tasks & Datasets

The framework evaluates models on Probability of Default (PD) and Loss Given Default (LGD).

| Task | Count | Examples | Description |
|------|-------|----------|-------------|
| PD (Classification) | 15 | `gmsc`, `home_credit`, `lendingclub` | Ranges from 1K to 300K+ rows. |
| LGD (Regression) | 5 | `heloc`, `axa`, `loss2` | Insurance and credit loss severity. |

### Methods

Each method is evaluated in two modes: Default vs. Tuned (Optuna HPO).

| Category | Methods Included | Notes |
|----------|------------------|-------|
| Foundation Models | TabPFN, TabPFN v2 | Pre-trained. No HPO required. (V1 limit: 10k rows; V2 limit: 50k rows) |
| Classical ML | XGBoost, CatBoost, LightGBM, RandomForest, SVM, KNN, LogReg, NaiveBayes | Robust baselines. |
| Deep Learning | TabNet, ResNet, FT-Transformer, SAINT, NODE, MLP, AutoInt, DCN2 + 20 more | Full list in `CONFIG_METHOD.yaml`. |

> **Note:** Methods generally incompatible with HPO (TabPFN, NaiveBayes, Dummy) automatically skip the tuning phase to save compute.

## 📂 Repository Structure

```
TabPFNCredit/
├── config/                 # Experiment configuration (YAML)
├── notebooks/              # Data exploration & analysis notebooks
├── results/                # Output directory (created at runtime)
├── scripts/                # Entry points & SLURM job scripts
└── src/
    ├── data/               # Data loaders & preprocessing logic
    ├── methods/            # TALENT method wrappers & HPO runners
    └── utils/              # Config reading & result storage
```

## 📈 Results & Metrics

Results are saved to `results/{experiment_name}/` as pickle files containing predictions, ground truth, and metrics.

- **PD Metrics:** AUC, Gini, KS, Brier Score, LogLoss, F1, Accuracy.
- **LGD Metrics:** R², RMSE, MAE, Spearman Correlation.

## 📜 Citation

```bibtex
@software{tabpfncredit2025,
  author = {Goethals, Andreas},
  title = {TabPFNCredit: Benchmarking Tabular Foundation Models for Credit Risk},
  year = {2025},
  url = {https://github.com/andreasgoethals/tabpfncredit}
}
```

## Acknowledgments

- [TALENT Framework](https://github.com/LAMDA-Tabular/TALENT)
- [TabPFN](https://github.com/automl/TabPFN)