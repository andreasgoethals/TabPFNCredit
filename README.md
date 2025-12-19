
# TabPFNCredit

**Benchmarking Tabular Foundation Models for Credit Risk Prediction**

A rigorous framework comparing Tabular Foundation Models (TabPFN, TabPFN v2) against classical machine learning (XGBoost, CatBoost) and deep learning baselines (TabNet, ResNet, TabR) on credit risk tasks. Built on the [TALENT](https://github.com/LAMDA-Tabular/TALENT) framework.

## 🚀 Quick Start

### 1. Installation

**Local Machine**

```bash
git clone [https://github.com/andreasgoethals/tabpfncredit.git](https://github.com/andreasgoethals/tabpfncredit.git)
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

### 2. Configuration

Control the benchmark via YAML files in `config/`:

| File | Purpose |
| --- | --- |
| `CONFIG_DATA.yaml` | Toggle datasets (True/False), set paths, and define split ratios (default: 5-fold CV). |
| `CONFIG_METHOD.yaml` | Enable/disable specific models for PD (classification) or LGD (regression). |
| `CONFIG_EXPERIMENT.yaml` | Set training parameters: `n_trials` (HPO), `max_epochs`, `batch_size`, `early_stopping`. |

### 3. Running Experiments

The framework uses an automated setup script to categorize methods into **CPU tasks** (Classical ML) and **GPU tasks** (Deep Learning + Tree Boosting) and generate the appropriate SLURM scripts.

#### Step A: Generate SLURM Scripts (Crucial)

Before submitting jobs, run the setup script. This reads your config, counts the tasks, and generates `Experiment1_GPU.slurm` and `Experiment1_CPU.slurm` with the correct array ranges.

```bash
python scripts/Experiment1/Experiment1_Setup.py

```

#### Step B: Submit Jobs (SLURM)

```bash
# Submit GPU-accelerated methods (Deep Learning, XGBoost, CatBoost, etc.)
sbatch scripts/Experiment1/Experiment1_GPU.slurm

# Submit CPU-only methods (RandomForest, LR, KNN, etc.)
sbatch scripts/Experiment1/Experiment1_CPU.slurm

```

#### Alternative: Run Locally (Sequential)

```bash
# Run the full benchmark sequentially (uses logic from Experiment1.py)
python scripts/Experiment1/Experiment1.py

# Debug a specific method on a specific dataset
python notebooks/Individual_Method_Runner.ipynb

```

#### Debugging

To quickly test *all* methods on a small subset of data (checking for crashes):

```bash
python src/methods/method_debugger.py

```

### 4. Summarizing Results

Once experiments are complete, aggregate the results into CSV summaries and pivot tables:

```bash
python src/utils/summarize_results.py --experiment experiment1

```

Outputs are saved to `results/experiment1/summary/`:

* `summary_{task}_aggregated.csv`: Mean ± Std metrics across folds.
* `pivot_{task}_{metric}_{hpo}.csv`: Comparison tables (Methods vs Datasets).

## 📊 Supported Assets

### Tasks & Datasets

The framework evaluates models on **Probability of Default (PD)** and **Loss Given Default (LGD)**.

| Task | Datasets | Description |
| --- | --- | --- |
| **PD** (Classification) | `gmsc`, `home_credit`, `lendingclub`, `german`, `taiwan`... (15 total) | Binary classification. Metrics: AUC, Gini, F1. |
| **LGD** (Regression) | `heloc`, `axa`, `loss2`, `base_model`... (7 total) | Regression (0-1 range). Metrics: R², RMSE. |

### Methods & Categorization

Methods are categorized by their hardware requirements for efficient scheduling. Each method runs in two modes: **Default (NO_HPO)** and **Tuned (HPO)**.

| Category | Execution | Methods Included |
| --- | --- | --- |
| **Foundation Models** | GPU | **TabPFN**, **TabPFN v2**. <br>

<br>*(V1 limit: 10k rows; V2 limit: 50k rows. Auto-capped)*. |
| **Deep Learning** | GPU | TabNet, ResNet, FT-Transformer, TabR, NODE, SAINT, MLP, AutoInt, DCN2, ModernNCA, etc. |
| **Tree Boosting** | GPU* | XGBoost, CatBoost, LightGBM (Running on GPU nodes for consistency). |
| **Classical ML** | CPU | RandomForest, SVM, KNN, LogisticRegression, NaiveBayes, NCM, LinearRegression. |

> **Note:** "NO_HPO" methods (TabPFN, NaiveBayes, Dummy) automatically skip the tuning phase. Their results are duplicated to the "HPO" key in outputs to facilitate consistent plotting.

## 📂 Repository Structure

```
TabPFNCredit/
├── config/                 # Experiment configuration (YAML)
├── config_hpo/             # Storage for tuned hyperparameters (created at runtime)
├── data/
│   ├── raw/                # Place raw datasets here
│   └── processed/          # Cached TALENT-formatted datasets
├── notebooks/              # Individual Method Runner & Analysis
├── results/                # Output directory
│   └── experiment1/
│       ├── pd/             # Pickle files for PD tasks
│       ├── lgd/            # Pickle files for LGD tasks
│       └── summary/        # Aggregated CSVs
├── scripts/                # Entry points & SLURM job scripts
│   └── Experiment1/        # Experiment 1 logic & Setup script
└── src/
    ├── data/               # DataFeeder, Preprocessing, Sampling logic
    ├── methods/            # Method runners, Configs, & Metric calculations
    └── utils/              # Storage handler & Result summarizer

```

## 📈 Metrics

Metrics are calculated internally to ensure consistency across all method types (Classical & Deep).

**PD (Classification):**

* **Probability-based:** AUC, Gini, KS, Brier Score, LogLoss, Avg Precision.
* **Prediction-based:** Accuracy, Balanced Accuracy, F1, Precision, Recall, MCC.
* *Note:* Thresholds for prediction-based metrics are optimized to maximize F1.

**LGD (Regression):**

* **Error:** R², MSE, RMSE, MAE, MedAE, MaxError.
* **Correlation:** Pearson, Spearman.
* **Other:** MAPE, Explained Variance.
* *Note:* LGD predictions are clipped to [0, 1].

## 📜 Citation

```bibtex
@software{tabpfncredit2025,
  author = {Goethals, Andreas},
  title = {TabPFNCredit: Benchmarking Tabular Foundation Models for Credit Risk},
  year = {2025},
  url = {[https://github.com/andreasgoethals/tabpfncredit](https://github.com/andreasgoethals/tabpfncredit)}
}

```

## Acknowledgments

* [TALENT Framework](https://github.com/LAMDA-Tabular/TALENT)
* [TabPFN](https://github.com/automl/TabPFN)

```

```