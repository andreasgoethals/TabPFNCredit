# Credit Scoring

This project implements and benchmarks different models for credit scoring, focusing on the two key metrics probability
of default (pd) as well as loss
given default (lgd).

## Setup

- Python 3.12 in a conda environment
- Install required packages: `pip install -r requirements.txt`

## Configuration

The project uses a modular configuration system with multiple YAML files in the `config` directory:

### 1. Data Configuration (CONFIG_DATA.yaml)

- Controls dataset selection for each task (PD and LGD)
- Only one dataset should be active (set to `true`) per task
- All other datasets should be set to `false`

### 2. Method Configuration (CONFIG_METHOD.yaml)

- Specifies which models to use for experiments (set to `true`)
- Controls preprocessing
    1. How to handle missing values
    2. How to encode categorical features
    3. How to standardize numerical data

### 3. Experiment Configuration (CONFIG_EXPERIMENT.yaml)

- Controls broader parameters for experiment
    1. Task
    2. Number of cross-validation splits
    3. Size of validation set
    4. Row limit for datasets
    5. Imbalance

### 4. Evaluation Configuration (CONFIG_EVALUATION.yaml)

- Specifies which evaluation metrics use compute (set to `true`)

### 5. Tuning Configuration (CONFIG_TUNING.yaml)

- Contains hyperparameter search spaces
- Defines tuning strategies and optimization parameters

## Project Structure

    config/     - configuration files
    data/       - datasets for different tasks
    outputs/    - results for different tasks split by dataset
        logs/   - logs of specific runs
    run/        - main.py to execute benchmarking
    src/        
    classes/    - files for evaluation, tuning and experiment
        data/   - data loading and preprocessing
        models/ - all models
    streamlit_app/ - code for interactive streamlit application

## Running Experiments

To run an experiment, make sure the configuration files are set up according to your requirements, and run `main.py`.

Alternatively, the `multiprocess_run_all.py` allows to run several experiments at once by specifying a number of available
GPUs. Therefore, in the `CONFIG_DATA.yaml` multiple datasets should be set to `true`.

## Start Streamlit Application

To get an interactive visualization of the results, you can locally run the Streamlit application with the following
command:

`streamlit run streamlit_app/app.py`