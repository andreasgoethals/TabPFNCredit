# Credit Scoring

This project benchmarks different models for credit scoring in terms of probability of default (pd) as well as loss
given default (lgd).

## Configuration

The project uses multiple config files located in the `config` directory to set up the experiment.

### Data Configuration (`CONFIG_DATA`)

The data configuration allows the selection of a single dataset for either of the available tasks by setting only one
dataset per task to `true` while all others are set to `false`

- CONFIG_METHOD: model configuration
- CONFIG_EXPERIMENT: training configuration
- CONFIG_EVALUATION: evaluation configuration
- CONFIG_TUNING: hyperparameter tuning configuration