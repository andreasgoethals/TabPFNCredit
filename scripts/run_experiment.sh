#!/bin/bash

set -euo pipefail

PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_PATH}"

python "$PROJECT_PATH/run/main.py" \
    --data "$PROJECT_PATH/config/CONFIG_DATA.yaml" \
    --experiment "$PROJECT_PATH/config/CONFIG_EXPERIMENT.yaml" \
    --method "$PROJECT_PATH/config/CONFIG_METHOD.yaml" \
    --evaluation "$PROJECT_PATH/config/CONFIG_EVALUATION.yaml" \
    --tuning "$PROJECT_PATH/config/CONFIG_TUNING.yaml" \
    --workdir "$PROJECT_PATH"