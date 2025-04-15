#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/home/linux_vmedina/projects/CreditScoring

python run/main.py \
    --data config/CONFIG_DATA.yaml \
    --experiment config/CONFIG_EXPERIMENT.yaml \
    --method config/CONFIG_METHOD.yaml \
    --evaluation config/CONFIG_EVALUATION.yaml \
    --workdir /home/linux_vmedina/projects/CreditScoring
