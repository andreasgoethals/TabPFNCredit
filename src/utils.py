import random
import logging
import os

import numpy as np
import torch
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def _assert_experimentconfig(experimentconfig):
    # check wheter the task is pd or lgd; giver error otherwise:
    if experimentconfig['task'] not in ['pd', 'lgd']:
        raise ValueError('Invalid task in experimentconfig - change CONFIG_EXPERIMENT.yaml')
    pass

def _assert_dataconfig(dataconfig, experimentconfig):

    task = experimentconfig['task']

    # check whether one dataset is selected (only one dataset can be set True):
    if sum(1 for value in dataconfig[f'dataset_{task}'].values() if value) != 1:
        raise ValueError(f'Only exactly one dataset can be selected (task: {task}) - change CONFIG_DATA.yaml')
    pass

def _assert_methodconfig(methodconfig):
    # todo: implement
    pass

def _assert_evaluationconfig(evaluationconfig):
    # todo: implement
    pass

def setup_logger(path, log_level="INFO"):
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(path),
            logging.StreamHandler()
        ]
    )

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.mps.is_available():
        torch.mps.manual_seed(seed)

    elif torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available():
        raise SystemError(
            'GPU device not found. For fast training, please enable GPU. See section above for instructions.'
        )