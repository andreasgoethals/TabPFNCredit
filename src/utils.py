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
    pass

def _assert_evaluationconfig(evaluationconfig):
    pass