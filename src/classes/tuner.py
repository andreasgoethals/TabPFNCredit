# hyperparameter_tuning.py
from abc import ABC, abstractmethod
import itertools
import optuna
from sklearn.metrics import roc_auc_score, mean_squared_error
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class HyperparameterTuner(ABC):
    @abstractmethod
    def tune(self, model_factory, model_trainer, data, method: str, task: str) -> Dict[str, Any]:
        pass

    def _evaluate_model(self, model, data, task: str) -> float:
        """Common evaluation logic for all tuners"""
        if task == 'pd':
            score = roc_auc_score(
                data.y_val,
                model.predict_proba(data.x_val)[:, 1]
            )
            return score
        else:
            score = mean_squared_error(
                data.y_val,
                model.predict(data.x_val)
            )
            return -score  # Negative because Optuna maximizes


class GridSearchTuner(HyperparameterTuner):
    def __init__(self, param_grid: Dict[str, list]):
        self.param_grid = param_grid

    def tune(self, model_factory, model_trainer, data, method: str, task: str) -> Dict[str, Any]:
        param_names = list(self.param_grid.keys())
        param_combinations = list(itertools.product(*self.param_grid.values()))

        best_model = None
        best_score = float('-inf')
        best_params = None

        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            param_dict = {k: None if v == 'None' else v for k, v in param_dict.items()}

            model = (model_factory.create_classifier(method, param_dict)
                     if task == 'pd'
                     else model_factory.create_regressor(method, param_dict))

            model = model_trainer.train_model(
                model, method, data.x_train, data.y_train,
                data.x_val, data.y_val
            )

            score = self._evaluate_model(model, data, task)

            if score > best_score:
                best_score = score
                best_model = model
                best_params = param_dict

        return best_params


class OptunaTuner(HyperparameterTuner):
    def __init__(self, param_space: Dict[str, Dict[str, Any]], n_trials: int = 100):
        self.param_space = param_space
        self.n_trials = n_trials

    def tune(self, model_factory, model_trainer, data, method: str, task: str) -> Dict[str, Any]:
        logger.info(f"Starting external hyperparameter tuning for {method} with {self.__class__.__name__}...")
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            params = self._create_trial_params(trial)

            model = (model_factory.create_classifier(method, params)
                     if task == 'pd'
                     else model_factory.create_regressor(method, params))

            model = model_trainer.train_model(
                model, method, data.x_train, data.y_train,
                data.x_val, data.y_val
            )

            return self._evaluate_model(model, data, task)

        study.optimize(objective, n_trials=self.n_trials)

        logger.info(f"Finished external hyperparameter tuning for {method}")
        return study.best_params

    def _create_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        for param_name, param_config in self.param_space.items():
            param_type = param_config['type']
            if param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['values']
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', 1)
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )

        return params

class Tuner:
    INTERNAL_TUNING_METHODS = {'tabpfn_hpo'}
    @staticmethod
    def create_tuner(task: str,method: str, tuning_config: Dict[str, Dict[str, Any]]) -> HyperparameterTuner:
        if method in Tuner.INTERNAL_TUNING_METHODS:
            return None

        if not tuning_config.get('tune_hyperparameters', False):
            return None

        tuner_type = tuning_config.get('tuning_method', 'grid')

        if tuner_type == 'grid':
            param_grid = tuning_config['tuning_params'][task][method]['param_grid']
            return GridSearchTuner(param_grid)
        elif tuner_type == 'optuna':
            param_space = tuning_config['tuning_params'][task][method]['param_space']
            n_trials = tuning_config.get('n_trials', 100)
            return OptunaTuner(param_space, n_trials)
        else:
            raise ValueError(f"Unknown tuner type: {tuner_type}")