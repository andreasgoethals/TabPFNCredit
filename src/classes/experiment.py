# tooling:
import datetime
import warnings
from typing import Dict, Any, Optional

import numpy as np
from tqdm import tqdm

# metrics:

# foundation models:
from pytorch_tabnet.tab_model import TabNetClassifier

# Proprietary imports
from src.classes.data.data import Data
from src.classes.evaluation import ModelEvaluator
from src.classes.models.models import Models, ModelConfiguration
from src.classes.data.preprocessing import standardize_data, encode_cat_vars, handle_missing_values
from src.classes.tuner import Tuner
from src.utils import _assert_dataconfig, _assert_experimentconfig, _assert_methodconfig, _assert_evaluationconfig

# hide warnings
warnings.filterwarnings("ignore")


class ModelTrainer:
    @staticmethod
    def train_model(model: Any, method: str, x_train: np.ndarray, y_train: np.ndarray,
                    x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        if method == 'tabnet':
            model.fit(
                x_train, y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train,
                eval_set=[(x_train, y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train),
                          (x_val, y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val)],
                eval_name=['train', 'val'],
                eval_metric=['auc' if isinstance(model, TabNetClassifier) else 'rmse'],
                max_epochs=200,
                patience=10,
                batch_size=512,
                virtual_batch_size=512,
                num_workers=0,
                weights=1,
                drop_last=False
            )
        else:
            model.fit(x_train, y_train)
        return model


class Experiment:
    def __init__(self, dataconfig: Dict, experimentconfig: Dict, methodconfig: Dict, evaluationconfig: Dict, tuningconfig: Dict):
        self.methodconfig = methodconfig
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        print('\nExperiment created at: ', now)

        self.config = ModelConfiguration(
            task=experimentconfig['task'],
            cv_splits=experimentconfig['cv_splits'],
            binary_threshold=evaluationconfig['binary_threshold'],
            round_digits=evaluationconfig['round_digits'],
            metrics_pd=evaluationconfig['metrics_pd'],
            metrics_lgd=evaluationconfig['metrics_lgd'],
            tune_hyperparameters=tuningconfig['tune_hyperparameters'],
            hyperparameters_pd=methodconfig['hyperparameters_pd'],
            hyperparameters_lgd=methodconfig['hyperparameters_lgd'],
            methods_pd=methodconfig['methods_pd'],
            methods_lgd=methodconfig['methods_lgd']
        )

        _assert_experimentconfig(experimentconfig)
        _assert_dataconfig(dataconfig, experimentconfig)
        _assert_methodconfig(methodconfig)
        _assert_evaluationconfig(evaluationconfig)

        self.data = Data(dataconfig, experimentconfig)
        self.results = {}
        self.model_factory = Models()
        self.model_evaluator = ModelEvaluator(self.config)
        self.model_trainer = ModelTrainer()

        print(f'Task: {self.config.task}')
        for key, value in dataconfig[f'dataset_{self.config.task}'].items():
            if value:
                print(f'Dataset: {key}')
        print('CV splits: ', self.config.cv_splits)

    def run(self):
        self.data.load_preprocess_data()
        self.data.split_data()
        self.train_evaluate()

    def train_evaluate(self):
        results = {}
        tuned_hyperparams = {}  # Track hyperparams per method
        methods = (self.config.methods_pd if self.config.task == 'pd'
                   else self.config.methods_lgd)

        for fold, indices in tqdm(self.data.split_indices.items(), desc="Cross-validation loop:"):
            x_train, y_train = self.data.x[indices['train']], self.data.y[indices['train']]
            x_val, y_val = self.data.x[indices['val']], self.data.y[indices['val']]
            x_test, y_test = self.data.x[indices['test']], self.data.y[indices['test']]

            # Preprocess data
            x_train, x_val, x_test, y_train, y_val, y_test = self._preprocess_data(
                x_train, x_val, x_test, y_train, y_val, y_test)

            # Store processed data
            self.data.x_train, self.data.y_train = x_train, y_train
            self.data.x_val, self.data.y_val = x_val, y_val
            self.data.x_test, self.data.y_test = x_test, y_test

            for method, use_method in methods.items():
                if use_method:
                    # Tune only once per method
                    if method not in tuned_hyperparams:
                        print(f"---- Tuning {method} ----")
                        tuned_hyperparams[method] = self._get_optimal_hyperparameters(fold, indices, method)
                    #optimal_params = self._get_optimal_hyperparameters(fold, indices, method)

                    # Create and train model
                    model = (self.model_factory.create_classifier(method, tuned_hyperparams.get(method, {})) #optimal_params)
                             if self.config.task == 'pd'
                             else self.model_factory.create_regressor(method, tuned_hyperparams.get(method, {}))) #optimal_params))

                    model = self.model_trainer.train_model(
                        model, method, x_train, y_train, x_val, y_val)

                    # Evaluate model
                    if method not in results:
                        results[method] = {}

                    if self.config.task == 'pd':
                        y_pred_proba = model.predict_proba(x_test)[:, 1]
                        results[method][fold] = self.model_evaluator.evaluate_classification(
                            y_test, y_pred_proba)
                    else:
                        y_pred = model.predict(x_test)
                        results[method][fold] = self.model_evaluator.evaluate_regression(
                            y_test, y_pred)

        self.results = results

    def _preprocess_data(self, x_train, x_val, x_test, y_train, y_val, y_test):
        # Handle missing values
        x_train, x_val, x_test, y_train, y_val, y_test = handle_missing_values(
            x_train, x_val, x_test, y_train, y_val, y_test,
            self.methodconfig, self.data.cols_num_idx, self.data.cols_cat_idx)

        # Encode categorical variables
        x_train, x_val, x_test, y_train, y_val, y_test = encode_cat_vars(
            x_train, x_val, x_test, y_train, y_val, y_test,
            self.methodconfig, self.data.cols_cat, self.data.cols_cat_idx)

        # Standardize data
        x_train, x_val, x_test, y_train, y_val, y_test = standardize_data(
            x_train, x_val, x_test, y_train, y_val, y_test, self.methodconfig)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def _get_optimal_hyperparameters(self, fold, indices, method):
        if self.config.tune_hyperparameters:
            return self._tune_hyperparameters(method)
        else:
            return self._read_hyperparameters_from_config(method)

    def _tune_hyperparameters(self, method: str) -> Dict[str, Any]:
        task = self.config.task
        tuning_config = self.config.hyperparameters_pd[method]

        tuner = Tuner.create_tuner(tuning_config)

        optimal_params = tuner.tune(
            self.model_factory,
            self.model_trainer,
            self.data,
            method,
            task
        )

        print(f"*Best hyperparameters ({method})* {optimal_params}")
        return optimal_params

    def _read_hyperparameters_from_config(self, method):
        params = {}
        hyperparams = (self.config.hyperparameters_pd[method]
                       if self.config.task == 'pd'
                       else self.config.hyperparameters_lgd[method])

        for param in hyperparams:
            params[param] = hyperparams[param][0]
        print(f"*Selected hyperparameters ({method})* {params}")
        return params
