# tooling:
import datetime
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

# metrics:
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, precision_score,
                             brier_score_loss, average_precision_score, mean_squared_error,
                             mean_absolute_error, r2_score, root_mean_squared_error, recall_score)
from hmeasure import h_score

# classification and regression methods:
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

# foundation models:
from tabpfn import TabPFNClassifier, TabPFNRegressor
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

# Proprietary imports
from src.classes.data import Data
from src.classes.models.ann import NNClassifier, NNRegressor
from src.classes.preprocessing import standardize_data, encode_cat_vars, handle_missing_values
from src.utils import _assert_dataconfig, _assert_experimentconfig, _assert_methodconfig, _assert_evaluationconfig

# hide warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelConfig:
    task: str
    cv_splits: int
    binary_threshold: float
    round_digits: int
    metrics_pd: Dict[str, bool]
    metrics_lgd: Dict[str, bool]
    tune_hyperparameters: bool
    hyperparameters_pd: Dict[str, Any]
    hyperparameters_lgd: Dict[str, Any]
    methods_pd: Dict[str, bool]
    methods_lgd: Dict[str, bool]


class ModelFactory:
    @staticmethod
    def create_classifier(method: str, params: Dict[str, Any]) -> Any:
        models = {
            'ab': lambda p: AdaBoostClassifier(random_state=0, **p),
            'ann': lambda p: NNClassifier(**p),
            'bnb': lambda p: BernoulliNB(**p),
            'cb': lambda p: CatBoostClassifier(random_state=0, **p),
            'dt': lambda p: DecisionTreeClassifier(random_state=0, **p),
            'gnb': lambda p: GaussianNB(),
            'knn': lambda p: KNeighborsClassifier(**p),
            'lda': lambda p: LinearDiscriminantAnalysis(),
            'lgbm': lambda p: LGBMClassifier(random_state=0, verbose=-1, **p),
            'lr': lambda p: LogisticRegression(random_state=0, **p),
            'rf': lambda p: RandomForestClassifier(random_state=0, **p),
            'svm': lambda p: SVC(random_state=0, probability=True, **p),
            'tabnet': lambda p: TabNetClassifier(verbose=0, optimizer_fn=torch.optim.Adam,
                                                 scheduler_fn=torch.optim.lr_scheduler.StepLR, **p),
            'tabpfn': lambda p: TabPFNClassifier(**p),
            'xgb': lambda p: XGBClassifier(random_state=0, **p)
        }
        return models[method](params)

    @staticmethod
    def create_regressor(method: str, params: Dict[str, Any]) -> Any:
        models = {
            'ab': lambda p: AdaBoostRegressor(random_state=0, **p),
            'ann': lambda p: NNRegressor(**p),
            'cb': lambda p: CatBoostRegressor(random_state=0, **p),
            'dt': lambda p: DecisionTreeRegressor(random_state=0, **p),
            'en': lambda p: ElasticNet(random_state=0, **p),
            'knn': lambda p: KNeighborsRegressor(**p),
            'lgbm': lambda p: LGBMRegressor(random_state=0, verbose=-1, **p),
            'lr': lambda p: LinearRegression(**p),
            'rf': lambda p: RandomForestRegressor(random_state=0, **p),
            'svr': lambda p: SVR(**p),
            'tabnet': lambda p: TabNetRegressor(verbose=0, optimizer_fn=torch.optim.Adam,
                                                scheduler_fn=torch.optim.lr_scheduler.StepLR, **p),
            'tabpfn': lambda p: TabPFNRegressor(**p),
            'xgb': lambda p: XGBRegressor(random_state=0, **p)
        }
        return models[method](params)


class ModelEvaluator:
    def __init__(self, config: ModelConfig):
        self.config = config

    def evaluate_classification(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        results = {}
        t = float(self.config.binary_threshold)
        y_pred = (y_pred_proba > t).astype(int)

        metrics = {
            'accuracy': lambda: accuracy_score(y_true, y_pred),
            'brier': lambda: brier_score_loss(y_true, y_pred_proba),
            'f1': lambda: f1_score(y_true, y_pred),
            'precision': lambda: precision_score(y_true, y_pred, zero_division=0.0),
            'recall': lambda: recall_score(y_true, y_pred, zero_division=0.0),
            'h_measure': lambda: h_score(y_true, y_pred_proba),
            'aucroc': lambda: roc_auc_score(y_true, y_pred_proba),
            'aucpr': lambda: average_precision_score(y_true, y_pred_proba)
        }

        for metric_name, metric_func in metrics.items():
            if self.config.metrics_pd.get(metric_name, False):
                results[metric_name] = round(metric_func(), self.config.round_digits)

        return results

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        results = {}
        metrics = {
            'mse': lambda: mean_squared_error(y_true, y_pred),
            'mae': lambda: mean_absolute_error(y_true, y_pred),
            'r2': lambda: r2_score(y_true, y_pred),
            'rmse': lambda: root_mean_squared_error(y_true, y_pred)
        }

        for metric_name, metric_func in metrics.items():
            if self.config.metrics_lgd.get(metric_name, False):
                results[metric_name] = round(metric_func(), self.config.round_digits)

        return results


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
    def __init__(self, dataconfig: Dict, experimentconfig: Dict, methodconfig: Dict, evaluationconfig: Dict):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        print('\nExperiment created at: ', now)

        self.config = ModelConfig(
            task=experimentconfig['task'],
            cv_splits=experimentconfig['cv_splits'],
            binary_threshold=evaluationconfig['binary_threshold'],
            round_digits=evaluationconfig['round_digits'],
            metrics_pd=evaluationconfig['metrics_pd'],
            metrics_lgd=evaluationconfig['metrics_lgd'],
            tune_hyperparameters=methodconfig['tune_hyperparameters'],
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
        self.model_factory = ModelFactory()
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
                    optimal_params = self._get_optimal_hyperparameters(fold, indices, method)

                    # Create and train model
                    model = (self.model_factory.create_classifier(method, optimal_params)
                             if self.config.task == 'pd'
                             else self.model_factory.create_regressor(method, optimal_params))

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

    def _tune_hyperparameters(self, method):
        hyperpara_grid = (self.config.hyperparameters_pd[method]
                          if self.config.task == 'pd'
                          else self.config.hyperparameters_lgd[method])

        param_names = list(hyperpara_grid.keys())
        param_combinations = list(itertools.product(*hyperpara_grid.values()))

        best_model = None
        best_score = float('-inf') if self.config.task == 'pd' else float('inf')

        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            param_dict = {k: None if v == 'None' else v for k, v in param_dict.items()}

            model = (self.model_factory.create_classifier(method, param_dict)
                     if self.config.task == 'pd'
                     else self.model_factory.create_regressor(method, param_dict))

            model = self.model_trainer.train_model(
                model, method, self.data.x_train, self.data.y_train,
                self.data.x_val, self.data.y_val)

            if self.config.task == 'pd':
                score = roc_auc_score(
                    self.data.y_val, model.predict_proba(self.data.x_val)[:, 1])
                if score > best_score:
                    best_score = score
                    best_model = model
            else:
                score = mean_squared_error(
                    self.data.y_val, model.predict(self.data.x_val))
                if score < best_score:
                    best_score = score
                    best_model = model

        optimal_params = {k: v for k, v in best_model.get_params().items()
                          if k in param_names}
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