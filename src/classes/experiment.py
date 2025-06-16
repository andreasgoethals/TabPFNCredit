# Tooling
import datetime
import warnings
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm
import os
import logging

# Foundation models
from pytorch_tabnet.tab_model import TabNetClassifier

# Proprietary imports
from src.classes.data.data import Data
from src.classes.evaluation import ModelEvaluator
from src.classes.models.models import Models, ModelConfiguration
from src.classes.data.preprocessing import standardize_data, encode_cat_vars, handle_missing_values
from src.classes.tabpfn_tuner import create_classifier as create_tabpfn_classifier, \
    create_regressor as create_tabpfn_regressor
from src.classes.tuner import Tuner
from src.utils import _assert_dataconfig, _assert_experimentconfig, _assert_methodconfig, _assert_evaluationconfig, \
    setup_logger

logger = logging.getLogger(__name__)

# Hide warnings
warnings.filterwarnings("ignore")


class ModelTrainer:
    @staticmethod
    def train_model(model: Any, method: str, x_train: np.ndarray, y_train: np.ndarray,
                    x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        """
        Trains a given model using the specified method and training data. The training method is selected
        based on the provided `method` parameter. If `method` is 'tabnet', the model is trained using the
        TabNet framework with additional configurations like evaluation set, metric, batch size, and
        early stopping patience. For all other methods, the model is trained on the given training data
        directly.

        :param model: The machine learning model to be trained. It can be of any type depending on the
                      chosen `method`.
        :param method: Specifies the method to be used for training the model. If 'tabnet', the function
                       applies TabNet-specific configurations. Otherwise, it defaults to the standard
                       model fitting procedure.
        :param x_train: A NumPy array representing the training feature set used for model training.
        :param y_train: A NumPy array representing the training target variable.
        :param x_val: An optional NumPy array representing the validation feature set used for evaluation
                      during training, applicable for TabNet training.
        :param y_val: An optional NumPy array representing the validation target variable corresponding
                      to the validation features, applicable for TabNet training.
        :return: The trained model instance after the completion of the training process.
        """
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
    def __init__(self, dataconfig: Dict, experimentconfig: Dict, methodconfig: Dict, evaluationconfig: Dict,
                 tuningconfig: Dict, log_path: Optional[str] = "outputs", paramconfig: Optional[Dict] = None):

        log_path = os.path.join(log_path, "experiment.log")
        setup_logger(log_path, "DEBUG")
        logger = logging.getLogger(__name__)

        self.methodconfig = methodconfig
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        logger.info(f'Experiment created at: {now}')

        self.config = ModelConfiguration(
            task=experimentconfig['task'],
            cv_splits=experimentconfig['cv_splits'],
            binary_threshold=evaluationconfig['binary_threshold'],
            round_digits=evaluationconfig['round_digits'],
            metrics=evaluationconfig['metrics'],
            tune_hyperparameters=tuningconfig['tune_hyperparameters'],
            hyperparameters=tuningconfig,
            optimal_params=paramconfig,
            methods=methodconfig['methods'],
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

        logger.info(f'Task: {self.config.task}')
        for key, value in dataconfig[f'dataset_{self.config.task}'].items():
            if value:
                logger.info(f'Dataset: {key}')
        logger.info(f'CV splits: {self.config.cv_splits}')

    def run(self):
        self.data.load_preprocess_data()
        self.data.split_data()
        self.train_evaluate()

    def train_evaluate(self):
        results = {}
        tuned_hyperparams = {}
        methods = (self.config.methods['pd'] if self.config.task == 'pd'
                   else self.config.methods['lgd'])

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
                if not use_method:
                    continue  # skip disabled methods

                # if method == 'tabpfn_hpo':
                #     logger.debug(f"Using INTERNAL tuning for {method} (config params will be used)")
                #     # Always fetch params directly from config for internal-tuning
                #     tabpfn_params = (self.config.hyperparameters['tuning_params']['pd'].get(method, {})
                #                      if self.config.task == 'pd'
                #                      else self.config.hyperparameters['tuning_params']['lgd'].get(method, {}))
                #     logger.debug(f"Params for {method}: {tabpfn_params}")
                #     if self.config.task == 'pd':
                #         model = create_tabpfn_classifier(method, tabpfn_params)
                #     else:
                #         model = create_tabpfn_regressor(method, tabpfn_params)
                else:
                    # All other models (including tabpfn_rf): tune if not already tuned
                    if method not in tuned_hyperparams:
                        logger.info(f"Tuning {method}")
                        if self.config.hyperparameters['tuning_params'][self.config.task][
                            method] is not None and 'param_space' in \
                                self.config.hyperparameters['tuning_params'][self.config.task][method]:
                            tuned_hyperparams[method] = [
                                {} if self.config.hyperparameters['tuning_params'][self.config.task][
                                          method] is None else self._get_optimal_hyperparameters(fold, indices, method)]
                        elif self.config.hyperparameters['tuning_params'][self.config.task][
                            method] is not None and 'param_space' not in \
                                self.config.hyperparameters['tuning_params'][self.config.task][method]:
                            tuned_hyperparams[method] = self.config.hyperparameters['tuning_params'][
                                self.config.task].get(method, {})
                        else:
                            tuned_hyperparams[method] = {}
                    else:
                        logger.debug(f"{method} already tuned, using cached params.")

                    if self.config.task == 'pd':
                        logging.info(f"Creating classifier with method: {method}")
                        model = self.model_factory.create_classifier(method, tuned_hyperparams[method])
                    else:
                        logging.info(f"Creating regressor with method: {method}")
                        model = self.model_factory.create_regressor(method, tuned_hyperparams[method])

                logger.debug(f"Training model for {method} on fold {fold}...")
                # Train the model
                start = datetime.datetime.now()
                logger.debug(f"Starttime: {start}")
                model = self.model_trainer.train_model(
                    model, method, x_train, y_train, x_val, y_val
                )

                # Evaluate model
                if method not in results:
                    results[method] = {}

                if fold not in results[method]:
                    results[method][fold] = {}

                if self.config.task == 'pd':
                    y_pred_proba = model.predict_proba(x_test)[:, 1]
                    results[method][fold] = self.model_evaluator.evaluate_classification(
                        y_test, y_pred_proba)
                else:
                    y_pred = model.predict(x_test)
                    results[method][fold] = self.model_evaluator.evaluate_regression(
                        y_test, y_pred)

                end = datetime.datetime.now()
                logger.debug(f"Endtime: {end}")
                training_time = (end - start).total_seconds()
                logger.debug(f"Traintime: {training_time}")
                results[method][fold]['training_time'] = training_time

                logger.debug(f"Finished fold {fold} for {method}\n{'-' * 60}")
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

        logger.info(f'Data preprocessing finished')

        return x_train, x_val, x_test, y_train, y_val, y_test

    def _get_optimal_hyperparameters(self, fold, indices, method):
        if self.config.tune_hyperparameters:
            if self.config.hyperparameters['tuning_method'] == 'none':
                return self._read_hyperparameters_from_config(method)
            else:
                return self._tune_hyperparameters(method)
        else:
            return {}

    def _tune_hyperparameters(self, method: str) -> Dict[str, Any]:
        task = self.config.task
        tuning_config = self.config.hyperparameters
        tuner = Tuner.create_tuner(task, method, tuning_config)

        optimal_params = tuner.tune(
            self.model_factory,
            self.model_trainer,
            self.data,
            method,
            task
        )
        logger.info(f"Best hyperparameters for {method}: {optimal_params}")
        return optimal_params

    def _read_hyperparameters_from_config(self, method):
        params = self.config.optimal_params
        if not params:
            logger.info("No optimal hyperparameters found in config. Using default hyperparameters.")
            return {}

        current_dataset = None
        for dataset in params:
            if dataset in str(self.data.dataset_name):
                current_dataset = dataset
                break

        if not current_dataset or method not in params[current_dataset]:
            return {}

        return params[current_dataset][method]
