# Tooling:
from dataclasses import dataclass
from typing import Dict, Any
import logging

import torch

# Classification and regression methods:
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
from src.classes.tabpfn_tuner import create_classifier as tabpfn_create_classifier, create_regressor as tabpfn_create_regressor

# Foundation models:
from tabpfn import TabPFNClassifier, TabPFNRegressor
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from src.classes.models.ann import NNClassifier, NNRegressor

@dataclass
class ModelConfiguration:
    """
    Represents the configuration for a machine learning model.

    This class encapsulates all the necessary details for configuring a machine
    learning model for credit scoring. It includes task definition, cross-validation splits,
    threshold settings for binary classification, numerical details like rounding
    digits, as well as options for hyperparameter tuning, metrics, and methods
    applied during modeling.

    :ivar task: The task for which the model is configured (e.g., classification,
        regression).
    :type task: str
    :ivar cv_splits: The number of cross-validation splits to be used in model
        evaluation.
    :type cv_splits: int
    :ivar binary_threshold: The threshold to be applied for binary classification
        tasks.
    :type binary_threshold: float
    :ivar round_digits: The number of digits to round to in results or metrics.
    :type round_digits: int
    :ivar metrics: A dictionary defining the metrics to evaluate the model.
        Includes details of whether specific metrics are enabled.
    :type metrics: Dict[str, Dict[str, bool]]
    :ivar tune_hyperparameters: Specifies whether hyperparameter tuning is to be
        enabled.
    :type tune_hyperparameters: bool
    :ivar hyperparameters: A dictionary defining the hyperparameters that can
        be tuned. Includes specification of which hyperparameters are selected.
    :type hyperparameters: Dict[str, Dict[str, bool]]
    :ivar optimal_params: A dictionary containing optimal hyperparameter
        values for training.
    :type optimal_params: Dict[str, Dict[str, bool]]
    :ivar methods: A dictionary defining the methods to be used in the modeling
        process. Includes details of enabled or disabled methods.
    :type methods: Dict[str, Dict[str, bool]]
    """
    task: str
    cv_splits: int
    binary_threshold: float
    round_digits: int
    metrics: Dict[str, Dict[str, bool]]
    tune_hyperparameters: bool
    hyperparameters: Dict[str, Dict[str, bool]]
    optimal_params: Dict[str, Dict[str, bool]]
    methods: Dict[str, Dict[str, bool]]


class Models:
    """
    Provides methods for creating machine learning classifiers and regression models.

    This class includes static methods to initialize machine learning models for classification
    and regression tasks. Users can specify the model type using shorthand method names and
    customize the behavior of the models through parameter dictionaries. The available models
    include popular machine learning libraries such as Scikit-learn, XGBoost, LightGBM, and more.

    The class is designed to provide a unified, easy-to-use interface for initializing models
    without requiring users to write repetitive code for model configuration and instantiation.
    """
    @staticmethod
    def create_classifier(method: str, params: Dict[str, Any]) -> Any:
        """
        Creates and returns a machine learning classifier based on the specified method and parameters.

        This method initializes a classifier from a selection of predefined types, including
        popular models like AdaBoost, Logistic Regression, Random Forest, XGBoost, and others.
        The selected model is determined by the `method` argument, and specific parameters
        can be passed to configure the classifier through `params`.

        :param method: The shorthand string identifier for the desired machine learning
            classifier. Supported keys include:
            - 'ab': AdaBoostClassifier
            - 'ann': NNClassifier
            - 'bnb': BernoulliNB
            - 'cb': CatBoostClassifier
            - 'dt': DecisionTreeClassifier
            - 'gnb': GaussianNB
            - 'knn': KNeighborsClassifier
            - 'lda': LinearDiscriminantAnalysis
            - 'lgbm': LGBMClassifier
            - 'lr': LogisticRegression
            - 'rf': RandomForestClassifier
            - 'svm': Support Vector Classifier
            - 'tabnet': TabNetClassifier
            - 'tabpfn': TabPFNClassifier
            - 'xgb': XGBClassifier

        :param params: A dictionary of configuration parameters for customizing the
            selected model. Keys and values depend on the model type.

        :return: An instance of the specified machine learning classifier, configured with
            the provided parameters.
        """
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
            'lr': lambda p: LogisticRegression(random_state=0, solver='liblinear', **p),
            'rf': lambda p: RandomForestClassifier(random_state=0, **p),
            'svm': lambda p: SVC(random_state=0, probability=True, **p),
            'tabnet': lambda p: TabNetClassifier(verbose=0, optimizer_fn=torch.optim.Adam,
                                                 scheduler_fn=torch.optim.lr_scheduler.StepLR, **p),
            'tabpfn': lambda p: TabPFNClassifier(**p),
            'tabpfn_rf': lambda p: tabpfn_create_classifier('tabpfn_rf', **p),
            'tabpfn_hpo': lambda p: tabpfn_create_classifier('tabpfn_hpo', **p),
            'xgb': lambda p: XGBClassifier(random_state=0, **p)
        }
        return models[method](params)

    @staticmethod
    def create_regressor(method: str, params: Dict[str, Any]) -> Any:
        """
        Creates and returns a regression model instance based on the specified method and parameters.

        This static method is designed to provide a uniform interface for creating regression models from
        a predefined set of regression techniques. It uses keywords to distinguish between various
        regression models and allows users to customize the configuration of the chosen model by passing a
        dictionary of parameters.

        :param method:
            A string that specifies the regression model to create. Must be one of the predefined keys
            in the `models` dictionary:
            - 'ab': AdaBoostRegressor
            - 'ann': NNRegressor
            - 'cb': CatBoostRegressor
            - 'dt': DecisionTreeRegressor
            - 'en': ElasticNet
            - 'knn': KNeighborsRegressor
            - 'lgbm': LGBMRegressor
            - 'lr': LinearRegression
            - 'rf': RandomForestRegressor
            - 'svr': SVR
            - 'tabnet': TabNetRegressor
            - 'tabpfn': TabPFNRegressor
            - 'xgb': XGBRegressor

        :param params:
            A dictionary of parameters to configure the specific regression model. The keys and
            values within the dictionary are expected to align with the parameter names and
            supported values of the corresponding regression model.

        :return:
            An instance of the regression model specified by the `method` parameter, initialized
            with the provided `params` configuration.
        """
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
            'tabpfn_rf': lambda p: tabpfn_create_regressor('tabpfn_rf', p),
            'tabpfn_hpo': lambda p: tabpfn_create_regressor('tabpfn_hpo', p),
            'xgb': lambda p: XGBRegressor(random_state=0, **p)
        }
        return models[method](params)