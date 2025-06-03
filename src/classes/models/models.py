# tooling:
from dataclasses import dataclass
from typing import Dict, Any, List

import torch

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

from src.classes.models.ann import NNClassifier, NNRegressor

@dataclass
class ModelConfiguration:
    task: str
    cv_splits: int
    binary_threshold: float
    round_digits: int
    metrics: Dict[str, Dict[str, bool]]
    tune_hyperparameters: bool
    hyperparameters: Dict[str, Dict[str, bool]]
    methods: Dict[str, Dict[str, bool]]


class Models:
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
            'lr': lambda p: LogisticRegression(random_state=0, solver='liblinear', **p),
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