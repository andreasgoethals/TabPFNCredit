# --- Classical (Sklearn) Models from TALENT ---
from TALENT.model.lib.sklearn_models.sklearn_wrappers import (
    LogisticRegressionClassifier,
    LinearRegressionRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    SVRRegressor,
    SVMClassifier,
    KNNClassifier,
    KNNRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

# --- Neural and Foundation Models ---
from TALENT.model.lib.realmlp.api import MLPClassifier, MLPRegressor
from TALENT.model.lib.tabnet.tabnet import TabNetClassifier, TabNetRegressor
from TALENT.model.lib.pfn_v2.tabpfn.classifier import TabPFNClassifier
from TALENT.model.lib.pfn_v2.tabpfn.regressor import TabPFNRegressor
from TALENT.model.lib.lgbm.lgbm import LGBMClassifier, LGBMRegressor
from TALENT.model.lib.xgb.xgb import XGBClassifier, XGBRegressor
from TALENT.model.lib.cb.cb import CatBoostClassifier, CatBoostRegressor



MODEL_REGISTRY = {
    "pd": {
        "ab": adaboost.AdaBoostClassifier,
        "ann": MLPClassifier,
        "bnb": naivebayes.BernoulliNB,
        "cb": catboost.CatBoostClassifier,
        "dt": decisiontree.DecisionTreeClassifier,
        "gnb": naivebayes.GaussianNB,
        "knn": knn.KNNClassifier,
        "lgbm": lightgbm.LGBMClassifier,
        "lr": logreg.LogisticRegressionClassifier,
        "rf": randomforest.RandomForestClassifier,
        "svm": svm.SVMClassifier,
        "tabnet": TabNetClassifier,
        "tabpfn": TabPFNClassifier,
        "tabpfn_rf": TabPFNClassifier,
        "tabpfn_hpo": TabPFNClassifier,
        "tabpfn_auto": TabPFNClassifier,
        "xgb": xgboost.XGBClassifier,
    },

    "lgd": {
        "ab": adaboost.AdaBoostRegressor,
        "ann": MLPRegressor,
        "cb": catboost.CatBoostRegressor,
        "dt": decisiontree.DecisionTreeRegressor,
        "en": elasticnet.ElasticNetRegressor,
        "knn": knn.KNNRegressor,
        "lgbm": lightgbm.LGBMRegressor,
        "lr": linear_regression.LinearRegressionRegressor,
        "rf": randomforest.RandomForestRegressor,
        "svr": svm.SVRRegressor,
        "tabnet": TabNetRegressor,
        "tabpfn": TabPFNRegressor,
        "tabpfn_rf": TabPFNRegressor,
        "tabpfn_hpo": TabPFNRegressor,
        "xgb": xgboost.XGBRegressor,
    },
}


def get_talent_model(task_type: str, method_name: str):
    """Return the TALENT model class for the given task and method."""
    try:
        model_class = MODEL_REGISTRY[task_type][method_name]
        if model_class is None:
            raise NotImplementedError(f"{method_name} not implemented yet in TALENT.")
        return model_class
    except KeyError:
        raise ValueError(f"Unknown method '{method_name}' for task '{task_type}'.")