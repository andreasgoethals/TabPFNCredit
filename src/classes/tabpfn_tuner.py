from typing import Optional, Dict, Any, Union
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)
from tabpfn_extensions.hpo import TunedTabPFNClassifier, TunedTabPFNRegressor

def create_classifier(
    method: str,
    params: Optional[Dict[str, Any]] = None
) -> Union[TabPFNClassifier, RandomForestTabPFNClassifier, TunedTabPFNClassifier]:
    """
    Create a TabPFN-based classifier according to the specified method and parameters.

    Parameters
    ----------
    method : str
        The classifier variant to create. Supported values are:
        - 'tabpfn_rf': Random Forest TabPFN classifier
        - 'tabpfn_hpo': Hyperparameter-optimized TabPFN classifier

    params : dict, optional
        Dictionary of hyperparameters to pass to the classifier. If None, defaults are used.

    Returns
    -------
    classifier : TabPFNClassifier or RandomForestTabPFNClassifier or TunedTabPFNClassifier
        Instantiated classifier object, ready to be trained.

    Raises
    ------
    ValueError
        If the specified method is not recognized.
    """
    if params is None:
        params = {}

    if method == 'tabpfn_rf':
        print("[TabPFN_RF] Creating RandomForestTabPFNClassifier (external tuning expected)...")
        clf_base = TabPFNClassifier(
            ignore_pretraining_limits=True,
            inference_config={"SUBSAMPLE_SAMPLES": 10000}
        )
        return RandomForestTabPFNClassifier(
            tabpfn=clf_base,
            verbose=1,
            max_predict_time=60,
            fit_nodes=True,
            adaptive_tree=True,
            **(params or {})
        )
    elif method == 'tabpfn_hpo':
        print("[TabPFN_HPO] Creating TunedTabPFNClassifier with internal tuning:")
        print(f"           Params: n_trials={params.get('n_trials', 50)}, metric={params.get('metric', 'ROC_AUC')}")
        return TunedTabPFNClassifier(
                n_trials=params.get('n_trials', 50),
            metric=params.get('metric', 'roc_auc'),
            categorical_feature_indices=params.get('categorical_feature_indices', []),
            random_state=params.get('random_state', None),
            **{k: v for k, v in params.items() if
               k not in {'n_trials', 'metric', 'categorical_feature_indices', 'random_state'}}
        )
    else:
        raise ValueError(f"Unknown TabPFN classifier method: {method}")


def create_regressor(
    method: str,
    params: Optional[Dict[str, Any]] = None
) -> Union[TabPFNRegressor, RandomForestTabPFNRegressor, TunedTabPFNRegressor]:
    """
    Create a TabPFN-based regressor according to the specified method and parameters.

    Parameters
    ----------
    method : str
        The regressor variant to create. Supported values are:
        - 'tabpfn_rf': Random Forest TabPFN regressor
        - 'tabpfn_hpo': Hyperparameter-optimized TabPFN regressor

    params : dict, optional
        Dictionary of hyperparameters to pass to the regressor. If None, defaults are used.

    Returns
    -------
    regressor : TabPFNRegressor or RandomForestTabPFNRegressor or TunedTabPFNRegressor
        Instantiated regressor object, ready to be trained.

    Raises
    ------
    ValueError
        If the specified method is not recognized.
    """
    if params is None:
        params = {}

    if method == 'tabpfn_rf':
        reg_base = TabPFNRegressor(
            ignore_pretraining_limits=True,
            inference_config={"SUBSAMPLE_SAMPLES": 10000}
        )

        return RandomForestTabPFNRegressor(
            tabpfn=reg_base,
            verbose=1,
            max_predict_time=60,
            fit_nodes=True,
            adaptive_tree=True,
            **(params or {})
        )
    elif method == 'tabpfn_hpo':
        return TunedTabPFNRegressor(
            n_trials=params.get('n_trials', 50),
            metric=params.get('metric', 'mse'),
            categorical_feature_indices=params.get('categorical_feature_indices', []),
            random_state=params.get('random_state', None),
            **{k: v for k, v in params.items() if
               k not in {'n_trials', 'metric', 'categorical_feature_indices', 'random_state'}}
        )
    else:
        raise ValueError(f"Unknown TabPFN regressor method: {method}")
