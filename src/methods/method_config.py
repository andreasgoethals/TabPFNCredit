"""
TALENT method configuration constants and preprocessing requirements.

This module contains all the constant sets that define:
- Which methods are deep learning vs classical
- Method-specific preprocessing requirements
- Methods with special handling needs (row limits, HPO, etc.)

These constants are used by method_runner.py to properly configure
TALENT methods before training.
"""

from __future__ import annotations

# ======================================================================================
#                          CONFIGURATION - METHOD CATEGORIES
# ======================================================================================

# Deep learning methods - EXACT NAMES AS TALENT EXPECTS THEM
DEEP_METHODS = {
    "mlp", "tabnet", "tabpfn", "tabpfn_v2", "tabpfn_real",
    "resnet", "node", "ftt", "tabptm", "tabr",
    "saint", "tabtransformer", "grownet", "autoint",
    "snn", "danets", "tabcaps", "dcn2", "tangos",
    "ptarl", "switchtab", "dnnr", "modernNCA",
    "hyperfast", "bishop", "realmlp", "protogate",
    "mlp_plr", "excelformer", "grande", "amformer",
    "trompt", "tabm", "t2gformer", "tabautopnpnet", "tabicl",
    "limix", "mitra"
}

# Classical methods - EXACT NAMES AS TALENT EXPECTS THEM
CLASSICAL_METHODS = {
    "xgboost", "catboost", "lightgbm", "RandomForest",
    "LogReg", "LinearRegression", "knn", "svm",
    "NaiveBayes", "NCM", "dummy"
}

# Methods that don't benefit from HPO (pre-trained or too simple)
NO_HPO_METHODS = {
    'tabpfn', 'tabpfn_v2', 'tabpfn_real', 'dummy', 'NCM', 
    'NaiveBayes', 'LinearRegression'
}

# Deep learning methods that return logits (require softmax/sigmoid)
LOGIT_METHODS = {
    'mlp', 'resnet', 'node', 'snn', 'danets', 'tabcaps', 'dcn2',
    'switchtab', 'dnnr', 'tangos', 'protogate', 'hyperfast',
    'bishop', 'realmlp', 'mlp_plr', 'excelformer', 'grande',
    'amformer', 'trompt', 'tabm', 't2gformer', 'tabautopnpnet'
}

# Methods that return probabilities directly
PROBABILITY_METHODS = {
    'xgboost', 'catboost', 'lightgbm', 'RandomForest', 'LogReg',
    'knn', 'svm', 'NaiveBayes', 'NCM', 'dummy',
    'tabpfn', 'tabpfn_v2', 'tabpfn_real', 'tabnet', 'ftt', 'tabptm', 'tabr',
    'saint', 'tabtransformer', 'grownet', 'autoint', 'ptarl',
    'modernNCA', 'tabicl', 'limix', 'mitra'
}

# Methods with dataset size limitations (row limits)
METHOD_ROW_LIMITS = {
    'tabpfn': 10_000,
    'tabpfn_v2': 50_000,
}

# ======================================================================================
#                       CONFIGURATION - PREPROCESSING REQUIREMENTS
# ======================================================================================

# Methods requiring cat_policy='indices'
REQUIRES_CAT_INDICES = {
    'amformer', 'autoint', 'bishop', 'catboost', 'dcn2', 'ftt', 'grande', 'grownet',
    'hyperfast', 'ptarl', 'realmlp', 'saint', 'snn',
    't2gformer', 'tabm', 'tabtransformer', 'trompt'
}

# Methods requiring cat_policy='tabr_ohe'
REQUIRES_CAT_TABR_OHE = {
    'modernNCA', 'tabr', 'mlp_plr', 'tabautopnpnet'
}

# Methods requiring cat_policy='ohe'
REQUIRES_CAT_OHE = {
    'tabptm'
}

# Methods that forbid cat_policy='indices'
FORBIDS_CAT_INDICES = {
    'mlp', 'resnet', 'switchtab', 'danets', 'dnnr', 'excelformer',
    'node', 'protogate', 'tabcaps', 'tabnet', 'tangos'
}

# TabPFN variants - special preprocessing (indices, no normalization, no num encoding)
TABPFN_VARIANTS = {
    'tabpfn', 'tabpfn_v2', 'tabpfn_real'
}

# Methods requiring normalization='none'
REQUIRES_NO_NORMALIZATION = {
    'hyperfast', 'tabicl', 'tabpfn', 'tabpfn_v2', 'tabpfn_real'
}

# Methods requiring num_policy='none'
REQUIRES_NO_NUM_ENCODING = {
    'hyperfast', 'modernNCA', 'tabicl', 'tabptm', 'tabr',
    'tabpfn', 'tabpfn_v2', 'tabpfn_real'
}

# Methods requiring normalization='standard'
REQUIRES_STANDARD_NORMALIZATION = {
    'tabptm'
}