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
from typing import Any, Dict

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


# ======================================================================================
#                    PREPROCESSING POLICY APPLICATION LOGIC
# ======================================================================================

# Sentinel values that indicate "missing" or "not specified"
_MISSING_SENTINELS = {None, "", "nothing", "Nothing", "NONE", "None"}


def _is_missing(x) -> bool:
    """Check if a value represents "missing" or "not specified"."""
    try:
        return x in _MISSING_SENTINELS
    except TypeError:
        return False


def apply_preprocessing_policies(args: Any, method: str, user_specified: Dict[str, bool]) -> None:
    """
    Apply preprocessing policy defaults and method-specific requirements.
    Uses EXACT method names as TALENT expects them.
    
    This function is responsible for ensuring that each method gets the correct
    preprocessing configuration. It:
    1. Fills in project defaults for any unspecified preprocessing options
    2. Applies method-specific requirements (e.g., TabPFN needs cat_policy='indices')
    3. Validates user-specified options don't conflict with method requirements
    
    Args:
        args: TALENT argument namespace (modified in-place)
        method: TALENT method name (canonical name)
        user_specified: Dict tracking which options user explicitly provided
        
    Raises:
        ValueError: If user-specified options conflict with method requirements
    """
    
    # ==========================================================================
    # Step 1: Fill project defaults for missing values
    # ==========================================================================
    defaults = {
        'cat_policy': 'ordinal',
        'num_policy': 'none',
        'normalization': 'standard',
        'num_nan_policy': 'median',
        'cat_nan_policy': 'new'
    }
    
    for attr, default_value in defaults.items():
        if _is_missing(getattr(args, attr, None)):
            setattr(args, attr, default_value)

    # ==========================================================================
    # Step 2: Apply method-specific categorical encoding requirements
    # ==========================================================================
    
    # Determine required cat_policy
    if method in TABPFN_VARIANTS or method in REQUIRES_CAT_INDICES:
        required_cat = 'indices'
    elif method in REQUIRES_CAT_TABR_OHE:
        required_cat = 'tabr_ohe'
    elif method in REQUIRES_CAT_OHE:
        required_cat = 'ohe'
    else:
        required_cat = None
    
    # Apply or validate cat_policy
    if required_cat:
        if user_specified.get('cat_policy', False):
            if args.cat_policy != required_cat:
                raise ValueError(f"{method} requires cat_policy='{required_cat}' but got '{args.cat_policy}'")
        else:
            args.cat_policy = required_cat
    
    # Handle methods that forbid 'indices'
    elif method in FORBIDS_CAT_INDICES:
        if user_specified.get('cat_policy', False):
            if args.cat_policy == 'indices':
                raise ValueError(f"{method} does not support cat_policy='indices'")
        else:
            if args.cat_policy == 'indices':
                args.cat_policy = 'ordinal'

    # ==========================================================================
    # Step 3: Apply normalization requirements
    # ==========================================================================
    
    if method in REQUIRES_NO_NORMALIZATION:
        if user_specified.get('normalization', False):
            if args.normalization != 'none':
                raise ValueError(f"{method} requires normalization='none' but got '{args.normalization}'")
        else:
            args.normalization = 'none'
    
    elif method in REQUIRES_STANDARD_NORMALIZATION:
        if user_specified.get('normalization', False):
            if args.normalization != 'standard':
                raise ValueError(f"{method} requires normalization='standard' but got '{args.normalization}'")
        else:
            args.normalization = 'standard'

    # ==========================================================================
    # Step 4: Apply numerical encoding requirements
    # ==========================================================================
    
    if method in REQUIRES_NO_NUM_ENCODING:
        if user_specified.get('num_policy', False):
            if args.num_policy != 'none':
                raise ValueError(f"{method} requires num_policy='none' but got '{args.num_policy}'")
        else:
            args.num_policy = 'none'
    
    # TabR OHE methods also require no num encoding
    if method in REQUIRES_CAT_TABR_OHE:
        if user_specified.get('num_policy', False):
            if args.num_policy != 'none':
                raise ValueError(f"{method} requires num_policy='none' but got '{args.num_policy}'")
        else:
            args.num_policy = 'none'


def apply_method_row_limit(method: str, row_limit: int | None) -> int | None:
    """
    Apply method-specific row limits for methods with inherent dataset size constraints.
    
    Some methods have architectural limitations on the number of rows they can process:
    - TabPFN: Maximum 10,000 rows (in-context learning limitation)
    - PFN-v2: Maximum 50,000 rows (larger context window than TabPFN)
    
    If user specifies a row_limit larger than the method's maximum, it will be capped.
    If user specifies a row_limit smaller than the maximum, it will be preserved.
    If user doesn't specify row_limit (None), the method maximum will be applied.
    
    Args:
        method: TALENT method name
        row_limit: User-specified row limit (or None for no limit)
        
    Returns:
        Capped row limit respecting both user preference and method constraints
    """
    if method not in METHOD_ROW_LIMITS:
        return row_limit
    
    method_max = METHOD_ROW_LIMITS[method]
    
    if row_limit is None:
        return method_max
    else:
        return min(row_limit, method_max)