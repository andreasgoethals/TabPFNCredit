"""
TALENT Method Configuration and Preprocessing Requirements

This module provides a centralized configuration system for all TALENT methods,
defining their categorization, preprocessing policies, and architectural requirements.

NOTE: GPU vs CPU execution categorization is defined in Experiment1_Setup.py,
      as it's experiment-specific rather than a fundamental method property.

Organization:
    1. Method Categorization (by architecture and optimization needs)
    2. Output Format Requirements (logits vs probabilities)
    3. Preprocessing Requirements (categorical, numerical, normalization)
    4. Helper Functions (policy application logic)
    5. Validation & Sanity Checks

Usage:
    from src.methods.method_config import NO_HPO_METHODS, DEEP_METHODS
    from src.methods.method_config import apply_preprocessing_policies
"""

from __future__ import annotations
import typing as ty
from dataclasses import dataclass

# Type aliases for clarity
MethodName = str
MethodSet = ty.Set[MethodName]

# ======================================================================================
#                    SECTION 1: METHOD CATEGORIZATION BY ARCHITECTURE
# ======================================================================================

# Deep learning methods - Neural network architectures
DEEP_METHODS: MethodSet = {
    # Basic neural architectures
    'mlp', 'resnet',
    
    # Attention-based transformers
    'ftt', 'saint', 'tabtransformer', 'tabptm', 'trompt',
    
    # Specialized deep learning
    'tabnet', 'node', 'tabr', 'grownet',
    
    # Advanced architectures
    'autoint', 'snn', 'danets', 'tabcaps', 'dcn2',
    'tangos', 'ptarl', 'switchtab', 'dnnr',
    
    # Modern architectures
    'modernNCA', 'hyperfast', 'bishop', 'realmlp',
    'protogate', 'mlp_plr', 'excelformer', 'grande',
    'amformer', 'tabm', 't2gformer', 'tabautopnpnet',
    'tabicl', 'limix', 'mitra',
    
    # Foundation models
    'tabpfn', 'tabpfn_v2', 'tabpfn_real',
}

# Classical methods - Traditional ML algorithms
CLASSICAL_METHODS: MethodSet = {
    # Tree-based gradient boosting
    'xgboost', 'catboost', 'lightgbm',
    
    # Traditional ML models
    'RandomForest', 'LogReg', 'LinearRegression',
    'knn', 'svm', 'NaiveBayes', 'NCM',
    
    # Baseline models
    'dummy',
}

# Methods that don't benefit from hyperparameter optimization
NO_HPO_METHODS: MethodSet = {
    # Foundation models (pre-trained, no tuning needed)
    'tabpfn', 'tabpfn_v2', 'tabpfn_real','tabicl'
    
    # Simple baselines (no hyperparameters or already optimal)
    'dummy', 'NCM', 'NaiveBayes', 'LinearRegression',
}


# ======================================================================================
#                    SECTION 2: OUTPUT FORMAT REQUIREMENTS
# ======================================================================================

# Methods that return raw logits (need softmax for probabilities)
LOGIT_METHODS: MethodSet = {
    # Basic neural networks
    'mlp', 'resnet', 'node',
    
    # Specialized deep architectures
    'snn', 'danets', 'tabcaps', 'dcn2', 'switchtab',
    'dnnr', 'tangos', 'protogate', 'hyperfast',
    'bishop', 'realmlp', 'mlp_plr',
    
    # Modern transformers
    'excelformer', 'grande', 'amformer', 'trompt',
    'tabm', 't2gformer', 'tabautopnpnet',
    
    # FT-Transformer returns logits
    'ftt',
}

# Methods that return calibrated probabilities directly
PROBABILITY_METHODS: MethodSet = {
    # Classical methods (all return probabilities or raw predictions)
    'xgboost', 'catboost', 'lightgbm', 'RandomForest',
    'LogReg', 'LinearRegression', 'knn', 'svm', 
    'NaiveBayes', 'NCM', 'dummy',
    
    # Foundation models (calibrated probabilities)
    'tabpfn', 'tabpfn_v2', 'tabpfn_real',
    
    # Deep learning methods with probability output
    'tabnet', 'tabptm', 'tabr', 'saint', 'tabtransformer',
    'grownet', 'autoint', 'ptarl', 'modernNCA', 'tabicl',
    'limix', 'mitra',
}


# ======================================================================================
#                    SECTION 3: PREPROCESSING REQUIREMENTS
# ======================================================================================

@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Immutable configuration for preprocessing policies.
    Defines default preprocessing strategies for tabular data.
    """
    cat_policy: str = 'ordinal'          # Categorical encoding: ordinal, ohe, indices, tabr_ohe
    num_policy: str = 'none'             # Numerical encoding: none, Q_PLE, T_PLE, etc.
    normalization: str = 'standard'      # Normalization: standard, minmax, quantile, none
    num_nan_policy: str = 'median'       # Numerical NaN: mean, median
    cat_nan_policy: str = 'new'          # Categorical NaN: new, most_frequent


# Default preprocessing configuration (applied when user doesn't specify)
DEFAULT_PREPROCESSING = PreprocessingConfig()


# -----------------------------------------------------------------------------
# Categorical Encoding Requirements
# -----------------------------------------------------------------------------

# Methods requiring cat_policy='indices' (keep categories as integer codes)
REQUIRES_CAT_INDICES: MethodSet = {
    # Transformers with embedding layers
    'amformer', 'autoint', 'bishop', 'dcn2', 'ftt', 'grande',
    'grownet', 'hyperfast', 'ptarl', 'realmlp', 'saint', 'snn',
    't2gformer', 'tabm', 'tabtransformer', 'trompt','tabicl','limix','mitra',
    
    # Tree-based methods with native categorical support
    'catboost',
}

# Methods requiring cat_policy='tabr_ohe' (TabR-specific one-hot encoding)
REQUIRES_CAT_TABR_OHE: MethodSet = {
    'modernNCA', 'tabr', 'mlp_plr', 'tabautopnpnet',
}

# Methods requiring cat_policy='ohe' (standard one-hot encoding)
REQUIRES_CAT_OHE: MethodSet = {
    'tabptm',
}

# Methods that cannot handle cat_policy='indices' (need numerical features)
FORBIDS_CAT_INDICES: MethodSet = {
    'mlp', 'resnet', 'switchtab', 'danets', 'dnnr',
    'excelformer', 'node', 'protogate', 'tabcaps',
    'tabnet', 'tangos',
}

# TabPFN variants (special case: indices + no normalization + no num encoding)
TABPFN_VARIANTS: MethodSet = {
    'tabpfn', 'tabpfn_v2', 'tabpfn_real',
}


# -----------------------------------------------------------------------------
# Normalization Requirements
# -----------------------------------------------------------------------------

# Methods requiring normalization='none' (expect raw scale)
REQUIRES_NO_NORMALIZATION: MethodSet = {
    'hyperfast', 'tabicl','limix',
    # TabPFN variants
    'tabpfn', 'tabpfn_v2', 'tabpfn_real','mitra',
}

# Methods requiring normalization='standard' (z-score normalization)
REQUIRES_STANDARD_NORMALIZATION: MethodSet = {
    'tabptm',
}


# -----------------------------------------------------------------------------
# Numerical Encoding Requirements
# -----------------------------------------------------------------------------

# Methods requiring num_policy='none' (no numerical encoding)
REQUIRES_NO_NUM_ENCODING: MethodSet = {
    'hyperfast', 'modernNCA', 'tabicl', 'tabptm', 'tabr',
    # TabPFN variants
    'tabpfn', 'tabpfn_v2', 'tabpfn_real',
}


# -----------------------------------------------------------------------------
# Dataset Size Constraints
# -----------------------------------------------------------------------------

# Methods with inherent architectural row limits
METHOD_ROW_LIMITS: ty.Dict[MethodName, int] = {
    'tabpfn': 10_000,        # In-context learning window limit
    'tabpfn_v2': 50_000,     # Larger context window than TabPFN
}


# ======================================================================================
#                    SECTION 4: VALIDATION & HELPER FUNCTIONS
# ======================================================================================

# Sentinel values indicating "missing" or "not specified"
_MISSING_SENTINELS: ty.Set[ty.Any] = {None, "", "nothing", "Nothing", "NONE", "None"}


def _is_missing(value: ty.Any) -> bool:
    """
    Check if a value represents "missing" or "not specified".
    
    Args:
        value: Value to check
        
    Returns:
        True if value is considered missing/unspecified
    """
    try:
        return value in _MISSING_SENTINELS
    except TypeError:
        # Handle unhashable types (e.g., lists, dicts)
        return False


def apply_preprocessing_policies(
    args: ty.Any,
    method: MethodName,
    user_specified: ty.Dict[str, bool]
) -> None:
    """
    Apply preprocessing policy defaults and method-specific requirements.
    
    This function modifies the args namespace in-place to ensure correct
    preprocessing configuration for each method. It follows a three-step process:
    
    1. Fill in default values for any unspecified preprocessing options
    2. Apply method-specific requirements (e.g., TabPFN needs cat_policy='indices')
    3. Validate that user-specified options don't conflict with requirements
    
    Args:
        args: TALENT argument namespace (modified in-place)
        method: Method name (must match TALENT's expected naming)
        user_specified: Dictionary tracking which options user explicitly provided
        
    Raises:
        ValueError: If user-specified options conflict with method requirements
        
    Example:
        >>> args = get_classical_args()
        >>> user_specified = {'cat_policy': False, 'normalization': True}
        >>> apply_preprocessing_policies(args, 'catboost', user_specified)
        >>> print(args.cat_policy)  # Will be 'indices' (CatBoost requirement)
    """
    
    # Step 1: Apply defaults for missing values
    # =========================================================================
    for attr in ['cat_policy', 'num_policy', 'normalization', 'num_nan_policy', 'cat_nan_policy']:
        if _is_missing(getattr(args, attr, None)):
            default_value = getattr(DEFAULT_PREPROCESSING, attr)
            setattr(args, attr, default_value)

    # Step 2: Apply method-specific categorical encoding requirements
    # =========================================================================
    required_cat_policy = _determine_required_cat_policy(method)
    
    if required_cat_policy:
        if user_specified.get('cat_policy', False):
            # User specified a policy - validate it matches requirement
            if args.cat_policy != required_cat_policy:
                raise ValueError(
                    f"{method} requires cat_policy='{required_cat_policy}' "
                    f"but got '{args.cat_policy}'"
                )
        else:
            # Apply required policy
            args.cat_policy = required_cat_policy
    
    elif method in FORBIDS_CAT_INDICES:
        # Method cannot handle 'indices' - enforce ordinal if indices was set
        if user_specified.get('cat_policy', False):
            if args.cat_policy == 'indices':
                raise ValueError(f"{method} does not support cat_policy='indices'")
        else:
            if args.cat_policy == 'indices':
                args.cat_policy = 'ordinal'

    # Step 3: Apply normalization requirements
    # =========================================================================
    _apply_normalization_requirements(args, method, user_specified)

    # Step 4: Apply numerical encoding requirements
    # =========================================================================
    _apply_num_encoding_requirements(args, method, user_specified)


def _determine_required_cat_policy(method: MethodName) -> ty.Optional[str]:
    """
    Determine the required categorical encoding policy for a method.
    
    Args:
        method: Method name
        
    Returns:
        Required cat_policy string, or None if no specific requirement
    """
    if method in TABPFN_VARIANTS or method in REQUIRES_CAT_INDICES:
        return 'indices'
    elif method in REQUIRES_CAT_TABR_OHE:
        return 'tabr_ohe'
    elif method in REQUIRES_CAT_OHE:
        return 'ohe'
    else:
        return None


def _apply_normalization_requirements(
    args: ty.Any,
    method: MethodName,
    user_specified: ty.Dict[str, bool]
) -> None:
    """Apply method-specific normalization requirements."""
    
    if method in REQUIRES_NO_NORMALIZATION:
        if user_specified.get('normalization', False):
            if args.normalization != 'none':
                raise ValueError(
                    f"{method} requires normalization='none' "
                    f"but got '{args.normalization}'"
                )
        else:
            args.normalization = 'none'
    
    elif method in REQUIRES_STANDARD_NORMALIZATION:
        if user_specified.get('normalization', False):
            if args.normalization != 'standard':
                raise ValueError(
                    f"{method} requires normalization='standard' "
                    f"but got '{args.normalization}'"
                )
        else:
            args.normalization = 'standard'


def _apply_num_encoding_requirements(
    args: ty.Any,
    method: MethodName,
    user_specified: ty.Dict[str, bool]
) -> None:
    """Apply method-specific numerical encoding requirements."""
    
    # Check if method requires no numerical encoding
    if method in REQUIRES_NO_NUM_ENCODING or method in REQUIRES_CAT_TABR_OHE:
        if user_specified.get('num_policy', False):
            if args.num_policy != 'none':
                raise ValueError(
                    f"{method} requires num_policy='none' "
                    f"but got '{args.num_policy}'"
                )
        else:
            args.num_policy = 'none'


def apply_method_row_limit(
    method: MethodName,
    row_limit: ty.Optional[int]
) -> ty.Optional[int]:
    """
    Apply method-specific row limits for methods with dataset size constraints.
    
    Some methods have architectural limitations on the number of rows:
    - TabPFN: Max 10,000 rows (in-context learning limitation)
    - TabPFN v2: Max 50,000 rows (larger context window)
    
    The function respects both user preferences and method constraints:
    - If user specifies a smaller limit, it's preserved
    - If user specifies a larger limit, it's capped to method maximum
    - If no limit specified, method maximum is applied
    
    Args:
        method: TALENT method name
        row_limit: User-specified row limit (None = no limit)
        
    Returns:
        Effective row limit respecting both user and method constraints
        
    Example:
        >>> apply_method_row_limit('tabpfn', 15000)  # User wants 15k
        10000  # Capped to TabPFN's 10k limit
        >>> apply_method_row_limit('tabpfn', 5000)   # User wants 5k
        5000   # Preserved (smaller than limit)
        >>> apply_method_row_limit('xgboost', None)  # No limit needed
        None   # XGBoost has no row limit
    """
    if method not in METHOD_ROW_LIMITS:
        return row_limit
    
    method_max = METHOD_ROW_LIMITS[method]
    
    if row_limit is None:
        return method_max
    else:
        return min(row_limit, method_max)


# ======================================================================================
#                    SECTION 5: VALIDATION & SANITY CHECKS
# ======================================================================================

def validate_configuration() -> None:
    """
    Validate that the configuration is internally consistent.
    
    Checks:
    - All methods are categorized (deep vs classical)
    - Logit and probability methods don't overlap
    - All methods have output type defined
    - Required preprocessing sets are disjoint where expected
    
    Raises:
        AssertionError: If configuration is inconsistent
    """
    # Check all methods are categorized
    all_methods = DEEP_METHODS | CLASSICAL_METHODS
    
    # Check logit vs probability methods cover all methods
    output_methods = LOGIT_METHODS | PROBABILITY_METHODS
    uncategorized = all_methods - output_methods
    assert not uncategorized, f"Methods without output type: {uncategorized}"
    
    # Check no overlap between logit and probability
    overlap_output = LOGIT_METHODS & PROBABILITY_METHODS
    assert not overlap_output, f"Methods in both LOGIT and PROBABILITY: {overlap_output}"
    
    # TabPFN variants should be in probability methods
    assert TABPFN_VARIANTS.issubset(PROBABILITY_METHODS), \
        "TabPFN variants should return probabilities"
    
    # Check preprocessing requirements don't conflict
    indices_vs_ohe = REQUIRES_CAT_INDICES & REQUIRES_CAT_OHE
    assert not indices_vs_ohe, f"Methods requiring both indices and OHE: {indices_vs_ohe}"
    
    print("✓ Configuration validation passed")


# Run validation when module is imported (only in development)
if __name__ == "__main__":
    validate_configuration()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("METHOD CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Total methods: {len(DEEP_METHODS | CLASSICAL_METHODS)}")
    print(f"  Deep learning: {len(DEEP_METHODS)}")
    print(f"  Classical: {len(CLASSICAL_METHODS)}")
    print(f"  No-HPO methods: {len(NO_HPO_METHODS)}")
    print(f"\nOutput formats:")
    print(f"  Logit methods: {len(LOGIT_METHODS)}")
    print(f"  Probability methods: {len(PROBABILITY_METHODS)}")
    print(f"\nPreprocessing requirements:")
    print(f"  Require cat_indices: {len(REQUIRES_CAT_INDICES)}")
    print(f"  Require tabr_ohe: {len(REQUIRES_CAT_TABR_OHE)}")
    print(f"  Forbid indices: {len(FORBIDS_CAT_INDICES)}")
    print(f"  Row limits: {len(METHOD_ROW_LIMITS)}")
    print("="*70)