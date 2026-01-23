"""
TALENT Method Configuration and Preprocessing Requirements

This module provides a centralized configuration system for all TALENT methods,
defining their categorization, preprocessing policies, and architectural requirements.

Organization:
    1. Method Categorization (Hardware & Architecture)
    2. Output Format Requirements (logits vs probabilities)
    3. Preprocessing Requirements (categorical, numerical, normalization)
    4. Helper Functions (policy application logic)
    5. Validation & Sanity Checks

Usage:
    from src.methods.method_config import NO_HPO_METHODS, DEEP_METHODS, GPU_METHODS
    from src.methods.method_config import apply_preprocessing_policies
"""

from __future__ import annotations
import typing as ty
from dataclasses import dataclass

# Type aliases for clarity
MethodName = str
MethodSet = ty.Set[MethodName]

# ======================================================================================
#                    SECTION 1: METHOD CATEGORIZATION
# ======================================================================================

# -----------------------------------------------------------------------------
# Hardware Execution Categorization
# -----------------------------------------------------------------------------

# GPU Methods - Neural architectures + tree boosting (run on GPU nodes)
GPU_METHODS: MethodSet = {
    # Basic neural architectures
    'mlp', 'resnet',
    
    # Attention-based transformers
    'ftt', 'saint', 'tabtransformer', 'trompt',
    
    # Specialized deep learning
    'tabnet', 'node', 'tabr', 'grownet',
    
    # Advanced architectures
    'autoint', 'snn', 'danets', 'tabcaps', 'dcn2',
    'tangos', 'ptarl', 'switchtab', 'dnnr',
    
    # Modern architectures
    'modernNCA', 'hyperfast', 'bishop', 'realmlp',
    'protogate', 'mlp_plr', 'excelformer', 'grande',
    'amformer', 'tabm', 't2gformer', 'tabautopnpnet',
    'limix',
    
    # Foundation models
    'tabpfn', 'tabpfn_v2', 'tabpfn_real', 
    'tabicl', 'tabptm', 'mitra',
    
    # Tree boosting (run on GPU nodes for speed/compatibility)
    'xgboost', 'catboost', 'lightgbm',
}

# CPU Methods - Only classical ML (excluding tree boosting)
CPU_METHODS: MethodSet = {
    # Traditional ML models
    'RandomForest', 'LogReg', 'LinearRegression',
    'knn', 'svm', 'NaiveBayes', 'NCM',

    # Baseline models
    'dummy',
}

# Foundation Models - Require high-memory GPUs (wICE cluster with H100/A100)
# These use in-context learning and need to load entire datasets into GPU memory
FOUNDATION_METHODS: MethodSet = {
    'tabpfn',        # TabPFN v1
    'tabpfn_v2',     # TabPFN v2
    'tabpfn_real',   # TabPFN v2.5 "Real" mode
    'mitra',         # Mitra foundation model
    'tabicl',        # TabICL (In-Context Learning)
    'tabptm',        # TabPTM
}


# -----------------------------------------------------------------------------
# Architectural Categorization
# -----------------------------------------------------------------------------

# Deep learning methods - Neural network architectures
DEEP_METHODS: MethodSet = {
    # Basic neural architectures
    'mlp', 'resnet',
    
    # Attention-based transformers
    'ftt', 'saint', 'tabtransformer', 'trompt',
    
    # Specialized deep learning
    'tabnet', 'node', 'tabr', 'grownet',
    
    # Advanced architectures
    'autoint', 'snn', 'danets', 'tabcaps', 'dcn2',
    'tangos', 'ptarl', 'switchtab', 'dnnr',
    
    # Modern architectures
    'modernNCA', 'hyperfast', 'bishop', 'realmlp',
    'protogate', 'mlp_plr', 'excelformer', 'grande',
    'amformer', 'tabm', 't2gformer', 'tabautopnpnet', 'limix', 
    
    # Foundation models
    'tabpfn', 'tabpfn_v2', 'tabpfn_real', 'tabicl', 'tabptm', 'mitra'
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

# -----------------------------------------------------------------------------
# Optimization Categorization
# -----------------------------------------------------------------------------

# Methods that don't benefit from hyperparameter optimization
NO_HPO_METHODS: MethodSet = {
    # Foundation models (pre-trained, no tuning needed)
    'tabpfn', 'tabpfn_v2', 'tabpfn_real', 'tabicl', 'tabptm', 'mitra','hyperfast',
    
    # Simple baselines (no hyperparameters or already optimal)
    'dummy', 'NCM', 'NaiveBayes', 'LinearRegression',
}


# ======================================================================================
#                    SECTION 2: OUTPUT FORMAT REQUIREMENTS
# ======================================================================================

# Methods that return raw logits (need softmax for probabilities)
LOGIT_METHODS: MethodSet = {
    # -----------------------------
    # Basic Neural Architectures
    # -----------------------------
    'mlp',           # Multilayer Perceptron
    'resnet',        # Residual Network
    'node',          # Neural Oblivious Decision Ensembles
    'snn',           # Self-Normalizing Neural Networks
    'realmlp',       # RealMLP
    'mlp_plr',       # MLP with Periodic Linear Embeddings

    # -----------------------------
    # Transformer-based Architectures
    # -----------------------------
    'ftt',           # Feature Tokenizer Transformer (FT-Transformer)
    'tabtransformer',# TabTransformer
    'saint',         # SAINT
    'autoint',       # AutoInt
    'excelformer',   # ExcelFormer
    'amformer',      # AMFormer
    't2gformer',     # T2G-Former
    'trompt',        # TROMPT
    'ptarl',         # PTARL (Tabular Representation Learning)

    # -----------------------------
    # Deep Tabular Specialists
    # -----------------------------
    'tabnet',        # TabNet
    'danets',        # Deep Abstract Networks
    'dcn2',          # Deep & Cross Network v2
    'switchtab',     # SwitchTab
    'grownet',       # Gradient Boosting Neural Networks (GrowNet)
    'tabcaps',       # TabCaps
    'tangos',        # TANGOS (Regularization)
    'tabr',          # TabR
    'tabm',          # TabM (Tabular Machines)

    # -----------------------------
    # Advanced / Regularized / Novel
    # -----------------------------
    'dnnr',          # Deep Neural Network Regression
    'protogate',     # ProtoGate
    'bishop',        # BISHOP
    'modernNCA',     # Modern Neighborhood Components Analysis
    'grande',        # GRANDE
    'tabautopnpnet', # TabAuto-PNPNet
    'limix',         # LiMiX (Mixup-based)

    # -----------------------------
    # Foundation Models (Returning Logits)
    # -----------------------------
    'mitra',         # Mitra (Outputs raw logits requiring Softmax)
    'hyperfast',     # HyperFast (Meta-learning model outputs logits)
    'tabptm',        # TabPTM (Pre-trained Model outputs logits)
    'tabicl',        # TabICL (In-Context Learning outputs logits)
    'tabpfn_real',   # TabPFN (Real version, often wraps logits unlike v1/v2)
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
    't2gformer', 'tabm', 'tabtransformer', 'trompt', 'tabicl', 'limix', 'mitra',
    
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
    'hyperfast', 'tabicl', 'limix',
    # TabPFN variants
    'tabpfn', 'tabpfn_v2', 'tabpfn_real', 'mitra',
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
    'hyperfast', 'modernNCA', 'tabicl', 'tabptm', 'tabr', 'mitra',
    # TabPFN variants
    'tabpfn', 'tabpfn_v2', 'tabpfn_real',
}


# -----------------------------------------------------------------------------
# Dataset Size Constraints
# -----------------------------------------------------------------------------

# Methods with inherent architectural row limits (applied to TRAINING set)
METHOD_ROW_LIMITS = {
    # --- Foundation Models (In-Context Learning) ---
    'tabpfn': 5000,       # Original v1 (Context ~2-4k tokens)
    'tabpfn_v2': 30_000,    # Standard v2 base model (Safe limit)
    'tabpfn_real': 30_000,  # v2.5 "Real" mode (Explicitly designed for 50k)

    'mitra': 5_000,        # Standard Transformer context window
    'tabicl': 50_000,       # Uses retrieval/kernel-attention for larger context

    # --- Deep Learning (Complexity Constraints) ---
    'amformer': 100_000,    # VSC Memory constraint (O(N^2) attention)
    'tangos': 100_000,      # VSC Memory constraint
}

# Limits for Validation and Test sets to prevent OOM during inference
# Foundation models use in-context learning and need to load entire datasets
# into GPU memory, causing OOM on large validation/test sets.
# Mitra is excluded because it handles larger inference sets differently.
METHOD_TEST_VAL_LIMITS = {
    'tabpfn': 2048,        # TabPFN v1: Limited context window
    'tabpfn_v2': 4096,     # TabPFN v2: Larger but still constrained
    'tabpfn_real': 4096,   # TabPFN v2.5 "Real" mode
    'tabicl': 2048,        # TabICL: In-context learning constraint
}


# ======================================================================================
#                    SECTION 4: VALIDATION & HELPER FUNCTIONS
# ======================================================================================

# Sentinel values indicating "missing" or "not specified"
_MISSING_SENTINELS: ty.Set[ty.Any] = {None, "", "nothing", "Nothing", "NONE", "None"}


def _is_missing(value: ty.Any) -> bool:
    """
    Check if a value represents "missing" or "not specified".
    """
    try:
        return value in _MISSING_SENTINELS
    except TypeError:
        return False


def apply_preprocessing_policies(
    args: ty.Any,
    method: MethodName,
    user_specified: ty.Dict[str, bool]
) -> None:
    """
    Apply preprocessing policy defaults and method-specific requirements.
    This function modifies the args namespace in-place.
    """
    
    # Step 1: Apply defaults for missing values
    for attr in ['cat_policy', 'num_policy', 'normalization', 'num_nan_policy', 'cat_nan_policy']:
        if _is_missing(getattr(args, attr, None)):
            default_value = getattr(DEFAULT_PREPROCESSING, attr)
            setattr(args, attr, default_value)

    # Step 2: Apply method-specific categorical encoding requirements
    required_cat_policy = _determine_required_cat_policy(method)
    
    if required_cat_policy:
        if user_specified.get('cat_policy', False):
            if args.cat_policy != required_cat_policy:
                raise ValueError(
                    f"{method} requires cat_policy='{required_cat_policy}' "
                    f"but got '{args.cat_policy}'"
                )
        else:
            args.cat_policy = required_cat_policy
    
    elif method in FORBIDS_CAT_INDICES:
        if user_specified.get('cat_policy', False):
            if args.cat_policy == 'indices':
                raise ValueError(f"{method} does not support cat_policy='indices'")
        else:
            if args.cat_policy == 'indices':
                args.cat_policy = 'ordinal'

    # Step 3: Apply normalization requirements
    _apply_normalization_requirements(args, method, user_specified)

    # Step 4: Apply numerical encoding requirements
    _apply_num_encoding_requirements(args, method, user_specified)


def _determine_required_cat_policy(method: MethodName) -> ty.Optional[str]:
    """Determine the required categorical encoding policy for a method."""
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
    Returns effective row limit respecting both user and method constraints.
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
    """
    # Architecture Sets
    all_methods = DEEP_METHODS | CLASSICAL_METHODS
    
    # Hardware Sets
    all_hardware_methods = GPU_METHODS | CPU_METHODS
    
    # 1. Check Hardware <-> Architecture consistency
    # (Note: GPU set includes some classical methods like XGBoost, 
    #  so we check that the Union of Hardware == Union of Architecture)
    missing_arch = all_hardware_methods - all_methods
    missing_hw = all_methods - all_hardware_methods
    assert not missing_arch, f"Methods in GPU/CPU but not in DEEP/CLASSICAL: {missing_arch}"
    assert not missing_hw, f"Methods in DEEP/CLASSICAL but not in GPU/CPU: {missing_hw}"
    
    # 2. Check Hardware Disjointness
    overlap_hw = GPU_METHODS & CPU_METHODS
    assert not overlap_hw, f"Methods in both GPU and CPU lists: {overlap_hw}"

    # 3. Check Output Formats
    output_methods = LOGIT_METHODS | PROBABILITY_METHODS
    uncategorized = all_methods - output_methods
    assert not uncategorized, f"Methods without output type: {uncategorized}"
    
    overlap_output = LOGIT_METHODS & PROBABILITY_METHODS
    assert not overlap_output, f"Methods in both LOGIT and PROBABILITY: {overlap_output}"
    
    # 4. Check TabPFN specific
    assert TABPFN_VARIANTS.issubset(PROBABILITY_METHODS), \
        "TabPFN variants should return probabilities"
    
    # 5. Check Preprocessing consistency
    indices_vs_ohe = REQUIRES_CAT_INDICES & REQUIRES_CAT_OHE
    assert not indices_vs_ohe, f"Methods requiring both indices and OHE: {indices_vs_ohe}"
    
    print("✓ Configuration validation passed")


if __name__ == "__main__":
    validate_configuration()
    
    print("\n" + "="*70)
    print("METHOD CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Total methods: {len(DEEP_METHODS | CLASSICAL_METHODS)}")
    print(f"  GPU Methods: {len(GPU_METHODS)}")
    print(f"  CPU Methods: {len(CPU_METHODS)}")
    print(f"  Deep learning: {len(DEEP_METHODS)}")
    print(f"  Classical: {len(CLASSICAL_METHODS)}")
    print(f"  No-HPO methods: {len(NO_HPO_METHODS)}")
    print(f"\nOutput formats:")
    print(f"  Logit methods: {len(LOGIT_METHODS)}")
    print(f"  Probability methods: {len(PROBABILITY_METHODS)}")
    print("="*70)