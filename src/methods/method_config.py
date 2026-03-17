"""
TALENT Method Configuration and Preprocessing Requirements

This module provides a centralized configuration system for all TALENT methods,
using a single-source-of-truth registry pattern. Each method is defined exactly
once with all its properties, eliminating the possibility of inconsistencies
between separate sets.

Organization:
    1. Method Registry (single dataclass per method)
    2. Derived Sets (computed from registry, never manually maintained)
    3. Preprocessing Policy Application
    4. Validation (runs at import time)

Usage:
    from src.methods.method_config import METHOD_REGISTRY, DEEP_METHODS, GPU_METHODS
    from src.methods.method_config import apply_preprocessing_policies

Design decisions documented:
    - Categorical encoding: Categories in credit risk datasets are NEVER ordinal.
      Therefore, ordinal encoding is never used as a default.
      Methods that support 'indices' (embedding-based) use indices.
      Methods that forbid 'indices' use 'ohe' (one-hot encoding).
      Methods with specialised handling (TabR family) use 'tabr_ohe'.

    - Output types verified against TALENT source code (updated version):
      * Methods using predict_proba() → PROBABILITIES
      * Methods using model(X) raw forward pass → LOGITS
      * SVM uses CalibratedClassifierCV wrapping LinearSVC → predict_proba() → PROBABILITIES
      * NCM has custom _predict_proba() (softmax over negative distances) → PROBABILITIES
      * NaiveBayes uses GaussianNB.predict_proba() → PROBABILITIES
      * Dummy uses model.predict_proba() → PROBABILITIES
      * RealMLP classifier's predict() modified to return predict_proba() → PROBABILITIES
      * LinearRegression is regression-only → CLASS_LABELS (continuous predictions)

    - TALENT constraints verified from assert statements in each method class:
      * assert(args.cat_policy == 'indices') → must use indices
      * assert(args.cat_policy != 'indices') → must NOT use indices
      * assert(args.cat_policy == 'ohe') → must use ohe
      * assert(args.cat_policy == 'tabr_ohe') → must use tabr_ohe
"""

from __future__ import annotations
import typing as ty
from dataclasses import dataclass
from enum import Enum

# Type aliases
MethodName = str
MethodSet = ty.Set[MethodName]


# ======================================================================================
#                    SECTION 1: ENUMERATIONS
# ======================================================================================

class OutputType(Enum):
    """What a method's predict() returns for classification tasks.

    Verified by reading each method's predict() in TALENT source code.
    """
    LOGITS = "logits"              # Raw network output; needs softmax/sigmoid
    PROBABILITIES = "probabilities" # predict_proba() output; already [0,1] summing to 1
    CLASS_LABELS = "class_labels"   # Hard predictions (0/1); no probability available


class Hardware(Enum):
    """Where the method runs."""
    GPU = "gpu"
    CPU = "cpu"


class Architecture(Enum):
    """High-level architecture category."""
    DEEP = "deep"
    CLASSICAL = "classical"


class CatPolicy(Enum):
    """Categorical encoding policy.

    Maps directly to TALENT's cat_policy argument. The value is the exact
    string TALENT expects.
    """
    INDICES = "indices"      # Keep as integer codes for embedding layers
    OHE = "ohe"              # Standard one-hot encoding, concatenated into N
    TABR_OHE = "tabr_ohe"   # TabR-specific OHE (separate N and C arrays)


# ======================================================================================
#                    SECTION 2: METHOD REGISTRY
# ======================================================================================

@dataclass(frozen=True)
class MethodSpec:
    """Immutable specification for a single TALENT method.

    This is the SINGLE SOURCE OF TRUTH for each method's configuration.
    All derived sets (GPU_METHODS, LOGIT_METHODS, etc.) are computed from
    this registry, so they can never go out of sync.

    Attributes:
        name:              Canonical TALENT method name (case-sensitive)
        hardware:          GPU or CPU execution
        architecture:      Deep learning or classical ML
        output_type:       What predict() returns (logits, probabilities, or class labels)
        cat_policy:        Required categorical encoding policy
        normalization:     Required normalization ('none', 'standard', or None for default)
        num_policy:        Required numerical encoding ('none' or None for default)
        train_row_limit:   Max training rows (None = no limit)
        test_val_limit:    Max val/test rows for inference (None = no limit)
        supports_hpo:      Whether HPO is meaningful for this method
        notes:             Human-readable notes about design decisions
    """
    name: str
    hardware: Hardware
    architecture: Architecture
    output_type: OutputType
    cat_policy: CatPolicy
    normalization: ty.Optional[str] = None      # None = use default (standard)
    num_policy: ty.Optional[str] = None         # None = use default
    train_row_limit: ty.Optional[int] = None
    test_val_limit: ty.Optional[int] = None
    supports_hpo: bool = True
    notes: str = ""


# -------------------------------------------------------------------------------------
# Registry: Every method defined exactly once.
#
# cat_policy rationale:
#   - INDICES: TALENT source has `assert(args.cat_policy == 'indices')`.
#     Used by methods with learned embedding layers for categorical features.
#   - OHE: TALENT source has `assert(args.cat_policy != 'indices')`.
#     Since our categorical features are NEVER ordinal, we use OHE instead
#     of ordinal encoding to avoid imposing false ordinal relationships.
#   - TABR_OHE: TALENT source has `assert(args.cat_policy == 'tabr_ohe')`.
#     Specialized OHE that keeps N and C arrays separate.
#
# output_type rationale:
#   Verified by reading each method's predict() in:
#     TALENT/model/methods/*.py
#     TALENT/model/classical_methods/*.py
#   See notes field for specifics.
# -------------------------------------------------------------------------------------

_REGISTRY_LIST: ty.List[MethodSpec] = [
    # ==================================================================================
    # FOUNDATION MODELS
    # ==================================================================================
    MethodSpec(
        name="tabpfn",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.INDICES,
        normalization="none",
        num_policy="none",
        train_row_limit=5_000,
        test_val_limit=50_000,
        supports_hpo=False,
        notes="TabPFN v1. predict_proba() returns calibrated probabilities.",
    ),
    MethodSpec(
        name="tabpfn_v2",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.INDICES,
        normalization="none",
        num_policy="none",
        train_row_limit=30_000,
        test_val_limit=50_000,
        supports_hpo=False,
        notes="TabPFN v2 (PFN_v2.py). predict_proba() returns probabilities.",
    ),
    MethodSpec(
        name="tabpfn_real",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.INDICES,
        normalization="none",
        num_policy="none",
        train_row_limit=30_000,
        test_val_limit=50_000,
        supports_hpo=False,
        notes="TabPFN v2.5 'Real' mode. Same PFN_v2.py class, predict_proba().",
    ),
    MethodSpec(
        name="mitra",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        normalization="none",
        num_policy="none",
        train_row_limit=5_000,
        test_val_limit=5_000,
        supports_hpo=False,
        notes="Mitra foundation model. Has custom predict() that returns raw model output, "
              "not predict_proba(). Needs softmax.",
    ),
    MethodSpec(
        name="tabicl",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.INDICES,
        normalization="none",
        num_policy="none",
        train_row_limit=30_000,
        test_val_limit=30_000,
        supports_hpo=False,
        notes="TabICL. predict_proba() in tabicl.py returns probabilities.",
    ),
    MethodSpec(
        name="tabptm",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.OHE,
        normalization="standard",
        num_policy="none",
        supports_hpo=False,
        notes="TabPTM. assert(cat_policy == 'ohe'). model(X_meta) returns raw logits.",
    ),
    MethodSpec(
        name="hyperfast",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.INDICES,
        normalization="none",
        num_policy="none",
        supports_hpo=False,
        notes="HyperFast. assert(cat_policy == 'indices'). predict_proba() returns probs.",
    ),

    # ==================================================================================
    # BASIC NEURAL ARCHITECTURES
    # ==================================================================================
    MethodSpec(
        name="mlp",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.OHE,
        notes="MLP. assert(cat_policy != 'indices'). Uses base Method.predict() → model(X) logits.",
    ),
    MethodSpec(
        name="resnet",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.OHE,
        notes="ResNet. assert(cat_policy != 'indices'). Uses base Method.predict() → logits.",
    ),
    MethodSpec(
        name="snn",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="Self-Normalizing NN. assert(cat_policy == 'indices'). Base predict() → logits.",
    ),
    MethodSpec(
        name="realmlp",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.INDICES,
        notes="RealMLP. assert(cat_policy == 'indices'). Classifier's predict() modified to "
              "return predict_proba() output (calibrated probabilities).",
    ),
    MethodSpec(
        name="mlp_plr",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.TABR_OHE,
        num_policy="none",
        notes="MLP with PLR embeddings. assert(cat_policy == 'tabr_ohe'). Base predict() → logits.",
    ),

    # ==================================================================================
    # TRANSFORMER-BASED ARCHITECTURES
    # ==================================================================================
    MethodSpec(
        name="ftt",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="FT-Transformer. assert(cat_policy == 'indices'). Base predict() → logits.",
    ),
    MethodSpec(
        name="saint",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="SAINT. assert(cat_policy == 'indices'). Base predict() → logits.",
    ),
    MethodSpec(
        name="tabtransformer",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="TabTransformer. assert(cat_policy == 'indices'). Base predict() → logits.",
    ),
    MethodSpec(
        name="trompt",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="TROMPT. assert(cat_policy == 'indices'). Has own predict() but returns model(X) logits.",
    ),
    MethodSpec(
        name="autoint",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="AutoInt. assert(cat_policy == 'indices'). Base predict() → logits.",
    ),
    MethodSpec(
        name="excelformer",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.OHE,
        notes="ExcelFormer. assert(cat_policy != 'indices'). Has own predict() with model() → logits.",
    ),
    MethodSpec(
        name="amformer",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        train_row_limit=100_000,
        notes="AMFormer. assert(cat_policy == 'indices'). O(N^2) attention. Base predict() → logits.",
    ),
    MethodSpec(
        name="t2gformer",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="T2G-Former. assert(cat_policy == 'indices'). Own predict() but model(X) → logits.",
    ),

    # ==================================================================================
    # DEEP TABULAR SPECIALISTS
    # ==================================================================================
    MethodSpec(
        name="tabnet",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.OHE,
        notes="TabNet. assert(cat_policy != 'indices'). predict_proba() returns probabilities.",
    ),
    MethodSpec(
        name="node",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.OHE,
        notes="NODE. assert(cat_policy != 'indices'). Base predict() → logits.",
    ),
    MethodSpec(
        name="tabr",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.TABR_OHE,
        num_policy="none",
        notes="TabR. assert(cat_policy == 'tabr_ohe'). Own predict() but model() → logits.",
    ),
    MethodSpec(
        name="grownet",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="GrowNet. assert(cat_policy == 'indices'). Own predict(): model.forward() → logits.",
    ),
    MethodSpec(
        name="danets",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.OHE,
        notes="DANets. assert(cat_policy != 'indices'). Base predict() → logits.",
    ),
    MethodSpec(
        name="tabcaps",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.OHE,
        notes="TabCaps. assert(cat_policy != 'indices'). model.predict() returns logits "
              "(not sklearn predict_proba).",
    ),
    MethodSpec(
        name="dcn2",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="DCN v2. assert(cat_policy == 'indices'). Base predict() → logits.",
    ),
    MethodSpec(
        name="tangos",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.OHE,
        train_row_limit=100_000,
        notes="TANGOS. assert(cat_policy != 'indices'). Base predict() → logits.",
    ),
    MethodSpec(
        name="ptarl",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="PTARL. assert(cat_policy == 'indices'). Own predict() but uses test() → logits.",
    ),
    MethodSpec(
        name="switchtab",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.OHE,
        notes="SwitchTab. assert(cat_policy != 'indices'). assert(C is None) after encoding. "
              "Base predict() → logits.",
    ),
    MethodSpec(
        name="dnnr",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.OHE,
        notes="DNNR. assert(cat_policy != 'indices'). model.predict(X) returns raw output.",
    ),

    # ==================================================================================
    # ADVANCED / REGULARIZED / NOVEL
    # ==================================================================================
    MethodSpec(
        name="modernNCA",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.TABR_OHE,
        num_policy="none",
        notes="ModernNCA. assert(cat_policy == 'tabr_ohe'). Own predict() but model() → logits.",
    ),
    MethodSpec(
        name="bishop",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="BISHOP. assert(cat_policy == 'indices'). Base predict() → logits.",
    ),
    MethodSpec(
        name="protogate",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.OHE,
        notes="ProtoGate. assert(cat_policy != 'indices'). Own predict() but proto_predict() → logits.",
    ),
    MethodSpec(
        name="grande",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="GRANDE. assert(cat_policy == 'indices'). Own predict() but model(X) → logits.",
    ),
    MethodSpec(
        name="tabautopnpnet",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.TABR_OHE,
        num_policy="none",
        notes="TabAutoPNPNet. assert(cat_policy == 'tabr_ohe'). Base predict() → logits.",
    ),
    MethodSpec(
        name="tabm",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        notes="TabM. assert(cat_policy == 'indices'). model(X).mean(1) → logits.",
    ),
    MethodSpec(
        name="limix",
        hardware=Hardware.GPU,
        architecture=Architecture.DEEP,
        output_type=OutputType.LOGITS,
        cat_policy=CatPolicy.INDICES,
        normalization="none",
        notes="LiMiX. Mixup-based. Uses indices. No normalization. Base predict() → logits.",
    ),

    # ==================================================================================
    # TREE-BASED GRADIENT BOOSTING (Classical, but run on GPU nodes)
    # ==================================================================================
    MethodSpec(
        name="xgboost",
        hardware=Hardware.GPU,
        architecture=Architecture.CLASSICAL,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.OHE,
        notes="XGBoost. assert(cat_policy != 'indices') in TALENT. predict_proba() for classification. "
              "OHE used because features are not ordinal.",
    ),
    MethodSpec(
        name="catboost",
        hardware=Hardware.GPU,
        architecture=Architecture.CLASSICAL,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.INDICES,
        notes="CatBoost. assert(cat_policy == 'indices'). Has native categorical support. "
              "predict_proba() for classification.",
    ),
    MethodSpec(
        name="lightgbm",
        hardware=Hardware.GPU,
        architecture=Architecture.CLASSICAL,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.OHE,
        notes="LightGBM. Inherits XGBoost in TALENT: assert(cat_policy != 'indices'). "
              "predict_proba() via XGBoost parent. OHE because features are not ordinal.",
    ),

    # ==================================================================================
    # CLASSICAL ML (CPU only)
    # ==================================================================================
    MethodSpec(
        name="RandomForest",
        hardware=Hardware.CPU,
        architecture=Architecture.CLASSICAL,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.OHE,
        notes="Inherits KNN in TALENT: assert(cat_policy != 'indices'). predict_proba(). "
              "OHE because features are not ordinal (trees could handle ordinal but we "
              "standardise across all classical methods for fairness).",
    ),
    MethodSpec(
        name="LogReg",
        hardware=Hardware.CPU,
        architecture=Architecture.CLASSICAL,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.OHE,
        notes="LogisticRegression. assert(cat_policy != 'indices'). predict_proba(). "
              "OHE required: ordinal encoding would impose false linear relationships.",
    ),
    MethodSpec(
        name="LinearRegression",
        hardware=Hardware.CPU,
        architecture=Architecture.CLASSICAL,
        output_type=OutputType.CLASS_LABELS,
        cat_policy=CatPolicy.OHE,
        supports_hpo=False,
        notes="LinearRegression (regression only). assert(cat_policy != 'indices'). "
              "model.predict() returns continuous values. OHE required.",
    ),
    MethodSpec(
        name="knn",
        hardware=Hardware.CPU,
        architecture=Architecture.CLASSICAL,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.OHE,
        notes="KNeighbors. assert(cat_policy != 'indices'). predict_proba() for classification. "
              "OHE required: ordinal encoding distorts distance metrics.",
    ),
    MethodSpec(
        name="svm",
        hardware=Hardware.CPU,
        architecture=Architecture.CLASSICAL,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.OHE,
        notes="LinearSVC wrapped in CalibratedClassifierCV(method='sigmoid', cv=3) for "
              "classification → predict_proba(). LinearSVR for regression. "
              "assert(cat_policy != 'indices'). OHE required.",
    ),
    MethodSpec(
        name="NaiveBayes",
        hardware=Hardware.CPU,
        architecture=Architecture.CLASSICAL,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.OHE,
        supports_hpo=False,
        notes="GaussianNB. Inherits NCM → assert(cat_policy != 'indices'). "
              "Uses predict_proba() for classification. OHE required.",
    ),
    MethodSpec(
        name="NCM",
        hardware=Hardware.CPU,
        architecture=Architecture.CLASSICAL,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.OHE,
        supports_hpo=False,
        notes="NearestCentroid. assert(cat_policy != 'indices'). Custom _predict_proba() "
              "computes softmax over negative Euclidean distances to class centroids. OHE required.",
    ),
    MethodSpec(
        name="dummy",
        hardware=Hardware.CPU,
        architecture=Architecture.CLASSICAL,
        output_type=OutputType.PROBABILITIES,
        cat_policy=CatPolicy.OHE,
        supports_hpo=False,
        notes="DummyClassifier/Regressor. assert(cat_policy != 'indices'). "
              "Uses predict_proba() for classification. Baseline only.",
    ),
]


# Build the registry as an immutable lookup dict
METHOD_REGISTRY: ty.Dict[str, MethodSpec] = {spec.name: spec for spec in _REGISTRY_LIST}

# Verify no duplicate names
assert len(METHOD_REGISTRY) == len(_REGISTRY_LIST), (
    f"Duplicate method names in registry! "
    f"Expected {len(_REGISTRY_LIST)}, got {len(METHOD_REGISTRY)}"
)


# ======================================================================================
#                    SECTION 3: DERIVED SETS (computed from registry)
# ======================================================================================
# These are NEVER manually maintained. They are computed from METHOD_REGISTRY.

# Hardware categorization
GPU_METHODS: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.hardware == Hardware.GPU
}
CPU_METHODS: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.hardware == Hardware.CPU
}

# Architecture categorization
DEEP_METHODS: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.architecture == Architecture.DEEP
}
CLASSICAL_METHODS: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.architecture == Architecture.CLASSICAL
}

# Output type categorization
LOGIT_METHODS: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.output_type == OutputType.LOGITS
}
PROBABILITY_METHODS: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.output_type == OutputType.PROBABILITIES
}
CLASS_LABEL_METHODS: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.output_type == OutputType.CLASS_LABELS
}

# Foundation models (pre-trained, limited context)
FOUNDATION_METHODS: MethodSet = {
    'tabpfn', 'tabpfn_v2', 'tabpfn_real', 'mitra', 'tabicl', 'tabptm',
}

# Methods without meaningful HPO
NO_HPO_METHODS: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if not spec.supports_hpo
}

# Categorical encoding requirement sets (for backward compatibility)
REQUIRES_CAT_INDICES: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.cat_policy == CatPolicy.INDICES
}
REQUIRES_CAT_OHE: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.cat_policy == CatPolicy.OHE
}
REQUIRES_CAT_TABR_OHE: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.cat_policy == CatPolicy.TABR_OHE
}

# TabPFN family
TABPFN_VARIANTS: MethodSet = {'tabpfn', 'tabpfn_v2', 'tabpfn_real'}

# Normalization requirements
REQUIRES_NO_NORMALIZATION: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.normalization == "none"
}
REQUIRES_STANDARD_NORMALIZATION: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.normalization == "standard"
}

# Numerical encoding requirements
REQUIRES_NO_NUM_ENCODING: MethodSet = {
    name for name, spec in METHOD_REGISTRY.items()
    if spec.num_policy == "none"
}

# Row limits
METHOD_ROW_LIMITS: ty.Dict[str, int] = {
    name: spec.train_row_limit
    for name, spec in METHOD_REGISTRY.items()
    if spec.train_row_limit is not None
}
METHOD_TEST_VAL_LIMITS: ty.Dict[str, int] = {
    name: spec.test_val_limit
    for name, spec in METHOD_REGISTRY.items()
    if spec.test_val_limit is not None
}

# Legacy compatibility: FORBIDS_CAT_INDICES is now just REQUIRES_CAT_OHE
# (methods that can't use indices must use OHE since we never use ordinal)
FORBIDS_CAT_INDICES: MethodSet = REQUIRES_CAT_OHE


# ======================================================================================
#                    SECTION 4: PREPROCESSING DEFAULTS
# ======================================================================================

@dataclass(frozen=True)
class PreprocessingConfig:
    """Default preprocessing strategies for tabular data."""
    cat_policy: str = 'ohe'              # Changed from 'ordinal' — categories are never ordinal
    num_policy: str = 'none'
    normalization: str = 'standard'
    num_nan_policy: str = 'median'
    cat_nan_policy: str = 'new'


DEFAULT_PREPROCESSING = PreprocessingConfig()


# ======================================================================================
#                    SECTION 5: HELPER FUNCTIONS
# ======================================================================================

# Sentinel values indicating "missing" or "not specified"
_MISSING_SENTINELS: ty.Set[ty.Any] = {None, "", "nothing", "Nothing", "NONE", "None"}


def _is_missing(value: ty.Any) -> bool:
    """Check if a value represents 'missing' or 'not specified'."""
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
    Modifies the args namespace in-place.

    Priority:
    1. Method-specific requirements (from registry) — always enforced
    2. User-specified values — used if compatible with method
    3. Defaults from PreprocessingConfig — fallback
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: '{method}'. Not in METHOD_REGISTRY.")

    spec = METHOD_REGISTRY[method]

    # Step 1: Apply defaults for any missing values
    for attr in ['cat_policy', 'num_policy', 'normalization', 'num_nan_policy', 'cat_nan_policy']:
        if _is_missing(getattr(args, attr, None)):
            default_value = getattr(DEFAULT_PREPROCESSING, attr)
            setattr(args, attr, default_value)

    # Step 2: Enforce method-specific categorical encoding
    required_cat = spec.cat_policy.value
    if user_specified.get('cat_policy', False):
        if args.cat_policy != required_cat:
            raise ValueError(
                f"{method} requires cat_policy='{required_cat}' "
                f"but got '{args.cat_policy}'"
            )
    else:
        args.cat_policy = required_cat

    # Step 3: Enforce normalization if required
    if spec.normalization is not None:
        if user_specified.get('normalization', False):
            if args.normalization != spec.normalization:
                raise ValueError(
                    f"{method} requires normalization='{spec.normalization}' "
                    f"but got '{args.normalization}'"
                )
        else:
            args.normalization = spec.normalization

    # Step 4: Enforce numerical encoding if required
    if spec.num_policy is not None:
        if user_specified.get('num_policy', False):
            if args.num_policy != spec.num_policy:
                raise ValueError(
                    f"{method} requires num_policy='{spec.num_policy}' "
                    f"but got '{args.num_policy}'"
                )
        else:
            args.num_policy = spec.num_policy

    # Also enforce num_policy='none' for tabr_ohe methods
    # (TALENT's tabr_ohe methods skip numerical encoding internally)
    if spec.cat_policy == CatPolicy.TABR_OHE:
        if user_specified.get('num_policy', False):
            if args.num_policy != 'none':
                raise ValueError(
                    f"{method} (tabr_ohe) requires num_policy='none' "
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
#                    SECTION 6: VALIDATION (runs at import time)
# ======================================================================================

def validate_configuration() -> None:
    """
    Validate that the configuration is internally consistent.
    Called at import time to fail fast on any misconfiguration.
    """
    all_methods = DEEP_METHODS | CLASSICAL_METHODS
    all_hardware = GPU_METHODS | CPU_METHODS

    # 1. Every method must appear in exactly one hardware set and one architecture set
    assert all_methods == all_hardware, (
        f"Hardware/Architecture mismatch.\n"
        f"  In hardware but not architecture: {all_hardware - all_methods}\n"
        f"  In architecture but not hardware: {all_methods - all_hardware}"
    )

    # 2. Hardware sets must be disjoint
    overlap_hw = GPU_METHODS & CPU_METHODS
    assert not overlap_hw, f"Methods in both GPU and CPU: {overlap_hw}"

    # 3. Architecture sets must be disjoint
    overlap_arch = DEEP_METHODS & CLASSICAL_METHODS
    assert not overlap_arch, f"Methods in both DEEP and CLASSICAL: {overlap_arch}"

    # 4. Output type sets must be exhaustive and disjoint
    all_output = LOGIT_METHODS | PROBABILITY_METHODS | CLASS_LABEL_METHODS
    assert all_output == all_methods, (
        f"Methods without output type: {all_methods - all_output}"
    )
    assert not (LOGIT_METHODS & PROBABILITY_METHODS), (
        f"Methods in both LOGIT and PROBABILITY: {LOGIT_METHODS & PROBABILITY_METHODS}"
    )
    assert not (LOGIT_METHODS & CLASS_LABEL_METHODS), (
        f"Methods in both LOGIT and CLASS_LABEL: {LOGIT_METHODS & CLASS_LABEL_METHODS}"
    )
    assert not (PROBABILITY_METHODS & CLASS_LABEL_METHODS), (
        f"Methods in both PROBABILITY and CLASS_LABEL: {PROBABILITY_METHODS & CLASS_LABEL_METHODS}"
    )

    # 5. Cat policy sets must be exhaustive
    all_cat = REQUIRES_CAT_INDICES | REQUIRES_CAT_OHE | REQUIRES_CAT_TABR_OHE
    assert all_cat == all_methods, (
        f"Methods without cat_policy: {all_methods - all_cat}"
    )

    # 6. TabPFN variants should return probabilities
    assert TABPFN_VARIANTS.issubset(PROBABILITY_METHODS), (
        f"TabPFN variants not returning probabilities: {TABPFN_VARIANTS - PROBABILITY_METHODS}"
    )

    # 7. No HPO methods must be a subset of all methods
    assert NO_HPO_METHODS.issubset(all_methods), (
        f"NO_HPO methods not in registry: {NO_HPO_METHODS - all_methods}"
    )

    # 8. Row limit methods must be in registry
    for name in METHOD_ROW_LIMITS:
        assert name in METHOD_REGISTRY, f"Row limit method not in registry: {name}"
    for name in METHOD_TEST_VAL_LIMITS:
        assert name in METHOD_REGISTRY, f"Test/val limit method not in registry: {name}"


# Run validation at import time — fail fast on any misconfiguration
validate_configuration()


# ======================================================================================
#                    SECTION 7: DIAGNOSTICS (for __main__ usage)
# ======================================================================================

if __name__ == "__main__":
    print("[OK] Configuration validation passed at import time")

    print(f"\n{'='*70}")
    print("METHOD CONFIGURATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total methods: {len(METHOD_REGISTRY)}")
    print(f"  GPU: {len(GPU_METHODS)}")
    print(f"  CPU: {len(CPU_METHODS)}")
    print(f"  Deep: {len(DEEP_METHODS)}")
    print(f"  Classical: {len(CLASSICAL_METHODS)}")
    print(f"  No-HPO: {len(NO_HPO_METHODS)}")

    print(f"\nOutput types:")
    print(f"  Logits: {len(LOGIT_METHODS)}")
    print(f"  Probabilities: {len(PROBABILITY_METHODS)}")
    print(f"  Class labels: {len(CLASS_LABEL_METHODS)}")

    print(f"\nCategorical encoding:")
    print(f"  indices: {len(REQUIRES_CAT_INDICES)}")
    print(f"  ohe: {len(REQUIRES_CAT_OHE)}")
    print(f"  tabr_ohe: {len(REQUIRES_CAT_TABR_OHE)}")

    print(f"\nRow limits:")
    for name, limit in sorted(METHOD_ROW_LIMITS.items()):
        print(f"  {name}: {limit:,} training rows")

    print(f"\n{'='*70}")
    print("\nPer-method details:")
    print(f"{'='*70}")
    for name, spec in sorted(METHOD_REGISTRY.items()):
        print(
            f"  {name:20s} | {spec.hardware.value:3s} | {spec.architecture.value:9s} | "
            f"{spec.output_type.value:14s} | cat={spec.cat_policy.value:8s} | "
            f"norm={str(spec.normalization):8s} | hpo={spec.supports_hpo}"
        )
