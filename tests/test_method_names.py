"""The canonical figure display-name map must be exact and complete."""
from __future__ import annotations

from src.methods.method_names import METHOD_DISPLAY_NAMES, display_name

# The standard names that MUST render exactly as the paper's method table.
EXPECTED = {
    # Foundation
    "tabpfn": "TabPFN", "tabpfn_v2": "TabPFNv2", "tabpfn_v2_5": "TabPFN-2.5",
    "tabpfn_v3": "TabPFN-3", "tabpfn_real": "Real-TabPFN", "tabicl": "TabICL",
    "tabicl_v2": "TabICLv2", "tabdpt": "TabDPT", "mitra": "Mitra",
    # Tree boosting
    "catboost": "CatBoost", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    # Transformers
    "ftt": "FT-Transformer", "autoint": "AutoInt", "excelformer": "ExcelFormer",
    "amformer": "AMFormer", "t2gformer": "T2G-Former", "ptarl": "PTARL",
    # MLP & specialized
    "mlp": "MLP", "resnet": "ResNet", "snn": "SNN", "realmlp": "RealMLP",
    "mlp_plr": "MLP-PLR", "danets": "DANets", "switchtab": "SwitchTab",
    "tabnet": "TabNet", "dcn2": "DCN2", "tabm": "TabM", "tangos": "TANGOS",
    "modernNCA": "ModernNCA",
    # Classical (regression baselines abbreviated for dense method axes)
    "LogReg": "log. reg", "LinearRegression": "lin. reg",
    "knn": "KNN", "RandomForest": "Random Forest", "svm": "SVM",
    "NaiveBayes": "Naive Bayes", "NCM": "NCM",
}

# Every method key registered in TALENT's registry (kept in sync with
# TALENT/model/method_registry.py) must have a label, so no figure ever shows a
# raw key for a method the project can run.
REGISTRY_KEYS = {
    "mlp", "resnet", "snn", "realmlp", "mlp_plr", "autoint", "saint", "ftt",
    "tabtransformer", "excelformer", "t2gformer", "amformer", "trompt", "dcn2",
    "node", "tabcaps", "tabnet", "danets", "grownet", "grande", "tabm", "tabr",
    "modernNCA", "dnnr", "tangos", "switchtab", "ptarl", "bishop", "protogate",
    "tabautopnpnet", "tabpfn", "tabpfn_v2", "tabpfn_v2_5", "tabpfn_v3",
    "tabpfn_real", "hyperfast", "tabptm", "tabicl", "tabicl_v2", "mitra", "limix",
    "tabdpt", "dummy", "LogReg", "LinearRegression", "xgboost", "catboost",
    "lightgbm", "RandomForest", "svm", "knn", "NCM", "NaiveBayes", "rfm", "xrfm",
}


def test_expected_display_names_exact():
    for key, want in EXPECTED.items():
        assert display_name(key) == want, (key, display_name(key))


def test_unknown_method_passes_through_unchanged():
    assert display_name("some_future_method") == "some_future_method"
    assert display_name("tabpfn_v3") == "TabPFN-3"   # idempotent on a known key


def test_every_registry_key_has_a_label():
    missing = sorted(k for k in REGISTRY_KEYS if k not in METHOD_DISPLAY_NAMES)
    assert not missing, f"registry keys with no figure label: {missing}"


def test_no_duplicate_display_labels():
    # Distinct methods must not collapse to the same on-figure label.
    seen = {}
    for key, label in METHOD_DISPLAY_NAMES.items():
        assert label not in seen, f"{key!r} and {seen.get(label)!r} share label {label!r}"
        seen[label] = key
