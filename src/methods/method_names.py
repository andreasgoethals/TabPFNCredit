"""Canonical display names for methods in figures (the paper standard).

Internal/registry keys (TALENT's canonical ``model_type`` strings, e.g.
``tabpfn_v3``, ``LogReg``) map to the single consistent label shown in every
generated figure (``TabPFN-3``, ``log. reg``). The data, file names,
focus lists and statistical tests all keep using the raw registry keys — ONLY
the text a reader sees in a figure is mapped through :func:`display_name`.

This module deliberately has **no third-party / TALENT dependency** so the
plotting and statistics modules can import it anywhere (notebooks, CI, a laptop
without the full stack). Any method not listed renders with its raw key
unchanged, so a new method never crashes a plot — it just shows its key until a
label is added here.

The four classical methods the source table abbreviates in parentheses
(``KNN``, ``SVM``, ``NCM``, ``FT-Transformer``) use the short form, since the
full names are unwieldy on a dense ~30-method axis. Change any label in one
place — this dict.
"""

from __future__ import annotations

METHOD_DISPLAY_NAMES: dict[str, str] = {
    # ---- Foundation models -------------------------------------------------
    "tabpfn": "TabPFN",
    "tabpfn_v2": "TabPFNv2",
    "tabpfn_v2_5": "TabPFN-2.5",
    "tabpfn_v3": "TabPFN-3",
    "tabpfn_real": "Real-TabPFN",
    "tabicl": "TabICL",
    "tabicl_v2": "TabICLv2",
    "tabdpt": "TabDPT",
    "tabfm": "TabFM",
    "mitra": "Mitra",
    "limix": "LimiX",
    "hyperfast": "HyperFast",
    "tabptm": "TabPTM",
    # ---- Tree boosting -----------------------------------------------------
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    # ---- Deep tabular — transformers --------------------------------------
    "ftt": "FT-Transformer",
    "autoint": "AutoInt",
    "excelformer": "ExcelFormer",
    "amformer": "AMFormer",
    "t2gformer": "T2G-Former",
    "ptarl": "PTARL",
    "saint": "SAINT",
    "tabtransformer": "TabTransformer",
    "trompt": "Trompt",
    # ---- Deep tabular — MLP & specialized ---------------------------------
    "mlp": "MLP",
    "resnet": "ResNet",
    "snn": "SNN",
    "realmlp": "RealMLP",
    "mlp_plr": "MLP-PLR",
    "danets": "DANets",
    "switchtab": "SwitchTab",
    "tabnet": "TabNet",
    "dcn2": "DCN2",
    "tabm": "TabM",
    "tangos": "TANGOS",
    "modernNCA": "ModernNCA",
    "node": "NODE",
    "grownet": "GrowNet",
    "grande": "GRANDE",
    "tabcaps": "TabCaps",
    "bishop": "BiSHop",
    "protogate": "ProtoGate",
    "tabr": "TabR",
    "dnnr": "DNNR",
    "tabautopnpnet": "TabAuto-PNPNet",
    # ---- Classical ML ------------------------------------------------------
    "LogReg": "log. reg",
    "LinearRegression": "lin. reg",
    "knn": "KNN",
    "RandomForest": "Random Forest",
    "svm": "SVM",
    "NaiveBayes": "Naive Bayes",
    "NCM": "NCM",
    "rfm": "RFM",
    "xrfm": "xRFM",
    "dummy": "Dummy",
}


def display_name(method) -> str:
    """Standard figure label for a method key (unchanged if it isn't mapped)."""
    return METHOD_DISPLAY_NAMES.get(str(method), str(method))


# ---------------------------------------------------------------------------
#  Method CLASSES -- the bar colour code for every many-method figure
# ---------------------------------------------------------------------------
# Bar charts that show ~30 methods used to be coloured by a green->red value
# gradient, which duplicated information already printed on each bar. Colouring
# by MODEL CLASS instead makes the family structure readable at a glance ("do
# the foundation models cluster at the top?"), which is the actual question
# those figures answer. Foundation models keep their crimson NAME on the tick
# axis, so the two codes complement rather than compete.

FOUNDATION = "Foundation model"
BOOSTING = "Gradient boosting"
DEEP = "Deep tabular"
CLASSICAL = "Classical ML"

#: Class ORDER for legends (most to least "modern"); also the colour order.
METHOD_CLASS_ORDER = (FOUNDATION, BOOSTING, DEEP, CLASSICAL)

#: Okabe-Ito-derived, colour-blind-safe, and deliberately distinct from both
#: the crimson foundation-name highlight and the observed/predicted blues.
METHOD_CLASS_COLORS: dict[str, str] = {
    FOUNDATION: "#0072B2",   # blue
    BOOSTING: "#E69F00",     # orange
    DEEP: "#009E73",         # green
    CLASSICAL: "#999999",    # grey
}

_METHOD_CLASSES: dict[str, str] = {
    # ---- Foundation models / in-context learners --------------------------
    **{m: FOUNDATION for m in (
        "tabpfn", "tabpfn_v2", "tabpfn_v2_5", "tabpfn_v3", "tabpfn_real",
        "tabicl", "tabicl_v2", "tabdpt", "mitra", "limix", "hyperfast",
        "tabptm", "tabfm",
    )},
    # ---- Gradient boosting (the credit-risk industry baselines) -----------
    **{m: BOOSTING for m in ("xgboost", "catboost", "lightgbm")},
    # ---- Deep tabular networks (transformers, MLPs, tree-mimics, ...) -----
    **{m: DEEP for m in (
        "ftt", "autoint", "excelformer", "amformer", "t2gformer", "ptarl",
        "saint", "tabtransformer", "trompt",
        "mlp", "resnet", "snn", "realmlp", "mlp_plr", "danets", "switchtab",
        "tabnet", "dcn2", "tabm", "node", "grownet", "grande",
        "tangos", "modernNCA", "tabcaps", "bishop", "protogate", "tabr",
        "dnnr", "tabautopnpnet",
    )},
    # ---- Classical ML ------------------------------------------------------
    **{m: CLASSICAL for m in (
        "LogReg", "LinearRegression", "knn", "RandomForest", "svm",
        "NaiveBayes", "NCM", "dummy", "rfm", "xrfm",
    )},
}


def method_class(method) -> str:
    """Model class of ``method`` (unknown methods -> ``"Classical ML"``).

    The fallback is deliberate: an unmapped method is far more likely to be a
    newly added baseline than a foundation model, and mislabelling something as
    a foundation model would misrepresent a figure.
    """
    return _METHOD_CLASSES.get(str(method), CLASSICAL)


def method_class_color(method) -> str:
    """Bar colour for ``method``, by model class."""
    return METHOD_CLASS_COLORS[method_class(method)]


__all__ = [
    "METHOD_DISPLAY_NAMES", "display_name",
    "FOUNDATION", "BOOSTING", "DEEP", "CLASSICAL",
    "METHOD_CLASS_ORDER", "METHOD_CLASS_COLORS",
    "method_class", "method_class_color",
]
