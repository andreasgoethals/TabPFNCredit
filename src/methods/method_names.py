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


__all__ = ["METHOD_DISPLAY_NAMES", "display_name"]
