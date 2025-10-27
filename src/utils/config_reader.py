# src/utils/config_reader.py
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict, List


# ---- Allowed values (exactly what your YAML supports) ----
ALLOWED_CAT_ENCODING = {"onehot", "ordinal", "embedding", "indices", "none"}
ALLOWED_NUM_ENCODING = {"standard", "quantile", "power", "discretize", "Q_bins", "none"}
ALLOWED_NORMALIZATION = {"standard", "minmax", "robust", "none"}
ALLOWED_NUM_NAN = {"mean", "median", "zero", "none"}
ALLOWED_CAT_NAN = {"most_frequent", "constant", "none"}

# Methods you currently support (and nothing else)
PD_METHODS_ALLOWED = {
    # classical
    "cb", "knn", "lgbm", "logreg", "nb", "rf", "svm", "xgb", "ncm", "dummy",
    # deep/foundation
    "mlp", "tabnet", "tabpfn",
}
LGD_METHODS_ALLOWED = {
    # classical
    "cb", "knn", "lgbm", "lr", "rf", "xgb",
    # deep/foundation
    "mlp", "tabnet", "tabpfn",
}
CLASS_ONLY = {"logreg", "nb", "svm", "ncm", "dummy"}  # invalid for LGD
REGR_ONLY  = {"lr"}                                   # invalid for PD


class ConfigReader:
    """
    Read & validate:
    - config/CONFIG_DATA.yaml
    - config/CONFIG_EXPERIMENT.yaml
    - config/CONFIG_EVALUATION.yaml
    - config/CONFIG_METHOD.yaml

    Sole responsibility: load, validate, and return a merged configs dict.
    """

    def __init__(self, config_dir: str = "config"):
        self.root = Path(config_dir)
        self.paths = {
            "data": self.root / "CONFIG_DATA.yaml",
            "experiment": self.root / "CONFIG_EXPERIMENT.yaml",
            "evaluation": self.root / "CONFIG_EVALUATION.yaml",
            "methods": self.root / "CONFIG_METHOD.yaml",
        }
        self.cfg: Dict[str, Dict[str, Any]] = {}

    # ---------------- Public API ----------------
    def load(self) -> "ConfigReader":
        for name, p in self.paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing config file: {p}")
            with open(p, "r") as f:
                self.cfg[name] = yaml.safe_load(f) or {}
        return self

    def validate(self) -> "ConfigReader":
        errs: List[str] = []
        self._validate_data(errs)
        self._validate_evaluation(errs)
        self._validate_experiment(errs)
        self._validate_methods(errs)
        if errs:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errs))
        return self

    def to_dict(self) -> Dict[str, Any]:
        task = self._infer_task(self.cfg["data"])
        return {
            "data": self.cfg["data"],
            "experiment": self.cfg["experiment"],
            "evaluation": self.cfg["evaluation"],
            "methods": self.cfg["methods"]["methods"],
            "task": task,  # "pd" | "lgd" | "both"
        }

    # ---------------- Internal: validation ----------------
    def _validate_data(self, errs: List[str]) -> None:
        d = self.cfg.get("data", {})
        split = d.get("split", {})
        paths = d.get("paths", {})
        ds_pd = d.get("dataset_pd") or {}
        ds_lgd = d.get("dataset_lgd") or {}

        # required split keys
        for key in ["test_size", "val_size", "cv_splits", "seed", "row_limit"]:
            if key not in split:
                errs.append(f"[DATA] split.{key} is missing")

        ts = split.get("test_size"); vs = split.get("val_size")
        if not (isinstance(ts, (int, float)) and 0 < ts < 1):
            errs.append("[DATA] test_size must be in (0,1)")
        if not (isinstance(vs, (int, float)) and 0 < vs < 1):
            errs.append("[DATA] val_size must be in (0,1)")
        if isinstance(ts, (int, float)) and isinstance(vs, (int, float)) and (ts + vs > 0.6):
            errs.append("[DATA] test_size + val_size must be ≤ 0.6")

        if not (isinstance(split.get("cv_splits"), int) and split["cv_splits"] >= 2):
            errs.append("[DATA] cv_splits must be an integer ≥ 2")
        if not isinstance(split.get("seed"), int):
            errs.append("[DATA] seed must be an integer")

        # row_limit = per-dataset cap, pre-split (allow None or int >= 100)
        rl = split.get("row_limit")
        if rl is not None and not (isinstance(rl, int) and rl >= 100):
            errs.append("[DATA] row_limit must be an integer ≥ 100 or null")

        # at least one dataset enabled
        if not any(ds_pd.values()) and not any(ds_lgd.values()):
            errs.append("[DATA] At least one dataset must be set to true (PD or LGD)")

        # folder paths must exist (relative to repo root)
        repo_root = Path.cwd()
        pd_dir = paths.get("pd_dir")
        lgd_dir = paths.get("lgd_dir")
        if not pd_dir or not (repo_root / pd_dir).exists():
            errs.append(f"[DATA] paths.pd_dir does not exist: {pd_dir}")
        if not lgd_dir or not (repo_root / lgd_dir).exists():
            errs.append(f"[DATA] paths.lgd_dir does not exist: {lgd_dir}")

        # ensure task can be inferred
        _ = self._infer_task(d)  # will return "pd"/"lgd"/"both" — no error here

    def _validate_evaluation(self, errs: List[str]) -> None:
        ev = self.cfg.get("evaluation", {})
        bt = ev.get("binary_threshold")
        if bt is None or not (0 <= bt <= 1):
            errs.append("[EVAL] binary_threshold must be in [0,1]")
        rd = ev.get("round_digits")
        if rd is None or not (isinstance(rd, int) and rd >= 0):
            errs.append("[EVAL] round_digits must be a non-negative integer")

        metrics = ev.get("metrics") or {}
        if "pd" not in metrics or "lgd" not in metrics:
            errs.append("[EVAL] metrics.pd and metrics.lgd must exist")

        # If a task is active, at least one metric for that task must be enabled
        d = self.cfg.get("data", {})
        ds_pd = d.get("dataset_pd") or {}
        ds_lgd = d.get("dataset_lgd") or {}
        if any(ds_pd.values()) and not any((metrics.get("pd") or {}).values()):
            errs.append("[EVAL] PD datasets selected but no PD metrics enabled")
        if any(ds_lgd.values()) and not any((metrics.get("lgd") or {}).values()):
            errs.append("[EVAL] LGD datasets selected but no LGD metrics enabled")

    def _validate_experiment(self, errs: List[str]) -> None:
        ex = self.cfg.get("experiment", {})

        # imbalance
        if not isinstance(ex.get("imbalance"), bool):
            errs.append("[EXP] imbalance must be boolean")
        ir = ex.get("imbalance_ratio")
        if ir is None or not (0 < ir < 1):
            errs.append("[EXP] imbalance_ratio must be in (0,1)")

        # preprocessing knobs
        cat = ex.get("categorical_encoding")
        num = ex.get("numerical_encoding")
        norm = ex.get("normalization")
        nnum = ex.get("num_nan_policy")
        ncat = ex.get("cat_nan_policy")

        if cat not in ALLOWED_CAT_ENCODING:
            errs.append(f"[EXP] categorical_encoding must be one of {sorted(ALLOWED_CAT_ENCODING)}")
        if num not in ALLOWED_NUM_ENCODING:
            errs.append(f"[EXP] numerical_encoding must be one of {sorted(ALLOWED_NUM_ENCODING)}")
        if norm not in ALLOWED_NORMALIZATION:
            errs.append(f"[EXP] normalization must be one of {sorted(ALLOWED_NORMALIZATION)}")
        if nnum not in ALLOWED_NUM_NAN:
            errs.append(f"[EXP] num_nan_policy must be one of {sorted(ALLOWED_NUM_NAN)}")
        if ncat not in ALLOWED_CAT_NAN:
            errs.append(f"[EXP] cat_nan_policy must be one of {sorted(ALLOWED_CAT_NAN)}")

        # training & tuning
        me = ex.get("max_epochs"); bs = ex.get("batch_size")
        tune = ex.get("tune"); n_trials = ex.get("n_trials")
        if not (isinstance(me, int) and me > 0):
            errs.append("[EXP] max_epochs must be an integer > 0")
        if not (isinstance(bs, int) and bs > 0):
            errs.append("[EXP] batch_size must be an integer > 0")
        if not isinstance(tune, bool):
            errs.append("[EXP] tune must be boolean")
        if tune and (not isinstance(n_trials, int) or n_trials < 1):
            errs.append("[EXP] n_trials must be an integer ≥ 1 when tune=true")

        # Extra rule from TALENT classical base: cat_policy must NOT be 'indices' for classical models
        # (TALENT/model/classical_methods/base.py asserts args.cat_policy != 'indices')
        classical_enabled = self._any_classical_enabled()
        if classical_enabled and cat == "indices":
            errs.append("[EXP] categorical_encoding='indices' is invalid when any classical method is enabled")

    def _validate_methods(self, errs: List[str]) -> None:
        m = self.cfg.get("methods", {}).get("methods", {})
        pd_cfg = m.get("pd") or {}
        lgd_cfg = m.get("lgd") or {}

        # unknown method keys
        unknown_pd = set(pd_cfg.keys()) - PD_METHODS_ALLOWED
        unknown_lgd = set(lgd_cfg.keys()) - LGD_METHODS_ALLOWED
        if unknown_pd:
            errs.append(f"[METHODS] Unknown/unsupported PD methods: {sorted(unknown_pd)}")
        if unknown_lgd:
            errs.append(f"[METHODS] Unknown/unsupported LGD methods: {sorted(unknown_lgd)}")

        # for each active task, at least one method must be true
        d = self.cfg.get("data", {})
        ds_pd = d.get("dataset_pd") or {}
        ds_lgd = d.get("dataset_lgd") or {}
        if any(ds_pd.values()) and not any(pd_cfg.values()):
            errs.append("[METHODS] PD dataset(s) selected but no PD method enabled")
        if any(ds_lgd.values()) and not any(lgd_cfg.values()):
            errs.append("[METHODS] LGD dataset(s) selected but no LGD method enabled")

        # disallow classification-only methods for LGD, and regression-only for PD
        pd_on = {k for k, v in pd_cfg.items() if v}
        lgd_on = {k for k, v in lgd_cfg.items() if v}
        bad_lgd = lgd_on.intersection(CLASS_ONLY)
        bad_pd = pd_on.intersection(REGR_ONLY)
        if bad_lgd:
            errs.append(f"[METHODS] Classification-only methods selected for LGD: {sorted(bad_lgd)}")
        if bad_pd:
            errs.append(f"[METHODS] Regression-only methods selected for PD: {sorted(bad_pd)}")

    # ---------------- Internal helpers ----------------
    def _infer_task(self, data_cfg: Dict[str, Any]) -> str:
        has_pd = any((data_cfg.get("dataset_pd") or {}).values())
        has_lgd = any((data_cfg.get("dataset_lgd") or {}).values())
        if has_pd and has_lgd:
            return "both"
        if has_pd:
            return "pd"
        if has_lgd:
            return "lgd"
        # guarded by validation
        return "pd"

    def _any_classical_enabled(self) -> bool:
        methods_cfg = self.cfg.get("methods", {}).get("methods", {})
        pd_cfg = methods_cfg.get("pd") or {}
        lgd_cfg = methods_cfg.get("lgd") or {}
        classical_pd = {"cb", "knn", "lgbm", "logreg", "nb", "rf", "svm", "xgb", "ncm", "dummy"}
        classical_lgd = {"cb", "knn", "lgbm", "lr", "rf", "xgb"}
        return any(pd_cfg.get(m, False) for m in classical_pd) or any(lgd_cfg.get(m, False) for m in classical_lgd)


# =============================================================================
# Expected Merged Configuration Dictionary Structure
# =============================================================================
# The ConfigReader.load().validate().to_dict() method returns a single
# validated and unified configuration dictionary with the following structure:
#
# {
#   "data": {
#       "split": {
#           "test_size": float,        # Fraction of full data reserved for testing (0 < x < 1)
#           "val_size": float,         # Fraction of training data used for validation (0 < x < 1)
#           "cv_splits": int,          # Number of cross-validation folds (>= 2)
#           "seed": int,               # Random seed for reproducibility
#           "row_limit": int | None,   # Max number of rows per dataset BEFORE splitting (None = all rows)
#       },
#       "paths": {
#           "pd_dir": str,             # Relative path to raw PD datasets folder
#           "lgd_dir": str,            # Relative path to raw LGD datasets folder
#       },
#       "dataset_pd": {
#           "<dataset_name>": bool,    # True = include dataset for PD task, False = exclude
#           ...
#       },
#       "dataset_lgd": {
#           "<dataset_name>": bool,    # True = include dataset for LGD task, False = exclude
#           ...
#       },
#   },
#
#   "experiment": {
#       # ---------------- Preprocessing & Encoding ----------------
#       "imbalance": bool,             # Whether to simulate artificial imbalance for PD data
#       "imbalance_ratio": float,      # Ratio for minority class when imbalance=True (0 < x < 1)
#
#       # --- Categorical encoding options (TALENT-supported) ---
#       # onehot | ordinal | embedding | indices | none
#       "categorical_encoding": str,
#
#       # --- Numerical encoding options (TALENT-supported) ---
#       # standard | quantile | power | discretize | Q_bins | none
#       "numerical_encoding": str,
#
#       # --- Normalization options (TALENT-supported) ---
#       # standard | minmax | robust | log | none
#       "normalization": str,
#
#       # --- Missing value handling ---
#       # Numeric: mean | median | zero | none
#       # Categorical: most_frequent | constant | none
#       "num_nan_policy": str,
#       "cat_nan_policy": str,
#
#       # ---------------- Training / Optimization ----------------
#       "max_epochs": int,             # Max number of epochs (for deep/foundation models)
#       "batch_size": int,             # Batch size (ignored by classical models)
#       "tune": bool,                  # Whether to perform hyperparameter tuning
#       "n_trials": int,               # Number of tuning trials if tune=True
#   },
#
#   "evaluation": {
#       "binary_threshold": float,     # Classification cutoff for PD (0 ≤ x ≤ 1)
#       "round_digits": int,           # Decimal precision for reported metrics
#       "metrics": {
#           "pd": {
#               "accuracy": bool,
#               "brier": bool,
#               "f1": bool,
#               "precision": bool,
#               "recall": bool,
#               "aucroc": bool,
#               "aucpr": bool,
#               "h_measure": bool,
#           },
#           "lgd": {
#               "mse": bool,
#               "mae": bool,
#               "r2": bool,
#               "rmse": bool,
#           },
#       },
#   },
#
#   "methods": {
#       "pd": {
#           # ---------------- Classical ML ----------------
#           "cb": bool, "knn": bool, "lgbm": bool, "logreg": bool,
#           "nb": bool, "rf": bool, "svm": bool, "xgb": bool,
#           "ncm": bool, "dummy": bool,
#
#           # ---------------- Deep / Foundation ------------
#           "mlp": bool, "tabnet": bool, "tabpfn": bool,
#       },
#       "lgd": {
#           # ---------------- Classical ML ----------------
#           "cb": bool, "knn": bool, "lgbm": bool, "lr": bool,
#           "rf": bool, "xgb": bool,
#
#           # ---------------- Deep / Foundation ------------
#           "mlp": bool, "tabnet": bool, "tabpfn": bool,
#       },
#   },
#
#   # Automatically inferred task based on dataset selection:
#   # "pd"   → only PD datasets selected
#   # "lgd"  → only LGD datasets selected
#   # "both" → both PD and LGD datasets active
#   "task": str,
# }
#
# Example usage:
#   cfg = ConfigReader().load().validate().to_dict()
#   print(cfg["task"])                     # -> "pd"
#   print(cfg["methods"]["pd"]["tabpfn"])  # -> True
#   print(cfg["data"]["split"]["row_limit"])
# =============================================================================