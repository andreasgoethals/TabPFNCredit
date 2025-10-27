# src/data/preprocessing.py
"""
Dataset-specific preprocessing and storage for TALENT-compatible data.
----------------------------------------------------------------------
This module loads each dataset from data/raw, performs dataset-specific
cleaning and formatting to satisfy TALENT’s requirements, and saves the
processed arrays into data/processed/{pd|lgd}/{dataset_name}/.

It does NOT:
- perform row subsampling (row_limit is ignored here)
- create train/val/test splits
- perform encoding, normalization, or imputation (handled by TALENT)

Each preprocessed dataset folder will contain:
    N.npy     -> numeric features  (float32)
    C.npy     -> categorical features (int64)
    y.npy     -> target array (int64 or float32)
    info.json -> metadata (task_type, n_num_features, n_cat_features)
"""

from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
pd.set_option("future.no_silent_downcasting", True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"


# =====================================================================
# Core helper utilities
# =====================================================================

def _save_processed_dataset(task: str, name: str, N: Optional[np.ndarray],
                            C: Optional[np.ndarray], y: np.ndarray, info: Dict):
    """Save N, C, y, and info.json to data/processed/{task}/{name}."""
    target_dir = PROC_DIR / task / name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if N is not None:
        np.save(target_dir / "N.npy", N.astype(np.float32))
    if C is not None:
        np.save(target_dir / "C.npy", C.astype(np.int64))
    np.save(target_dir / "y.npy", y)

    with open(target_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    logger.info(f"✅ Saved TALENT-ready dataset to {target_dir}")


def _split_categorical_numerical(df: pd.DataFrame, target_col: str):
    """Split DataFrame into numeric and categorical arrays according to TALENT format."""
    df = df.copy()
    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    # convert categoricals → integer codes
    C = None
    if cat_cols:
        C = X[cat_cols].astype("category").apply(lambda s: s.cat.codes).to_numpy(dtype=np.int64)

    N = None
    if num_cols:
        N = X[num_cols].to_numpy(dtype=np.float32)

    return N, C, y, cat_cols, num_cols


# =====================================================================
# Dataset-specific preprocessing logic
# =====================================================================

def preprocess_dataset(dataconfig: Dict, experimentconfig: Dict):
    """
    Run TALENT-compatible preprocessing for the selected dataset.
    Saves results to data/processed/{task}/{dataset_name}/
    """

    task = experimentconfig.get("task", "").lower()
    if task not in {"pd", "lgd"}:
        raise ValueError("experimentconfig['task'] must be 'pd' or 'lgd'.")

    # Identify dataset
    if task == "pd":
        selected = [k for k, v in dataconfig["dataset_pd"].items() if v]
        if len(selected) != 1:
            raise ValueError("Exactly one PD dataset must be True in dataconfig['dataset_pd'].")
        dataset = selected[0]
        logger.info(f"Preprocessing PD dataset: {dataset}")
        data_dir = RAW_DIR / "pd"
    else:
        selected = [k for k, v in dataconfig["dataset_lgd"].items() if v]
        if len(selected) != 1:
            raise ValueError("Exactly one LGD dataset must be True in dataconfig['dataset_lgd'].")
        dataset = selected[0]
        logger.info(f"Preprocessing LGD dataset: {dataset}")
        data_dir = RAW_DIR / "lgd"

    # --------------------------------------------------
    # Load raw dataset
    # --------------------------------------------------
    # Each dataset’s loading follows your current structure but simplified.
    # Below only a few examples; extend similarly for all.
    # --------------------------------------------------

    if dataset == "01_gmsc":
        path = data_dir / "01 kaggle_give me some credit" / "gmsc.csv"
        df = pd.read_csv(path)
        df = df.dropna(subset=["SeriousDlqin2yrs"])
        df = df.replace("na", np.nan)
        df["SeriousDlqin2yrs"] = df["SeriousDlqin2yrs"].astype(int)
        target_col = "SeriousDlqin2yrs"

    elif dataset == "02_taiwan_creditcard":
        path = data_dir / "02 taiwan creditcard" / "taiwan_creditcard.csv"
        df = pd.read_csv(path)
        df = df.drop(columns=["ID"])
        df["SEX"] = df["SEX"].replace({"2": 1, "1": 0}).astype(int)
        target_col = "default.payment.next.month"

    elif dataset == "03_vehicle_loan":
        path = data_dir / "03 vehicle loan" / "train.csv"
        df = pd.read_csv(path)
        df = df.drop(columns=["UniqueID", "branch_id", "supplier_id", "Current_pincode_ID",
                              "Employee_code_ID", "MobileNo_Avl_Flag"])
        # Date features → numeric age difference
        def _year(v):
            yr = int(v[-2:])
            return yr + (2000 if yr < 20 else 1900)
        df["Date.of.Birth"] = df["Date.of.Birth"].apply(_year)
        df["DisbursalDate"] = df["DisbursalDate"].apply(_year)
        df["Age"] = df["DisbursalDate"] - df["Date.of.Birth"]
        df = df.drop(columns=["Date.of.Birth", "DisbursalDate"])
        # Map risk description
        risk_map = {
            'C-Very Low Risk': 4, 'A-Very Low Risk': 4, 'D-Very Low Risk': 4,
            'B-Very Low Risk': 4, 'M-Very High Risk': 0, 'L-Very High Risk': 0,
            'F-Low Risk': 3, 'E-Low Risk': 3, 'G-Low Risk': 3,
            'H-Medium Risk': 2, 'I-Medium Risk': 2,
            'J-High Risk': 1, 'K-High Risk': 1,
            'No Bureau History Available': -1,
            'Not Scored: No Activity seen on the customer (Inactive)': -1,
            'Not Scored: Sufficient History Not Available': -1,
            'Not Scored: No Updates available in last 36 months': -1,
            'Not Scored: Only a Guarantor': -1,
            'Not Scored: More than 50 active Accounts found': -1,
            'Not Scored: Not Enough Info available on the customer': -1,
        }
        df["PERFORM_CNS.SCORE.DESCRIPTION"] = df["PERFORM_CNS.SCORE.DESCRIPTION"].map(risk_map)
        # Age features
        def _months(v):
            y, m = v.split(" ")
            return int(y.replace("yrs", "")) * 12 + int(m.replace("mon", ""))
        df["AVERAGE.ACCT.AGE"] = df["AVERAGE.ACCT.AGE"].apply(_months)
        df["CREDIT.HISTORY.LENGTH"] = df["CREDIT.HISTORY.LENGTH"].apply(_months)
        target_col = "loan_default"

    elif dataset == "14_german_credit":
        path = data_dir / "14 statlog german credit data" / "german.data"
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        df.columns = [*(f"feature_{i+1}" for i in range(df.shape[1]-1)), "target"]
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].replace({1: 0, 2: 1})
        target_col = "target"

    elif dataset == "22_bank_status":
        path = data_dir / "22 bank loan status dataset" / "credit_train.csv"
        df = pd.read_csv(path)
        df = df.dropna(how="all")
        df = df.drop(columns=["Loan ID", "Customer ID"])
        df["Loan Status"] = df["Loan Status"].replace({"Fully Paid": 0, "Charged Off": 1})
        df["Term"] = df["Term"].replace({"Short Term": 0, "Long Term": 1})
        df["Years in current job"] = (
            df["Years in current job"]
            .replace("< 1 year", 0)
            .str.replace(" years", "")
            .str.replace(" year", "")
            .replace("10+", 11)
            .astype(float)
        )
        target_col = "Loan Status"

    elif dataset == "01_heloc":
        path = data_dir / "01 heloc_lgd" / "heloc_lgd.csv"
        df = pd.read_csv(path)
        df = df.drop(columns=["REC", "DLGD_Econ", "PrinBal", "PayOff", "DefPayOff", "ObsDT", "DefDT"])
        df["LienPos"] = df["LienPos"].replace({"Unknow": 0, "First": 1, "Second": 2})
        target_col = "LGD_ACTG"

    else:
        raise NotImplementedError(f"Preprocessing for dataset {dataset} not yet implemented.")

    # --------------------------------------------------
    # Clean all “na” strings, ensure proper dtypes
    # --------------------------------------------------
    df = df.replace(["na", "NA", "NaN", "missing", "?"], np.nan)
    df = df.infer_objects(copy=False)

    # Drop constant columns (zero variance)
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        df = df.drop(columns=const_cols)
        logger.info(f"Dropped constant columns: {const_cols}")

    # --------------------------------------------------
    # Split into N, C, y (TALENT format)
    # --------------------------------------------------
    N, C, y, cat_cols, num_cols = _split_categorical_numerical(df, target_col)

    info = {
        "task_type": "regression" if task == "lgd" else "classification",
        "n_samples": len(y),
        "n_num_features": N.shape[1] if N is not None else 0,
        "n_cat_features": C.shape[1] if C is not None else 0,
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols,
    }

    # --------------------------------------------------
    # Save to disk
    # --------------------------------------------------
    _save_processed_dataset(task, dataset, N, C, y, info)

    logger.info(f"Finished preprocessing {dataset}: "
                f"{info['n_samples']} samples, "
                f"{info['n_num_features']} num, "
                f"{info['n_cat_features']} cat.")

    return info
