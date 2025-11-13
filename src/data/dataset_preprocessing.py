# src/data/dataset_preprocessing.py
"""
Dataset-specific preprocessing for TALENT.

Each supported dataset is loaded and cleaned into a standardized
format (df, target_col, categorical_cols, numerical_cols).

This module:
- Loads raw CSVs from the task directory (/pd or /lgd) as specified in CONFIG_DATA.yaml.
- Applies dataset-specific cleaning and value transformations.
- Identifies target, categorical, and numerical features.

Note:
This function only harmonizes raw data into a consistent schema.
Cross-validation, test/train/val splitting, other preprocessing is done later.
"""

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Import centralized config loader
from ..utils.config_reader import load_config

logger = logging.getLogger(__name__)
pd.set_option("future.no_silent_downcasting", True)


def _get_data_directory(task: str, config: dict = None) -> Path:
    """
    Get the data directory for a specific task from config.
    
    Parameters
    ----------
    task : str
        'pd' or 'lgd'
    config : dict, optional
        Configuration dictionary. If None, loads from config_reader.
        
    Returns
    -------
    Path
        Path to the data directory for the specified task.
    """
    if config is None:
        config = load_config()  # Use centralized loader
    
    if 'paths' not in config:
        raise ValueError("Config file missing 'paths' section")
    
    if task == 'pd':
        if 'pd_dir' not in config['paths']:
            raise ValueError("Config file missing 'paths.pd_dir'")
        return Path(config['paths']['pd_dir'])
    elif task == 'lgd':
        if 'lgd_dir' not in config['paths']:
            raise ValueError("Config file missing 'paths.lgd_dir'")
        return Path(config['paths']['lgd_dir'])
    else:
        raise ValueError(f"Invalid task '{task}'. Must be 'pd' or 'lgd'.")



def preprocess_dataset_specific(
    task: str, 
    dataset: str, 
    raw_dir: Path = None,
    config: dict = None
):
    """
    Load and preprocess a specific dataset by name and task (pd or lgd).
    
    Paths are loaded from CONFIG_DATA.yaml unless raw_dir is explicitly provided.

    Parameters
    ----------
    task : str
        'pd' or 'lgd'
    dataset : str
        e.g. '0014.hmeq', '0004.lendingclub', '0001.heloc'
    raw_dir : Path, optional
        Root directory containing /pd and /lgd folders.
        If None, reads from CONFIG_DATA.yaml.
    config : dict, optional
        Configuration dictionary. If None, loads from CONFIG_DATA.yaml.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe ready for TALENT ingestion.
    target_col : str
        Name of the target column.
    categorical_cols : list[str]
        List of categorical feature names.
    numerical_cols : list[str]
        List of numerical feature names.
    """
    
    # If raw_dir not provided, get it from config
    if raw_dir is None:
        if config is None:
            config = load_config()
        
        # Get the task-specific directory from config
        task_dir = _get_data_directory(task, config)
        dataset_path = task_dir / f"{dataset}.csv"
    else:
        # Use provided raw_dir (backward compatibility)
        dataset_path = raw_dir / task / f"{dataset}.csv"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path, low_memory=False)

    logger.info(f"Loaded raw dataset {dataset_path.name} ({task}), shape = {df.shape}")

    # -------------------------------
    #  Probability of Default (PD)
    # -------------------------------
    if task == "pd":

        # --- 0001.gmsc.csv ---
        if dataset == "0001.gmsc":
            target_col = "SeriousDlqin2yrs"
            if target_col not in df.columns:
                raise ValueError(f"Expected target '{target_col}' not in columns.")
            # No categorical features in this setup
            cat_cols: list[str] = []
            # All remaining are numeric features
            num_cols = [c for c in df.columns if c != target_col]
            return df, target_col, cat_cols, num_cols

        # --- 0002.taiwan_creditcard.csv ---
        elif dataset == "0002.taiwan_creditcard":
            # Drop ID and recode SEX exactly as in your code
            if "ID" in df.columns:
                df = df.drop(columns=["ID"])
            if "SEX" in df.columns:
                # Mapping used string keys {'2':1,'1':0}; unify to numeric mapping
                df["SEX"] = df["SEX"].replace({2: 1, 1: 0, "2": 1, "1": 0})
            target_col = "default.payment.next.month"
            if target_col not in df.columns:
                raise ValueError(f"Expected target '{target_col}' not in columns.")
            cat_cols: list[str] = []  # you specified none
            num_cols = [c for c in df.columns if c != target_col]
            return df, target_col, cat_cols, num_cols
        
        # --- 0003.vehicle_loan.csv ---
        elif dataset == "0003.vehicle_loan":
            drop_cols = [
                "UniqueID",            # unique identifier
                "branch_id",           # many categories
                "supplier_id",         # many categories
                "Current_pincode_ID",  # many categories
                "Employee_code_ID",    # many categories
                "MobileNo_Avl_Flag",   # single unique value
            ]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            # Age engineering from 'Date.of.Birth' and 'DisbursalDate'
            def _to_year(two_digit: str) -> int:
                # robust to int/str inputs like '01-01-98' OR '98'
                s = str(two_digit)
                yy = None
                # extract last 2-digit token
                digits = "".join([ch for ch in s if ch.isdigit()])
                if len(digits) >= 2:
                    yy = int(digits[-2:])
                else:
                    yy = int(digits) if digits else 0
                return 2000 + yy if 0 <= yy < 20 else 1900 + yy

            if "Date.of.Birth" in df.columns:
                df["Date.of.Birth"] = df["Date.of.Birth"].apply(_to_year)
            if "DisbursalDate" in df.columns:
                df["DisbursalDate"] = df["DisbursalDate"].apply(_to_year)

            if {"Date.of.Birth", "DisbursalDate"}.issubset(df.columns):
                df["Age"] = df["DisbursalDate"] - df["Date.of.Birth"]
                df = df.drop(columns=["DisbursalDate", "Date.of.Birth"])

            # Transform PERFORM_CNS.SCORE.DESCRIPTION to grouped buckets then numeric
            if "PERFORM_CNS.SCORE.DESCRIPTION" in df.columns:
                df.replace(
                    {
                        "PERFORM_CNS.SCORE.DESCRIPTION": {
                            "C-Very Low Risk": "Very Low Risk",
                            "A-Very Low Risk": "Very Low Risk",
                            "D-Very Low Risk": "Very Low Risk",
                            "B-Very Low Risk": "Very Low Risk",
                            "M-Very High Risk": "Very High Risk",
                            "L-Very High Risk": "Very High Risk",
                            "F-Low Risk": "Low Risk",
                            "E-Low Risk": "Low Risk",
                            "G-Low Risk": "Low Risk",
                            "H-Medium Risk": "Medium Risk",
                            "I-Medium Risk": "Medium Risk",
                            "J-High Risk": "High Risk",
                            "K-High Risk": "High Risk",
                        }
                    },
                    inplace=True,
                )
                risk_map = {
                    "No Bureau History Available": -1,
                    "Not Scored: No Activity seen on the customer (Inactive)": -1,
                    "Not Scored: Sufficient History Not Available": -1,
                    "Not Scored: No Updates available in last 36 months": -1,
                    "Not Scored: Only a Guarantor": -1,
                    "Not Scored: More than 50 active Accounts found": -1,
                    "Not Scored: Not Enough Info available on the customer": -1,
                    "Very Low Risk": 4,
                    "Low Risk": 3,
                    "Medium Risk": 2,
                    "High Risk": 1,
                    "Very High Risk": 0,
                }
                df["PERFORM_CNS.SCORE.DESCRIPTION"] = df["PERFORM_CNS.SCORE.DESCRIPTION"].map(risk_map)

            # Convert duration strings like "2 yrs 5 mon" to months
            def _months(s: str) -> int:
                if pd.isna(s):
                    return np.nan
                parts = str(s).split()
                years, months = 0, 0
                for i, tok in enumerate(parts):
                    if "yr" in tok:
                        try:
                            years = int(parts[i - 1])
                        except Exception:
                            pass
                    if "mon" in tok:
                        try:
                            months = int(parts[i - 1])
                        except Exception:
                            pass
                return years * 12 + months

            if "AVERAGE.ACCT.AGE" in df.columns:
                df["AVERAGE.ACCT.AGE"] = df["AVERAGE.ACCT.AGE"].apply(_months)
            if "CREDIT.HISTORY.LENGTH" in df.columns:
                df["CREDIT.HISTORY.LENGTH"] = df["CREDIT.HISTORY.LENGTH"].apply(_months)

            # Target and column partitions
            target_col = "loan_default"
            if target_col not in df.columns:
                raise ValueError(f"Expected target '{target_col}' not in columns.")

            # Your explicit column sets (filtered to existing columns for robustness)
            explicit_cat = ["manufacturer_id", "Employment.Type", "State_ID"]
            explicit_num = [
                "disbursed_amount", "asset_cost", "ltv", "Aadhar_flag", "PAN_flag",
                "VoterID_flag", "Driving_flag", "Passport_flag", "PERFORM_CNS.SCORE",
                "PERFORM_CNS.SCORE.DESCRIPTION", "PRI.NO.OF.ACCTS", "PRI.ACTIVE.ACCTS",
                "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE", "PRI.SANCTIONED.AMOUNT",
                "PRI.DISBURSED.AMOUNT", "SEC.NO.OF.ACCTS", "SEC.ACTIVE.ACCTS",
                "SEC.OVERDUE.ACCTS", "SEC.CURRENT.BALANCE", "SEC.SANCTIONED.AMOUNT",
                "SEC.DISBURSED.AMOUNT", "PRIMARY.INSTAL.AMT", "SEC.INSTAL.AMT",
                "NEW.ACCTS.IN.LAST.SIX.MONTHS", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS",
                "AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH", "NO.OF_INQUIRIES", "Age",
            ]
            cat_cols = [c for c in explicit_cat if c in df.columns and c != target_col]
            num_cols = [c for c in explicit_num if c in df.columns and c != target_col]

            # Add any remaining numeric columns that aren't in explicit lists (safety net)
            remaining_num = (df.select_dtypes(include=["number"]).columns.drop([target_col] + num_cols, errors="ignore").tolist())
            num_cols = num_cols + remaining_num

            # Ensure no target leakage in features
            cat_cols = [c for c in cat_cols if c != target_col]
            num_cols = [c for c in num_cols if c != target_col]
            return df, target_col, cat_cols, num_cols

        # --- 0004.lendingclub.csv ---
        elif dataset == "0004.lendingclub":
            target_col = "not.fully.paid"
            if target_col not in df.columns:
                raise ValueError("Expected 'not.fully.paid' in LendingClub dataset.")

            # Identify column types
            cat_cols = df.select_dtypes(include=["object", "category"]).drop(columns=[target_col], errors="ignore").columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

            logger.info("04_lendingclub preprocessed")
            return df, target_col, cat_cols, num_cols

        # --- 0005.Case Study.csv ---
        elif dataset == "0005.Case Study":
            if "status" in df.columns:
                df["status"] = df["status"].replace({
                    "RICH": 6, "POOR": 2, "MIDDLE": 4, "LOWMIDDLE": 3,
                    "VERYRICH": 7, "VERYMIDDLE": 5, "VERYPOOR": 1
                })

            target_col = "PaymentMissFlag"
            if target_col not in df.columns:
                raise ValueError("Expected 'PaymentMissFlag' in Case Study dataset.")

            cat_cols = df.select_dtypes(include=["object", "category"]).drop(columns=[target_col], errors="ignore").columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

            logger.info("05_case_study preprocessed")
            return df, target_col, cat_cols, num_cols

        # --- 0006.myhom.csv ---
        elif dataset == "0006.myhom":
            if "loan_id" in df.columns:
                df = df.drop(columns=["loan_id"], errors="ignore")

            target_col = "loan_default"
            if target_col not in df.columns:
                raise ValueError("Expected 'loan_default' in myhom dataset.")

            cat_cols = df.select_dtypes(include=["object", "category"]).drop(columns=[target_col], errors="ignore").columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

            logger.info("06_myhom preprocessed")
            return df, target_col, cat_cols, num_cols

        # --- 0007.hackerearth.csv ---
        elif dataset == "0007.hackerearth":
            # Drop irrelevant or high-cardinality identifiers
            drop_cols = ["member_id", "batch_enrolled", "emp_title", "pymnt_plan", "desc", "title", "zip_code"]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            # --- Clean emp_length ---
            if "emp_length" in df.columns:
                df["emp_length"] = df["emp_length"].replace("< 1 year", 0)
                df["emp_length"] = df["emp_length"].astype(str).str.replace(" years", "", regex=False)
                df["emp_length"] = df["emp_length"].astype(str).str.replace(" year", "", regex=False)
                df["emp_length"] = df["emp_length"].replace("10+", 11)
                df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce")

            # --- Clean last_week_pay ---
            if "last_week_pay" in df.columns:
                df["last_week_pay"] = df["last_week_pay"].astype(str).str.replace("th week", "", regex=False)
                df["last_week_pay"] = df["last_week_pay"].replace("NA", np.nan)
                df["last_week_pay"] = pd.to_numeric(df["last_week_pay"], errors="coerce")

            # --- Target column ---
            target_col = "loan_status"
            if target_col not in df.columns:
                raise ValueError("Expected 'loan_status' in hackerearth dataset.")
            df = df.dropna(subset=[target_col])

            # --- Identify categorical columns ---
            cat_cols = [
                "addr_state", "home_ownership", "verification_status",
                "purpose", "application_type", "grade", "sub_grade", "initial_list_status"
            ]
            cat_cols = [c for c in cat_cols if c in df.columns]

            # --- Encode all categorical columns  ---  
            for c in cat_cols:
                df[c] = df[c].astype("category")
                df[c] = df[c].cat.codes  

            # --- Identify numeric columns ---
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

            logger.info("07_hackerearth preprocessed")
            return df, target_col, cat_cols, num_cols


        # --- 0008.cobranded.csv ---
        elif dataset == "0008.cobranded":
            df = df.replace(["na", "missing"], np.nan)

            if "application_key" in df.columns:
                df = df.drop(columns=["application_key"], errors="ignore")

            if "mvar47" in df.columns:
                df["mvar47"] = df["mvar47"].replace({"C": 1, "L": 0})

            # Convert to numeric where possible
            for col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                except Exception:
                    pass

            target_col = "default_ind"
            if target_col not in df.columns:
                raise ValueError("Expected 'default_ind' in cobranded dataset.")

            cat_cols = df.select_dtypes(include=["object", "category"]).drop(columns=[target_col], errors="ignore").columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

            logger.info("08_cobranded preprocessed")
            return df, target_col, cat_cols, num_cols

        # --- 0009.german.csv ---
        elif dataset == "0009.german":
            # The original German dataset has no header row → it will load as unnamed columns: 0, 1, 2, ...
            if df.columns[0] == 0 or "Unnamed: 0" in df.columns:
                logger.info("Detected headerless German dataset, assigning synthetic feature names.")
                n_cols = df.shape[1]
                df.columns = [f"feature_{i}" for i in range(1, n_cols)] + ["target"]
            else:
                # Has header, ensure target exists
                if "target" not in df.columns:
                    n_cols = len(df.columns)
                    df.columns = [f"feature_{i}" for i in range(1, n_cols)] + ["target"]

            target_col = "target"

            # Replace missing or invalid values, drop empty target rows
            df = df.dropna(subset=[target_col])
            df[target_col] = df[target_col].replace({1: 0, 2: 1})

            # Define feature groups (based on original documentation)
            cat_cols = [
                "feature_1", "feature_3", "feature_4", "feature_6", "feature_7",
                "feature_9", "feature_10", "feature_12", "feature_14", "feature_15",
                "feature_17", "feature_19", "feature_20"
            ]
            num_cols = [
                "feature_2", "feature_5", "feature_8", "feature_11",
                "feature_13", "feature_16", "feature_18"
            ]

            cat_cols = [c for c in cat_cols if c in df.columns]
            num_cols = [c for c in num_cols if c in df.columns]

            logger.info(f"09_german_credit preprocessed: {len(num_cols)} num, {len(cat_cols)} cat.")
            return df, target_col, cat_cols, num_cols

        # --- 0010.bank_status.csv ---
        elif dataset == "0010.bank_status":
            df = df.dropna(how="all").reset_index(drop=True)
            df = df.drop(columns=[c for c in ["Loan ID", "Customer ID"] if c in df.columns], errors="ignore")

            # --- Binary flag cleanup ---
            if "Loan Status" in df.columns:
                df["Loan Status"] = df["Loan Status"].replace({"Fully Paid": 0, "Charged Off": 1})
                df["Loan Status"] = pd.to_numeric(df["Loan Status"], errors="coerce")

            if "Term" in df.columns:
                df["Term"] = df["Term"].replace({"Short Term": 0, "Long Term": 1})
                df["Term"] = pd.to_numeric(df["Term"], errors="coerce")

            # --- Encode categorical string columns ---
            # Home Ownership mapping based on observed values
            if "Home Ownership" in df.columns:
                home_map = {
                    "Own Home": 0,
                    "Home Mortgage": 1,
                    "HaveMortgage": 1,  # same meaning
                    "Rent": 2
                }
                df["Home Ownership"] = df["Home Ownership"].replace(home_map)
                df["Home Ownership"] = pd.to_numeric(df["Home Ownership"], errors="coerce")

            # Purpose mapping: consolidate semantically similar loan types
            if "Purpose" in df.columns:
                purpose_map = {
                    "Debt Consolidation": 0,
                    "Debt Consolidation Loan": 0,
                    "Home Improvements": 1,
                    "Home Improvement": 1,
                    "Buy House": 2,
                    "Buy a Car": 3,
                    "major_purchase": 4,
                    "Business Loan": 5,
                    "small_business": 5,
                    "Take a Trip": 6,
                    "Vacation": 6,
                    "Other": 7,
                    "other": 7
                }
                df["Purpose"] = df["Purpose"].replace(purpose_map)
                df["Purpose"] = pd.to_numeric(df["Purpose"], errors="coerce")

            # --- Clean up 'Years in current job' ---
            if "Years in current job" in df.columns:
                df["Years in current job"] = df["Years in current job"].replace("< 1 year", 0)
                df["Years in current job"] = df["Years in current job"].astype(str).str.replace(" years", "", regex=False)
                df["Years in current job"] = df["Years in current job"].astype(str).str.replace(" year", "", regex=False)
                df["Years in current job"] = df["Years in current job"].replace("10+", 11)
                df["Years in current job"] = pd.to_numeric(df["Years in current job"], errors="coerce")

            # --- Target column ---
            target_col = "Loan Status"
            df = df.dropna(subset=[target_col])

            # --- Column typing for TALENT ---
            cat_cols = df.select_dtypes(include=["object", "category"]).drop(columns=[target_col], errors="ignore").columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

            logger.info("10_bank_status preprocessed")
            return df, target_col, cat_cols, num_cols


        # --- 0011.thomas.csv ---
        elif dataset == "0011.thomas":
            target_col = "BAD"
            if target_col not in df.columns:
                raise ValueError("Expected 'BAD' in Thomas dataset.")

            cat_cols = df.select_dtypes(include=["object", "category"]).drop(columns=[target_col], errors="ignore").columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

            logger.info("11_thomas preprocessed")
            return df, target_col, cat_cols, num_cols

        # --- 0012.loan_default.csv ---
        elif dataset == "0012.loan_default":
            df = df.apply(pd.to_numeric, errors="coerce")
            if "id" in df.columns:
                df = df.drop(columns=["id"], errors="ignore")

            df = df.loc[:, (df != df.iloc[0]).any()]  # drop constant columns

            target_col = "loss"
            if target_col not in df.columns:
                raise ValueError("Expected 'loss' in loan_default dataset.")

            df[target_col] = np.where(df[target_col] == 0, 0, 1)

            cat_cols = []
            num_cols = [c for c in df.columns if c != target_col]

            logger.info("12_loan_default preprocessed")
            return df, target_col, cat_cols, num_cols

        # --- 0013.home_credit.csv ---
        elif dataset == "0013.home_credit":
            if "SK_ID_CURR" in df.columns:
                df = df.drop(columns=["SK_ID_CURR"], errors="ignore")

            target_col = "TARGET"
            if target_col not in df.columns:
                raise ValueError("Expected 'TARGET' in Home Credit dataset.")

            cat_cols = [
                "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
                "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE",
                "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"
            ]

            num_cols = [c for c in df.columns if c not in cat_cols + [target_col]]

            logger.info("13_home_credit preprocessed")
            return df, target_col, cat_cols, num_cols

        # --- 0014.hmeq.csv ---
        elif dataset == "0014.hmeq":
            target_col = "BAD"
            if target_col not in df.columns:
                raise ValueError("Expected 'BAD' in HMEQ dataset.")

            cat_cols = ["REASON", "JOB"]
            num_cols = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG",
                        "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]

            cat_cols = [c for c in cat_cols if c in df.columns]
            num_cols = [c for c in num_cols if c in df.columns]

            logger.info("14_hmeq preprocessed")
            return df, target_col, cat_cols, num_cols

        else:
            raise ValueError(f"No preprocessing routine defined for PD dataset: {dataset}")

    # -------------------------------
    #  Loss Given Default (LGD)
    # -------------------------------
    elif task == "lgd":

        # --- 0001.heloc.csv ---
        if dataset == "0001.heloc":
            df = df.drop(columns=["REC", "DLGD_Econ", "PrinBal", "PayOff", "DefPayOff", "ObsDT", "DefDT"], errors="ignore")

            if "LienPos" in df.columns:
                df["LienPos"] = df["LienPos"].replace({"Unknow": 0, "First": 1, "Second": 2})
                df = df.infer_objects(copy=False)

            target_col = "LGD_ACTG"
            if target_col not in df.columns:
                raise ValueError("Expected 'LGD_ACTG' in HELOC dataset.")

            # Columns as in _preprocess_01_heloc
            cat_cols = []
            num_cols = [c for c in ["PortNum", "AvailAmt", "LTV", "LienPos", "Age", "CurrEquifax",
                                    "Utilization", "DefPrinBal", "PD_Rnd"] if c in df.columns]

            logger.info("01_heloc preprocessed")
            return df, target_col, cat_cols, num_cols

        # --- 0002.loss2.csv ---
        elif dataset == "0002.loss2":
            drop_cols = [
                "_ELGDnum1", "_ELGDnum2", "id1", "Alltel_Client",
                "REO_Appraisal_Date", "Origination_Date", "date_vintage_year",
                "date_vintage_year_month", "Servicing_Loss"
            ]
            df = df.drop(columns=drop_cols, errors="ignore")

            target_col = "_ELGD"
            if target_col not in df.columns:
                raise ValueError("Expected '_ELGD' in loss2 dataset.")

            df = df.dropna(subset=[target_col])

            cat_cols = df.select_dtypes(include=["object", "category"]).drop(columns=[target_col], errors="ignore").columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

            logger.info("02_loss2 preprocessed")
            return df, target_col, cat_cols, num_cols

        # --- 0003.axa.csv ---
        elif dataset == "0003.axa":
            df = df.drop(columns=["Recovery_rate", "y_logistic", "lnrr", "Y_probit", "event"], errors="ignore")

            target_col = "lgd_time"
            if target_col not in df.columns:
                raise ValueError("Expected 'lgd_time' in axa dataset.")

            cat_cols = []
            num_cols = [c for c in ["LTV", "purpose1"] if c in df.columns]

            logger.info("03_axa preprocessed")
            return df, target_col, cat_cols, num_cols

        # --- 0004.base_model.csv ---
        elif dataset == "0004.base_model":
            drop_cols = [
                # IDs
                'DEAL_DocUNID', 'DEAL_MainID', 'DEAL_FacilityIdentifier', 'DEAL_StarWebIdentifier',
                'DFLT_MainID', 'DFLT_SPM', 'DFLT_DAI', 'DFLT_BDR', 'DFLT_LegalEntityName',
                'DFLT_StarWeb_PCRU', 'DFLT_ClientNAE', 'DFLT_ParentSPM', 'DFLT_ParentSIREN',
                'DFLT_ParentDAI', 'DFLT_ParentLegalEntityName', 'DFLT_ParentPCRU', 'DFLT_ParentNAE',
                'DFLT_subject', 'FCLT_DealUNID', 'FCLT_BCEIdentifier', 'FCLT_Identifier',
                'FCLT_BookingUnit', 'fclt_docunid',

                # Time-related
                'DEAL_TransactionStartDate', 'DEAL_TransactionEndDate', 'DEAL_DateComposed', 'DEAL_LastUpDate',
                'DFLT_SGDefaultDate', 'DFLT_PublicDefaultDate', 'DFLT_EndDefaultDate', 'DFLT_SGRatingDate',
                'DFLT_RatingDate1YPD', 'DFLT_ParentDefaultDateIf', 'DFLT_ParentSGRatingDate',
                'DFLT_ParentRatingDate1YPD', 'DFLT_DateComposed', 'DFLT_LastUpdate', 'DATE_DECLAR_CT',
                'FCLT_StartDate', 'FCLT_EndDate', 'FCLT_DefaultDate', 'FCLT_DateComposed', 'FCLT_LastUpdate', 'date',

                # Textual / description
                'FCLT_CommentsOnLimit', 'FCLT_subject', 'DEAL_GoverningLawRecovery', 'DEAL_PFRU', 'DEAL_subject',

                # Correlated / leakage
                'lgd_cat_15', 'lgd_cat_10', 'lgd_cat_5', 'LGD_log', 'LGD_deF', 'LGD_norm', 'sortie', 'RecAssoFlag',

                # High missing or leakage
                'DEAL_ConstructionEndDate', 'DEAL_ConstructionStartDate', 'DEAL_AverageRents',
                'DEAL_ExpectedVacancyRate', 'DEAL_StrikeLESSEEOption', 'DEAL_StatusUpDate', 'DEAL_DeleteDate',
                'DFLT_DeleteStatus', 'DFLT_DeletedDate', 'DFLT_JRIRating', 'DFLT_ParentJRIRating',
                'FCLT_DeleteStatus', 'FCLT_IrrevocableLocOffshore', 'FCLT_DeleteDate',
                'flag_eps', 'flag_fcltcurrency', 'fac_ss_commcov', 'flag_pme', 'Flag_specifique',
                'flag_specperi', 'Categorie_AV'
            ]
            df = df.drop(columns=drop_cols, errors="ignore")

            target_col = "LGD_brute"
            if target_col not in df.columns:
                raise ValueError("Expected 'LGD_brute' in base_model dataset.")
            df = df.dropna(subset=[target_col])

            # Remove columns with >80% missing values
            missing_ratio = df.isnull().mean()
            df = df.drop(columns=missing_ratio[missing_ratio > 0.8].index, errors="ignore")

            cat_cols = df.select_dtypes(include=["object", "category"]).drop(columns=[target_col], errors="ignore").columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

            logger.info("04_base_model preprocessed")
            return df, target_col, cat_cols, num_cols

        # --- 0005.base_modelisation.csv ---
        elif dataset == "0005.base_modelisation":
            df = df.drop(columns=["Ident_cliej_spm", "ID_CONC_ORIGIN_CDL", "id_crc", "id_unique"], errors="ignore")

            leakage_cols = [
                "lgd_5_sscout_ligne", "lgd_corr", "lgd_defaut_nt", "lgd_3class", "lgd_2class", "lgd_log",
                "lgd_t", "lgd_1log", "logit_lgd", "Dt_entree_defaut", "Dt_sortie_defaut",
                "flag_defaut_moins_1an", "auto_av_defaut", "util_av_defaut", "defaut_clos",
                "defaut_clos_4nonclos", "duree_1A_av_defaut", "util_av_defaut_tot",
                "auto_av_defaut_tot", "defaut_M1Y", "defaut_P1Y"
            ]
            df = df.drop(columns=leakage_cols, errors="ignore")

            target_col = "lgd_defaut"
            if target_col not in df.columns:
                raise ValueError("Expected 'lgd_defaut' in base_modelisation dataset.")

            df = df.dropna(subset=[target_col])
            cat_cols = df.select_dtypes(include=["object", "category"]).drop(columns=[target_col], errors="ignore").columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

            logger.info("05_base_modelisation preprocessed")
            return df, target_col, cat_cols, num_cols

        else:
            raise ValueError(f"No preprocessing routine defined for LGD dataset: {dataset}")

    else:
        raise ValueError("Task must be either 'pd' or 'lgd'")