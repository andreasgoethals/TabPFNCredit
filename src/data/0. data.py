from pathlib import Path
from typing import Dict, List
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from data.dataset_preprocessing import Dataset_Preprocessing

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"  # define a single base path for all datasets

class Data:
    def __init__(self, dataconfig, experimentconfig):
        self.dataconfig = dataconfig
        self.experimentconfig = experimentconfig

        # Initialize empty placeholders, stays constant over splits
        # Data placeholders:
        # - self.x, self.y: full dataset after preprocessing
        # - *_train/val/test: subsets after splitting
        # - cols_*: metadata about categorical/numerical features
        self.x = np.empty((0, 0))
        self.y = np.empty(0)

        # Initialize empty placeholders, is updated per split;
        # contains data after handle_missing_values, encode_cat, standardize
        self.x_train = np.empty((0, 0))
        self.y_train = np.empty(0)
        self.x_val = np.empty((0, 0))
        self.y_val = np.empty(0)
        self.x_test = np.empty((0, 0))
        self.y_test = np.empty(0)

        self.cols = []
        # These contain column names:
        self.cols_cat = []
        self.cols_num = []
        # These contain column indices:
        self.cols_cat_idx = []
        self.cols_num_idx = []

        self.split_indices = {}

        self.dataset_name = None

        self.preprocessing = Dataset_Preprocessing(dataconfig, experimentconfig)

    @property
    def metadata(self):
        return {
            "dataset_name": self.dataset_name,
            "n_samples": len(self.y),
            "n_features": self.x.shape[1],
            "n_categorical": len(self.cols_cat),
            "n_numeric": len(self.cols_num),
        }

    def subsample_if_necessary(self, _data):
        row_limit = self.experimentconfig.get("row_limit", None)
        if not row_limit or not isinstance(row_limit, int) or row_limit <= 0:
            logger.info("No subsampling applied (row_limit not set or invalid).")
            return _data

        if isinstance(_data, pd.DataFrame):
            if len(_data) > row_limit:
                logger.info(f"Subsampling DataFrame from {len(_data)} to {row_limit} rows.")
                _data = _data.sample(n=row_limit, random_state=42).reset_index(drop=True)
            else:
                logger.info(f"Dataset has {len(_data)} rows. No subsampling needed.")
            return _data

        elif hasattr(_data, "data") and hasattr(_data, "target"):
            if _data.data.shape[0] > row_limit:
                idx = np.random.RandomState(seed=42).choice(_data.data.shape[0], row_limit, replace=False)
                _data.data = _data.data[idx]
                _data.target = _data.target[idx]
                logger.info(f"Subsampled sklearn dataset to {row_limit} rows.")
            return _data

        else:
            logger.error("Unsupported data format for subsampling.")
            raise TypeError("Unsupported data format for subsampling.")


    def load_preprocess_data(self):
        _data = self.load_data()
        _data = self.subsample_if_necessary(_data)
        self.x, self.y, self.cols, self.cols_cat, self.cols_num, self.cols_cat_idx, self.cols_num_idx = self.preprocessing.preprocess_data(_data)
        return self.x, self.y


    def split_data(self) -> Dict[int, Dict[str, List[int]]]:

        # Retrieve the number of CV splits from experimentconfig
        n_splits = self.experimentconfig['cv_splits']

        # Initialize KFold with the number of splits
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

        # Split the data
        for i, (train_index, test_index) in enumerate(kf.split(self.x)):
            # Further split the train set into train and validation sets
            train_index, val_index = train_test_split(train_index, test_size=self.experimentconfig['val_size'], random_state=0)

            # Store the indices in the dictionary
            self.split_indices[i] = {
                'train': train_index.tolist(),
                'val': val_index.tolist(),
                'test': test_index.tolist()
            }

        return self.split_indices
    
    
    def get_talent_datasets(self, split_indices: dict, fold: int, task_type: str = "classification"):
        # Convert stored numpy arrays into TALENT-compatible dictionaries for a given CV fold.

        train_idx = split_indices[fold]['train']
        val_idx = split_indices[fold]['val']
        test_idx = split_indices[fold]['test']

        train_val_data = {
            "x": self.x[train_idx],
            "y": self.y[train_idx],
            "val_x": self.x[val_idx],
            "val_y": self.y[val_idx]
        }
        test_data = {
            "x": self.x[test_idx],
            "y": self.y[test_idx]
        }
        info = {"task_type": task_type}

        return train_val_data, test_data, info

    def load_data(self):
        
        ##########################################
        ### method definitions for pd datasets ###
        ##########################################
        def _load_00_pd_toydata():
            _data = load_breast_cancer()
            self.dataset_name = '00_pd_toydata'
            logger.info("00_pd_toydata loaded")
            return _data


        def _load_01_gmsc():
            path = DATA_ROOT / "pd" / "01 kaggle_give me some credit" / "gmsc.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path)
            self.dataset_name = '01_gmsc'
            logger.info("01_gmsc loaded")
            return _data


        def _load_02_taiwan_creditcard():
            path = DATA_ROOT / "pd" / "02 taiwan creditcard" / "taiwan_creditcard.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")            
            _data = pd.read_csv(path, sep=",")
            self.dataset_name = '02_taiwan_creditcard'
            logger.info("02_taiwan_creditcard loaded")
            return _data


        def _load_03_vehicle_loan():
            path = DATA_ROOT / "pd" / "03 vehicle loan" / "train.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, sep=",")
            self.dataset_name = '03_vehicle_loan'
            logger.info("03_vehicle_loan loaded")
            return _data


        def _load_06_lendingclub():
            path = DATA_ROOT / "pd" / "06 lendingclub" / "loan_data.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, sep=",")
            self.dataset_name = '06_lendingclub'
            logger.info("06_lendingclub loaded")
            return _data


        def _load_07_case_study():
            path = DATA_ROOT / "pd" / "07 case study" / "Case Study- Probability of Default.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, sep=",")
            self.dataset_name = '07_case_study'
            logger.info("07_case_study loaded")
            return _data


        def _load_09_myhom():
            path = DATA_ROOT / "pd" / "09 myhom" / "train_data.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, sep=",")
            self.dataset_name = '09_myhom'
            logger.info("09_myhom loaded")
            return _data


        def _load_10_hackerearth():
            path = DATA_ROOT / "pd" / "10 hackerearth" / "train_indessa.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, sep=",")
            self.dataset_name = '10_hackerearth'
            logger.info("10_hackerearth loaded")
            return _data


        def _load_11_cobranded():
            path = DATA_ROOT / "pd" / "11 cobranded" / "Training_dataset_Original.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, sep=",", low_memory=False)
            self.dataset_name = '11_cobranded'
            logger.info("11_cobranded loaded")
            return _data


        def _load_14_german_credit():
            path = DATA_ROOT / "pd" / "14 statlog german credit data" / "german.data"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, delim_whitespace=True, header=None)
            _data.columns = [f"feature_{i+1}" for i in range(_data.shape[1] - 1)] + ["target"]
            self.dataset_name = '14_german_credit'
            logger.info("14_german_credit loaded")
            return _data


        def _load_22_bank_status():
            path = DATA_ROOT / "pd" / "22 bank loan status dataset" / "credit_train.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, sep=",")
            self.dataset_name = '22_bank_status'
            logger.info("22_bank_status loaded")
            return _data


        def _load_28_thomas():
            path = DATA_ROOT / "pd" / "28 thomas" / "Loan Data.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, sep=";")
            self.dataset_name = '28_thomas'
            logger.info("28_thomas loaded")
            return _data


        def _load_29_loan_default():
            path = DATA_ROOT / "pd" / "29 loan default predictions - imperial college london" / "train_v2.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, sep=",", dtype=float)
            self.dataset_name = '29_loan_default'
            logger.info("29_loan_default loaded")
            return _data


        def _load_30_home_credit():
            path = DATA_ROOT / "pd" / "30 home credit default risk" / "home-credit-default-risk" / "application_train.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, sep=",")
            self.dataset_name = '30_home_credit'
            logger.info("30_home_credit loaded")
            return _data


        def _load_34_hmeq_data():
            path = DATA_ROOT / "pd" / "34 hmeq" / "hmeq.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, sep=",")
            self.dataset_name = '34_hmeq_data'
            logger.info("34_hmeq_data loaded")
            return _data


        ###########################################
        ### method definitions for lgd datasets ###
        ###########################################
        def _load_00_lgd_toydata():
            _data = load_diabetes()
            self.dataset_name = '00_lgd_toydata'
            logger.info("00_lgd_toydata loaded")
            return _data


        def _load_01_heloc_lgd():
            path = DATA_ROOT / "lgd" / "01 heloc_lgd" / "heloc_lgd.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path, low_memory=False)
            self.dataset_name = '01_heloc_lgd'
            logger.info("01_heloc_lgd loaded")
            return _data


        def _load_03_loss2():
            path = DATA_ROOT / "lgd" / "03 loss2" / "loss2.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path)
            self.dataset_name = '03_loss2'
            logger.info("03_loss2 loaded")
            return _data


        def _load_05_axa():
            path = DATA_ROOT / "lgd" / "05 lgd_axa" / "lgd_axa.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path)
            self.dataset_name = '05_axa'
            logger.info("05_axa loaded")
            return _data


        def _load_06_base_model():
            path = DATA_ROOT / "lgd" / "06 base_model" / "base_model.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path)
            self.dataset_name = '06_base_model'
            logger.info("06_base_model loaded")
            return _data


        def _load_07_base_modelisation():
            path = DATA_ROOT / "lgd" / "07 base_modelisation" / "base_modelisation.csv"
            if not path.exists():
                logger.error(f"File not found: {path.resolve()}")
                raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
            _data = pd.read_csv(path)
            self.dataset_name = '07_base_modelisation'
            logger.info("07_base_modelisation loaded")
            return _data


        ######################################
        ### Dataset-specific loading calls ###
        ######################################
        task = self.experimentconfig.get("task", "").lower()

        if task == "pd":
            pd_loaders = {
                "00_pd_toydata": _load_00_pd_toydata,
                "01_gmsc": _load_01_gmsc,
                "02_taiwan_creditcard": _load_02_taiwan_creditcard,
                "03_vehicle_loan": _load_03_vehicle_loan,
                "06_lendingclub": _load_06_lendingclub,
                "07_case_study": _load_07_case_study,
                "09_myhom": _load_09_myhom,
                "10_hackerearth": _load_10_hackerearth,
                "11_cobranded": _load_11_cobranded,
                "14_german_credit": _load_14_german_credit,
                "22_bank_status": _load_22_bank_status,
                "28_thomas": _load_28_thomas,
                "29_loan_default": _load_29_loan_default,
                "30_home_credit": _load_30_home_credit,
                "34_hmeq_data": _load_34_hmeq_data,
            }

            selected = [k for k, v in self.dataconfig["dataset_pd"].items() if v]
            if len(selected) != 1:
                raise ValueError("Exactly one PD dataset must be set to True in dataconfig['dataset_pd'].")

            dataset_key = selected[0]
            logger.info(f"Loading PD dataset: {dataset_key}")
            return pd_loaders[dataset_key]()

        elif task == "lgd":
            lgd_loaders = {
                "00_lgd_toydata": _load_00_lgd_toydata,
                "01_heloc": _load_01_heloc_lgd,
                "03_loss2": _load_03_loss2,
                "05_axa": _load_05_axa,
                "06_base_model": _load_06_base_model,
                "07_base_modelisation": _load_07_base_modelisation,
            }

            selected = [k for k, v in self.dataconfig["dataset_lgd"].items() if v]
            if len(selected) != 1:
                raise ValueError("Exactly one LGD dataset must be set to True in dataconfig['dataset_lgd'].")

            dataset_key = selected[0]
            logger.info(f"Loading LGD dataset: {dataset_key}")
            return lgd_loaders[dataset_key]()

        else:
            raise ValueError(f"Invalid or missing task in experimentconfig: {task}")


