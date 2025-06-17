from pathlib import Path
from typing import Dict, List
import logging

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, KFold

from src.classes.data.preprocessing import Preprocessing

logger = logging.getLogger(__name__)

class Data:
    def __init__(self, dataconfig, experimentconfig):
        self.dataconfig = dataconfig
        self.experimentconfig = experimentconfig

        # Initialize empty placeholders, stays constant over splits
        # contains data after load_data, preprocess_data; but before split_data; and before handle_missing_values, encode_cat, standardize
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

        self.preprocessing = Preprocessing(dataconfig, experimentconfig)



    '''
    def load_preprocess_data(self):

        _data = self._load_data()
        self.x, self.y, self.cols, self.cols_cat, self.cols_num = self._preprocess_data(_data)
    '''

    def load_preprocess_data(self):
        _data = self.load_data()
        _data = self.subsample_if_necessary(_data)
        self.x, self.y, self.cols, self.cols_cat, self.cols_num, self.cols_cat_idx, self.cols_num_idx = self.preprocessing.preprocess_data(_data)

        # check if we want artificial class imbalance
        if self.experimentconfig.get('imbalance', False):
            logger.info(f"Inducing class imbalance with ratio {self.experimentconfig['imbalance_ratio']} ...")
            self.x, self.y = self._introduce_class_imbalance(
                self.x, self.y,
                imbalance_ratio=self.experimentconfig.get('imbalance_ratio', 0.1),
            )

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

    @staticmethod
    def _introduce_class_imbalance(x, y, imbalance_ratio=0.1, random_state=0):
        """
        Downsample the minority class to achieve a given class imbalance.

        Parameters
        ----------
        x : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector (1D, integer or categorical).
        imbalance_ratio : float
            Desired ratio of minority class in the output data (e.g. 0.1 for 10%).
        random_state : int
            Random seed.

        Returns
        -------
        x_new, y_new : np.ndarray
            Feature matrix and labels with induced class imbalance.
        """

        # Count class occurrences
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Class distribution BEFORE imbalance: {dict(zip(unique, counts))}")

        # Find majority and minority classes
        class_counts = dict(zip(unique, counts))
        majority_class = unique[np.argmax(counts)]
        minority_class = unique[np.argmin(counts)]
        logger.info(f"Majority class: {majority_class}, Minority class: {minority_class}")

        idx_major = np.where(y == majority_class)[0]
        idx_minor = np.where(y == minority_class)[0]
        n_major = len(idx_major)

        # Calculate how many minority samples to keep
        n_minor_new = int(n_major * imbalance_ratio / (1 - imbalance_ratio))
        n_minor_new = min(len(idx_minor), n_minor_new)
        logger.info(f"Will keep {n_minor_new} of {len(idx_minor)} minority samples (ratio {imbalance_ratio})")

        # Randomly sample minority indices
        rng = np.random.RandomState(random_state)
        idx_minor_sampled = rng.choice(idx_minor, size=n_minor_new, replace=False)

        # Combine indices and shuffle
        idx_combined = np.concatenate([idx_major, idx_minor_sampled])
        rng.shuffle(idx_combined)

        # Final data
        x_new = x[idx_combined]
        y_new = y[idx_combined]

        unique_new, counts_new = np.unique(y_new, return_counts=True)
        logger.info(f"Class distribution AFTER imbalance: {dict(zip(unique_new, counts_new))}")
        logger.info(f"Total samples: {len(y_new)}\n")
        return x_new, y_new

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
            _data = pd.read_csv('data/pd/01 kaggle_give me some credit/gmsc.csv')
            self.dataset_name = '01_gmsc'
            logger.info("01_gmsc loaded")
            return _data

        def _load_02_taiwan_creditcard():
            try:
                _data = pd.read_csv('data/pd/02 taiwan creditcard/taiwan_creditcard.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../../../data/pd/02 taiwan creditcard/taiwan_creditcard.csv', sep=',')
            logger.info("02_taiwan_creditcard loaded")
            return _data

        def _load_03_vehicle_loan():
            try:
                _data = pd.read_csv('data/pd/03 vehicle loan/train.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/03 vehicle loan/train.csv', sep=',')
            logger.info("03_vehicle_loan loaded")
            return _data

        def _load_06_lendingclub():
            try:
                _data = pd.read_csv('data/pd/06 lendingclub/loan_data.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/06 lendingclub/loan_data.csv', sep=',')
            logger.info("06_lendingclub loaded")
            return _data

        def _load_07_case_study():
            try:
                _data = pd.read_csv('data/pd/07 case study/Case Study- Probability of Default.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/07 case study/Case Study- Probability of Default.csv', sep=',')
            logger.info("07_case_study loaded")
            return _data

        def _load_09_myhom():
            try:
                _data = pd.read_csv('data/pd/09 myhom/train_data.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/09 myhom/train_data.csv', sep=',')
            logger.info("09_myhom loaded")
            return _data

        def _load_10_hackerearth():
            try:
                _data = pd.read_csv('data/pd/10 hackerearth/train_indessa.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/10 hackerearth/train_indessa.csv', sep=',')
            logger.info("10_hackerearth loaded")
            return _data

        def _load_11_cobranded():
            try:
                _data = pd.read_csv('data/pd/11 cobranded/Training_dataset_Original.csv', sep=',', low_memory=False)
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/11 cobranded/Training_dataset_Original.csv', sep=',', low_memory=False)
            logger.info("11_cobranded loaded")
            return _data

        def _load_14_german_credit():
            try:
                _data = pd.read_csv('data/pd/14 statlog german credit data/german.csv')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/14 statlog german credit data/german.csv')
            logger.info("14_german_credit loaded")
            return _data

        def _load_22_bank_status():
            try:
                _data = pd.read_csv('data/pd/22 bank loan status dataset/credit_train.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/22 bank loan status dataset/credit_train.csv', sep=',')
            return _data

        def _load_28_thomas():
            try:
                _data = pd.read_csv('data/pd/28 thomas/Loan Data.csv', sep=';')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/28 thomas/Loan Data.csv', sep=';')
            logger.info("28_thomas loaded")
            return _data

        def _load_29_loan_default():
            try:
                _data = pd.read_csv(
                    'data/pd/29 loan default predictions - imperial college london/train_v2.csv/train_v2.csv', sep=',', dtype=float)
            except FileNotFoundError:
                _data = pd.read_csv(
                    '../data/pd/29 loan default predictions - imperial college london/train_v2.csv/train_v2.csv',
                    sep=',', dtype=float)
            logger.info("29_loan_default loaded")
            return _data

        def _load_30_home_credit():
            try:
                _data = pd.read_csv(
                    'data/pd/30 home credit default risk/home-credit-default-risk/application_train.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv(
                    '../data/pd/30 home credit default risk/home-credit-default-risk/application_train.csv', sep=',')
            logger.info("30_home_credit loaded")
            return _data

        def _load_34_hmeq_data():
            try:
                _data = pd.read_csv('data/pd/34 hmeq/hmeq.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/034 hmeq/hmeq.csv', sep=',')
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
            try:
                _data = pd.read_csv(Path('') / 'lgd' / '01 heloc_lgd' / 'heloc_lgd.csv', low_memory=False)
            except FileNotFoundError:
                _data = pd.read_csv(Path('../..') / 'data' / 'lgd' / '01 heloc_lgd' / 'heloc_lgd.csv', low_memory=False)
            logger.info("01_heloc_lgd loaded")
            return _data

        def _load_03_loss2():
            try:
                _data = pd.read_csv(Path('') / 'lgd' / '03 loss2' / 'loss2.csv')
            except FileNotFoundError:
                _data = pd.read_csv(Path('../..') / 'data' / 'lgd' / '03 loss2' / 'loss2.csv')
            logger.info("03_loss2 loaded")
            return _data

        def _load_05_axa():
            try:
                _data = pd.read_csv(Path('') / 'lgd' / '05 lgd_axa' / 'lgd_axa.csv')
            except FileNotFoundError:
                _data = pd.read_csv(Path('../..') / 'data' / 'lgd' / '05 lgd_axa' / 'lgd_axa.csv')
            logger.info("05_axa loaded")
            return _data

        def _load_06_base_model():
            try:
                _data = pd.read_csv(Path('') / 'lgd' / '06 base_model' / 'base_model.csv')
            except FileNotFoundError:
                _data = pd.read_csv(Path('../..') / 'data' / 'lgd' / '06 base_model' / 'base_model.csv')
            logger.info("06_base_model loaded")
            return _data

        def _load_07_base_modelisation():
            try:
                _data = pd.read_csv(Path('') / 'lgd' / '07 base_modelisation' / 'base_modelisation.csv')
            except FileNotFoundError:
                _data = pd.read_csv(Path('../..') / 'data' / 'lgd' / '07 base_modelisation' / 'base_modelisation.csv')
            logger.info("07_base_modelisation loaded")
            return _data

        """ Dataset-specific loading calls """
        # for PD datasets:
        if self.experimentconfig['task'] == 'pd':
            if self.dataconfig['dataset_pd']['00_pd_toydata']:
                self.dataset_name = '00_pd_toydata'
                return _load_00_pd_toydata()
            elif self.dataconfig['dataset_pd']['01_gmsc']:
                self.dataset_name = '01_gmsc'
                return _load_01_gmsc()
            elif self.dataconfig['dataset_pd']['02_taiwan_creditcard']:
                self.dataset_name = '02_taiwan_creditcard'
                return _load_02_taiwan_creditcard()

            elif self.dataconfig['dataset_pd']['03_vehicle_loan']:
                self.dataset_name = '03_vehicle_loan'
                return _load_03_vehicle_loan()

            elif self.dataconfig['dataset_pd']['06_lendingclub']:
                self.dataset_name = '06_lendingclub'
                return _load_06_lendingclub()
            elif self.dataconfig['dataset_pd']['07_case_study']:
                self.dataset_name = '07_case_study'
                return _load_07_case_study()
            elif self.dataconfig['dataset_pd']['09_myhom']:
                self.dataset_name = '09_myhom'
                return _load_09_myhom()
            elif self.dataconfig['dataset_pd']['10_hackerearth']:
                self.dataset_name = '10_hackerearth'
                return _load_10_hackerearth()
            elif self.dataconfig['dataset_pd']['11_cobranded']:
                self.dataset_name = '11_cobranded'
                return _load_11_cobranded()
            #elif self.dataconfig['dataset_pd']['12_loan_defaulter']:
            #    self.dataset_name = '12_loan_defaulter'
            #    return _load_12_loan_defaulter()
            #elif self.dataconfig['dataset_pd']['13_loan_data_2017']:
            #    self.dataset_name = '13_loan_data_2017'
            #    return _load_13_loan_data_2017


            elif self.dataconfig['dataset_pd']['14_german_credit']:
                self.dataset_name = '14_german_credit'
                return _load_14_german_credit()
            elif self.dataconfig['dataset_pd']['22_bank_status']:
                self.dataset_name = '22_bank_status'
                return _load_22_bank_status()
            elif self.dataconfig['dataset_pd']['28_thomas']:
                self.dataset_name = '28_thomas'
                return _load_28_thomas()

            elif self.dataconfig['dataset_pd']['29_loan_default']:
                self.dataset_name = '29_loan_default'
                return _load_29_loan_default()

            elif self.dataconfig['dataset_pd']['30_home_credit']:
                self.dataset_name = '30_home_credit'
                return _load_30_home_credit()

            elif self.dataconfig['dataset_pd']['34_hmeq_data']:
                self.dataset_name = '34_hmeq_data'
                return _load_34_hmeq_data()

        # for LGD datasets:
        elif self.experimentconfig['task'] == 'lgd':
            if self.dataconfig['dataset_lgd']['00_lgd_toydata']:
                self.dataset_name = '00_lgd_toydata'
                return _load_00_lgd_toydata()

            elif self.dataconfig['dataset_lgd']['01_heloc']:
                self.dataset_name = '01_heloc'
                return _load_01_heloc_lgd()

            elif self.dataconfig['dataset_lgd']['03_loss2']:
                self.dataset_name = '03_loss2'
                return _load_03_loss2()

            elif self.dataconfig['dataset_lgd']['05_axa']:
                self.dataset_name = '05_axa'
                return _load_05_axa()

            elif self.dataconfig['dataset_lgd']['06_base_model']:
                self.dataset_name = '06_base_model'
                return _load_06_base_model()

            elif self.dataconfig['dataset_lgd']['07_base_modelisation']:
                self.dataset_name = '07_base_modelisation'
                return _load_07_base_modelisation()
        else:
            raise ValueError('Invalid task in experimentconfig, or no dataset selected in dataconfig')

    def subsample_if_necessary(self, _data):
        row_limit = self.experimentconfig.get('row_limit', 0)
        if not isinstance(row_limit, int) or row_limit <= 0:
            raise ValueError("row_limit must be a positive integer.")
        if isinstance(_data, pd.DataFrame):
            if len(_data) > row_limit:
                logger.info(f"Dataset has {len(_data)} rows. Subsampling to {row_limit} rows.")
                _data = _data.sample(n=row_limit, random_state=42).reset_index(drop=True)
            else:
                logger.info(f"Dataset has {len(_data)} rows. No subsampling needed.")
            return _data
        elif hasattr(_data, 'data') and hasattr(_data, 'target'):
            if _data.data.shape[0] > row_limit:
                idx = np.random.RandomState(seed=42).choice(_data.data.shape[0], row_limit, replace=False)
                _data.data = _data.data[idx]
                _data.target = _data.target[idx]
                logger.info(f"Subsampled toy dataset to {row_limit} rows.")
            return _data
        else:
            raise TypeError("Unsupported data format for subsampling.")