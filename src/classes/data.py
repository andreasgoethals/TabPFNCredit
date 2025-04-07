from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold

from src.classes.preprocessing import Preprocessing


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
        _data = self.preprocessing.load_data()
        self.x, self.y, self.cols, self.cols_cat, self.cols_num, self.cols_cat_idx, self.cols_num_idx = self.preprocessing.preprocess_data(_data)

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
    '''
    def _load_data(self):

        """ method definitions for pd datasets """

        # Dataset-specific loading functions
        def _load_00_pd_toydata():
            # Load the toy data from sklearn
            _data = load_breast_cancer()
            print("00_pd_toydata loaded")
            return _data

        def _load_01_gmsc():
            _data = pd.read_csv('data/pd/01 kaggle_give me some credit/gmsc.csv')
            print("01_gmsc loaded")
            return _data

        def _load_02_taiwan_creditcard():
            try:
                _data = pd.read_csv('data/pd/02 taiwan creditcard/taiwan_creditcard.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/02 taiwan creditcard/taiwan_creditcard.csv', sep=',')
            print("02_taiwan_creditcard loaded")
            return _data


        """ method definitions for lgd datasets """

        def _load_01_heloc_lgd():
            _data = 'this is a 01 data to be implemetned'
            print("01_heloc_lgd loaded")
            return _data

        """ Dataset-specific loading calls """

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

        elif self.experimentconfig['task'] == 'lgd':
            if self.dataconfig['dataset']['01_heloc_lgd']:
                self.dataset_name = '01_heloc_lgd'
                return _load_01_heloc_lgd()

        else:
            raise ValueError('Invalid task in experimentconfig, or no dataset selected in dataconfig')

    def _preprocess_data(self, _data):

        """ method definitions for preprocessing pd datasets """

        def _preprocess_00_pd_toydata(_data):
            x = _data.data
            y = _data.target
            print("00_pd_toydata preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)
            cols = _data.feature_names
            cols_cat = []
            cols_num = cols
            return x, y, cols, cols_cat, cols_num

        def _preprocess_01_gmsc(_data):
            _data = _data.drop(_data.columns[0], axis=1) # drop first column:

            cols = _data.columns

            x = _data['SeriousDlqin2yrs']
            y = _data.drop('SeriousDlqin2yrs', axis=1)

            cols_cat = []
            cols_num = cols

            print("01_gmsc preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num

        def _preprocess_02_taiwan_creditcard(_data):

            return x, y, cols, cols_cat, cols_num

        """ method definitions for preprocessing lgd datasets """

        def _preprocess_00_lgd_toydata(_data):
            #todo: implement the preprocessing for lgd toydata
            return x, y, cols, cols_cat, cols_num

        def _preprocess_01_heloc(data):
            # todo: implement the preprocessing for heloc data
            return x, y, cols, cols_cat, cols_num

        """ Dataset-specific preprocessing calls """

        if self.experimentconfig['task'] == 'pd':
            if self.dataconfig['dataset_pd']['00_pd_toydata']:
                x, y, cols, cols_cat, cols_num = _preprocess_00_pd_toydata(_data)
                return x, y, cols, cols_cat, cols_num
            elif self.dataconfig['dataset_pd']['01_gmsc']:
                x, y, cols, cols_cat, cols_num = _preprocess_01_gmsc(_data)
                return x, y, cols, cols_cat, cols_num
            elif self.dataconfig['dataset_pd']['02_taiwan_creditcard']:
                x, y, cols, cols_cat, cols_num = _preprocess_02_taiwan_creditcard(_data)
                return x, y, cols, cols_cat, cols_num

        elif self.experimentconfig['task'] == 'lgd':
            if self.dataconfig['dataset_lgd']['01_heloc']:
                _preprocess_01_heloc(_data)
    '''