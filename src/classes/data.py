from typing import Dict, List

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold


class Data:
    def __init__(self, dataconfig, experimentconfig):
        self.dataconfig = dataconfig
        self.experimentconfig = experimentconfig

        self.x = np.empty((0, 0))
        self.y = np.empty(0)

        self.cols = []
        self.cols_cat = []
        self.cols_num = []

        self.split_indices = {}

        self.dataset_name = None


    def load_preprocess_data(self):

        _data = self._load_data()
        self.x, self.y, self.cols, self.cols_cat, self.cols_num = self._preprocess_data(_data)


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

    def _load_data(self):

        """ method definitions for pd datasets """

        # Dataset-specific loading functions
        def _load_00_pd_toydata():
            # Load the toy data from sklearn
            _data = load_breast_cancer()
            print("00_pd_toydata loaded")
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
            # todo: implement the preprocessing for gmsc data
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

        elif self.experimentconfig['task'] == 'lgd':
            if self.dataconfig['dataset_lgd']['01_heloc']:
                _preprocess_01_heloc(_data)



