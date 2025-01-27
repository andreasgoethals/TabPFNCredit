import datetime
import itertools
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier

# Proprietary imports
from src.classes.data import Data


class Experiment:
    def __init__(self, dataconfig, experimentconfig, methodconfig, evaluationconfig):

        now = datetime.datetime.now().strftime('%d-%m-%y_%H-%M')
        print('\nExperiment created at: ', now)

        self.dataconfig = dataconfig
        self.experimentconfig = experimentconfig
        self.methodconfig = methodconfig
        self.evaluationconfig = evaluationconfig

        #todo: based on CONFIG_DATA.yaml, Assess that only one dataset can be set selected for the classes (wrte this in a way that modularizes the code)

        # Initialize attributes with empty nd arrays:
        self.data = Data(dataconfig, experimentconfig)
        #self.x = np.empty((0, 0))
        #self.y = np.empty(0)
        #self.split_indices = {} # To store split indices for train/val/test

    def run(self):
        # Load and preprocess data
        self.data.load_preprocess_data()

        # Split data
        self.data.split_data()

        self.train_evaluate()

    def train_evaluate(self):

        #create a dictionary to store the results, based on the number of splits and the evaluation metrics in CONFIG_EVALUATION.yaml
        results = {}

        # Loop over the splits
        for fold, indices in self.data.split_indices.items():
            train_idx = indices['train']
            val_idx = indices['val']
            test_idx = indices['test']

            x_train, y_train = self.data.x[train_idx], self.data.y[train_idx]
            x_val, y_val = self.data.x[val_idx], self.data.y[val_idx]
            x_test, y_test = self.data.x[test_idx], self.data.y[test_idx]

            hyperparams = self._get_optimal_hyperparameters(self, fold, indices)

            # Train the model
            # todo: do this in separate function (and gerenaliize it for all models)
            model = DecisionTreeClassifier(random_state=0, **hyperparams)
            model.fit(x_train, y_train)

            y_pred_proba = model.predict_proba(x_test)

            # Evaluate the model in separate function, based on CONFIG_EVALUATION.yaml
            # todo: implement evaluation function, taking into account the different splits
            results[fold] = self._evaluate(y_test, y_pred_proba)




    def _get_optimal_hyperparameters(self, fold, indices):
        # if tuning is required, call the hyperparameter tuning function
        if self.methodconfig['tune_hyperparameters']:
            return self._tune_hyperparameters(fold, indices)
        # else, return the hyperparameters from the config
        else:
            # todo: read hyperparameters from self.methodconfig
            return

    def _tune_hyperparameters(self, fold, indices):
        # read hyperparameters in self.methodconfig
        hyperparams = self.methodconfig['hyperparameters']

        param_names = list(hyperparams.keys())
        param_combinations = list(itertools.product(*hyperparams.values()))

        x_train, y_train = self.data.x[indices['train']], self.data.y[indices['train']]
        x_val, y_val = self.data.x[indices['val']], self.data.y[indices['val']]

        best_model = None
        best_accuracy = 0

        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            model = DecisionTreeClassifier(**param_dict)
            model.fit(x_train, y_train)

            # todo: implement evaluation function for hyperpar  tuning
            accuracy = accuracy_score(model.predict(x_val), y_val)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        print(f"Best hyperparameters: {best_model.get_params()}")
        return best_model

    def _evaluate(self, y_test, y_pred_proba):
        t = self.evaluationconfig['threshold']
        y_pred = (y_pred_proba[:, 1] > t).astype(int)


        pass


