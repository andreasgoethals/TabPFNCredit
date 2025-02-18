import datetime
import itertools

from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, brier_score_loss, \
    average_precision_score

from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

# Proprietary imports
from src.classes.data import Data
from src.classes.preprocessing import standardize_data, encode_cat_vars, handle_missing_values
from src.utils import _assert_dataconfig, _assert_experimentconfig, _assert_methodconfig, _assert_evaluationconfig


class Experiment:
    def __init__(self, dataconfig, experimentconfig, methodconfig, evaluationconfig):

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        print('\nExperiment created at: ', now)

        self.experimentconfig = experimentconfig
        _assert_experimentconfig(experimentconfig)

        #todo: based on CONFIG_DATA.yaml, Assess that only one dataset can be set selected for the classes (wrte this in a way that modularizes the code)
        self.dataconfig = dataconfig
        _assert_dataconfig(dataconfig, experimentconfig)

        self.methodconfig = methodconfig
        _assert_methodconfig(methodconfig)

        self.evaluationconfig = evaluationconfig
        _assert_evaluationconfig(evaluationconfig)

        # Initialize attributes with empty nd arrays:
        self.data = Data(dataconfig, experimentconfig)
        self.results = {}  # To store results of the experiment

        task = self.experimentconfig['task']
        print(f'Task: {task}')
        for key, value in self.dataconfig[f'dataset_{task}'].items():
            if value:
                print(f'Dataset: {key}')
        print('CV splits: ', self.experimentconfig['cv_splits'])

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
        for fold, indices in tqdm(self.data.split_indices.items(), desc="Cross-validation loop:"):

            train_idx = indices['train']
            val_idx = indices['val']
            test_idx = indices['test']

            x_train, y_train = self.data.x[train_idx], self.data.y[train_idx]
            x_val, y_val = self.data.x[val_idx], self.data.y[val_idx]
            x_test, y_test = self.data.x[test_idx], self.data.y[test_idx]

            # solve nan values
            x_train, x_val, x_test, y_train, y_val, y_test = handle_missing_values(x_train, x_val, x_test, y_train, y_val, y_test, self.methodconfig)

            # encode categorical variables
            x_train, x_val, x_test, y_train, y_val, y_test = encode_cat_vars(x_train, x_val, x_test, y_train, y_val, y_test, self.methodconfig, self.data.cols_cat, self.data.cols_cat_idx)

            # standardize the data
            x_train, x_val, x_test, y_train, y_val, y_test = standardize_data(x_train, x_val, x_test, y_train, y_val, y_test, self.methodconfig)

            # store the data in the data object - this is done per split
            self.data.x_train, self.data.y_train = x_train, y_train
            self.data.x_val, self.data.y_val = x_val, y_val
            self.data.x_test, self.data.y_test = x_test, y_test

            # Loop over selected methods based on config_method
            for method, use_method in self.methodconfig['methods_pd'].items():
                # Check if the method is selected
                if use_method:

                    optimal_hyperparams = self._get_optimal_hyperparameters(fold, indices, method)

                    # Train the model with optimal hyperparameters
                    if method == 'ab':
                        model = AdaBoostClassifier(random_state=0, **optimal_hyperparams)
                        model.fit(x_train, y_train)
                        y_pred_proba = model.predict_proba(x_test)
                        y_pred_proba = y_pred_proba[:, 1]

                    elif method == 'ann':
                        # todo: implement custom neural network class
                        pass

                    elif method == 'bnb':
                        model = BernoulliNB(**optimal_hyperparams)
                        model.fit(x_train, y_train)
                        y_pred_proba = model.predict_proba(x_test)
                        y_pred_proba = y_pred_proba[:, 1]

                    elif method == 'cb':
                        model = CatBoostClassifier(random_state=0, **optimal_hyperparams)
                        model.fit(x_train, y_train)
                        y_pred_proba = model.predict_proba(x_test)
                        y_pred_proba = y_pred_proba[:, 1]

                    elif method == 'dt':
                        model = DecisionTreeClassifier(random_state=0, **optimal_hyperparams)
                        model.fit(x_train, y_train)

                        y_pred_proba = model.predict_proba(x_test)
                        y_pred_proba = y_pred_proba[:, 1]  # dt in sklearn returns probabilities for each class; select the probability of the positive class

                    elif method == 'gnb':
                        model = GaussianNB()
                        model.fit(x_train, y_train)
                        y_pred_proba = model.predict_proba(x_test)
                        y_pred_proba = y_pred_proba[:, 1]

                    elif method == 'knn':
                        model = KNeighborsClassifier(**optimal_hyperparams)
                        model.fit(x_train, y_train)
                        y_pred_proba = model.predict_proba(x_test)
                        y_pred_proba = y_pred_proba[:, 1]

                    elif method == 'lda':
                        model = LinearDiscriminantAnalysis()
                        model.fit(x_train, y_train)
                        y_pred_proba = model.predict_proba(x_test)
                        y_pred_proba = y_pred_proba[:, 1]
                    elif method == 'lgbm':
                        model = LGBMClassifier(random_state=0, verbose=-1, **optimal_hyperparams)
                        model.fit(x_train, y_train)
                        y_pred_proba = model.predict_proba(x_test)
                        y_pred_proba = y_pred_proba[:, 1]

                    elif method == 'lr':
                        model = LogisticRegression(random_state=0, **optimal_hyperparams)
                        y_pred_proba = model.fit(x_train, y_train).predict_proba(x_test)[:, 1]

                    elif method == 'rf':
                        model = RandomForestClassifier(random_state=0, **optimal_hyperparams)
                        model.fit(x_train, y_train)
                        y_pred_proba = model.predict_proba(x_test)
                        y_pred_proba = y_pred_proba[:, 1]

                    elif method == 'svm':
                        model = SVC(random_state=0, probability=True, **optimal_hyperparams)
                        model.fit(x_train, y_train)
                        y_pred_proba = model.predict_proba(x_test)
                        y_pred_proba = y_pred_proba[:, 1]

                    elif method == 'xgb':
                        model = XGBClassifier(random_state=0, **optimal_hyperparams)
                        model.fit(x_train, y_train)
                        y_pred_proba = model.predict_proba(x_test)
                        y_pred_proba = y_pred_proba[:, 1]


                    # Evaluate the model in separate function, based on CONFIG_EVALUATION.yaml
                    # todo: implement evaluation function, taking into account the different splits
                    # Evaluate the model
                    if method not in results:
                        results[method] = {}
                    #todo: check that the evaluation function is implemented correctly (with key and value)
                    results[method][fold] = self._evaluate(y_test, y_pred_proba)

        self.results = results

    def _get_optimal_hyperparameters(self, fold, indices, method):
        """
        This function returns the optimal hyperparameters for the model; either from the config file or by tuning them
        :param fold:
        :param indices:
        :param method:
        :return:
        """

        # if tuning is required, call the hyperparameter tuning function
        if self.methodconfig['tune_hyperparameters']:
            optimal_hyperparameters = self._tune_hyperparameters(fold, indices, method)
            return optimal_hyperparameters

        # else, return the hyperparameters from the config
        else:
            # todo: read hyperparameters from self.methodconfig
            optimal_hyperparameters = self._read_hyperparameters_from_config(method)
            return optimal_hyperparameters

    def _tune_hyperparameters(self, fold, indices, method):
        # read hyperparameters in self.methodconfig
        hyperpara_grid = self.methodconfig['hyperparameters'][method]

        param_names = list(hyperpara_grid.keys())

        # enumerate all combinations of in hyperpara_grid:
        param_combinations = list(itertools.product(*hyperpara_grid.values()))

        # todo: old method: work with indices (requires new missing value handling, encoding, standardization)
        #x_train, y_train = self.data.x[indices['train']], self.data.y[indices['train']]
        #x_val, y_val = self.data.x[indices['val']], self.data.y[indices['val']]
        # todo: new method: work with x_train, x_val, y_train, y_val (as stored in data object) ; updated per split
        x_train, y_train = self.data.x_train, self.data.y_train
        x_val, y_val = self.data.x_val, self.data.y_val

        best_model = None
        best_score = 0

        # loop over all combinations of hyperparameters
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))

            if method == 'ab':
                model = AdaBoostClassifier(**param_dict)
            elif method == 'ann':
                #todo: implement

                # raise warning that 'ann' has not been implmeneted yet:

                pass
            elif method == 'bnb':
                model = BernoulliNB(**param_dict)
            elif method == 'cb':
                model = CatBoostClassifier(**param_dict)
            elif method == 'dt':
                model = DecisionTreeClassifier(**param_dict)
            elif method == 'gnb':
                model = GaussianNB(**param_dict)
            elif method == 'knn':
                model = KNeighborsClassifier(**param_dict)
            elif method == 'lda':
                model = LinearDiscriminantAnalysis(**param_dict)
            elif method == 'lgbm':
                model = LGBMClassifier(verbose=-1, **param_dict)
            elif method == 'lr':
                model = LogisticRegression(**param_dict)
            elif method == 'rf':
                model = RandomForestClassifier(**param_dict)
            elif method == 'svm':
                model = SVC(probability=True, **param_dict)
            elif method == 'xgb':
                model = XGBClassifier(**param_dict)

            model.fit(x_train, y_train)

            # todo: implement evaluation function for hyperpar  tuning
            # todo: other evaluatio metric for model selection?
            score = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])

            if score > best_score:
                best_score = score
                best_model = model

        _optimal_hyperparameters = best_model.get_params()
        print(f"*Best hyperparameters ({method})* {_optimal_hyperparameters}")

        # only keep the hyperparameters that are in the grid (stored in param_names):
        _optimal_hyperparameters = {k: v for k, v in _optimal_hyperparameters.items() if k in param_names}

        return _optimal_hyperparameters

    def _evaluate(self, y_test, y_pred_proba):
        _results = {}

        t = float(self.evaluationconfig['binary_threshold'])  # Ensure t is a float
        y_pred_proba = y_pred_proba.astype(float)  # Ensure y_pred_proba is a float array

        y_pred = (y_pred_proba > t).astype(int)

        if self.evaluationconfig['metrics_pd']['accuracy']:
            accuracy = accuracy_score(y_test, y_pred)
            _results['accuracy'] = accuracy.__round__(self.evaluationconfig['round_digits'])

        # Threshold-dependent metrics:

        if self.evaluationconfig['metrics_pd']['brier']:
            brier = brier_score_loss(y_test, y_pred_proba)
            _results['brier'] = brier.__round__(self.evaluationconfig['round_digits'])

        if self.evaluationconfig['metrics_pd']['f1']:
            f1 = f1_score(y_test, y_pred)
            _results['f1'] = f1.__round__(self.evaluationconfig['round_digits'])

        if self.evaluationconfig['metrics_pd']['precision']:
            precision = precision_score(y_test, y_pred)
            _results['precision'] = precision.__round__(self.evaluationconfig['round_digits'])

        if self.evaluationconfig['metrics_pd']['recall']:
            recall = precision_score(y_test, y_pred)
            _results['recall'] = recall.__round__(self.evaluationconfig['round_digits'])

        # Threshold-independent metrics:

        if self.evaluationconfig['metrics_pd']['aucroc']:
            aucroc = roc_auc_score(y_test, y_pred)
            _results['aucroc'] = aucroc.__round__(self.evaluationconfig['round_digits'])

        if self.evaluationconfig['metrics_pd']['aucpr']:
            aucpr = average_precision_score(y_test, y_pred)
            _results['aucpr'] = aucpr.__round__(self.evaluationconfig['round_digits'])

        # cost-sensitive metrics:

        # todo: implement cost-sensitive metrics: EMP?

        # return a dictionary with the evaluation metrics,
        # to be stored in the results dictionary
        return _results

    def _read_hyperparameters_from_config(self, method):
        """
        This function reads the hyperparameters from the config file - if hyperparameters are tuned in previous run
        :param method:
        :return:
        """
        # todo: read hyperparameters from self.methodconfig; depends on both method and dataset
        pass


