import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
from collections import Counter
from typing import Dict, List
import logging
import category_encoders

logger = logging.getLogger(__name__)

""" Generic method definitions """

def handle_missing_values(x_train, x_val, x_test, y_train, y_val, y_test, methodconfig: Dict, cols_num_idx, cols_cat_idx):
    if methodconfig['missing_values'] == 0:
        # don't handle missing values
        pass

    elif methodconfig['missing_values'] == 1:

        # Store original lengths
        original_train_size = len(x_train)
        original_val_size = len(x_val)
        original_test_size = len(x_test)

        # Convert to pandas DataFrame and drop rows with missing values
        x_train_df = pd.DataFrame(x_train).dropna()
        x_val_df = pd.DataFrame(x_val).dropna()
        x_test_df = pd.DataFrame(x_test).dropna()

        # Compute number of dropped rows
        dropped_train = original_train_size - len(x_train_df)
        dropped_val = original_val_size - len(x_val_df)
        dropped_test = original_test_size - len(x_test_df)

        # Drop corresponding rows from the target variable
        y_train = np.array(y_train)[x_train_df.index]
        y_val = np.array(y_val)[x_val_df.index]
        y_test = np.array(y_test)[x_test_df.index]

        # Convert x_train, x_val, x_test back to numpy arrays
        x_train = x_train_df.to_numpy()
        x_val = x_val_df.to_numpy()
        x_test = x_test_df.to_numpy()

        total_dropped_rows = dropped_train + dropped_val + dropped_test

        logger.info(f"Omitting rows with missing values: {total_dropped_rows} rows left out")

    elif methodconfig['missing_values'] == 2:
        # impute numeric with mean:
        # make check that cols_num_idx is not empty:
        if len(cols_num_idx)!=0:
            imputer_num = SimpleImputer(strategy='mean')
            x_train[:,cols_num_idx] = imputer_num.fit_transform(x_train[:,cols_num_idx])
            x_val[:,cols_num_idx] = imputer_num.transform(x_val[:,cols_num_idx])
            x_test[:,cols_num_idx] = imputer_num.transform(x_test[:,cols_num_idx])

        # impute categorical with most frequent:
        # make check that cols_cat_idx is not empty:
        if len(cols_cat_idx) != 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            x_train[:, cols_cat_idx] = imputer_cat.fit_transform(x_train[:, cols_cat_idx])
            x_val[:, cols_cat_idx] = imputer_cat.transform(x_val[:, cols_cat_idx])
            x_test[:, cols_cat_idx] = imputer_cat.transform(x_test[:, cols_cat_idx])

        logger.info('Imputed missing values with (num: mean) and (cat: mode)')

    elif methodconfig['missing_values'] == 3:
        # impute with median:
        if len(cols_num_idx) != 0: # check that cols_num_idx is not empty:
            imputer_num = SimpleImputer(strategy='median')
            x_train[:, cols_num_idx] = imputer_num.fit_transform(x_train[:, cols_num_idx])
            x_val[:, cols_num_idx] = imputer_num.transform(x_val[:, cols_num_idx])
            x_test[:, cols_num_idx] = imputer_num.transform(x_test[:, cols_num_idx])

        # impute categorical with most frequent:
        # make check that cols_cat_idx is not empty:
        if len(cols_cat_idx) != 0:

            imputer_cat = SimpleImputer(strategy='most_frequent')
            x_train[:, cols_cat_idx] = imputer_cat.fit_transform(x_train[:, cols_cat_idx])
            x_val[:, cols_cat_idx] = imputer_cat.transform(x_val[:, cols_cat_idx])
            x_test[:, cols_cat_idx] = imputer_cat.transform(x_test[:, cols_cat_idx])

        logger.info('Imputed missing values with (num: median) and (cat: mode)')

    else:
        # throw error that the methodconfig is not valid
        raise ValueError('Invalid methodconfig for handling missing data; change CONFIG_METHOD.yaml')

    return x_train, x_val, x_test, y_train, y_val, y_test

def encode_cat_vars(x_train, x_val, x_test, y_train, y_val, y_test, methodconfig: Dict, cols_cat: List, cols_cat_idx: List):

    # check if there are any categorical variables to encode;
    # if not, return the data as is
    if len(cols_cat)==0:
        return x_train, x_val, x_test, y_train, y_val, y_test

    elif methodconfig['encode_cat']==0:
        # don't encode categorical variables
        pass

    elif methodconfig['encode_cat'] == 1:
        # one-hot encode categorical variables
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        x_train_cat = encoder.fit_transform(x_train[:, cols_cat_idx])
        x_val_cat = encoder.transform(x_val[:, cols_cat_idx])
        x_test_cat = encoder.transform(x_test[:, cols_cat_idx])

        # Concatenate the encoded categorical columns with the numerical columns
        x_train = np.concatenate([x_train[:, ~np.isin(range(x_train.shape[1]), cols_cat_idx)], x_train_cat], axis=1)
        x_val = np.concatenate([x_val[:, ~np.isin(range(x_val.shape[1]), cols_cat_idx)], x_val_cat], axis=1)
        x_test = np.concatenate([x_test[:, ~np.isin(range(x_test.shape[1]), cols_cat_idx)], x_test_cat], axis=1)

    elif methodconfig['encode_cat'] == 2:

        # if y is not binary, then raise warning that WoE only works for binary targets:
        if len(np.unique(y_train))>2:
            warnings.warn('Weight of Evidence encoding is only applicable for binary targets; change settings in CONFIG_METHOD.yaml')

        # implement weight of evidence encoding:
        encoder = category_encoders.WOEEncoder()
        x_train_cat = encoder.fit_transform(x_train[:, cols_cat_idx], y_train)
        x_val_cat = encoder.transform(x_val[:, cols_cat_idx])
        x_test_cat = encoder.transform(x_test[:, cols_cat_idx])

        # Concatenate the encoded categorical columns with the numerical columns
        x_train = np.concatenate([x_train[:, ~np.isin(range(x_train.shape[1]), cols_cat_idx)], x_train_cat], axis=1)
        x_val = np.concatenate([x_val[:, ~np.isin(range(x_val.shape[1]), cols_cat_idx)], x_val_cat], axis=1)
        x_test = np.concatenate([x_test[:, ~np.isin(range(x_test.shape[1]), cols_cat_idx)], x_test_cat], axis=1)

    else:
        # throw error that the methodconfig is not valid
        raise ValueError('Invalid methodconfig for encoding categorical data; change CONFIG_METHOD.yaml')

    return x_train, x_val, x_test, y_train, y_val, y_test

def standardize_data(x_train, x_val, x_test, y_train, y_val, y_test, methodconfig: Dict):

    if methodconfig['standardize']==0:
        # don't standardize data
        pass

    elif methodconfig['standardize'] == 1:
        # use StandardScaler
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    elif methodconfig['standardize'] == 2:
        # use MinMaxScaler
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    else:
        # throw error that the methodconfig is not valid
        raise ValueError('Invalid methodconfig for standardizing data; change CONFIG_METHOD.yaml')

    return x_train, x_val, x_test, y_train, y_val, y_test

def _introduce_class_imbalance(x, y, imbalance_ratio=0.1, random_state=0):
    """
    Create artificial class imbalance in the dataset. Downsamples or oversamples
    the minority class to reach the desired imbalance ratio.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    imbalance_ratio : float
        Desired minority class ratio (e.g., 0.1 for 10% minority).
    random_state : int
        Random seed.

    Returns
    -------
    x_new, y_new : np.ndarray
        New dataset with induced class imbalance.
    """
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution BEFORE imbalance: {dict(zip(unique, counts))}")

    # Identify majority and minority classes
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]
    idx_major = np.where(y == majority_class)[0]
    idx_minor = np.where(y == minority_class)[0]

    current_ratio = len(idx_minor) / len(idx_major)
    logger.info(f"Current minority ratio: {current_ratio:.2f}, Desired: {imbalance_ratio}")

    rng = np.random.RandomState(random_state)

    # CASE 1: UNDERSAMPLING (if desired ratio < current)
    if imbalance_ratio < current_ratio:
        n_major = len(idx_major)
        n_minor_new = int(n_major * imbalance_ratio)
        n_minor_new = min(len(idx_minor), n_minor_new)

        idx_minor_sampled = rng.choice(idx_minor, size=n_minor_new, replace=False)
        idx_combined = np.concatenate([idx_major, idx_minor_sampled])
        rng.shuffle(idx_combined)
        x_new, y_new = x[idx_combined], y[idx_combined]

    # CASE 2: OVERSAMPLING (if desired ratio > current)
    else:
        smote = SMOTE(sampling_strategy=imbalance_ratio, random_state=random_state)
        x_new, y_new = smote.fit_resample(x, y)

    # Log final distribution
    final_counts = dict(Counter(y_new))
    logger.info(f"Class distribution AFTER imbalance: {final_counts}")
    logger.info(f"Total samples: {len(y_new)}\n")
    return x_new, y_new