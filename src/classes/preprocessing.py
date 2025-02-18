from typing import Dict, List

import category_encoders
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

""" Class definition and class-specific method definitions """
class Preprocessing:
    def __init__(self, dataconfig, experimentconfig):
        self.dataconfig = dataconfig
        self.experimentconfig = experimentconfig
        self.dataset_name = None

    def load_data(self):
        """ method definitions for pd datasets """
        def _load_00_pd_toydata():
            _data = load_breast_cancer()
            self.dataset_name = '00_pd_toydata'
            print("00_pd_toydata loaded")
            return _data

        def _load_01_gmsc():
            _data = pd.read_csv('data/pd/01 kaggle_give me some credit/gmsc.csv')
            self.dataset_name = '01_gmsc'
            print("01_gmsc loaded")
            return _data

        def _load_02_taiwan_creditcard():
            try:
                _data = pd.read_csv('data/pd/02 taiwan creditcard/taiwan_creditcard.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../../data/pd/02 taiwan creditcard/taiwan_creditcard.csv', sep=',')
            print("02_taiwan_creditcard loaded")
            return _data

        def _load_03_vehicle_loan():
            try:
                _data = pd.read_csv('data/pd/03 vehicle loan/train.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/03 vehicle loan/train.csv', sep=',')
            print("03_vehicle_loan loaded")
            return _data

        def _load_34_hmeq_data():
            try:
                _data = pd.read_csv('data/pd/34 hmeq/hmeq.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/034 hmeq/hmeq.csv', sep=',')
            print("34_hmeq_data loaded")
            return _data


        """ method definitions for lgd datasets """
        def _load_01_heloc_lgd():
            _data = 'this is a 01 data to be implemented'
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

            elif self.dataconfig['dataset_pd']['03_vehicle_loan']:
                self.dataset_name = '03_vehicle_loan'
                return _load_03_vehicle_loan()

            elif self.dataconfig['dataset_pd']['34_hmeq_data']:
                self.dataset_name = '34_hmeq_data'
                return _load_34_hmeq_data()

        elif self.experimentconfig['task'] == 'lgd':
            if self.dataconfig['dataset']['01_heloc_lgd']:
                self.dataset_name = '01_heloc_lgd'
                return _load_01_heloc_lgd()
        else:
            raise ValueError('Invalid task in experimentconfig, or no dataset selected in dataconfig')

    def preprocess_data(self, _data):
        """ method definitions for preprocessing pd datasets """
        def _preprocess_00_pd_toydata(_data):
            x = _data.data
            y = _data.target

            cols = list(_data.feature_names)
            cols_cat = []
            cols_num = cols
            cols_cat_idx = []
            cols_num_idx = list(range(len(cols)))

            print("00_pd_toydata preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_01_gmsc(_data: pd.DataFrame) -> tuple[
            np.ndarray, np.ndarray, list[str], list[str], list[str], list[int], list[int]]:

            y = _data['SeriousDlqin2yrs'].values.astype(int)
            x = _data.drop('SeriousDlqin2yrs', axis=1).values

            cols = list(_data.drop('SeriousDlqin2yrs', axis=1).columns)

            cols_cat = []
            cols_num = cols

            cols_cat_idx = []
            cols_num_idx = list(range(len(cols)))

            print("01_gmsc preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx


        def _preprocess_02_taiwan_creditcard(_data):

            # Drop ID and useless columns
            _data = _data.drop('ID', axis=1)

            # Transform
            _data['SEX'] = _data['SEX'].replace({'2': 1, '1': 0})

            # Split into covariates, labels
            y = _data['default.payment.next.month'].values.astype(int)
            x = _data.drop('default.payment.next.month', axis=1).values

            cols = list(_data.drop('default.payment.next.month', axis=1).columns)

            cols_cat = []
            cols_num = cols

            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("02_taiwan_creditcard preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_03_vehicle_loan(_data):

            #todo: currently performance is bad. Check this link for extra preprocessing? https://www.kaggle.com/code/jagannathrk/vehicle-loan-default-prediction

            # Drop ID and useless columns
            _data = _data.drop('UniqueID', axis=1)  # unique identifier
            _data = _data.drop('branch_id', axis=1)  # 82 unique categorical values
            _data = _data.drop('supplier_id', axis=1)  # 2953 unique categorical values
            _data = _data.drop('Current_pincode_ID', axis=1)  # 6698 unique categorical values
            _data = _data.drop('Employee_code_ID', axis=1)  # 3270 unique categorical values
            _data = _data.drop('MobileNo_Avl_Flag', axis=1)  # 1 unique value

            def age(dob):
                yr = int(dob[-2:])
                if yr >= 0 and yr < 20:
                    return yr + 2000
                else:
                    return yr + 1900

            _data['Date.of.Birth'] = _data['Date.of.Birth'].apply(age)
            _data['DisbursalDate'] = _data['DisbursalDate'].apply(age)
            _data['Age'] = _data['DisbursalDate'] - _data['Date.of.Birth']
            _data = _data.drop(['DisbursalDate', 'Date.of.Birth'], axis=1)

            def transform_PERFORM_CNS_SCORE_DESCRIPTION(df):
                # Replacing all the values into Common Group
                df.replace({'PERFORM_CNS.SCORE.DESCRIPTION': {
                    'C-Very Low Risk': 'Very Low Risk',
                    'A-Very Low Risk': 'Very Low Risk',
                    'D-Very Low Risk': 'Very Low Risk',
                    'B-Very Low Risk': 'Very Low Risk',
                    'M-Very High Risk': 'Very High Risk',
                    'L-Very High Risk': 'Very High Risk',
                    'F-Low Risk': 'Low Risk',
                    'E-Low Risk': 'Low Risk',
                    'G-Low Risk': 'Low Risk',
                    'H-Medium Risk': 'Medium Risk',
                    'I-Medium Risk': 'Medium Risk',
                    'J-High Risk': 'High Risk',
                    'K-High Risk': 'High Risk'
                }}, inplace=True)

                # Transforming them into Numeric Features
                risk_map = {
                    'No Bureau History Available': -1,
                    'Not Scored: No Activity seen on the customer (Inactive)': -1,
                    'Not Scored: Sufficient History Not Available': -1,
                    'Not Scored: No Updates available in last 36 months': -1,
                    'Not Scored: Only a Guarantor': -1,
                    'Not Scored: More than 50 active Accounts found': -1,
                    'Not Scored: Not Enough Info available on the customer': -1,
                    'Very Low Risk': 4,
                    'Low Risk': 3,
                    'Medium Risk': 2,
                    'High Risk': 1,
                    'Very High Risk': 0
                }

                df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].map(risk_map)

            transform_PERFORM_CNS_SCORE_DESCRIPTION(_data)

            def convert_to_months(value):
                years, months = value.split(' ')
                years = int(years.replace('yrs', ''))
                months = int(months.replace('mon', ''))
                return years * 12 + months

            _data['AVERAGE.ACCT.AGE'] = _data['AVERAGE.ACCT.AGE'].apply(convert_to_months)
            _data['CREDIT.HISTORY.LENGTH'] = _data['CREDIT.HISTORY.LENGTH'].apply(convert_to_months)

            # Split into covariates, labels
            y = _data['loan_default'].values.astype(int)
            x = _data.drop('loan_default', axis=1).values

            cols = list(_data.drop('loan_default', axis=1).columns)

            cols_cat = ['manufacturer_id', 'Employment.Type', 'State_ID']

            cols_num = ['disbursed_amount', 'asset_cost', 'ltv', 'Aadhar_flag', 'PAN_flag',
                        'VoterID_flag', 'Driving_flag', 'Passport_flag', 'PERFORM_CNS.SCORE',
                        'PERFORM_CNS.SCORE.DESCRIPTION', 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',
                        'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',
                        'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS',
                        'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',
                        'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT',
                        'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
                        'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 'NO.OF_INQUIRIES', 'Age']

            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("03_vehicle_loan preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_34_hmeq_data(_data):

            # Split into covariates, labels
            y = _data['BAD'].values.astype(int)
            x = _data.drop('BAD', axis=1).values

            cols = list(_data.drop('BAD', axis=1).columns)

            cols_cat = ['REASON', 'JOB']
            cols_num = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx


        """ method definitions for preprocessing lgd datasets """
        def _preprocess_00_lgd_toydata(_data):
            # todo: implement the preprocessing for lgd toydata
            return x, y, cols, cols_cat, cols_num

        def _preprocess_01_heloc(data):
            # todo: implement the preprocessing for heloc data
            return x, y, cols, cols_cat, cols_num

        """ Dataset-specific preprocessing calls """
        if self.experimentconfig['task'] == 'pd':

            if self.dataconfig['dataset_pd']['00_pd_toydata']:
                return _preprocess_00_pd_toydata(_data)
            elif self.dataconfig['dataset_pd']['01_gmsc']:
                return _preprocess_01_gmsc(_data)
            elif self.dataconfig['dataset_pd']['02_taiwan_creditcard']:
                return _preprocess_02_taiwan_creditcard(_data)
            elif self.dataconfig['dataset_pd']['03_vehicle_loan']:
                return _preprocess_03_vehicle_loan(_data)
            elif self.dataconfig['dataset_pd']['34_hmeq_data']:
                return _preprocess_34_hmeq_data(_data)

        elif self.experimentconfig['task'] == 'lgd':
            if self.dataconfig['dataset_lgd']['01_heloc']:
                return _preprocess_01_heloc(_data)

""" Generic method definitions """

def handle_missing_values(x_train, x_val, x_test, y_train, y_val, y_test, methodconfig: Dict):
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

        print(f"Omitting rows with missing values: {total_dropped_rows} rows left out")

    elif methodconfig['missing_values'] == 2:
        # todo: insert code
        pass

    elif methodconfig['missing_values'] == 3:
        # todo: insert code
        pass

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

    elif methodconfig['encode_cat']==1:
        # one-hot encode categorical variables
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        x_train_cat = encoder.fit_transform(x_train[:, cols_cat_idx])
        x_val_cat = encoder.transform(x_val[:, cols_cat_idx])
        x_test_cat = encoder.transform(x_test[:, cols_cat_idx])

        # Concatenate the encoded categorical columns with the numerical columns
        x_train = np.concatenate([x_train[:, ~np.isin(range(x_train.shape[1]), cols_cat_idx)], x_train_cat], axis=1)
        x_val = np.concatenate([x_val[:, ~np.isin(range(x_val.shape[1]), cols_cat_idx)], x_val_cat], axis=1)
        x_test = np.concatenate([x_test[:, ~np.isin(range(x_test.shape[1]), cols_cat_idx)], x_test_cat], axis=1)

    elif methodconfig['encode_cat']==2:
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
