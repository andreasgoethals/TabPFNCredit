from typing import Dict, List

import category_encoders
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

""" Class definition and class-specific method definitions """
class Preprocessing:
    def __init__(self, dataconfig, experimentconfig):
        self.dataconfig = dataconfig
        self.experimentconfig = experimentconfig
        self.dataset_name = None

    def load_data(self):
        ##########################################
        ### method definitions for pd datasets ###
        ##########################################
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

        def _load_29_loan_default():
            try:
                _data = pd.read_csv(
                    'data/pd/29 loan default predictions - imperial college london/train_v2.csv/train_v2.csv', sep=',', dtype=float)
            except FileNotFoundError:
                _data = pd.read_csv(
                    '../data/pd/29 loan default predictions - imperial college london/train_v2.csv/train_v2.csv',
                    sep=',', dtype=float)
                print("29_loan_default loaded")
            return _data

        def _load_30_home_credit():
            try:
                _data = pd.read_csv(
                    'data/pd/30 home credit default risk/home-credit-default-risk/application_train.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv(
                    '../data/pd/30 home credit default risk/home-credit-default-risk/application_train.csv', sep=',')
            print("30_home_credit loaded")
            return _data

        def _load_34_hmeq_data():
            try:
                _data = pd.read_csv('data/pd/34 hmeq/hmeq.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/034 hmeq/hmeq.csv', sep=',')
            print("34_hmeq_data loaded")
            return _data

        ###########################################
        ### method definitions for lgd datasets ###
        ###########################################
        def _load_01_heloc_lgd():
            _data = 'this is a 01 data to be implemented'
            print("01_heloc_lgd loaded")
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
            if self.dataconfig['dataset']['01_heloc_lgd']:
                self.dataset_name = '01_heloc_lgd'
                return _load_01_heloc_lgd()
        else:
            raise ValueError('Invalid task in experimentconfig, or no dataset selected in dataconfig')

    def preprocess_data(self, _data):
        ########################################################
        ### method definitions for preprocessing pd datasets ###
        ########################################################
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


        def _preprocess_29_loan_default(_data):
            # convert all columns to numeric:
            _data = _data.apply(pd.to_numeric, errors='coerce')

            # Drop ID and useless columns
            _data = _data.drop('id', axis=1)

            # Split into covariates, labels
            y = _data['loss'].values.astype(int)

            #convert y to 0 if 0 and to 1 if not zero:
            y = np.where(y==0, 0, 1)
            x = _data.drop('loss', axis=1).values

            # remove duplicate features by checking if columns have the same values:
            _data = _data.loc[:, (_data != _data.iloc[0]).any()]



            # Replace infinity values with a large finite number
            x = np.where(np.isinf(x), np.finfo(np.float32).max, x)
            # Ensure all values are within a valid range for float32
            x = np.clip(x, np.finfo(np.float32).min, np.finfo(np.float32).max)

            # only numeric cols:
            cols = list(_data.drop('loss', axis=1).columns)
            cols_cat = []
            cols_num = cols
            cols_cat_idx = []
            cols_num_idx = list(range(len(cols)))

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx


        def _preprocess_30_home_credit(_data):
            # Drop ID and useless columns
            _data = _data.drop('SK_ID_CURR', axis=1)

            # Split into covariates, labels
            y = _data['TARGET'].values.astype(int)
            x = _data.drop('TARGET', axis=1).values

            cols = list(_data.drop('TARGET', axis=1).columns)

            # List of categorical variables
            cols_cat = [
                'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
                'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE',
                'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
            ]

            # List of numerical variables
            cols_num = [
                'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
                'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
                'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
                'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
                'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
                'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
                'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
                'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'EXT_SOURCE_1',
                'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
                'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG',
                'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG',
                'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
                'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE',
                'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
                'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE',
                'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
                'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
                'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
                'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
                'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
                'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
                'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',
                'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
                'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
                'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
                'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
                'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
            ]

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

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

        #########################################################
        ### method definitions for preprocessing lgd datasets ###
        #########################################################
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
            elif self.dataconfig['dataset_pd']['29_loan_default']:
                return _preprocess_29_loan_default(_data)
            elif self.dataconfig['dataset_pd']['30_home_credit']:
                return _preprocess_30_home_credit(_data)
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

        print(f"- Omitting rows with missing values: {total_dropped_rows} rows left out")

    elif methodconfig['missing_values'] == 2:
        # impute with mean:
        imputer = SimpleImputer(strategy='mean')
        x_train = imputer.fit_transform(x_train)
        x_val = imputer.transform(x_val)
        x_test = imputer.transform(x_test)

        print('- Imputed missing values with mean')

    elif methodconfig['missing_values'] == 3:
        # impute with median:
        imputer = SimpleImputer(strategy='median')
        x_train = imputer.fit_transform(x_train)
        x_val = imputer.transform(x_val)
        x_test = imputer.transform(x_test)

        print('- Imputed missing values with median')

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
