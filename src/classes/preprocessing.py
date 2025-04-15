import warnings
from typing import Dict, List
from pathlib import Path
import category_encoders
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# toy datasets:
from sklearn.datasets import load_breast_cancer, load_diabetes

pd.set_option('future.no_silent_downcasting', True)

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

        def _load_06_lendingclub():
            try:
                _data = pd.read_csv('data/pd/06 lendingclub/loan_data.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/06 lendingclub/loan_data.csv', sep=',')
            print("06_lendingclub loaded")
            return _data

        def _load_07_case_study():
            try:
                _data = pd.read_csv('data/pd/07 case study/Case Study- Probability of Default.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/07 case study/Case Study- Probability of Default.csv', sep=',')
            print("07_case_study loaded")
            return _data

        def _load_09_myhom():
            try:
                _data = pd.read_csv('data/pd/09 myhom/train_data.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/09 myhom/train_data.csv', sep=',')
            print("09_myhom loaded")
            return _data

        def _load_10_hackerearth():
            try:
                _data = pd.read_csv('data/pd/10 hackerearth/train_indessa.csv', sep=',')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/10 hackerearth/train_indessa.csv', sep=',')
            print("10_hackerearth loaded")
            return _data

        def _load_11_cobranded():
            try:
                _data = pd.read_csv('data/pd/11 cobranded/Training_dataset_Original.csv', sep=',', low_memory=False)
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/11 cobranded/Training_dataset_Original.csv', sep=',', low_memory=False)
            print("11_cobranded loaded")
            return _data

        def _load_14_german_credit():
            try:
                _data = pd.read_csv('data/pd/14 statlog german credit data/german.csv')
            except FileNotFoundError:
                _data = pd.read_csv('../data/pd/14 statlog german credit data/german.csv')
            print("14_german_credit loaded")
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
            print("28_thomas loaded")
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
        def _load_00_lgd_toydata():
            _data = load_diabetes()
            self.dataset_name = '00_lgd_toydata'
            print("00_lgd_toydata loaded")
            return _data

        def _load_01_heloc_lgd():
            try:
                _data = pd.read_csv(Path('data') / 'lgd' / '01 heloc_lgd' / 'heloc_lgd.csv', low_memory=False)
            except FileNotFoundError:
                _data = pd.read_csv(Path('..') / 'data' / 'lgd' / '01 heloc_lgd' / 'heloc_lgd.csv', low_memory=False)
            print("01_heloc_lgd loaded")
            return _data

        def _load_03_loss2():
            try:
                _data = pd.read_csv(Path('data') / 'lgd' / '03 loss2' / 'loss2.csv')
            except FileNotFoundError:
                _data = pd.read_csv(Path('..') / 'data' / 'lgd' / '03 loss2' / 'loss2.csv')
            print("03_loss2 loaded")
            return _data

        def _load_05_axa():
            try:
                _data = pd.read_csv(Path('data') / 'lgd' / '05 lgd_axa' / 'lgd_axa.csv')
            except FileNotFoundError:
                _data = pd.read_csv(Path('..') / 'data' / 'lgd' / '05 lgd_axa' / 'lgd_axa.csv')
            print("05_axa loaded")
            return _data

        def _load_06_base_model():
            try:
                _data = pd.read_csv(Path('data') / 'lgd' / '06 base_model' / 'base_model.csv')
            except FileNotFoundError:
                _data = pd.read_csv(Path('..') / 'data' / 'lgd' / '06 base_model' / 'base_model.csv')
            print("06_base_model loaded")
            return _data

        def _load_07_base_modelisation():
            try:
                _data = pd.read_csv(Path('data') / 'lgd' / '07 base_modelisation' / 'base_modelisation.csv')
            except FileNotFoundError:
                _data = pd.read_csv(Path('..') / 'data' / 'lgd' / '07 base_modelisation' / 'base_modelisation.csv')
            print("07_base_modelisation loaded")
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
                print(f"- Dataset has {len(_data)} rows. Subsampling to {row_limit} rows.")
                _data = _data.sample(n=row_limit, random_state=42).reset_index(drop=True)
            else:
                print(f"- Dataset has {len(_data)} rows. No subsampling needed.")
            return _data
        elif hasattr(_data, 'data') and hasattr(_data, 'target'):
            if _data.data.shape[0] > row_limit:
                idx = np.random.RandomState(seed=42).choice(_data.data.shape[0], row_limit, replace=False)
                _data.data = _data.data[idx]
                _data.target = _data.target[idx]
                print(f"- Subsampled toy dataset to {row_limit} rows.")
            return _data
        else:
            raise TypeError("Unsupported data format for subsampling.")

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

        def _preprocess_01_gmsc(_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str], list[int], list[int]]:

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

        def _preprocess_06_lendingclub(_data):
            # Drop ID and useless columns

            # Split into covariates, labels
            target_col = 'not.fully.paid'
            y = _data[target_col].values.astype(int)
            x = _data.drop(target_col, axis=1).values

            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = _data.drop(columns=[target_col]).select_dtypes(include=['object', 'category']).columns.tolist()
            cols_num = _data.drop(columns=[target_col]).select_dtypes(include=['number']).columns.tolist()

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("06_lendingclub preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_07_case_study(_data):

            # replace column values:
            _data['status'] = _data['status'].replace({'RICH': 6, 'POOR': 2, 'MIDDLE': 4, 'LOWMIDDLE': 3, 'VERYRICH': 7, 'VERYMIDDLE': 5, 'VERYPOOR': 1})

            target_col = 'PaymentMissFlag'

            # Split into covariates, labels
            y = _data[target_col].values.astype(int)
            x = _data.drop(target_col, axis=1).values

            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = _data.drop(columns=[target_col]).select_dtypes(include=['object', 'category']).columns.tolist()
            cols_num = _data.drop(columns=[target_col]).select_dtypes(include=['number']).columns.tolist()

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("07_case_study preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_09_myhom(_data):

            target_col = 'loan_default'

            _data = _data.drop('loan_id', axis=1)

            # Split into covariates, labels
            y = _data[target_col].values.astype(int)
            x = _data.drop(target_col, axis=1).values

            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = _data.drop(columns=[target_col]).select_dtypes(include=['object', 'category']).columns.tolist()
            cols_num = _data.drop(columns=[target_col]).select_dtypes(include=['number']).columns.tolist()

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("09_myhom preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_10_hackerearth(_data):
            target_col = 'loan_status'

            _data = _data.drop(columns=['member_id', 'batch_enrolled', 'emp_title', 'desc', 'title', 'zip_code'])

            # clean _data['emp_length']
            _data['emp_length'] = _data['emp_length'].replace('< 1 year', 0)
            _data['emp_length'] = _data['emp_length'].str.replace(' years', '')
            _data['emp_length'] = _data['emp_length'].str.replace(' year', '')
            _data['emp_length'] = _data['emp_length'].replace('10+', 11)
            _data['emp_length'] = _data['emp_length'].astype(float)

            # clean _data['last_week_pay']
            _data['last_week_pay'] = _data['last_week_pay'].str.replace('th week', '')
            _data['last_week_pay'] = _data['last_week_pay'].replace('NA', np.nan)
            _data['last_week_pay'] = _data['last_week_pay'].astype(float)

            # Split into covariates, labels
            y = _data[target_col].values.astype(int)
            x = _data.drop(target_col, axis=1).values

            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = _data.drop(columns=[target_col]).select_dtypes(include=['object', 'category']).columns.tolist()
            cols_num = _data.drop(columns=[target_col]).select_dtypes(include=['number']).columns.tolist()

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("10_hackerearth preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_11_cobranded(_data):

            _data = _data.replace('na', np.nan)
            _data = _data.replace('missing', np.nan)

            _data = _data.drop(columns=['application_key'])

            _data.replace({'mvar47': {'C': 1, 'L': 0}}, inplace=True)

            for col in _data.columns:
                _data[col] = _data[col].astype(float)

            # Split into covariates, labels
            target_col = 'default_ind'
            y = _data[target_col].values.astype(int)
            x = _data.drop(target_col, axis=1).values

            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = _data.drop(columns=[target_col]).select_dtypes(include=['object', 'category']).columns.tolist()
            cols_num = _data.drop(columns=[target_col]).select_dtypes(include=['number']).columns.tolist()

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("11_cobranded preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx


        def _preprocess_14_german_credit(_data):
            target_col = '1.1'

            # drop any row where the target column is missing:
            _data = _data.dropna(subset=[target_col])

            # in target_col, set value 1 to 0 and 2 to 1
            _data[target_col] = _data[target_col].replace({1: 0, 2: 1})

            # Split into covariates, labels
            y = _data[target_col].values
            x = _data.drop(target_col, axis=1).values

            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = ['A11','A34','A43','A65','A75','A93','A101','A121','A143','A152','A173','A192','A201']
            cols_num = ['6','1169','4','4.1','2','1']

            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("14_german_credit preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_22_bank_status(_data):
            _data = _data.drop(columns=['Loan ID', 'Customer ID'])

            _data['Loan Status'] = _data['Loan Status'].replace('Fully Paid', 0)
            _data['Loan Status'] = _data['Loan Status'].replace('Charged Off', 1)

            _data['Term'] = _data['Term'].replace('Short Term', 0)
            _data['Term'] = _data['Term'].replace('Long Term', 1)

            _data['Years in current job'] = _data['Years in current job'].replace('< 1 year', 0)
            _data['Years in current job'] = _data['Years in current job'].str.replace(' years', '')
            _data['Years in current job'] = _data['Years in current job'].str.replace(' year', '')
            _data['Years in current job'] = _data['Years in current job'].replace('10+', 11)
            _data['Years in current job'] = _data['Years in current job'].astype(float)

            # Split into covariates, labels
            target_col = 'Loan Status'

            _data = _data.dropna(subset=[target_col])

            y = _data[target_col].values.astype(int)
            x = _data.drop(target_col, axis=1).values

            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = _data.drop(columns=[target_col]).select_dtypes(include=['object', 'category']).columns.tolist()
            cols_num = _data.drop(columns=[target_col]).select_dtypes(include=['number']).columns.tolist()

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("22_bank_status preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_28_thomas(_data):
            target_col = 'BAD'

            # Split into covariates, labels
            y = _data[target_col].values.astype(int)
            x = _data.drop(target_col, axis=1).values

            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = _data.drop(columns=[target_col]).select_dtypes(include=['object', 'category']).columns.tolist()
            cols_num = _data.drop(columns=[target_col]).select_dtypes(include=['number']).columns.tolist()

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("28_thomas preprocessed")
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

            print("34_hmeq_data preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        #########################################################
        ### method definitions for preprocessing lgd datasets ###
        #########################################################
        def _preprocess_00_lgd_toydata(_data):

            x = _data.data
            y = _data.target

            cols = list(_data.feature_names)
            cols_cat = []
            cols_num = cols
            cols_cat_idx = []
            cols_num_idx = list(range(len(cols)))

            print("00_lgd_toydata preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_01_heloc(_data):

            # Drop necessary columns (example: dropping 'ID' column)
            _data = _data.drop(columns=['REC'])
            _data = _data.drop(columns=['DLGD_Econ'])

            # double columns
            _data = _data.drop(columns=['PrinBal', 'PayOff', 'DefPayOff'])

            # time-related columns (we consider cross-sectional)
            _data = _data.drop(columns=['ObsDT', 'DefDT'])

            # Transform
            _data['LienPos'] = _data['LienPos'].replace({'Unknow': 0, 'First': 1, 'Second': 2}) # Downcasting behavior in `replace` is deprecated and will be removed in a future version.
            _data = _data.infer_objects(copy=False)

            # Split into covariates, labels
            y = _data['LGD_ACTG'].values
            x = _data.drop('LGD_ACTG', axis=1).values

            cols = list(_data.drop('LGD_ACTG', axis=1).columns)

            cols_cat = []
            cols_num = ['PortNum', 'AvailAmt', 'LTV', 'LienPos', 'Age', 'CurrEquifax', 'Utilization', 'DefPrinBal', 'PD_Rnd']

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("01_heloc preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_03_loss2(_data):
            # drop unnecessary columns
            # correlated with target:
            _data = _data.drop(columns=['_ELGDnum1', '_ELGDnum2'])

            # identifier
            _data = _data.drop(columns=['id1', 'Alltel_Client'])

            # date-related cols
            _data = _data.drop(columns=['REO_Appraisal_Date', 'Origination_Date', 'date_vintage_year', 'date_vintage_year_month'])

            # all nan values
            _data = _data.drop(columns=['Servicing_Loss'])

            # Define the target column
            target_col = '_ELGD'

            # drop any row where the target column is missing:
            _data = _data.dropna(subset=[target_col])

            x = _data.drop(columns=[target_col]).values
            y = _data[target_col].values

            # define cols
            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = _data.drop(columns=[target_col]).select_dtypes(include=['object', 'category']).columns
            cols_num = _data.drop(columns=[target_col]).select_dtypes(include=['number']).columns

            """
            cols_cat = ['Loan_Category', 'State', 'Analyst', 'Credit_Bureau', 'Loan_Type',
                        'Recourse_Type', 'ARM_Indicator', 'MI_Company_Name',
                        'amortization_type', 'broker_flag', 'business_line_crm_new',
                        'construction_indicator', 'doc_option', 'documentation_type',
                        'firsttime_homebuyer_indicator', 'jumbo_indicator',
                        'living_units_number', 'loan_purpose_type',
                        'occupancy_code_description', 'primary_borrower_self_employed',
                        'product_code', 'product_family_name', 'property_type_description']
            cols_num = ['UPB_At_Resolution', 'Unpaid_Interest', 'Total_Debt',
                        'REO_Sales_Price', 'Original_Appraised_Value', 'REO_Appraisal_Amount',
                        'Original_UPB', 'Analysis_Age', 'Investor_Category',
                        'Annual_Interest_Rate', 'alt_fico_code', 'amount_appraised',
                        'amount_funded', 'amount_note', 'housing_ratio', 'interest_only_term',
                        'loan_term', 'ltv_calculated_crm', 'ltv_combined_crm',
                        'primary_borrower_number', 'score_fico_used', 'total_debt_ratio',
                        'percent_of_primary_pmi_coverage', '_reo_sales_price', '_SellingCosts',
                        '_adv_interest1M', '_adv_interest', '_ELAO', '_Accrued_int', '_EAD',
                        '_Net_sales_Proceeds', '_Miclaimbal', '_Mirecovery', '_Proceeds',
                        '_Loss_Amount', 'lr1', 'lss_amt', 'lss_rt', 'MI_ind']
            """
            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("03_loss2 preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_05_axa(_data):
            # drop unnecessary columns
            # correlated with target:
            _data = _data.drop(columns=['Recovery_rate'])  # this is 1-LGD
            _data = _data.drop(columns=['y_logistic', 'lnrr', 'Y_probit', 'event'])

            # split into covariates and labels
            target_col = 'lgd_time'
            y = _data[target_col].values
            x = _data.drop(target_col, axis=1).values

            # define cols
            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = []
            cols_num = ['LTV','purpose1']

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("05_axa preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_06_base_model(_data):
            # drop unnecessary columns
            # id columns:
            _data = _data.drop(columns=['DEAL_DocUNID', 'DEAL_MainID', 'DEAL_FacilityIdentifier', 'DEAL_StarWebIdentifier',
                                  'DFLT_MainID', 'DFLT_SPM', 'DFLT_DAI', 'DFLT_BDR', 'DFLT_LegalEntityName',
                                  'DFLT_StarWeb_PCRU', 'DFLT_ClientNAE', 'DFLT_ParentSPM', 'DFLT_ParentSIREN',
                                  'DFLT_ParentDAI', 'DFLT_ParentLegalEntityName', 'DFLT_ParentPCRU', 'DFLT_ParentNAE',
                                  'DFLT_subject', 'FCLT_DealUNID', 'FCLT_BCEIdentifier', 'FCLT_Identifier',
                                  'FCLT_BookingUnit', 'fclt_docunid'])

            # time-related cols:
            _data = _data.drop(
                columns=['DEAL_TransactionStartDate', 'DEAL_TransactionEndDate', 'DEAL_DateComposed', 'DEAL_LastUpDate',
                         'DFLT_SGDefaultDate', 'DFLT_PublicDefaultDate', 'DFLT_EndDefaultDate', 'DFLT_SGRatingDate',
                         'DFLT_RatingDate1YPD', 'DFLT_ParentDefaultDateIf', 'DFLT_ParentSGRatingDate',
                         'DFLT_ParentRatingDate1YPD', 'DFLT_DateComposed', 'DFLT_LastUpdate', 'DATE_DECLAR_CT',
                         'FCLT_StartDate', 'FCLT_EndDate', 'FCLT_DefaultDate', 'FCLT_DateComposed', 'FCLT_LastUpdate',
                         'date'])

            # cols with textual description (as in: full sentences):
            _data = _data.drop(columns=['FCLT_CommentsOnLimit', 'FCLT_subject', 'DEAL_GoverningLawRecovery', 'DEAL_PFRU',
                                  'DEAL_subject'])

            # drop cols with only nan values:
            _data = _data.drop(columns=['DEAL_ConstructionEndDate', 'DEAL_ConstructionStartDate',
                                  'DEAL_AverageRents', 'DEAL_ExpectedVacancyRate', 'DEAL_StrikeLESSEEOption',
                                  'DEAL_StatusUpDate', 'DEAL_DeleteDate', 'DFLT_DeleteStatus',
                                  'DFLT_DeletedDate', 'DFLT_JRIRating', 'DFLT_ParentJRIRating',
                                  'FCLT_DeleteStatus', 'FCLT_IrrevocableLocOffshore', 'FCLT_DeleteDate',
                                  'flag_eps', 'flag_fcltcurrency', 'fac_ss_commcov', 'flag_pme',
                                  'Flag_specifique', 'flag_specperi', 'Categorie_AV'])

            # correlated with target:
            _data = _data.drop(columns=['lgd_cat_15', 'lgd_cat_10', 'lgd_cat_5', 'LGD_log', 'LGD_deF', 'LGD_norm', 'sortie',
                                  'RecAssoFlag'])

            # other data leakage::
            _data = _data.drop(columns=["DFLT_SIREN", "DFLT_CorporateAssets", "DFLT_OperatorCompanyIndicator",
                                  "DFLT_PCRU", "DFLT_LegalForm", "DFLT_ClientNationality",
                                  "DFLT_ClientAssetLocation", "DFLT_ClientSIC", "DFLT_HasSGtrigger",
                                  "DFLT_RationaleDefault", "DFLT_RecoveryApproach", "DFLT_RationalEndDefault",
                                  "DFLT_SGLocalCurRating", "DFLT_SGForeignCurRating", "DFLT_SPRating",
                                  "DFLT_MoodyRating", "DFLT_KMVRating", "DFLT_BilanYear",
                                  "DFLT_ClientCurrency", "DFLT_TaxFreeTO1YPD", "DFLT_TotalAsset1YPD",
                                  "DFLT_IntangibleAsset1YPD", "DFLT_CurrentAsset1YPD", "DFLT_DebtAmount1YPD",
                                  "DFLT_EquityAmount1YPD", "DFLT_ParentBDR", "DFLT_ParentLegalForm",
                                  "DFLT_ParentNationality", "DFLT_ParentAssetLocation", "DFLT_ParentSIC",
                                  "DFLT_ParentSGLocalCurRating", "DFLT_ParentSGForeignCurRating",
                                  "DFLT_ParentSPRating", "DFLT_ParentMoodyRating", "DFLT_ParentKMVRating",
                                  "DFLT_ParentBilanYear", "DFLT_ParentCurrency", "DFLT_ParentTaxFreeTO1YPD",
                                  "DFLT_ParentTotalAsset1YPD", "DFLT_ParentIntangibleAsset1YPD",
                                  "DFLT_ParentCurrentAsset1YPD", "DFLT_ParentDebtAmount1YPD",
                                  "DFLT_ParentEquityAmount1YPD", "DFLT_taux", "DFLT_Parenttaux", "DFLT_duree",
                                  "prorata_dflt", "FCLT_duree_pre_DfLT", "cat_DFLT_duree", "cat_FCLT_duree_pre_DfLT",
                                  "classe_DFLT_duree", "classe_FCLT_duree_pre_DfLT", "classe_DFLT_dicho_2A"
                                  ])

            # print columns with more than 80% missing values:
            missing_values = _data.isnull().mean()
            missing_values = missing_values[missing_values > 0.8]
            # drop these columns:
            _data = _data.drop(columns=missing_values.index)


            # split into covariates and labels
            # Define the target column
            target_col = 'LGD_brute'

            # drop any row where the target column is missing:
            _data = _data.dropna(subset=[target_col])

            x = _data.drop(columns=[target_col]).values
            y = _data[target_col].values

            # define cols
            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = _data.drop(columns=[target_col]).select_dtypes(include=['object', 'category']).columns.tolist()
            cols_num = _data.drop(columns=[target_col]).select_dtypes(include=['number']).columns.tolist()

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("06_base_model preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

        def _preprocess_07_base_modelisation(_data):
            # drop unnecessary columns
            # drop identifiers
            _data = _data.drop(columns=['Ident_cliej_spm', 'ID_CONC_ORIGIN_CDL', 'id_crc',
                                        'id_unique'])

            # drop leakage: lgd / loss / defaut / ... related cols:
            _data = _data.drop(
                columns=['lgd_5_sscout_ligne', 'lgd_corr', 'lgd_defaut_nt', 'lgd_3class', 'lgd_2class', 'lgd_log',
                         'lgd_t', 'lgd_1log', 'logit_lgd', 'Dt_entree_defaut', 'Dt_sortie_defaut',
                         'flag_defaut_moins_1an', 'auto_av_defaut', 'util_av_defaut', 'defaut_clos',
                         'defaut_clos_4nonclos', 'duree_1A_av_defaut', 'util_av_defaut_tot', 'auto_av_defaut_tot',
                         'defaut_M1Y', 'defaut_P1Y', ])

            # split into covariates and labels
            # Define the target column
            target_col = 'lgd_defaut'
            y = _data[target_col].values
            x = _data.drop(target_col, axis=1).values

            # define cols
            cols = list(_data.drop(target_col, axis=1).columns)

            cols_cat = _data.drop(columns=[target_col]).select_dtypes(include=['object', 'category']).columns.tolist()
            cols_num = _data.drop(columns=[target_col]).select_dtypes(include=['number']).columns.tolist()

            # define the indices of the categorical and numerical columns (in x):
            cols_cat_idx = [cols.index(col) for col in cols_cat if col in cols]
            cols_num_idx = [cols.index(col) for col in cols_num if col in cols]

            print("07_base_modelisation preprocessed")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)

            return x, y, cols, cols_cat, cols_num, cols_cat_idx, cols_num_idx

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

            elif self.dataconfig['dataset_pd']['06_lendingclub']:
                return _preprocess_06_lendingclub(_data)
            elif self.dataconfig['dataset_pd']['07_case_study']:
                return _preprocess_07_case_study(_data)
            elif self.dataconfig['dataset_pd']['09_myhom']:
                return _preprocess_09_myhom(_data)
            elif self.dataconfig['dataset_pd']['10_hackerearth']:
                return _preprocess_10_hackerearth(_data)
            elif self.dataconfig['dataset_pd']['11_cobranded']:
                return _preprocess_11_cobranded(_data)
            #elif self.dataconfig['dataset_pd']['12_loan_defaulter']:
            #    return _preprocess_12_loan_defaulter(_data)
            #elif self.dataconfig['dataset_pd']['13_loan_data_2017']:
            #    return _preprocess_13_loan_data_2017(_data)

            elif self.dataconfig['dataset_pd']['14_german_credit']:
                return _preprocess_14_german_credit(_data)
            elif self.dataconfig['dataset_pd']['22_bank_status']:
                return _preprocess_22_bank_status(_data)
            elif self.dataconfig['dataset_pd']['28_thomas']:
                return _preprocess_28_thomas(_data)
            elif self.dataconfig['dataset_pd']['29_loan_default']:
                return _preprocess_29_loan_default(_data)
            elif self.dataconfig['dataset_pd']['30_home_credit']:
                return _preprocess_30_home_credit(_data)
            elif self.dataconfig['dataset_pd']['34_hmeq_data']:
                return _preprocess_34_hmeq_data(_data)

            else:
                raise ValueError('Invalid dataset_pd in dataconfig')

        elif self.experimentconfig['task'] == 'lgd':

            if self.dataconfig['dataset_lgd']['00_lgd_toydata']:
                return _preprocess_00_lgd_toydata(_data)
            elif self.dataconfig['dataset_lgd']['01_heloc']:
                return _preprocess_01_heloc(_data)
            elif self.dataconfig['dataset_lgd']['03_loss2']:
                return _preprocess_03_loss2(_data)
            elif self.dataconfig['dataset_lgd']['05_axa']:
                return _preprocess_05_axa(_data)
            elif self.dataconfig['dataset_lgd']['06_base_model']:
                return _preprocess_06_base_model(_data)
            elif self.dataconfig['dataset_lgd']['07_base_modelisation']:
                return _preprocess_07_base_modelisation(_data)

            else:
                raise ValueError('Invalid dataset_lgd in dataconfig')

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

        print(f"- Omitting rows with missing values: {total_dropped_rows} rows left out")

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

        print('- Imputed missing values with (num: mean) and (cat: mode)')

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

        print('- Imputed missing values with (num: median) and (cat: mode)')

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
