import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
import copy
#importing user defined functions
os.chdir(r'//Users//mac_air//Documents//Documents//Side Projects//Kaggle_Anomaly_Detection//Scripts')
import ads_creation as ads_crtn
import eda_analysis as eda

def create_lag_feature_placeholder(df,variable_name,lags):
    '''
    This function creates place holders for lagged variables in the ads
    
    Input:
    1. df: pandas dataframe, this is the ads in which the column has to be made
    2. variable_name: the column name based on which the lagged variable has to be made
    3. lags: the list of lags for which the columns have to be made

    Returns:
    1. df: pandas dataframe with the lag feature placeholders
    '''
    for lag in lags:
        df[variable_name + '_lag_' + str(lag)] = np.nan
    return df

def create_lag_feature(df,variable_name,lags):
    '''
    This function creates the lag features in the ads

    Input:
    1. df: pandas dataframe, this is the ads in which the column has to be made
    2. variable_name: the column name based on which the lagged variable has to be made
    3. lags: the list of lags for which the columns have to be made

    Returns:
    1. df: pandas dataframe with the lag feature placeholders
    '''
    for lag in lags:
        df[variable_name + '_lag_' + str(lag)] = df[variable_name].shift(lag)
    return df

def create_lagged_features(df,variable_list,lags_list,panel_id_col,date_col):
    '''
    Creates the lagged feature for the target variable

    Input
    1. df: pandas dataframe, this is the ads in which the lagged columns are to be made
    2. variable_list: list, the list of time series that need to be lagged
    3. lags_list: list, the list of lags that will be applied onto the timeseries
    4. panel_id_col: str, this is the name of the column that contains the panel IDs
    5. date_col: str, this is the name of the column that contains the dates in the ads

    Returns:
    1. df: pandas dataframe, this is the ads with all the lagging for all the variables done

    https://stackoverflow.com/questions/575196/why-can-a-function-modify-some-arguments-as-perceived-by-the-caller-but-not-oth
    '''
    #This step is being done because pandas dataframes are mutable and here since we are changing the dataframe object
    #some bugs are coming in so, just doing a deep copy to avoid that
    #the copy below breaks prevents the argument df that we received (train_df) value to get changed at source
    df = copy.deepcopy(df)
    #before the deep copy df was changing because in the first line in for loop we insert a column in df 
    #this step happens before it is asigned back to df, so that is why this change is retained when we exit the function
    for variable in variable_list:
        df = create_lag_feature_placeholder(df,variable,lags_list)
        df = df.groupby(panel_id_col).apply(lambda x: create_lag_feature(x,variable,lags_list))
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.sort_values(by=[panel_id_col, date_col],ignore_index=True)
    return df

def rfe_feature_selection(train_df,estimator,features,label,cv_folds=5):
    '''
    This function implements feature selection through rfecv

    Input
    1. train_df: pandas dataframe, this is the training dataset
    2. estimator: this is the initialized model objects that will be used for rfecv
    3. features: list, this is the initial list of features from which the selection will happen
    4. label: str, target variable
    5. cv_folds: int, the number number cross validation folds to be created in each cycle of rfe

    Returns:
    1. feature_set: list, this is the list of the most significant features found through rfecv
    '''
    scaler = StandardScaler()
    scaler.fit(train_df[features])

    train_df_scaled = scaler.transform(train_df[features])
    selector = RFECV(estimator, step=1, cv=cv_folds)
    selector = selector.fit(train_df_scaled, train_df[label])

    feature_set = [sel_feature[1] for sel_feature in zip(selector.ranking_,features) if sel_feature[0] == 1]
    
    return feature_set
  
if __name__ == '__main__':
    #creating ads + outlier treatment
    ads = ads_crtn.create_ads()
    ads = ads.groupby('INVERTER_ID').apply(lambda x: eda.outlier_treatment(x,'INVERTER_ID'))

    #making ADS stationary
    non_stnry_invtr_list = eda.return_non_stnry_invtr_list(ads,'INVERTER_ID')
    ads = eda.make_ads_stnry(ads,non_stnry_invtr_list,'INVERTER_ID')

    #getting lag values for the target variables
    lag_values = ads[['INVERTER_ID','PER_TS_YIELD']].groupby('INVERTER_ID').agg(lambda x: eda.pick_pacf(x))
    distinct_lag_values = np.unique(lag_values.PER_TS_YIELD.sum())