import pandas as pd
import numpy as np
import os

#importing user defined functions
os.chdir(r'//Users//mac_air//Documents//Documents//Side Projects//Kaggle_Anomaly_Detection//Scripts')
import ads_creation as ads_crtn
import eda_analysis as eda

def create_lag_feature_placeholder(df,variable_name,lags):
    for lag in lags:
        df[variable_name + '_lag_' + str(lag)] = np.nan
    return df

def create_lag_feature(df,variable_name,lags):
    for lag in lags:
        df[variable_name + '_lag_' + str(lag)] = df[variable_name].shift(lag)
    return df

def create_features(df,variable_name,distinct_lag_values):
    '''
    Creates the lagged feature for the target variable
    '''
    df = create_lag_feature_placeholder(df,variable_name,distinct_lag_values)
    df = df.groupby('INVERTER_ID').apply(lambda x: create_lag_feature(x,variable_name,distinct_lag_values))
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.sort_values(by=['INVERTER_ID','DATE'],ignore_index=True)
    return df

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

    #creating lagged features
    ads = create_features(ads,'PER_TS_YIELD',distinct_lag_values)