import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV

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

def create_lagged_features(df,variable_list,lags_list,panel_id_col,date_col):
    '''
    Creates the lagged feature for the target variable
    '''
    for variable in variable_list:
        df = create_lag_feature_placeholder(df,variable,lags_list)
        df = df.groupby(panel_id_col).apply(lambda x: create_lag_feature(x,variable,lags_list))
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.sort_values(by=[panel_id_col, date_col],ignore_index=True)
    return df

def rfe_feature_selection(train_df,estimator,features,label):
    scaler = StandardScaler()
    scaler.fit(train_df[features])

    train_df_scaled = scaler.transform(train_df[features])
    selector = RFECV(estimator, step=1, cv=5)
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

    #creating lagged features
    ads = create_features(ads,'PER_TS_YIELD',distinct_lag_values)