import pandas as pd
import numpy as np
import os
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'//Users//mac_air//Documents//Documents//Side Projects//Kaggle_Anomaly_Detection//Scripts//')
import ads_creation as ads_crtn
import eda_analysis as eda
import feature_engg as feat_engg
import winsorize as wz
import regression as regr
from constants import ATTR, OUTLIER_METHOD, TIME_INVARIANT_ATTR, TIME_VARIANT_ATTR, SPLIT_PCT

def make_ar_model(train_df,test_df,lags):
    model = AutoReg(train_df,lags = lags)
    model_fit = model.fit()

    y_pred = model_fit.predict(start=len(train_df),end=len(train_df) + len(test_df) - 1)
    r2 = r2_score(test_df,y_pred)

    return r2

def make_ma_model(train_df,test_df,lags):
    model = ARIMA(train_df,order=(0,0,lags[0]))
    model_fit = model.fit()

    y_pred = model_fit.predict(start=len(train_df),end=len(train_df) + len(test_df) - 1)
    r2 = r2_score(test_df,y_pred)

    return r2  


ads = ads_crtn.create_ads()
train_ads, test_ads = eda.train_test_split(ads,'INVERTER_ID',SPLIT_PCT)

outlier_feature = [feature for feature in ATTR if feature != 'TOTAL_YIELD'] 
clip_model = wz.winsorize(OUTLIER_METHOD)
clip_model.fit(train_ads[outlier_feature],'INVERTER_ID')
train_ads[outlier_feature] = clip_model.transform(train_ads[outlier_feature])
test_ads[outlier_feature] = clip_model.transform(test_ads[outlier_feature])


non_stnry_invtr_list = eda.return_non_stnry_invtr_list(train_ads,'INVERTER_ID')
train_ads = eda.make_ads_stnry(train_ads,non_stnry_invtr_list,'INVERTER_ID','DATE')

non_stnry_invtr_list = eda.return_non_stnry_invtr_list(test_ads,'INVERTER_ID')
test_ads = eda.make_ads_stnry(test_ads,non_stnry_invtr_list,'INVERTER_ID','DATE')

lag_values = train_ads[['INVERTER_ID','PER_TS_YIELD']].groupby('INVERTER_ID').agg(lambda x: eda.pick_pacf(x))
acf_values = train_ads[['INVERTER_ID','PER_TS_YIELD']].groupby('INVERTER_ID').agg(lambda x: eda.pick_acf(x))

#Experiment 1
#Making an AR model for each panel. AR model is just a linear regression model with lagged
#versions of the target variable

ar_metrics = {}
for invtr in train_ads['INVERTER_ID'].unique():
    print('Modelling for panel:',invtr)
    lags = lag_values.loc[lag_values.index == invtr,'PER_TS_YIELD'][invtr]
    train = train_ads.loc[train_ads['INVERTER_ID']==invtr,'PER_TS_YIELD']
    test = test_ads.loc[test_ads['INVERTER_ID']==invtr,'PER_TS_YIELD']

    ar_metrics[invtr] = make_ar_model(train,test,lags)

#This was good and bad. Good because no model had negative R-squared so every model was better than
#an average model! However, The maximum R square was 65% only, which falls much behind our RF 
#supervised model

#Experiment 2
#Making a moving average model of the using the order found from ACF
#p will be 0 since this is a pure MA model
#the models are very slow to converge and all of them give negative r-squared
ma_metrics = {}
for invtr in train_ads['INVERTER_ID'].unique():
    print('Modelling for panel:',invtr)
    lags = acf_values.loc[lag_values.index == invtr,'PER_TS_YIELD'][invtr]
    train = train_ads.loc[train_ads['INVERTER_ID']==invtr,'PER_TS_YIELD']
    test = test_ads.loc[test_ads['INVERTER_ID']==invtr,'PER_TS_YIELD']

    ma_metrics[invtr] = make_ma_model(train,test,lags)





