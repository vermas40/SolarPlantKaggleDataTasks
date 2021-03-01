import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'//Users//mac_air//Documents//Documents//Side Projects//Kaggle_Anomaly_Detection//Scripts')
import ads_creation as ads_crtn
import eda_analysis as eda
import feature_engg as feat_engg
import regression as regr
from constants import ATTR, TIME_INVARIANT_ATTR, TIME_VARIANT_ATTR, SPLIT_PCT

def create_ts_list(df,date_col):
    '''
    This function creates the list of timestamps from first ts to last ts
    '''
    dates = pd.date_range(start=df[date_col].min(), end = df[date_col].max(),freq = '15min')
    dates = dates.to_frame(index=False)
    dates = dates.rename(columns = {0:'DATE'})
    return dates

def complt_ts(df,date_col):
    '''
    This function introduces the missing time stamps in ads
    '''
    date_list = create_ts_list(df,date_col)
    df = pd.merge(left=date_list,right=df,how='left', on='DATE')
    return df

ads = ads_crtn.create_ads()
ads = ads.groupby('INVERTER_ID').apply(lambda x: eda.outlier_treatment(x,'INVERTER_ID'))

#Experiment 1
#Creating lagged features 
#While creating lagged features, we see that some timestamps are missing
#to correct this we will have to complete timestamps in our ads

#when you run groupby-apply functions in such a way that the numerical index get changed 
#and for eg, in previous df at index 100 inverter 1 was present but now inverter 0 will be there
#due to our operation in function now inverter 0 will be at index 100
#to circumvent this problem, pandas changes the index to groupby column and then the index
ads = ads.groupby('INVERTER_ID').apply(lambda x: complt_ts(x, 'DATE'))
ads = ads.reset_index(drop=True)
#if we dont do this ffill here then the all the obs for inverter ids won't reach
ads['INVERTER_ID'] = ads['INVERTER_ID'].ffill()
ads = ads.groupby('INVERTER_ID').apply(lambda x: ads_crtn.missing_value_treatment(x))

#making ADS stationary
non_stnry_invtr_list = eda.return_non_stnry_invtr_list(ads,'INVERTER_ID')
ads = eda.make_ads_stnry(ads,non_stnry_invtr_list,'INVERTER_ID')

for variable in TIME_VARIANT_ATTR:
    ads = feat_engg.create_lag_feature_placeholder(ads,variable,[192])
    ads = ads.groupby('INVERTER_ID').apply(lambda x: feat_engg.create_lag_feature(x,variable,[192]))
ads = ads.dropna()
ads = ads.reset_index(drop=True)
ads = ads.sort_values(by=['INVERTER_ID','DATE'],ignore_index=True)

drop_features = [feature for feature in TIME_VARIANT_ATTR if feature != 'PER_TS_YIELD']
features = [feature + '_lag_192' for feature in TIME_VARIANT_ATTR]
features = features + TIME_INVARIANT_ATTR

models,metrics = regr.panel_wise_model(ads,'INVERTER_ID','PER_TS_YIELD',features=features)

low_acc_invtr = [(key,metrics[key][3]) for key in metrics.keys() if metrics[key][3] < 0.8]
invtr_list = [val_pair[0] for val_pair in low_acc_invtr]

#these are all the inverters that had high amount of outliers in them
eda.get_box_plot(ads.loc[ads['INVERTER_ID'].isin(invtr_list),ATTR])
#as confirmed by line plots as well
eda.line_plot(ads.loc[ads['INVERTER_ID'].isin(invtr_list),['INVERTER_ID','PER_TS_YIELD','DATE']],'DATE','PER_TS_YIELD','INVERTER_ID')
