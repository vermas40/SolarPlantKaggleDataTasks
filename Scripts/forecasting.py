import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

os.chdir(r'//Users//mac_air//Documents//Documents//Side Projects//Kaggle_Anomaly_Detection//Scripts//')
import ads_creation as ads_crtn
import eda_analysis as eda
import feature_engg as feat_engg
import winsorize as wz
import regression as regr
from constants import ATTR, OUTLIER_METHOD, TIME_INVARIANT_ATTR, TIME_VARIANT_ATTR, SPLIT_PCT

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

def complt_ts_panel_lvl(df,panel_id_col,date_col):
    df = df.groupby(panel_id_col).apply(lambda x: complt_ts(x, date_col))
    df = df.reset_index(drop=True)
    #if we dont do this ffill here then the all the obs for inverter ids won't reach
    df[panel_id_col] = df[panel_id_col].ffill()
    df = df.groupby(panel_id_col).apply(lambda x: ads_crtn.missing_value_treatment(x))
    return df

#ADS creation, train-test split, outlier treatment
ads = ads_crtn.create_ads()
train_ads, test_ads = eda.train_test_split(ads,'INVERTER_ID',SPLIT_PCT)

#leaving out total yield since it value since it value increase exponentially and rightly so
outlier_feature = [feature for feature in ATTR if feature != 'TOTAL_YIELD'] 
clip_model = wz.winsorize(OUTLIER_METHOD)
clip_model.fit(train_ads[outlier_feature],'INVERTER_ID')
train_ads[outlier_feature] = clip_model.transform(train_ads[outlier_feature])
test_ads[outlier_feature] = clip_model.transform(test_ads[outlier_feature])


#Experiment 1
#Creating lagged features 
#While creating lagged features, we see that some timestamps are missing
#to correct this we will have to complete timestamps in our ads

#when you run groupby-apply functions in such a way that the numerical index get changed 
#and for eg, in previous df at index 100 inverter 1 was present but now inverter 0 will be there
#due to our operation in function now inverter 0 will be at index 100
#to circumvent this problem, pandas changes the index to groupby column and then the index
train_ads = complt_ts_panel_lvl(train_ads,'INVERTER_ID','DATE')
test_ads = complt_ts_panel_lvl(test_ads,'INVERTER_ID','DATE')

#making ADS stationary
non_stnry_invtr_list = eda.return_non_stnry_invtr_list(train_ads,'INVERTER_ID')
train_ads = eda.make_ads_stnry(train_ads,non_stnry_invtr_list,'INVERTER_ID')

non_stnry_invtr_list = eda.return_non_stnry_invtr_list(test_ads,'INVERTER_ID')
test_ads = eda.make_ads_stnry(test_ads,non_stnry_invtr_list,'INVERTER_ID')

#creating lagged features
train_ads = feat_engg.create_lagged_features(train_ads,TIME_VARIANT_ATTR,[192],'INVERTER_ID','DATE')
test_ads = feat_engg.create_lagged_features(test_ads, TIME_VARIANT_ATTR, [192], 'INVERTER_ID','DATE')

features = [feature + '_lag_192' for feature in TIME_VARIANT_ATTR]
features = features + TIME_INVARIANT_ATTR

models,metrics = regr.panel_wise_model(train_ads,test_ads,'INVERTER_ID','PER_TS_YIELD',features=features)

low_acc_invtr = [(key,metrics[key][3]) for key in metrics.keys() if metrics[key][3] < 0.8]
invtr_list = [val_pair[0] for val_pair in low_acc_invtr]

#these are all the inverters that had high amount of outliers in them
eda.get_box_plot(train_ads.loc[train_ads['INVERTER_ID'].isin(invtr_list),ATTR])
#as confirmed by line plots as well
eda.line_plot(ads.loc[ads['INVERTER_ID'].isin(invtr_list),['INVERTER_ID','PER_TS_YIELD','DATE']],'DATE','PER_TS_YIELD','INVERTER_ID')

#Some panels gave 97% accuracy which is pretty good
#however, some panel are giving very low accuracy for these panels, after looking at the features, found out
#that there is lot of noise, this lets try to solve this through feature selection

#Experiment 2
#Performing rfecv based on RF on the panels that gave poor accuracy in the previous step
metrics_rfe = {}
for invertor in invtr_list:
    feat_set = feat_engg.rfe_feature_selection(train_ads.loc[train_ads['INVERTER_ID']==invertor,],
                                                RandomForestRegressor(random_state=42),
                                                features,'PER_TS_YIELD')
    
    models,metrics = regr.panel_wise_model(train_ads.loc[train_ads['INVERTER_ID']==invertor,],
                                            test_ads.loc[test_ads['INVERTER_ID']==invertor,],
                                            'INVERTER_ID',
                                            'PER_TS_YIELD',features=feat_set)
    print('Added metric for :',invertor)
    metrics_rfe[invertor] = metrics[invertor]

#No improvemnt through rfecv as well, maybe because all the features are being taken as important
#which does not change anything