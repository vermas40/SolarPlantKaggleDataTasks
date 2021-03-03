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

    Input:
    1. df: pandas dataframe, this is the ads subsetted for one panel
    2. date_col: str, this is the name of the column containing the dates

    Return
    1. dates: pandas dataframe, containing all the timestamps from start to end
    '''
    dates = pd.date_range(start=df[date_col].min(), end = df[date_col].max(),freq = '15min')
    dates = dates.to_frame(index=False)
    dates = dates.rename(columns = {0:'DATE'})
    return dates

def complt_ts(df,date_col):
    '''
    This function introduces the missing time stamps in ads

    Input:
    1. df: pandas dataframe, this is the ads subsetted for one panel
    2. date_col: str, this is the name of the column containing the dates

    Return
    1. df: pandas dataframe, containing all the timestamps from start to end
    '''
    date_list = create_ts_list(df,date_col)
    df = pd.merge(left=date_list,right=df,how='left', on='DATE')
    return df

def complt_ts_panel_lvl(df,panel_id_col,date_col):
    '''
    This function completes the time stamps for all panels and applies MVT as well

    Input:
    1. df: pandas dataframe, this is the ads for which timestamps need to be completed
    2. panel_id_col: str, the name of the column containing the panel IDs
    3. date_col, str, the name of the column containing the dates

    Returns
    1. df: pandas dataframe, containing all the timestamps from start to end
    '''
    df = df.groupby(panel_id_col).apply(lambda x: complt_ts(x, date_col))
    df = df.reset_index(drop=True)
    #if we dont do this ffill here then the all the obs for inverter ids won't reach
    df[panel_id_col] = df[panel_id_col].ffill()
    df = df.groupby(panel_id_col).apply(lambda x: ads_crtn.missing_value_treatment(x))
    return df

def get_optimal_lag(train_df,init_lag,pass_threshold,mode,feature_selection=False):
    '''
    This function gets the lag that is most optimal in predicting the target variable
    Either in isolated mode or cumulative mode

    Input:
    1. train_df: pandas dataframe, this is the ads
    2. init_lag: int, the largest lag to begin with
    3. pass_threshold: float, the minimum r-squared values required
    4. mode: str, isolated or cumulative mode
    - Isolated mode: just creates one set of lags and check in silo
    - Cumulative mode: creates a group of lags like 2 day and 1 day lagged variables and checks accuracy together

    Return:
    1. model_dict: python dictionary, this dict contains model objects for each panel
    2. metric_dict: python dictionary, this dict contains error metrics for each panel
    '''

    #Creating a validation df because here we will be choosing the optimal lag
    #this is a form of tuning and test dataset should not be used
    train_df,validation_df = eda.train_test_split(train_df,'INVERTER_ID',0.9)
    r2 = 0
    lag = init_lag
    lag_list = [init_lag]

    #while loop runs till the time the condition is true
    #here we want the loop to stop as soon as one condition is met and the condition to become false
    #if we do or between the two conditions, while loop will still keep running because 0 + 1 = 1 and it will keep running
    #but if we put 'and' then 0 * 1 = 0 and we get the desired result
    while (r2 < pass_threshold) and (lag > 0):
        print('Doing for lag: ',lag)
        train_df_lagged = feat_engg.create_lagged_features(train_df,TIME_VARIANT_ATTR,lag_list,'INVERTER_ID','DATE')
        validation_df_lagged = feat_engg.create_lagged_features(validation_df,TIME_VARIANT_ATTR,lag_list,'INVERTER_ID','DATE')

        features = [feature for feature in train_df_lagged.columns if '_lag_' in feature]
        features = features + TIME_INVARIANT_ATTR
        
        if feature_selection == True:
            features = feat_engg.rfe_feature_selection(train_df_lagged,
                                                RandomForestRegressor(random_state=42),
                                                features,'PER_TS_YIELD')

        models,metrics = regr.panel_wise_model(train_df_lagged,validation_df_lagged,'INVERTER_ID','PER_TS_YIELD',
                                              features=features)
        r2 = metrics[train_df_lagged['INVERTER_ID'].unique()[0]][3]
        print(r2)

        if mode == 'isolated':
            lag -= 8
            lag_list = [lag]
        elif mode == 'cumulative':
            lag -= 8
            lag_list.append(lag)
    return models,metrics

def fwd_step_rf_model(train_df,test_df,model_dict,panel_name):
    
    predictions = []
    scaler = StandardScaler()
    scaler.fit(train_df.values.reshape(-1,1))
    train_df_scaled = scaler.transform(train_df.values.reshape(-1,1))
    pred_obs = train_df_scaled[-1]
    
    for iter in range(len(test_df)):
        if iter == 0:
            pred_obs = model_dict[panel_name].predict(pred_obs.reshape(-1,1))
            predictions.append(pred_obs)
        else:
            pred_obs = model_dict[panel_name].predict(scaler.transform(pred_obs.reshape(-1,1)))
            predictions.append(pred_obs)

    return predictions

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
train_ads = eda.make_ads_stnry(train_ads,non_stnry_invtr_list,'INVERTER_ID','DATE')

non_stnry_invtr_list = eda.return_non_stnry_invtr_list(test_ads,'INVERTER_ID')
test_ads = eda.make_ads_stnry(test_ads,non_stnry_invtr_list,'INVERTER_ID','DATE')

#Creating lagged features which are lagged by exactly 2 days so that they can then be used for prediction

train_ads_exp = feat_engg.create_lagged_features(train_ads,TIME_VARIANT_ATTR,[192],'INVERTER_ID','DATE')
test_ads_exp = feat_engg.create_lagged_features(test_ads, TIME_VARIANT_ATTR, [192], 'INVERTER_ID','DATE')

features = [feature + '_lag_192' for feature in TIME_VARIANT_ATTR]
features = features + TIME_INVARIANT_ATTR

models,metrics = regr.panel_wise_model(train_ads_exp,test_ads_exp,'INVERTER_ID','PER_TS_YIELD',features=features)

low_acc_invtr = [(key,metrics[key][3]) for key in metrics.keys() if metrics[key][3] < 0.9]
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
    feat_set = feat_engg.rfe_feature_selection(train_ads_exp.loc[train_ads_exp['INVERTER_ID']==invertor,],
                                                RandomForestRegressor(random_state=42),
                                                features,'PER_TS_YIELD')
    
    models,metrics = regr.panel_wise_model(train_ads_exp.loc[train_ads_exp['INVERTER_ID']==invertor,],
                                            test_ads_exp.loc[test_ads_exp['INVERTER_ID']==invertor,],
                                            'INVERTER_ID',
                                            'PER_TS_YIELD',features=feat_set)
    print('Added metric for :',invertor)
    metrics_rfe[invertor] = metrics[invertor]

#No improvemnt through rfecv as well, maybe because all the features are being taken as important
#which does not change anything


#Experiment 3
#Finding the lag which gives the best accuracy, starting from 2 days out and moving closer to t-1
#All the different lags are looked at in isolation or cumulatively
#conducting this experiment with only one panel, if successful only then will scale up to other panels

trial_df = train_ads.loc[train_ads['INVERTER_ID']==invtr_list[1],]

#running it in isolated mode first
try_model,try_metric = get_optimal_lag(trial_df,192,0.8,'isolated')

#running the pipeline in cumulative mode
#better performance as compared to before but still did not cross 80% on validation dataset
try_model,try_metric = get_optimal_lag(trial_df,192,0.8,'cumulative')

#Experiment 4
#Tried to get optimal lag with feature selection & ran the code on google colab
#it took a lot of time and did not give favorable results

#Experiment 5
#Forecast only 15 min in advance and use that forecast as input and forecast for another 15 min

train_ads_exp = feat_engg.create_lagged_features(train_ads,['PER_TS_YIELD'],[1],'INVERTER_ID','DATE')
test_ads_exp = feat_engg.create_lagged_features(test_ads, ['PER_TS_YIELD'], [1], 'INVERTER_ID','DATE')

features = [feature for feature in train_ads_exp.columns if '_lag_' in feature]

models,metrics = regr.panel_wise_model(train_ads_exp,test_ads_exp,'INVERTER_ID','PER_TS_YIELD',features=features)

low_acc_invtr = [(key,metrics[key][3]) for key in metrics.keys() if metrics[key][3] < 0.8]
invtr_list = [val_pair[0] for val_pair in low_acc_invtr]

#Checking how the single step forward model works
fwd_models = {}
for invtr in invtr_list:
    print('Modelling for panel: ',invtr)
    preds = fwd_step_rf_model(train_ads_exp.loc[train_ads_exp['INVERTER_ID'] == invtr,features],
                            test_ads_exp.loc[test_ads_exp['INVERTER_ID'] == invtr,features],
                            models,invtr)


    y_pred = [prediction for sublist in preds for prediction in sublist]

    r2 = r2_score(test_ads_exp.loc[test_ads_exp['INVERTER_ID'] == invtr,'PER_TS_YIELD'],y_pred)
    fwd_models[invtr] = r2

#the models failed miserable with all of them being worse than an average model!
#This marks the end of supervised methods for forecasting into the future
#They have not performed well and maybe out of depth when it comes to forecasting into the future
#and not just approximating a function
#in the next module, we can look at the performance of actual time series forecast models