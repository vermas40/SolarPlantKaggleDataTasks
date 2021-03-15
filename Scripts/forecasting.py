import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import xgboost
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'//Users//mac_air//Documents//Documents//Side Projects//Kaggle_Anomaly_Detection//Scripts//')
import ads_creation as ads_crtn
import eda_analysis as eda
import feature_engg as feat_engg
import winsorize as wz
import regression as regr
import clustering as clst
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
    df['TIME_OF_DAY'] = df['DATE'].apply(lambda x: 1 if x.hour in range(6,18) else 0)
    #if we dont do this ffill here then the all the obs for inverter ids won't reach
    df[panel_id_col] = df[panel_id_col].ffill()
    df = df.groupby(panel_id_col).apply(lambda x: ads_crtn.missing_value_treatment(x))
    return df

#need to clean up this function
def get_optimal_lag(train_df,init_lag,pass_threshold,mode,panel_id_col,model_obj,feature_selection=False):
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

        models,metrics = regr.panel_wise_model(train_df_lagged,validation_df_lagged,panel_id_col,'PER_TS_YIELD',
                                              model_obj,features=features)
        r2 = metrics[train_df_lagged[panel_id_col].unique()[0]][3]
        print(r2)

        if mode == 'isolated':
            lag -= 8
            lag_list = [lag]
        elif mode == 'cumulative':
            lag -= 8
            lag_list.append(lag)
    return models,metrics

def fwd_step_rf_model(train_df,test_df,model_dict,panel_name):
    '''
    This function implements a perseverence model i.e. predicts 15 min in advance then uses that prediction 
    to churn out one more prediction

    Input
    1. train_df: pandas dataframe, training dataset
    2. test_df: pandas dataframe, testing dataset
    3. model_dict: dictionary, dict containing the prediciton models
    4. panel_name: str, string containing panel name for which the model will be run

    Returns:
    1. predicitions: list, which contains the predicitons made for 2 days
    '''
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

def act_vs_fcst(y_true,y_pred,date_series):
    '''
    This function plots actuals vs forecasts

    Input
    1. y_true: pandas series, ground truth
    2. y_pred: pandas series, predictions
    3. date_series: pandas series, series with the dates
    '''
    actual_df = pd.DataFrame({'DATE':date_series, 'Panel':'Actual',
                        'Actual':y_true})
    pred_df = pd.DataFrame({'DATE':date_series, 'Panel':'Predicted',
                        'Predicted':y_pred})

    plot_df = pd.concat([actual_df,pred_df])
    fig = px.line(plot_df,'DATE',['Actual','Predicted'])
    fig.show("svg")

    return

def remove_anomaly(df):
    '''
    This function removes instances wherein we have irradiation but AC power is 0

    Input:
    1. df: pandas dataframe, dataset containing noise

    Returns:
    df: pandas dataframe, dataset that is clean
    '''
    anam_idx = df.index[(df['IRRADIATION'] != 0) & (df['AC_POWER'] == 0)]
    df = df.drop(anam_idx)
    return df

if __name__ == '__main__':
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

    models,metrics = regr.panel_wise_model(train_ads_exp,test_ads_exp,'INVERTER_ID','PER_TS_YIELD',
                                            RandomForestRegressor(random_state = 42),features=features)


    low_acc_invtr = [(key,metrics[key][3]) for key in metrics.keys() if metrics[key][3] < 0.8]
    invtr_list = [val_pair[0] for val_pair in low_acc_invtr]

    #These are all the inverters that are in the plant 4136001. This plant is giving a lot of problems, and should
    #have a more specialised model made for it

    #these are all the inverters that had high amount of outliers in them
    eda.get_box_plot(train_ads.loc[train_ads['INVERTER_ID'].isin(invtr_list),ATTR])
    #as confirmed by line plots as well
    eda.line_plot(ads.loc[ads['INVERTER_ID'].isin(invtr_list),['INVERTER_ID','PER_TS_YIELD','DATE']],'DATE','PER_TS_YIELD','INVERTER_ID')

    #Creating lineage table
    metrics_r2 = {key:metrics[key][3] for key in metrics.keys()}

    lineage = pd.DataFrame(metrics_r2,index=[0])
    lineage = pd.melt(lineage, value_vars = lineage.columns, value_name = 'r2')

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
                                                'PER_TS_YIELD',
                                                RandomForestRegressor(random_state=42),
                                                features=feat_set)
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
    try_model,try_metric = get_optimal_lag(trial_df,192,0.8,'isolated',True)

    #running the pipeline in cumulative mode
    #better performance as compared to before but still did not cross 80% on validation dataset
    try_model,try_metric = get_optimal_lag(trial_df,192,0.8,'cumulative')

    #Experiment 4
    #Tried to get optimal lag with feature selection & ran the code on google colab
    #it took a lot of time and did not give favorable results

    #Experiment 5
    #Perseverence model
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

    #Experiment 6
    #Lets try XGBReggressor on the panels that did not perform well in experiment 1

    model = xgboost.XGBRegressor(random_state=42)

    train_ads_exp_xgb = train_ads_exp.loc[train_ads_exp['INVERTER_ID'].isin(invtr_list),]
    test_ads_exp_xgb = test_ads_exp.loc[test_ads_exp['INVERTER_ID'].isin(invtr_list),]
    models_xgb,metrics_xgb = regr.panel_wise_model(train_ads_exp_xgb,test_ads_exp_xgb,'INVERTER_ID','PER_TS_YIELD',model,features=features)

    low_acc_invtr = [(key,metrics_xgb[key][3]) for key in metrics_xgb.keys() if metrics_xgb[key][3] < 0.8]
    invtr_list = [val_pair[0] for val_pair in low_acc_invtr]

    metrics_r2_xgb = {key:metrics[key][3] for key in metrics_xgb.keys()}

    lineage_xgb = pd.DataFrame(metrics_r2_xgb,index=[0])
    lineage_xgb = pd.melt(lineage_xgb, value_vars = lineage_xgb.columns, value_name = 'r2')

    #none of the inverters achieved more than 80% accuracy. Will have to move on to clustered model maybe!

    #Experiment 7
    #making a model at a plant level
    #i.e. a model for each plant. plant 4135001 is modelling nicely and is giving a 95% accuracy
    #whereas plant 4136001 is the one which is giving only 52% accuracy and we will focus on 
    #it more closely now
    models,metrics = regr.panel_wise_model(train_ads_exp,test_ads_exp,
                                            'PLANT_ID','PER_TS_YIELD',
                                            RandomForestRegressor(random_state=42),
                                            features=features)


    #Experiment 8
    #Clustering within plant 4136001 and making a model for each cluster

    train_ads_clst = train_ads.loc[train_ads['PLANT_ID']==4136001.0].reset_index(drop=True)
    test_ads_clst = test_ads.loc[test_ads['PLANT_ID']==4136001.0].reset_index(drop=True)


    #running the below loop to give us the number of clusters that are optimal for this clustering exercise
    #10 clusters are coming out to be most optimal, so lets go ahead witthat

    n_clust = clst.get_n_clusters(train_ads_clst.loc[:,~train_ads_clst.columns.isin(IDENTIFIERS)],2,11)

    #kmeans clustering with 10 clusters
    clusterer = KMeans(n_clusters=n_clust, random_state=42)
    cluster_labels = clusterer.fit_predict(train_ads_clst.loc[:,~train_ads_clst.columns.isin(IDENTIFIERS)])

    train_ads_clst = pd.concat([pd.DataFrame({'CLUSTER_ID':list(cluster_labels)}),train_ads_clst],axis=1)

    #putting the cluster IDs in the test ads as well
    invtr_clst_map = train_ads_clst[['INVERTER_ID','CLUSTER_ID']].drop_duplicates().reset_index(drop=True)
    test_ads_clst = pd.merge(test_ads_clst,invtr_clst_map,on='INVERTER_ID',how='inner')

    train_ads_clst = feat_engg.create_lagged_features(train_ads_clst,TIME_VARIANT_ATTR,[192],'INVERTER_ID','DATE')
    test_ads_clst = feat_engg.create_lagged_features(test_ads_clst, TIME_VARIANT_ATTR, [192], 'INVERTER_ID','DATE')

    features = [feature + '_lag_192' for feature in TIME_VARIANT_ATTR]
    features = features + TIME_INVARIANT_ATTR

    models,metrics = regr.panel_wise_model(train_ads_clst,test_ads_clst,
                                            'CLUSTER_ID','PER_TS_YIELD',
                                            RandomForestRegressor(random_state=42),
                                            features=features)

    #getting pretty bad accuracy with clustered ads as well 

    #Experiment 9
    #Finding the lag at which we can get atleast 80% accuracy 
    #Do not get high accuracy at any lag from 2 days out to 2 hours
    #cannot make a perseverence model at a plan level using this technique
    train_ads_exp = train_ads.loc[train_ads['PLANT_ID']==4136001.0].reset_index(drop=True)
    test_ads_exp = test_ads.loc[test_ads['PLANT_ID']==4136001.0].reset_index(drop=True)

    _,metrics = get_optimal_lag(train_ads_exp,192,0.8,'isolated','PLANT_ID',RandomForestRegressor(random_state=42))


    #Experiment 10
    #creating a time of the day feature and running a model on top of that
    #Useless since IRRADIATION was already capturing that info

    train_ads_exp = complt_ts_panel_lvl(train_ads,'INVERTER_ID','DATE')
    test_ads_exp = complt_ts_panel_lvl(test_ads,'INVERTER_ID','DATE')

    train_ads_exp = feat_engg.create_lagged_features(train_ads_exp,TIME_VARIANT_ATTR,[192],'INVERTER_ID','DATE')
    test_ads_exp = feat_engg.create_lagged_features(test_ads_exp, TIME_VARIANT_ATTR, [192], 'INVERTER_ID','DATE')

    features = [feature + '_lag_192' for feature in TIME_VARIANT_ATTR]
    features = features + TIME_INVARIANT_ATTR + ['TIME_OF_DAY']

    train_ads_exp = train_ads_exp.loc[train_ads_exp['PLANT_ID']==4136001.0].reset_index(drop=True)
    test_ads_exp = test_ads_exp.loc[test_ads_exp['PLANT_ID']==4136001.0].reset_index(drop=True)
    
    models,metrics = regr.panel_wise_model(train_ads_exp,test_ads_exp,
                                        'PLANT_ID','PER_TS_YIELD',
                                        RandomForestRegressor(random_state=42),
                                        features=features)
    
    #on close inspection for Inverter 'Qf4GUc1pJu5T6c6', the bad performance is due to 
    #anomalies in data itself. Lets clean test data once and then see the performance
    #this difference in actual vs predicted may help in anomaly detection as well

    #Experiment 11
    #training and testing the model on clean data that is free of noise/anomalies

    train_ads_exp = remove_anomaly(train_ads)
    test_ads_exp = remove_anomaly(test_ads)

    train_ads_exp = complt_ts_panel_lvl(train_ads_exp,'INVERTER_ID','DATE')
    test_ads_exp = complt_ts_panel_lvl(test_ads_exp,'INVERTER_ID','DATE')

    train_ads_exp = feat_engg.create_lagged_features(train_ads_exp,TIME_VARIANT_ATTR,[192],'INVERTER_ID','DATE')
    test_ads_exp = feat_engg.create_lagged_features(test_ads_exp, TIME_VARIANT_ATTR, [192], 'INVERTER_ID','DATE')

    features = [feature + '_lag_192' for feature in TIME_VARIANT_ATTR]
    features = features + TIME_INVARIANT_ATTR + ['TIME_OF_DAY']
    
    models,metrics = regr.panel_wise_model(train_ads_exp,test_ads_exp,
                                        'INVERTER_ID','PER_TS_YIELD',
                                        RandomForestRegressor(random_state=42),
                                        features=features)

    #training and testing on noise free data gets us 80+% accuracy on both the datasets
    #The difference between predicted and actual can also be used as a robust anomaly detection
    #piece
