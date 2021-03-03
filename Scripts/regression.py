import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import warnings
from merf import merf
warnings.filterwarnings('ignore')

os.chdir(r'//Users//mac_air//Documents//Documents//Side Projects//Kaggle_Anomaly_Detection//Scripts')
import ads_creation as ads_crtn
import eda_analysis as eda
import feature_engg as feat_engg
import winsorize as wz
from constants import ATTR, TIME_INVARIANT_ATTR, TIME_VARIANT_ATTR, SPLIT_PCT, OUTLIER_METHOD


def panel_wise_model(train,test,panel_id_col,label,features=ATTR):
    '''
    This function creates one random forest model for each panel in ADS
    and returns 2 dictionaries housing the models and metrics

    Input:
    1. train: pandas dataframe, this is the training dataset
    2. test: pandas dataframe, this is the testing dataset
    3. panel_id_col: str, this is the name of the column that contains the panel IDs
    4. label: str, target variable
    5. features: list, this is the list of variables to be used for model creation

    Return:
    1. model_dict: python dictionary, this dict contains model objects for each panel
    2. metric_dict: python dictionary, this dict contains error metrics for each panel
    '''

    model_dict = {}
    metric_dict = {}
    
    for panel in train[panel_id_col].unique():
        train_df = train.loc[train[panel_id_col]==panel,]
        test_df = test.loc[test[panel_id_col]==panel,]

        scaler = StandardScaler()
        scaler.fit(train_df[features])

        train_df_scaled = scaler.transform(train_df[features])
        test_df_scaled = scaler.transform(test_df[features])
        print('Modelling for panel:',panel)

        model_dict[panel] = RandomForestRegressor(random_state=42)
        model_dict[panel].fit(train_df_scaled, train_df[label])

        y_pred = model_dict[panel].predict(test_df_scaled)
        
        mape = mean_absolute_percentage_error(test_df[label],y_pred)
        mae = mean_absolute_error(test_df[label],y_pred)
        mse = mean_squared_error(test_df[label],y_pred)
        r2 = r2_score(test_df[label],y_pred)

        metric_dict[panel] = (mape,mae,mse,r2)

    return model_dict, metric_dict

#creating ads + outlier treatment x2
#TODO: Can you do Z score treatment twice?
#Even with stricter Z scores, the outliers are not getting capped
#implemented winsorizing for outlier treatment
if __name__ == '__main__':
    ads = ads_crtn.create_ads()
    train_ads, test_ads = eda.train_test_split(ads,'INVERTER_ID',SPLIT_PCT)

    outlier_feature = [feature for feature in ATTR if feature != 'TOTAL_YIELD'] 
    clip_model = wz.winsorize(OUTLIER_METHOD)
    clip_model.fit(train_ads[outlier_feature],'INVERTER_ID')
    train_ads[outlier_feature] = clip_model.transform(train_ads[outlier_feature])
    test_ads[outlier_feature] = clip_model.transform(test_ads[outlier_feature])


    #making ADS stationary
    non_stnry_invtr_list = eda.return_non_stnry_invtr_list(train_ads,'INVERTER_ID')
    train_ads = eda.make_ads_stnry(train_ads,non_stnry_invtr_list,'INVERTER_ID','DATE')
    
    non_stnry_invtr_list = eda.return_non_stnry_invtr_list(test_ads,'INVERTER_ID')
    test_ads = eda.make_ads_stnry(test_ads,non_stnry_invtr_list,'INVERTER_ID','DATE')
    
    #getting lag values for the target variables
    lag_values = train_ads[['INVERTER_ID','PER_TS_YIELD']].groupby('INVERTER_ID').agg(lambda x: eda.pick_pacf(x))
    distinct_lag_values = np.unique(lag_values.PER_TS_YIELD.sum())

    #creating lagged features
    train_ads = feat_engg.create_lagged_features(train_ads,['PER_TS_YIELD'],distinct_lag_values,'INVERTER_ID','DATE')
    test_ads = feat_engg.create_lagged_features(test_ads,['PER_TS_YIELD'],distinct_lag_values,'INVERTER_ID','DATE')
    #Experiment 1
    #one Random Forest model for all panels, only one lagged feature, no scaling

    features = [feature for feature in ATTR if feature not in ['PER_TS_YIELD','INVERTER_ID']]
    features.append('PER_TS_YIELD_lag_1')
    label = 'PER_TS_YIELD'

    model = RandomForestRegressor(random_state=42)
    model.fit(train_ads[features],train_ads[label])
    y_pred = model.predict(test_ads[features])

    #mape is not preferred here because our time series values can be 0 and mape becomes inf if y_true can be 0
    #and hence unreliable

    #mse is scale dependent, therefore using it across different time series is not possible
    #r-squared is the most reliable metric that can be used here and we will go with that
    mape = mean_absolute_percentage_error(test_ads[label],y_pred)
    mae = mean_absolute_error(test_ads[label],y_pred)
    mse = mean_squared_error(test_ads[label],y_pred)
    r2 = r2_score(test_ads[label],y_pred)

    #Experiment 2
    #Same setup as above but with scaling done

    scaler = StandardScaler()
    scaler.fit(train_ads[features])

    train_ads_scaled = scaler.transform(train_ads[features])
    test_ads_scaled = scaler.transform(test_ads[features])

    model = RandomForestRegressor(random_state=42)
    model.fit(train_ads_scaled,train_ads[label])

    y_pred = model.predict(test_ads_scaled)

    mape = mean_absolute_percentage_error(test_ads[label],y_pred)
    mae = mean_absolute_error(test_ads[label],y_pred)
    mse = mean_squared_error(test_ads[label],y_pred)
    r2 = r2_score(test_ads[label],y_pred)

    #Experiment 3
    #Making individual models for each panel and seeing the accuracy 
    #Seeing very varied performance across different panels, some have 99%, some have 65 and some have negative R2

    models, metrics = panel_wise_model(train_ads,test_ads,'INVERTER_ID','PER_TS_YIELD',features)

    #lets look at the inverters with low accuracy scores
    low_acc_invtr = [(key,metrics[key][3]) for key in metrics.keys() if metrics[key][3] < 0.9]
    invtr_list = [val_pair[0] for val_pair in low_acc_invtr]

    #these are all the inverters that had high amount of outliers in them
    eda.get_box_plot(ads.loc[ads['INVERTER_ID'].isin(invtr_list),ATTR])
    #as confirmed by line plots as well
    eda.line_plot(ads.loc[ads['INVERTER_ID'].isin(invtr_list),['INVERTER_ID','PER_TS_YIELD','DATE']],'DATE','PER_TS_YIELD','INVERTER_ID')

    #earlier on we were getting low accuracies for some panels because of outliers
    #but with winsorizing outlier treatment average accuracy is around 99%


    #Experiment 4
    #Creating a mixed effects random forest model
    features_time_variant = [feature for feature in TIME_VARIANT_ATTR if feature not in ['PER_TS_YIELD','INVERTER_ID']]
    features_time_variant.append('PER_TS_YIELD_lag_1')
    label = 'PER_TS_YIELD'

    merf = merf.MERF()
    merf.fit(train_ads[TIME_INVARIANT_ATTR], train_ads[TIME_VARIANT_ATTR], train_ads['INVERTER_ID'], train_ads[label])
    #merf model took a lot of time. Hence got abandoned


    #with these experiments, one mistake we have made is that we cannot predict for the future
    #without having the future values of our input variables
    #what we have done excellently till now if panel regression; However, the task is panel forecasting
    #we might not have these values with us in the future so our model could be a bust. We can remedy this
    #by creating a dataset with lagged variables