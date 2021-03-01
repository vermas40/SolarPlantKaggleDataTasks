#library imports
import pandas as pd
import os
from datetime import datetime
import numpy as np
os.chdir(r'//Users//mac_air//Documents//Documents//Side Projects//Kaggle_Anomaly_Detection')

def merge_data(plant_name):
    '''
    This function creates the ads for a plant
    '''

    #reading in the file
    path = '//Users//mac_air//Documents//Documents//Side Projects//Kaggle_Anomaly_Detection'
    gen_data = pd.read_csv(path + '/data/{}_Generation_Data.csv'.format(plant_name))
    #formatting the date to date formate
    gen_data['DATE_TIME'] = gen_data['DATE_TIME'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y %H:%M"))
    
    sns_weather_data = pd.read_csv(path + '/data/{}_Weather_Sensor_Data.csv'.format(plant_name))
    #formatting the date
    sns_weather_data['DATE_TIME'] = sns_weather_data['DATE_TIME'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    
    #below line has to be uncommented when I am doing the missing timestamp treatment
    ads = pd.merge(gen_data, sns_weather_data, left_on=['DATE_TIME','PLANT_ID'], right_on=['DATE_TIME','PLANT_ID'],how='left')
    ads.rename(columns = {'SOURCE_KEY_x':'INVERTER_ID', 'SOURCE_KEY_y':'PANEL_ID'}, inplace=True)
    return ads

def create_daily_yield(df):
    df['PER_TS_YIELD'] = np.nan
    yield_df = df.groupby('INVERTER_ID')['TOTAL_YIELD'].agg(lambda x: list(x.diff())).to_frame().reset_index()

    for inverter in df['INVERTER_ID'].unique():
        df.loc[df['INVERTER_ID'] == inverter, 'PER_TS_YIELD'] = yield_df.loc[yield_df['INVERTER_ID'] == inverter,'TOTAL_YIELD'].values[0]

    return df

def create_features(df):
    #creating per timestamp yield
    df = create_daily_yield(df)
    df = df.rename(columns={'DATE_TIME':'DATE'})
    return df

def missing_value_treatment(df):
    '''
    1. Forward filling the missing values, assuming that if the sensor did not record the reading,
    then the next value can be rightly picked up as the value

    2. Filling the rest of the missing values with 0
    '''
    df.ffill(inplace=True)    
    df.fillna(0, inplace=True)
    return df

def create_ads():
    #creating ads for plant 1 and plant 2
    ads = pd.DataFrame()
    #merging sensor and weather data
    ads_plant_1 = merge_data('Plant_1')
    ads_plant_2 = merge_data('Plant_2')
    ads = pd.concat([ads_plant_1, ads_plant_2]).reset_index(drop=True)
    #creating date and per time stamp yields
    ads = create_features(ads)
    #performing missing value treatment
    ads = ads.groupby('INVERTER_ID').apply(lambda x: missing_value_treatment(x))
    ads = ads.reset_index(drop=True)
    return ads