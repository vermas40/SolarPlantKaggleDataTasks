#library imports
import pandas as pd
import os
from datetime import datetime
import numpy as np
os.chdir(r'/kaggle/working/SolarPlantKaggleDataTasks/Scripts')
from constants import IDENTIFIERS
def merge_data(plant_name):
    '''
    This function creates the ads for a plant
    Input:
    1. plant_name: Takes in the plant name as specified in the csv files for both the plants

    Return:
    ads: Merged dataset of generation ads and sensor ads with their dates formatted properly
    '''

    #reading in the file
    path = r'/kaggle/working/SolarPlantKaggleDataTasks'
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
    '''
    This function creates per timestamp yield generated at the plant using total yield column

    Input:
    1. df: pandas dataframe with all the required columns

    Return:
    1. df: pandas dataframe with the per timestamp yield
    '''
    df['PER_TS_YIELD'] = np.nan
    yield_df = df.groupby('INVERTER_ID')['TOTAL_YIELD'].agg(lambda x: list(x.diff())).to_frame().reset_index()

    for inverter in df['INVERTER_ID'].unique():
        df.loc[df['INVERTER_ID'] == inverter, 'PER_TS_YIELD'] = yield_df.loc[yield_df['INVERTER_ID'] == inverter,'TOTAL_YIELD'].values[0]

    return df

def create_features(df):
    '''
    This function just calls the daily yield function as well as renames the date column
    '''
    #creating per timestamp yield
    df = create_daily_yield(df)
    df = df.rename(columns={'DATE_TIME':'DATE'})
    #creating time of day flag
    #night is 0, day is 1
    df['TIME_OF_DAY'] = df['DATE'].apply(lambda x: 1 if x.hour in range(6,18) else 0)

    return df

#data leak is happening in this function
def missing_value_treatment(df):
    '''
    Input:
    1. df: pandas dataframe that is subsetted for one inverter ID only

    Return:
    1. df: input pandas dataframe + missing values treated through forward filling and imputation with 0
    '''
    #Forward filling all the identifier columns
    df[IDENTIFIERS] = df[IDENTIFIERS].ffill()

    #At night time, irradiation, ac & dc power are 0
    night_idx = df.index[df['TIME_OF_DAY'] == 0].tolist()
    df.at[night_idx,['IRRADIATION','DC_POWER','AC_POWER','PER_TS_YIELD']] = 0

    #When there are no values, we assume that daily and total yield remains the same
    df[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','DAILY_YIELD','TOTAL_YIELD']] \
        = df[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','DAILY_YIELD','TOTAL_YIELD']].ffill()

    #all the incidents when irradiation & per time stamp yield is na
    #this would now be only day time
    na_idx = df.index[(np.isnan(df['IRRADIATION'])) | (np.isnan(df['PER_TS_YIELD'])) == True].tolist()

    #replacing the na values with average
    #Making the assumption that during day time irradiation can be the average amount 
    #consequently the power generated would also be average
    for attribute in ['IRRADIATION','AC_POWER','DC_POWER','PER_TS_YIELD']:
        df.at[na_idx,attribute] = df[attribute].mean()

    return df

def create_ads():
    '''
    This is a wrapper function for all the other functions in this module to ease ads creation 
    in other modules
    '''
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