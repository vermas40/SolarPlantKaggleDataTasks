import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf,pacf, adfuller, kpss
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#importing user defined modules
os.chdir(r'//Users//mac_air//Documents//Documents//Side Projects//Kaggle_Anomaly_Detection//Scripts')
import ads_creation as ads_crtn
import winsorize as wz
from constants import CORR_THRESHOLD, OUTLIER_METHOD, ATTR, TIME_VARIANT_ATTR, \
                      TIME_INVARIANT_ATTR, SPLIT_PCT

def train_test_split(df,panel_id_col,split_pct):
    '''
    This function does train-test split at a panel level dividing each panel into training and testing
    observations depending on the split percentage

    Input
    1. df: pandas dataframe, this is the ads that needs to be split
    2. panel_id_col: str, this is the column name that has panel IDs
    3. split_pct: float, contains the percentage split

    Return
    1. train_data: pandas dataframe, dataset that is to be used for training
    2. test_data: pandas dataframe, dataset that is to be used for testing purposes
    '''
    assert split_pct < 1, 'Split Percentage should be divided by 100'
    train_indices = []
    test_indices = []
    for panel in df[panel_id_col].unique():
        obs_index = df.loc[df[panel_id_col]==panel].index
        train_upper_limit = int(split_pct * len(obs_index))
        train_indices.append(obs_index[0:train_upper_limit])
        test_indices.append(obs_index[train_upper_limit:len(obs_index)])
    train_indices = [item for sublist in train_indices for item in sublist]
    test_indices = [item for sublist in test_indices for item in sublist]

    train_data = df.loc[train_indices,].reset_index(drop=True)
    test_data = df.loc[test_indices,].reset_index(drop=True)
    return train_data, test_data

def box_plot(df, x_axis_attribute, cluster_name):
    '''
    This function creates a box plot for each panel ID 
    df: the dataset that will be used for plotting
    x_axis_attribute: the column that will be plotted
    cluster_name: the column that identifies the different clusters present in the data
    '''
    fig = px.box(df, 
            x = x_axis_attribute, 
            color = cluster_name, 
            facet_row = cluster_name, 
            facet_row_spacing = 0.02)
    for anno in fig['layout']['annotations']:
        anno['text']=''

    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.YAxis:
            fig.layout[axis].title.text = ''
        if type(fig.layout[axis]) == go.layout.XAxis:
            fig.layout[axis].title.text = ''

    fig.update_layout(
        # keep the original annotations and add a list of new annotations:
        annotations = list(fig.layout.annotations) + 
        [go.layout.Annotation(
                x=0.5,
                y=-0.08,
                font=dict(
                    size=16, color = 'blue'
                ),
                showarrow=False,
                text=x_axis_attribute,
                textangle=-0,
                xref="paper",
                yref="paper"
            )
        ]
    )
    
    fig.show()
    return 

def line_plot(df, x_axis_attribute, y_axis_attribute, panel_id_col):
    '''
    This function makes line plots for each panel in the dataset
    Input
    1. df: Pandas dataframe, this is the ads
    2. attributes: both x and y attributes are supplied. They can be a list (when plotting 2 lines) or str
    3. panel_id_col: str, it is the column name that contains the panel IDs
    '''
    if type(y_axis_attribute) != str:
        text_for_y_axis = ' & '.join(y_axis_attribute)
    else:
        text_for_y_axis = y_axis_attribute
    
    if type(x_axis_attribute) != str:
        text_for_x_axis = ' & '.join(x_axis_attribute)
    else:
        text_for_x_axis = x_axis_attribute

    fig = px.line(df,
            x = x_axis_attribute,
            y = y_axis_attribute,
            color = panel_id_col, 
            facet_row = panel_id_col,
            facet_row_spacing = 0.02)

    for anno in fig['layout']['annotations']:
        anno['text']=''

    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.YAxis:
            fig.layout[axis].title.text = ''
        if type(fig.layout[axis]) == go.layout.XAxis:
            fig.layout[axis].title.text = ''

    fig.update_layout(
        # keep the original annotations and add a list of new annotations:
        annotations = list(fig.layout.annotations) + 
        [go.layout.Annotation(
                x=-0.07,
                y=0.5,
                font=dict(
                    size=16, color = 'blue'
                ),
                showarrow=False,
                text=text_for_y_axis,
                textangle=-90,
                xref="paper",
                yref="paper"
            )
        ] +
        [go.layout.Annotation(
                x=0.5,
                y=-0.08,
                font=dict(
                    size=16, color = 'blue'
                ),
                showarrow=False,
                text=text_for_x_axis,
                textangle=-0,
                xref="paper",
                yref="paper"
            )
        ]
    )
    
    fig.show()

    return 

def pick_pacf(df,alpha=0.05,nlags=192):
    '''
    This function returns the lags in the timeseries which are highly correlated with the original timeseries
    Input
    1. df: pandas series, this is the column for which we are trying to find AR lag
    2. metric: str, what metric to be calculated - acf/pacf
    3. alpha: float, confidence interval
    4. nlags: int, the no. of lags to be tested

    Return
    1. lags: list, this contain the list of all the lags (# of timestamps) that are highly correlated
    '''
    

    values,conf_int = pacf(df.values,alpha=alpha,nlags=nlags)

    lags = []
    #in the pacf function, confidence interval is centered around pacf values
    #we need them to be centered around 0, this will produce the intervals we see in the graph
    conf_int_cntrd = [value[0] - value[1] for value in zip(conf_int,values)]
    for obs_index, obs in enumerate(zip(conf_int_cntrd,values)):
        if (obs[1] >= obs[0][1]) & (obs[1] >= CORR_THRESHOLD): #obs[0][1] contains the high value of the conf int
            lags.append(obs_index)
        elif (obs[1] <= obs[0][0]) & (obs[1] <= -1 * CORR_THRESHOLD): #obs[0][0] contains the low value of the conf_int
            lags.append(obs_index)
    lags.remove(0) #removing the 0 lag for auto-corr with itself
    #keeping statistically significant and highly correlated lags
    return lags 

def pick_acf(df,nlags=192):
    '''
    This funciton takes returns the ACF value for a MA model for a time series

    Input
    1. df: pandas series, this is the series for which we want to find ACF value
    2. nlags: the number of lags to be taken into consideration for ACF

    Returns
    1. The lags value at which ACF cuts off
    '''
    acf_values = acf(df.values)
    acf_values = np.round(acf_values,1)
    q = np.where(acf_values <= 0.2)[0][0]
    return [q]

def correlated_var(df, target_variable):
    '''
    This function finds all the variables that are highly correlated with the target variable
    
    Input:
    1. df: pandas dataframe, this is the ads
    2. target_variable: str, the variable name with which we want to find correlation

    return:
    corr_var: list, list of variables that are highly correlated with the target variable
    '''
    corr_df = df.corr().reset_index()
    corr_df = pd.melt(corr_df, id_vars = 'index', var_name = 'variable 2', value_name = 'corr_val')
    corr_var = corr_df.loc[(abs(corr_df['corr_val']) > CORR_THRESHOLD) & (abs(corr_df['corr_val']!=1)) & (corr_df['index'] == target_variable),]
    corr_var = corr_var['variable 2'].unique()
    return corr_var

def get_box_plot(df):
    '''
    This funciton creates box plots for all the columns and panels present in the data
    '''
    for variable in [column for column in df.columns if column != 'INVERTER_ID']:
        for i in range(0,df['INVERTER_ID'].nunique(),6):
            box_plot(df.loc[df['INVERTER_ID'].isin(df['INVERTER_ID'].unique()[i:i+6].tolist()),], variable, 'INVERTER_ID')
    return 

def adf_kpss_test(df):

    '''
    This function conducts the ADS and KPSS tests for stationarity
    --------------
    Case 1: Both tests conclude that the series is not stationary - The series is not stationary
    Case 2: Both tests conclude that the series is stationary - The series is stationary
    Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. 
    Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
    Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary.
    Differencing is to be used to make series stationary. The differenced series is checked for stationarity.
    ---------------

    Input:
    1. df: pandas series, the series which we want to test if it is stationary or not

    Return:
    1. 0/1: 0 means stationary, 1 means non-stationary
    '''

    p_value_kpss = kpss(df)[1]
    p_value_adf = adfuller(df)[1]

    if (p_value_adf < 0.05) & (p_value_kpss > 0.05):
        #Stationary
        return 0
    elif (p_value_adf > 0.05) & (p_value_kpss < 0.05):
        #Not Stationary
        return 1
    elif (p_value_adf > 0.05) & (p_value_kpss > 0.05):
        #Trend Stationary
        return 1
    elif (p_value_adf < 0.05) & (p_value_kpss < 0.05):
        #Difference Stationary
        return 1

def return_non_stnry_invtr_list(df,panel_id_col):
    '''
    This function takes ADS as input and panel_id column name 
    and return the list of those panels which have even one attribute non stationary

    Input
    1. df: pandas dataframe, this is the ads
    2. panel_id_col: str, this is the column name of the column containing the panel IDs in the ads

    Return:
    stnry_check_ads: pandas dataframe, index is panel IDs and for each panel and for each time series, we get
    to know if it is stationary or not
    '''
    stnry_check_ads = df[TIME_VARIANT_ATTR + [panel_id_col]].groupby(panel_id_col).agg(lambda x: adf_kpss_test(x))
    stnry_ads_list = stnry_check_ads.apply(sum,axis=1)
    stnry_ads_list = stnry_ads_list[stnry_ads_list != 0].index

    stnry_check_ads = stnry_check_ads.loc[stnry_check_ads.index.isin(stnry_ads_list),]
    return stnry_check_ads

def make_series_stnry(df,date_col):
    '''
    This function makes a series stationary by continuous differencing

    Input:
    1. df: pandas series, the series that has to be differenced
    2. date_col: str, the name of the column that contains the dates in the ads

    Return:
    1. df: pandas series, series that is stationary
    '''
    result = 1
    df.set_index([date_col],inplace=True)
    while result != 0:
        df = df.diff().dropna()
        result = adf_kpss_test(df)
    return df

def make_ads_stnry(df,stnry_check_ads,panel_id_col,date_col):
    '''
    This function goes inverter by inverter and column by column 
    making each column in each inverter stationary (if not already)

    Input:
    1. df: pandas dataframe, this is the dataset that has to be made stationary
    2. stnry_check_ads: pandas dataframe, this is the df that contains info on what all attributes are non
    stationary for what all panels in the data
    3. panel_id_col: str, this is the name of the column that contains the panel IDs in the data
    4. date_col: str, this is the name of the column that contains the dates in the ads

    Returns:
    1. df: pandas dataframe, this is the ads that has all the attributes stationary
    '''
    while len(stnry_check_ads) != 0:
        for invtr in stnry_check_ads.index:
            for column in stnry_check_ads.columns:
                if stnry_check_ads.loc[invtr,column] == 1:
                    strct_stnry_ads = make_series_stnry(df.loc[df[panel_id_col] == invtr,[column,date_col]],date_col)
                    strct_stnry_ads.reset_index(inplace=True)
                    drop_idx = df.loc[(~df[date_col].isin(strct_stnry_ads[date_col]) & (df[panel_id_col]==invtr))].index
                    df = df.drop(drop_idx)
                    #different indices make it very tough to just write values at certain places
                    df.at[(df.DATE.isin(strct_stnry_ads[date_col])) & (df[panel_id_col] == invtr),column] = strct_stnry_ads[column].to_list()
                    df.reset_index(drop=True,inplace=True)
        stnry_check_ads = return_non_stnry_invtr_list(df,panel_id_col)
    
    return df

if __name__ == '__main__':
    #creating the ads
    ads = ads_crtn.create_ads()

    #creating box plots to look at outliers
    #visually inspecting the plots show outliers in different panels
    #reading of different panels is outright way beyond range for some
    get_box_plot(ads[ATTR])

    #lets create line plots for each inverter, showcasing how their total yield per 15 min varies    
    for inverter_no in range(0,ads['INVERTER_ID'].nunique(),10):
        line_plot(ads.loc[ads['INVERTER_ID'].isin(list(ads['INVERTER_ID'].unique()[inverter_no : inverter_no + 10]))],
                'DATE',
                'PER_TS_YIELD',
                'INVERTER_ID')
    #We need to perform train-test split before doing any outlier/missing value treatment
    #so that there is no data leak from test to train dataset
    train_ads, test_ads = train_test_split(ads,'INVERTER_ID',SPLIT_PCT)
    #As we were still facing outliers with z-score, switched to Winsorizing
    #It is usually capping +ve side outliers to 99ptile
    #and -ve side outliers to 20-25%
    #found using the check_ptile function

    outlier_feature = [feature for feature in ATTR if feature != 'TOTAL_YIELD'] 
    clip_model = wz.winsorize(OUTLIER_METHOD)
    clip_model.fit(train_ads[outlier_feature],'INVERTER_ID')
    train_ads[outlier_feature] = clip_model.transform(train_ads[outlier_feature])
    test_ads[outlier_feature] = clip_model.transform(test_ads[outlier_feature])

    #lets create line plots for each inverter again to see the target variable without any outliers
    #we see much better graphs
    for inverter_no in range(0,train_ads['INVERTER_ID'].nunique(),6):
        line_plot(train_ads.loc[train_ads['INVERTER_ID'].isin(list(train_ads['INVERTER_ID'].unique()[inverter_no : inverter_no + 6]))],
                'DATE',
                'PER_TS_YIELD',
                'INVERTER_ID')

    #checking for stationarity in the data inverter by inverter
    non_stnry_invtr_list = return_non_stnry_invtr_list(train_ads,'INVERTER_ID')

    #making sure all the time series in our dataset is stationary
    train_ads = make_ads_stnry(train_ads,non_stnry_invtr_list,'INVERTER_ID','DATE')

    #repeating the process for test dataset as well
    non_stnry_invtr_list = return_non_stnry_invtr_list(test_ads,'INVERTER_ID')
    test_ads = make_ads_stnry(test_ads,non_stnry_invtr_list,'INVERTER_ID','DATE')

    #skipped scatter plots since that information is captured through correlation
    #looking at the data using correlation values
    #not all panels share the same correlated variables
    #indication of panel specific models
    train_ads.groupby('INVERTER_ID').apply(lambda x: list(correlated_var(x, 'PER_TS_YIELD')))
    
    #Partial Autocorrelation tells us how many extreme values of lags can be used to predict the target variable
    #not all panels have the same autocorrelation lags
    #getting too many lag values when aggregating to a day level
    #not enough data points, therefore pacf is not that reliable at a day level, however at 15 min mark it is
    lag_values = train_ads[['INVERTER_ID','PER_TS_YIELD']].groupby('INVERTER_ID').agg(lambda x: pick_pacf(x))

    acf_values = train_ads[['INVERTER_ID','PER_TS_YIELD']].groupby('INVERTER_ID').agg(lambda x: pick_acf(x))
