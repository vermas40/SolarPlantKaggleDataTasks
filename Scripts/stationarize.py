import pandas as pd
import numpy as np
import copy
from statsmodels.tsa.stattools import acf,pacf, adfuller, kpss

#keep instance variables as instance variables EVERYWHERE!
class stationarize(object):

    def __init__(self,df):
        self.diff_order = {}
        self.stnry_check_ads = pd.DataFrame()
        self.init_value = {}
        self.non_stnry_ads = copy.deepcopy(df)

    def adf_kpss_test(self,df):

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

    def return_non_stnry_invtr_list(self,df,panel_id_col,first_run=True):
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
        self.stnry_check_ads = df[TIME_VARIANT_ATTR + [panel_id_col]].groupby(panel_id_col).agg(lambda x: self.adf_kpss_test(x))
        stnry_ads_list = self.stnry_check_ads.apply(sum,axis=1)
        stnry_ads_list = stnry_ads_list[stnry_ads_list != 0].index

        self.stnry_check_ads = self.stnry_check_ads.loc[self.stnry_check_ads.index.isin(stnry_ads_list),]

        if first_run:
            for invtr in self.stnry_check_ads.index:
                self.diff_order[invtr] = {}
                self.init_value[invtr] = {}
                for column in self.stnry_check_ads.columns:
                    self.diff_order[invtr][column] = 0
                    self.init_value[invtr][column] = {}
        return 

    def make_series_stnry(self,df,date_col):
        '''
        This function makes a series stationary by continuous differencing

        Input:
        1. df: pandas series, the series that has to be differenced
        2. date_col: str, the name of the column that contains the dates in the ads

        Return:
        1. df: pandas series, series that is stationary
        '''

        df.set_index([date_col],inplace=True)
        result = self.adf_kpss_test(df)
        while result != 0:
            self.diff_order[self.invtr][self.column] += 1
            order = self.diff_order[self.invtr][self.column]
            self.init_value[self.invtr][self.column][order] = df.loc[df.index[0]][0]
            df = df.diff().dropna()
            result = self.adf_kpss_test(df)
        return df

    def make_ads_stnry(self,df,panel_id_col,date_col):
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

        while len(self.stnry_check_ads) != 0:
            for self.invtr in self.stnry_check_ads.index:
                for self.column in self.stnry_check_ads.columns:
                    if self.stnry_check_ads.loc[self.invtr,self.column] == 1:
                        strct_stnry_ads = self.make_series_stnry(df.loc[df[panel_id_col] == self.invtr,[self.column,date_col]],\
                                                                date_col)

                        strct_stnry_ads = strct_stnry_ads.reset_index() #to make date a column again
                        #checking if removing 1st value has an impact on stationarity or not
                        drop_idx = df.index[(~df[date_col].isin(strct_stnry_ads[date_col]) & (df[panel_id_col]==self.invtr))]
                        df = df.drop(drop_idx)
                        #different indices make it very tough to just write values at certain places
                        replace_idx = df.index[(df[date_col].isin(strct_stnry_ads[date_col])) & (df[panel_id_col] == self.invtr)]
                        df.at[replace_idx,self.column] = strct_stnry_ads[self.column].to_list()
                        df = df.reset_index(drop=True)
            self.return_non_stnry_invtr_list(df,panel_id_col,first_run=False)
        
        return df

    def inverse_series(self,diff_srs,org_srs=None):
        order = self.diff_order[self.invtr][self.column]
        nan_srs = pd.Series([np.nan for i in range(order)])
        int_srs = pd.concat([nan_srs,diff_srs]).reset_index(drop=True)
        
        #this function keeps integrating the series till it reaches order 0 of differencing
        while order != 0:
            int_srs[order-1:] = int_srs[order-1:].cumsum().fillna(0) + self.init_value[self.invtr][self.column][order]
            order -= 1

        #if integrated series is not equal in length to original series
        #due to it being cut off because of some other series differencing
        #then the below code puts back the original values to the top of the 
        #integrated series
        if len(int_srs) != len(org_srs): 
            len_diff = len(org_srs) - len(int_srs)
            pad_idx = [idx for idx in range(org_srs.index[0],org_srs.index[0] + len_diff)]
            int_srs = pd.concat([org_srs[pad_idx].reset_index(drop=True),int_srs]).reset_index(drop=True)
        
        assert round(int_srs.sum(),3) == round(org_srs.sum(),3), 'Inverse logic not working'
        
        return int_srs

    def fit(self,df,panel_id_col):
        self.return_non_stnry_invtr_list(df,panel_id_col)
        return
    
    def transform(self,df,panel_id_col,date_col):
        df = self.make_ads_stnry(df,panel_id_col,date_col)
        return df
    
    def inverse_transform(self,df,panel_id_col,attribute):
        #remove 'fcst from attribute name'
        for key in self.diff_order.keys():
            if self.diff_order[key][attribute] != 0:
                self.invtr = key
                self.column = attribute
                stnry_df_panel_idx = df.index[df[panel_id_col] == key]
                non_stnry_panel_idx = self.non_stnry_ads.index[self.non_stnry_ads[panel_id_col]==key]
                int_srs = self.inverse_series(df.loc[stnry_df_panel_idx,attribute],\
                                              self.non_stnry_ads.loc[non_stnry_panel_idx,attribute])                
                
                self.non_stnry_ads.at[non_stnry_panel_idx,attribute] = int_srs.to_list()
        return self.non_stnry_ads