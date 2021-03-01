import pandas as pd
import numpy as np

class winsorize(object):
    def __init__(self,method,z_scr_thresh=None):
        self.params = {}
        self.method = method
        if self.method == 'Z_SCR':
            #don't prefer z-score since here we are making the assumption that the data is coming from
            #a normal distribution
            assert z_scr_thresh is not None, 'Please provide z-score threshold'
            self.z_scr_thresh = z_scr_thresh
        
    def num_cat(self,df):
        num = df.select_dtypes(include=[np.number]).columns
        cat = list(set(df.columns) - set(num))
        
        datatypes = {
            'numeric' : num,
            'categorical' : cat
        }
        
        return datatypes

    def iqr_fence(self,df):
        '''
        This function calculates the minimum and maximum values for an attribute
        '''
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        Lower_Fence = Q1 - (1.5 * IQR)
        Upper_Fence = Q3 + (1.5 * IQR)
        try:
            #this step is done to find the actual whiskers
            #whiskers are actual data points in the series whereas fences are theoretical cutoff points
            u = max(df[df<Upper_Fence])
            l = min(df[df>Lower_Fence])
        except ValueError:
            # This is for columns that are having the same value throughout the series
            u = Upper_Fence
            l = Lower_Fence
        return u,l

    def get_params(self,df):
        '''
        This function calculates the mean, std for z score and upper/lower fences for winsorizing
        '''
        if self.method == 'Z_SCR':
            upper_ptile, lower_ptile = self.iqr_fence(df)
            params = (upper_ptile,lower_ptile, df.mean(), df.std())
        else:
            upper_ptile, lower_ptile = self.iqr_fence(df)
            params = (upper_ptile,lower_ptile)    
        return params
    
    def fit(self,df,panel_id_col):
        '''
        This function creates params (mean,std,upper/lower fence) at panel-attribute level 
        in a dictionary and stores it in class object
        '''
        self.datatypes = self.num_cat(df)
        self.panel_id_col = panel_id_col
        for inverter in df[panel_id_col].unique():
            self.params[inverter] = {}
            for attribute in self.datatypes['numeric']:
                self.params[inverter][attribute] = self.get_params(df.loc[df[panel_id_col]==inverter,attribute])
        return
    
    def treat_outliers(self,df,param_dict):
        '''
        This function checks at an obs level if it is an outlier or not, if it is then the value
        is clipped and it is given upper or lower limit value
        '''
        upper_lim = param_dict[0]
        lower_lim = param_dict[1]
        if self.method == 'Z_SCR':
            df_mean = param_dict[2]
            df_std = param_dict[3]

            for obs in df.index:
                z_score = (df[obs] - df_mean)/df_std
                if abs(z_score) > self.z_scr_thresh:
                    if z_score < 0:
                        df.at[obs] = lower_lim
                    else:
                        df.at[obs] = upper_lim
        else:        
            for obs in df.index:
                if df[obs] > upper_lim:
                    df.at[obs] = upper_lim
                elif df[obs] < lower_lim:
                    df.at[obs] = lower_lim
            
        return df

    def transform(self,df):
        '''
        This function takes in a dataframe and goes panel-attribute at a time and does outlier treatment
        '''
        for inverter in df[self.panel_id_col].unique():
            for attribute in self.datatypes['numeric']:
                df.at[df[self.panel_id_col]==inverter,attribute] = self.treat_outliers(df.loc[df[self.panel_id_col]==inverter,attribute],self.params[inverter][attribute])
        return df