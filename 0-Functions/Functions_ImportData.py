#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import os.path

from Functions_Indicators import str_month
from Functions_Indicators import add_year_month_season


# ## NOAA

# import_treat_obs_NOAA aims to import the original file containing the NOAA observation data. It used in 'Treat DATA NOAA Station'

# In[ ]:


# this function is meant to import the NOAA observation data
def import_treat_obs_NOAA():
    # path where the file is placed
    path_file_NOAA = r'C:\Users\CLMRX\COWI\A248363 - Climate analysis - Documents\General\CRVA_tool\Master_thesis\Project\3 - Implementation\1 - Data\1-BC\NOAA-ClimateDataOnline\3370204.csv'
    # read the information in the file
    data_obs_NOAA = pd.read_csv(path_file_NOAA)
    # unit of PRCP are mm
    # unit of temperature are degrees Celsius

    return data_obs_NOAA


# treat_NOAA_data aims to add information to the dataframe of the observation data. It is used in 'Treat DATA NOAA Station'

# In[ ]:


def treat_NOAA_data(daily_sum_obs_from_NOAA):
    
    daily_sum_obs_from_NOAA = add_year_month_season(daily_sum_obs_from_NOAA,'DATE')
    
    daily_sum_obs_from_NOAA = daily_sum_obs_from_NOAA[daily_sum_obs_from_NOAA['Year'].between(1970,2014)]
    
    daily_sum_obs_from_NOAA=count_na_in_df_NOAA(daily_sum_obs_from_NOAA,'PRCP')
    daily_sum_obs_from_NOAA=count_na_in_df_NOAA(daily_sum_obs_from_NOAA,'TAVG')
    daily_sum_obs_from_NOAA=count_na_in_df_NOAA(daily_sum_obs_from_NOAA,'TMAX')
    daily_sum_obs_from_NOAA=count_na_in_df_NOAA(daily_sum_obs_from_NOAA,'TMIN')
    
    return daily_sum_obs_from_NOAA


# In[ ]:


# this funciton count the missing values in the columns named name_col in the dataframe df
def count_na_in_df_NOAA(df,name_col): # function used in treat_NOAA_data
    df[name_col+' MISSING']=0
    df[name_col+' MISSING'][df[name_col].isna()]=1
    return df


# import_filtered_NOAA_obs aims to import the filtered observation NOAA data

# In[ ]:


def import_filtered_NOAA_obs():
    # path where the filtered file is placed
    path_file_NOAA = r'C:\Users\CLMRX\COWI\A248363 - Climate analysis - Documents\General\CRVA_tool\Master_thesis\Project\3 - Implementation\1 - Data\1-BC\NOAA-ClimateDataOnline\filtred_NOAA_obs_data.csv'
    # read the information in the file
    data_obs_NOAA = pd.read_csv(path_file_NOAA) 
    
    data_obs_NOAA = data_obs_NOAA.drop('Unnamed: 0',axis=1)
    
    return data_obs_NOAA


# ## Precipitation data from Gorongosa

# In[ ]:


def import_gorongosa_obs_pr():
    path=r'C:\Users\CLMRX\COWI\A248363 - Climate analysis - Documents\General\CRVA_tool\Master_thesis\Project\3 - Implementation\1 - Data\1-BC\DirecltyfromMoz\Precipitation_Gorongosa_reformat_ready_to_use.csv'
    df = pd.read_csv(path)
    df = df.drop('Unnamed: 0',axis=1)
    return df


# ## CMIP6
# 
# function 'import_CMIP6_past_close_to_NOAA' aims to import past CMIP6 data, at the same emplacement than the NOAA station

# In[ ]:


def import_CMIP6_past_close_to_NOAA(global_variable,climate_var=''):
    if 'pr' in global_variable:
        path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\precipitation\Copernicus-CMIP6\csv\1950-2014'
    else:
        out_path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets'
        path = os.path.join(out_path,global_variable,climate_var,'Copernicus-CMIP6','csv','1950-2014')
    
    for name in os.listdir(path):
        if 'Close_to_NOAA_Station' in name:
            path_csv = os.path.join(path,name)
            
    df = pd.read_csv(path_csv)
    
    df = add_year_month_season(df,'Date') # add month, year and season
    
    return df


# ## NEX-GDDP-CMIP6

# import_treat_modeled_NEX_GDDP_CMIP6_close_to_stationNOAA aims to import and treat the NEX GDDP CMIP6 data close to the NOAA station
# 

# In[ ]:


def import_treat_modeled_NEX_GDDP_CMIP6_close_to_stationNOAA(climate_var, unit):
    # import data
    
    path_NEX_GDDP_CMIP6_EmplacementStation=os.path.join(r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file',climate_var,climate_var+'_'+unit+'_day_1950-2100_Closest_to_NOAA','NEXGDDPCMIP6_at_same_emplacement_as_NOAA_stationPembaChimoioBeira_'+climate_var+'_1950-2100.csv')
    
    data_NEX_GDDP_CMIP6_EmplacementStation = pd.read_csv(path_NEX_GDDP_CMIP6_EmplacementStation)
    
    data_NEX_GDDP_CMIP6_EmplacementStation = add_year_month_season(data_NEX_GDDP_CMIP6_EmplacementStation,'Date')
    
    return data_NEX_GDDP_CMIP6_EmplacementStation


# import_treat_modeled_NEX_GDDP_CMIP6 aims to import and treat the NEX GDDP CMIP6 data at the emplacement of the project of interest

# In[ ]:


def import_treat_modeled_NEX_GDDP_CMIP6(climate_var, unit,temporal_resolution,start_y,stop_y):
    # import data
    
    #if climate_var =='pr':
    path_NEX_GDDP_CMIP6=os.path.join(r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file',climate_var,climate_var+'_'+unit+'_'+temporal_resolution+'_'+str(start_y)+'-'+str(stop_y),climate_var+'_'+str(start_y)+'-'+str(stop_y)+'_projectsMoz.csv')
    #else: # temperature
        #path_NEX_GDDP_CMIP6=os.path.join(r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file',climate_var,climate_var+'_Celsius_day_1950-2100',climate_var+'_1950-2100_projectsMoz.csv')
        
    data_NEX_GDDP_CMIP6 = pd.read_csv(path_NEX_GDDP_CMIP6)
    
    data_NEX_GDDP_CMIP6 = add_year_month_season(data_NEX_GDDP_CMIP6,'Date')
    
    return data_NEX_GDDP_CMIP6


# import_BC_NOAA_NEX_GDDP_CMIP6 aims to import NEX-GDDP-CMIP6 data that were bias corrected with NOAA data.

# In[ ]:


def import_BC_NOAA_NEX_GDDP_CMIP6(climate_var,start_y,stop_y,EmplacementStationNOAA='yes',resolution ='day'):
    path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file'
    if climate_var =='pr':
        unit ='mm_per_day'
    if 'tas' in climate_var:
        unit ='Celsius'
    if EmplacementStationNOAA=='yes':
        print('read csv')
        df=pd.read_csv(os.path.join(path,climate_var,climate_var+'_'+unit+'_'+resolution+'_'+str(start_y)+'-'+str(stop_y)+'_BiasCorrected',climate_var+'_'+unit+resolution+str(start_y)+'-'+str(stop_y)+'_BiasCorrected_EmplacementStationNOAA.csv'))
    else:
        print('read csv')
        df=pd.read_csv(os.path.join(path,climate_var,climate_var+'_'+unit+'_'+resolution+'_'+str(start_y)+'-'+str(stop_y)+'_BiasCorrected',climate_var+'_'+unit+resolution+str(start_y)+'-'+str(stop_y)+'_BiasCorrected.csv'))
    df=df.drop('Unnamed: 0',axis=1)
    return df


# import_BC_NOAA_NEX_GDDP_CMIP6 aims to import precipitation NEX-GDDP-CMIP6 data that were bias corrected with precipitation data from Gorongosa

# In[ ]:


def import_BC_Gorongosa_NEX_GDDP_CMIP6(start_y,stop_y, climate_var='pr',resolution='day',unit='mm_per_day'):
    path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file'
    df=pd.read_csv(os.path.join(path,climate_var,climate_var+'_'+unit+'_'+resolution+'_'+str(start_y)+'-'+str(stop_y)+'_BiasCorrected',climate_var+'_'+unit+resolution+str(start_y)+'-'+str(stop_y)+'_BiasCorrected_OnlyGorongosa.csv'))
    df=df.drop('Unnamed: 0',axis=1)
    return df


# # Sensitivity informations

# In[64]:


# this function imports the sentivity for the town of gorongosa
# even if the user modifies the path and the initial document, also need to adapt the code, to permit to deal with several projects
def sensitivity():
    path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file\GorongosaSensitivity.csv'
    df = pd.read_csv(path)
    # register names of projects in the matrix
    names = list(df['Unnamed: 0'].dropna())
    names.remove('Project')
    # register Sensitivity theme
    ST = list(df['Unnamed: 1'].dropna())
    ST.remove('Sensitivity theme')
    # list climate variables
    CV = list(df.iloc[0,2:len(df.columns)])

    # create a new dataframe, with multiple indexes andmultiples columns
    midx = pd.MultiIndex.from_product([names,ST],names=['Name project','Sensitivity theme']) # multiple index
    cols = pd.MultiIndex.from_product([('Sensitivity level',),CV]) # create mutliple columns
    # create empty dataframe
    df_ = pd.DataFrame(data = [], 
                                index = midx,
                                columns = cols)
    # set a default value for every element in the new dataframe
    df_.loc[:,:]='No'

    # register sensitivity values from initial csv
    for i in np.arange(0,len(df_.index)):
        for j in np.arange(0,len(df_.columns)):
            if df.iloc[i+1,j+2]!= 'No':
                df_.iloc[i,j] = df.iloc[i+1,j+2]
    return df_
