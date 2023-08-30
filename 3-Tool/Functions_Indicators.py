#!/usr/bin/env python
# coding: utf-8

# This notebook aims to contain all functions for indicators.

# In[ ]:


from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gumbel_r
from scipy.stats import gumbel_l
import os
import os.path
import math


# # Treat data

# In[ ]:


def select_station(df,name_col,name_station):
    df_name_station = df[df[name_col]==name_station]
    return df_name_station


# In[ ]:


## Add Year, Month and Season to df


# In[ ]:


def add_year_month_season(df,column_date):
    # add Year, month and season columns for graphs
    Year = df[[column_date]].values.reshape(len(df[[column_date]].values),)
    Month = df[[column_date]].values.reshape(len(df[[column_date]].values),)
    Season = df[[column_date]].values.reshape(len(df[[column_date]].values),)
    
    if str(Year[1]).find('-')==2 or str(Year[1]).find('/')==2:
        for i in np.arange(0,len(df[[column_date]].values)):
            Year[i]=int(Year[i][6:10])
            Month[i]=int(Month[i][3:5])
            if Month[i]>3 and Month[i]<10: # dry season in Mozambique is between April and September
                Season[i]='Dry'
            else:# humid season is between October and March
                Season[i]='Humid'
            
            Month[i]=str_month(Month[i])
            
    if str(Year[1]).find('-')==4 or str(Year[1]).find('/')==4:
        for i in np.arange(0,len(df[[column_date]].values)):
            Year[i]=int(Year[i][0:4])
            Month[i]=int(Month[i][5:7])
            if Month[i]>3 and Month[i]<10: # dry season in Mozambique is between April and September
                Season[i]='Dry'
            else:# humid season is between October and March
                Season[i]='Humid'
            # paper talking about the season in Mozambique: 
            # https://pdf.sciencedirectassets.com/277910/1-s2.0-S1876610217X00350/1-s2.0-S1876610217351081/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEO7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDZKtKpBNFodsR6x3m1JZHMHJgHlr%2FK6sMWgVM97BtgmQIge46Zf6Qxp6M%2FGJsZOOv8IEgLeqnCG7tVmQF6dIR4%2FkgquwUIp%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDJQ24kvPuoPzBE80WCqPBRiSv%2B5GquSrLthGw%2F5FqS7nkk%2FuxrwYE%2BYQ7m6xTz3kzMVHQWoKkDNI3VkHM21LHeguWBbZ8UvprGsp%2FS%2BYR%2BjgF9fBkI57EH%2F6EeKAI%2B8tnLtAduTAhFA8ExJOMN1l8Y871ZSPy2X%2FJhBl5Pr5dboHZ5W9fkR%2FcSW9YFXRXyKreLaji2%2FgEmX3qeuQRGvmE2Tbz2VxGuGsNw%2FiwCs4t0BRvCSemEG6%2Bmyz9pw7KsQey6ojF51y82F0ufSgz48ZcI0Fj1CrYEDnj7js6M8B08wU3n4zBhReUM%2FBAqY1dJUp0cO1SkZ4cDe3NgA88rNuLG8cV%2BGh5pfGH2vWqGs8tF%2BQEXFJMu2Svxn04xqwX1AxykpeHFHDwhTno9iQOKqK1ZghpFXziEk0CpWi9LHOBCUaAjcEuxVpJoHM4pNhGsGU%2FvvLBSuJcCkg267NpLwwwtdyNLZEcuyPpfHo1pN9Yc%2Bev4lMBMkokV9oK%2F2WmXMuduQu5OxJI4QHBeLFkNxWIdKNmHH%2FkmcGD8AlZ%2BfPuKllSLEPmRhA5PGv1u7JyA1TtpRrpjbO3ZCMMyXjUpyiihlgT6Y2ifZL9A0IS9lxz0ucm%2FVRsa%2BP%2Bo%2FONuDZK3Z10mXCO6Fx1u%2F%2BaHg5chtDmbd2ByPdQuytjPiOYJhBomP7dE4ggSfy1G4Zff1hLmc0AN%2FdLNxo88iY3W0cf7dqWSkUo%2BnRNrRlRTxAX5fFl9oC9Wkqf3phJmT8vY73LiizwG6HuXnVLmgYvjkEb8M04jKUpUSe0xFsKrKNO1vemq0pvSdTfq7oY4p7XIeshvvclChLTQPHNDjndrS5vNZhBX%2F5js4QLbEjlFmJRjFgZI5xJ88%2BhY783L2cF2UPQEUw3ouDpwY6sQFRdp5XujWpzdWYY3QLfECY9aJXZdpvzDxtbuBCrmep4FPpcoGya2t5N1C3Nh7bsrkbsdx3Ny689MunpOvPIsAcTF5cJ18QQo0Pi72zNAiNV5eZj4oIUYDCdpgihnK%2Bk%2FJYr4EVuk9M0uwxvIjWRAZYWeF%2FEJWxtnCY3QzojO3vFn%2BGaKESKxIX%2Fcig0UGY4wdu0RIRZFutktfI1ht7ejNnOB6rKfu%2BxPfAuBP7AFRm%2Bl0%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230819T141836Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2G3LIM7P%2F20230819%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=27ac7cf0872c7a68d97e0a99292c5ca1c98fba84fa3d9ff0fd1978a20a485fbc&hash=b4bcb944de8a8adaeef8435431522b46ab79732518c76334b8924be0dddf96af&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1876610217351081&tid=spdf-8eb3a4f2-2183-4a80-93fc-cee7ffcec8a3&sid=cfbd58eb54dba641ff79dab-46f13144aa3egxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=010c5807585455530754&rr=7f9306d72df310ef&cc=dk
            
            Month[i]=str_month(Month[i])
                
    df['Year'] = Year
    df['Month'] = Month
    df['Season'] = Season
    return df


# In[ ]:


def str_month(int_m):
    if int_m==1:
        str_m = 'Jan'
    if int_m==2:
        str_m = 'Feb'    
    if int_m==3:
        str_m = 'Mar'
    if int_m==4:
        str_m = 'Apr'
    if int_m==5:
        str_m = 'May'
    if int_m==6:
        str_m = 'Jun'
    if int_m==7:
        str_m = 'Jul'
    if int_m==8:
        str_m = 'Aug'    
    if int_m==9:
        str_m = 'Sep'
    if int_m==10:
        str_m = 'Oct'
    if int_m==11:
        str_m = 'Nov'
    if int_m==12:
        str_m = 'Dec'
    return str_m


# In[ ]:


# this function is meant to filter the data wanted
# the function use the function model_to_kill (to take out the models the user does not want, the funciton is defined below in the same file) 
# and the function add_year_month_season to add to the dataframe the information about the year, the month and the season for each row
def filter_dataframe(df,name_projects,list_model_to_kill,start_y=1950,stop_y=2100):
    df = df.reset_index() # to take out multiindex that may exist and complicate filtering process
    
    if list_model_to_kill!=[]:
        df = model_to_kill(df,list_model_to_kill)
    
    df_final= pd.DataFrame()
    # find name of the colum containing the name of the localization of interest
    name_col_name=[i for i in df.columns if 'Name ' in i][0]
    for name_project in name_projects:
        df_temp = df[df[name_col_name]==name_project] # select only data of interest
        df_final = pd.concat([df_final,df_temp])
    
    if 'Year' not in list(df_final.columns):
        df_final=add_year_month_season(df_final,'Date') # add column 'Year', 'Month', 'Season'
    
    if start_y!=1950 or stop_y!=2100:
        df_final = df_final[df_final['Year'].between(start_y,stop_y)] # select only the years of interest
    if 'index' in df_final.columns:
        df_final= df_final.drop('index',axis=1)
    
    return df_final


# In[1]:


# this function is meant to take out the results of the models the user does not want
# inputs are
# df a dataframe, with no indexes (used .reset_index() if there are some indexes)
# list_model_to_kill: a list of string containing the names of the models that need to be taken out of the dataframe
# returns
# the dataframe without the rows corresponding to elements in the list_model_to_kill
def model_to_kill(df,list_model_to_kill):
    for name_model in list_model_to_kill:
        df = df[df['Model']!=name_model]
    return df


# In[ ]:


# this function aims to find the correct name of the column of interest

def find_name_col(df,climate_var_longName):
    try:
        try:
            try:
                old_title_column=df.filter(like=climate_var_longName, axis=1).columns[0]
            except:
                old_title_column=df.filter(like=climate_var_longName.capitalize(), axis=1).columns[0]
        except:
            old_title_column=df.filter(like=climate_var_longName.upper(), axis=1).columns[0]
    except:
        old_title_column=df.filter(like=climate_var_longName.lower(), axis=1).columns[0]
    return old_title_column


# In[ ]:


# the function df_to_csv aims to return the filled dataframe in a csv format
# Inputs are:
#       df: the dataframe that should be register in a csv file
#      path_for_csv: this is the path where the csv file should be registered, in a string format
#      title_file: this is the name of the csv file to be created in a string format
#                  CAREFUL --> title_file MUST have the extension of the file in the string (.csv for example)
# Output is:
#      in the case where the dataframe is not empty, the ouput is the full path to the created csv file
#      in the case where the dataframe is empty, the output is an empty list

def df_to_csv(df,path_for_csv,title_file):
    # test if dataframe is empty, if values exist for this period
    if not df.empty: 
        # if dataframe is not empty, value were registered, the first part is run : 
        # a path to register the csv file is created, .....
        if not os.path.isdir(path_for_csv):
            # the path to the file does not exist
            os.makedirs(path_for_csv) # to ensure creation of the folder
            # creation of the path for the csv file, in a string format
        full_name = os.path.join(path_for_csv,title_file)
        # ..... and the dataframe is registered in a csv file
        df.to_csv(full_name) # register dataframe in csv file
        print('Path for csv file is: ' + full_name)
        return full_name # return the full path that leads to the created csv file
    else: # if the dataframe is empty, no value were found, there is no value to register or to return
        print('The dataframe is empty')
        return []


# # General functions

# ### Return period

# In[ ]:


# return period for each project, model, scenario


# In[ ]:


# function value for return period


# In[ ]:


# this function return
def threshold_coresponding_to_return_period(loc,scale,T):
    p_non_exceedance = 1 - (1/T)
    try:
        threshold_coresponding = round(gumbel_r.ppf(p_non_exceedance,loc,scale))
    except OverflowError: # the result is not finite
        if math.isinf(gumbel_r.ppf(p_non_exceedance,loc,scale)) and gumbel_r.ppf(p_non_exceedance,loc,scale)<0:
            # ppf is the inverse of cdf
            # the result is -inf
            threshold_coresponding = 0 # the value of zero is imposed
    return threshold_coresponding
    # ppf: Percent point function
    #print('Threshold '+str(threshold_coresponding)+' mm/day will be exceeded at least once in '+str(n)+' year, with a probability of '+str(round(p_exceedance*100))+ ' %')
    #print('This threshold corresponds to a return period of '+str(round(return_period))+ ' year event over a '+str(n)+' year period')


# In[ ]:


# return return period for dataframe of modelled data
def dataframe_threshold_coresponding_to_return_period_model(df,name_col):
    df_copy=df.copy(deep=True)
    df_copy=df_copy.drop(labels='Date',axis=1)
    df_max = df_copy.groupby(['Name project','Experiment','Model','Year']).max() # maximum    
    midx = pd.MultiIndex.from_product([list(set(df_copy[df_copy.columns[0]])),list(set(df_copy[df_copy.columns[1]])),list(set(df_copy[df_copy.columns[2]]))],names=['Name project','Experiment', 'Model'])
    cols = ['Value for return period 50 years mm/day','Value for return period 100 years mm/day']
    return_period = pd.DataFrame(data = [], 
                                index = midx,
                                columns = cols)
    for name_p in return_period.index.levels[0].tolist():
        for ssp in return_period.index.levels[1].tolist():
            for model in return_period.index.levels[2].tolist():
                print('Name project '+name_p+ ' ssp '+ssp+ ' model '+model)
                Z=df_max.loc[(name_p,ssp,model)][name_col].values.reshape(len(df_max.loc[(name_p,ssp,model)][name_col]),)
                (loc1,scale)=stats.gumbel_r.fit(Z) # return the parameters necessary to establish the continous function
                # choice of gumbel because suits to extreme precipitation
                return_period.loc[(name_p,ssp,model),('Value for return period 50 years mm/day')] = threshold_coresponding_to_return_period(loc1,scale,50)
                return_period.loc[(name_p,ssp,model),('Value for return period 100 years mm/day')] = threshold_coresponding_to_return_period(loc1,scale,100)
                
    return return_period


# In[ ]:


# return return period for dataframe of modelled data
def dataframe_return_period_coresponding_to_past_100year_event_model(df,name_col,df_past):
    df_copy=df.copy(deep=True)
    df_copy=df_copy.drop(labels='Date',axis=1)
    df_max = df_copy.groupby(['Name project','Experiment','Model','Year']).max() # maximum    
    midx = pd.MultiIndex.from_product([list(set(df_copy[df_copy.columns[0]])),list(set(df_copy[df_copy.columns[1]])),list(set(df_copy[df_copy.columns[2]]))],names=['Name project','Experiment', 'Model'])
    cols = ['Return period for past 50 year event','Return period for past 100 year event']
    return_period = pd.DataFrame(data = [], 
                                index = midx,
                                columns = cols)
    for name_p in return_period.index.levels[0].tolist():
        for ssp in return_period.index.levels[1].tolist():
            for model in return_period.index.levels[2].tolist():
                print('Name project '+name_p+ ' ssp '+ssp+ ' model '+model)
                Z=df_max.loc[(name_p,ssp,model)][name_col].values.reshape(len(df_max.loc[(name_p,ssp,model)][name_col]),)
                (loc1,scale)=stats.gumbel_r.fit(Z) # return the parameters necessary to establish the continous function
                # choice of gumbel because suits to extreme precipitation
                return_period.loc[(name_p,ssp,model),('Return period for past 50 year event')] = return_period_coresponding_to_threshold(df_past.loc[(name_p,'historical',model)]['Value for return period 50 years mm/day'],loc1,scale)
                return_period.loc[(name_p,ssp,model),('Return period for past 100 year event')] = return_period_coresponding_to_threshold(df_past.loc[(name_p,'historical',model)]['Value for return period 100 years mm/day'],loc1,scale)
                
    return return_period


# In[ ]:


# return return period for dataframe of observed data. Mostly used for figures
# use only with future
def dataframe_threshold_coresponding_to_return_period_obs(df,name_col):
    df_copy=df.copy(deep=True)
    df_copy=df_copy.drop(labels='Date',axis=1)
    df_max = df_copy.groupby(['Name project','Year'])[[name_col]].max() # maximum    
    midx = pd.MultiIndex.from_product([list(set(df_copy[df_copy.columns[0]]))],names=['Name project'])
    cols = ['Value for return period 50 years mm/day','Value for return period 100 years mm/day']
    return_period = pd.DataFrame(data = [], 
                                index = midx,
                                columns = cols)
    for name_p in return_period.index.levels[0].tolist():
        print('Name project '+name_p)
        Z=df_max.loc[(name_p)][name_col].values.reshape(len(df_max.loc[(name_p)][name_col]),)
        (loc1,scale)=stats.gumbel_r.fit(Z) # return the function necessary to establish the continous function
        # choice of gumbel because suits to extreme precipitation
        return_period.loc[(name_p,ssp,model),('Value for return period 50 years mm/day')] = threshold_coresponding_to_return_period(loc1,scale,50)
        return_period.loc[(name_p,ssp,model),('Value for return period 100 years mm/day')] = threshold_coresponding_to_return_period(loc1,scale,100)

    return return_period


# In[ ]:


# function a check
def return_period_coresponding_to_threshold(threshold,loc,scale):
    proba = gumbel_r.cdf(threshold,loc,scale) # probability of non exceedance
    T = round(1/(1-proba))
    return T


# In[ ]:


# return value corresponding to return period T
def return_period(df,T,start_y,stop_y):
    Z = df[df['Year'].between(start_y,stop_y)].groupby('Year')[['pr']].agg(np.nanmax)#.reshape(len(pr_obs_gorongosa_from_gorongosa.groupby('Year')[['pr']].max()),)
    #Z = Z[~np.isnan(Z)]
    (loc1,scale1)=scipy.stats.gumbel_r.fit(Z) # return the function necessary to establish the continous function
    value_for_T=threshold_coresponding_to_return_period(loc1,scale1,T)
    return value_for_T


# In[ ]:


# questions Temps retour :
#      
#      besoin de caler mieux distribution ? package pour le faire automatiquement ? si on fiat pas avec maxima, mais on va faire que avec maxima pour le moment


# ## calculation Yearly average

# In[ ]:


# this function only works for projections
# example of use df_monthly_avg_tas_NEXGDDPCMIP6_gorongosa=temporal_avg(df_tas_NEXGDDPCMIP6_gorongosa,'temperature','Monthly average temperature','month')
def temporal_avg(df,climate_var_long_name,title_column,temporal_resolution):
    df_yearly_avg = df.copy(deep =True)
    old_name_column = find_name_col(df,climate_var_long_name)
    df_yearly_avg=df_yearly_avg.rename(columns={old_name_column:title_column})
    if 'pr' in title_column.lower():
        if temporal_resolution == 'year':
            df_yearly_avg = df_yearly_avg.groupby(['Name project','Experiment','Model','Year'])[[title_column]].mean()*365.25
        if temporal_resolution == 'month':
            df_yearly_avg = df_yearly_avg.groupby(['Name project','Experiment','Model','Month'])[[title_column]].mean()*30
    else:
        if temporal_resolution == 'year':
            df_yearly_avg = df_yearly_avg.groupby(['Name project','Experiment','Model','Year'])[[title_column]].mean()
        if temporal_resolution == 'month':
            df_yearly_avg = df_yearly_avg.groupby(['Name project','Experiment','Model','Month'])[[title_column]].mean()
    return df_yearly_avg


# In[ ]:


# this function only works for projections
# example of use df_monthly_avg_tas_NEXGDDPCMIP6_gorongosa=temporal_avg(df_tas_NEXGDDPCMIP6_gorongosa,'temperature','Monthly average temperature','month')
def temporal_max(df,climate_var_long_name,title_column,temporal_resolution):
    df_yearly_avg = df.copy(deep =True)
    old_name_column = find_name_col(df,climate_var_long_name)
    df_yearly_avg=df_yearly_avg.rename(columns={old_name_column:title_column})
    if 'pr' in title_column.lower():
        if temporal_resolution == 'year':
            df_yearly_avg = df_yearly_avg.groupby(['Name project','Experiment','Model','Year'])[[title_column]].max()*365.25
        if temporal_resolution == 'month':
            df_yearly_avg = df_yearly_avg.groupby(['Name project','Experiment','Model','Month'])[[title_column]].max()*30
    else:
        if temporal_resolution == 'year':
            df_yearly_avg = df_yearly_avg.groupby(['Name project','Experiment','Model','Year'])[[title_column]].max()
        if temporal_resolution == 'month':
            df_yearly_avg = df_yearly_avg.groupby(['Name project','Experiment','Model','Month'])[[title_column]].max()
    return df_yearly_avg


# ### N - number of days above threshold

# In[ ]:


def number_day_above_threshold(df,climate_var_longName,old_title_column,threshold):
    new_name='Average annual number of days with '+climate_var_longName+' above '+str(threshold)
    #df = df.rename(columns={old_title_column:new_name})
    
    df = df.drop(['Date','Month','Season'],axis=1) 
    df=df.reset_index()
    #df=df.groupby(['Experiment','Model','Year']).apply(lambda x: x[x[new_name]>40].count()).reset_index()
    df[new_name]=0
    df[new_name].iloc[np.where(df[old_title_column]>40)[0]]=1    
    df = df.groupby(['Name project','Experiment','Model','Year'])[[new_name]].sum()
    
    return df


# # Precipitation

# ### N-day event 

# In[ ]:


# some models to not have any values for some scenarios
# need to delete them from the global dataset

# PROBLEM AVEC CETTE FUNCTION

def delete_NaN_model(df):
    df_copy=df.copy(deep=True) # copy the original dataframe, not to modify the original one    
    model_to_delete =[]
    longitude=[]
    for project in df_copy.index.levels[0].tolist(): # projects
        # look value of longitude for each project
        for j in np.arange(0,len(df_copy.loc[[project]].columns)):
            if ~df_copy[[df_copy.columns[j]]].isnull().values.all():
                # can check if a pandas DataFrame contains NaN/None values in any cell 
                # (all rows & columns ). This method returns True if it finds NaN/None 
                # on any cell of a DataFrame, returns False when not found
                longitude.append(df_copy.columns[j])
                continue
        
        for scenario in df_copy.index.levels[1].tolist(): # scenarios
            for model in df_copy.index.levels[2].tolist(): # models
                if df_copy.loc[(project,scenario, model)].isnull().values.all():
                    print('No data for Project '+ project+', scenario '+scenario+', model '+model)
                    # all the values for the given project, scenario and model are NaN
                    if model not in model_to_delete:
                        model_to_delete.append(model)# keep unique values
    
    if model_to_delete!=[]:
        # for some given project, scenario and model, there is no values
        for model in model_to_delete:
            models_index = df_copy.index.levels[2].tolist()
            models_index.remove(model)
            df_copy.drop(labels=model,level=2,inplace=True)
        
        return models_index
        # create new dataframe with correct index
    return []


# In[ ]:


# this functions aims to calculate the n_day_event
def n_day_rainfall(number_day,df):
    df1=df.copy(deep=True)
    title_col=[i for i in list(df1.columns) if 'precipitation' in i.lower()][0]
    # df.use function rolling(n).sum() to calculate cumulative precipitation over n days
    df1[[title_col]]=df1[[title_col]].rolling(number_day).sum()
    time=df1[['Date']].reset_index(drop=True)
    for k in np.arange(len(time)-number_day,-1,-1):
        time.iloc[number_day-1+k] = time.iloc[k].values + ' to '+time.iloc[number_day-1+k].values
    df1.drop(df1.index[np.arange(0,number_day-1)], inplace=True) # delete first elements which are NaNs
    time = time.iloc[number_day-1:len(time)]
    midx = pd.MultiIndex.from_product([list(set(df1[['Name project']].values.reshape(len(df1),))), list(time.values.reshape(len(time),)),list(set(df1[['Model']].values.reshape(len(df1),))),list(set(df1[['Experiment']].values.reshape(len(df1),))),list(set(df1[['Year']].values.reshape(len(df1),)))],names=['Name project','Period','Model','Experiment','Year'])
    name_col = [str(number_day)+' days rainfall mm']
    Dataframe_n_day_event = pd.DataFrame(data = df1[[title_col]].values, 
                                index = midx,
                                columns = name_col)
    Dataframe_n_day_event = Dataframe_n_day_event.reset_index()
    return Dataframe_n_day_event


# In[ ]:


# this function aims to create the empty dataframe that will be filled

def fill_dataframe(name_project,scenario,model,time,data_df,name_col):
    #df = pd.DataFrame()
    #for i in np.arange(0,len(name_project)):
    midx = pd.MultiIndex.from_product([name_project,scenario,model , time],names=['Name project','Experiment', 'Model', 'Date'])
    name_col = [name_col]#['Precipitation '+str(number_day)+' day event mm']
    Variable_dataframe = pd.DataFrame(data = data_df, 
                                index = midx,
                                columns = name_col)
        #df = pd.concat([df,Variable_dataframe])
    return Variable_dataframe


# In[ ]:


# function dataframe_n_day_event produce a dataframe, with the n_day event precipitation for the period, models and scenarios asked
# this function use the function : 'n_day_rainfall'


def dataframe_n_day_event(df,number_day):
    df_copy=df.copy(deep=True) # copy the original dataframe, not to modify the original one    
    df_n_day_event = pd.DataFrame() # create empty dataframe, that will be filled later
    # extract years of the period of interest, make a vector containing all the years of interest
    years = np.arange(min(df_copy['Year']),max(df_copy['Year']))
    for project in list(set(df_copy['Name project'])): # projects
        for scenario in list(set(df_copy['Experiment'])): # scenarios
            for model in list(set(df_copy['Model'])): # models
                print('Project '+ project+', scenario '+scenario+', model '+model)
                # select on project, one scenario, one model and drop Latitude index
                df_temp_all_years = df_copy[(df_copy['Name project']==project)&(df_copy['Experiment']==scenario)&(df_copy['Model']==model)]#.dropna()
                for year in years:
                    print(year)
                    df_temp_one_year = df_temp_all_years[df_temp_all_years['Year']==year] # select only data for one year
                    df_temp_one_year_n_event=n_day_rainfall(number_day,df_temp_one_year) # use function to calculate cumulative precipitation
                    df_n_day_event = pd.concat([df_n_day_event,df_temp_one_year_n_event])
    return df_n_day_event # return a dataframe, with all the projects, scenarios, models and period of n day


# In[ ]:


# would be nice bettering this function by registering the period or month associated with the maximum 5 day event occuring each year
def dataframe_max_5_days_event(df,number_day):
    df1=df.copy(deep=True)
    df2=dataframe_n_day_event(df1,5)
    df2 = df2.groupby(['Name project','Model','Experiment','Year'])[[str(number_day)+' days rainfall mm']].max()
    
    return df2


# In[ ]:


def dataframe_1_day_event(df):
    df_copy = df.copy(deep=True)
    df_max = df_copy.groupby(['Name project','Experiment','Model','Year']).max() # maximum
    df_max=df_max.drop(labels='Date',axis=1)# drop columns Date
    df_max=df_max.rename(columns={df_max.columns[0]:'Maximum 1 day rainfall mm '+str(df_max.index.levels[3][0])+'-'+str(df_max.index.levels[3][len(df_max.index.levels[3])-1])})
    return df_max


# ### Seasonal average precipitation

# In[ ]:


# function a check
def avg_dry_season_precipitation(df,title_column):
    df_season = df.copy(deep=True)
    df_season=df_season.rename(columns={df_season.columns[4]:title_column})
    
    Month = df_season[['Date']].values.reshape(len(df_season[['Date']].values),)
    Season = df_season[['Date']].values.reshape(len(df_season[['Date']].values),)
    for i in np.arange(0,len(df_season[['Date']].values)):
        Month[i]=Month[i][3:5]
        if int(Month[i])>3 and int(Month[i])<10:
            Season[i]='Dry'
        else:
            Season[i]='Humid'

    #df_season['Month'] = Month
    df_season['Season'] = Season
    df_season=df_season.drop(labels='Date',axis=1)
    df_season=df_season.groupby(['Name project','Experiment','Model','Season','Year']).sum()
    df_season=df_season.groupby(['Name project','Experiment','Model','Season']).mean()
    df_season=df_season.groupby(['Name project','Season']).describe(percentiles=[.1, .5, .9])
    pr_dry_season_mean_distribution=df_season.query('Season=="Dry"')
    pr_dry_season_mean_distribution=pr_dry_season_mean_distribution.reset_index().drop('Season',axis=1).set_index('Name project')
    return pr_dry_season_mean_distribution


# # Changes in indicators

# In[ ]:


# inverse columns and rows for final df

def changes_in_indicators(df_past,df_futur,title_indicator, unit,climate_var):
    # create empty dataframe
    #midx = pd.MultiIndex.from_product([df_years_avg_2041_2060_distribution.index.tolist(),precipitation_2021_2060_copy.index.levels[1].tolist(),models],names=['Name project','Experiment', 'Model'])
    cols = pd.MultiIndex.from_product([(climate_var,),(title_indicator,),('Median for the past period '+unit,'Change in the median in %','10-th percentile for the past period '+unit, 'Change in 10-th percentile %','90-th percentile for the past period '+unit,'Change in 90-th percentile %')])
    changes_past_future_indicator = pd.DataFrame(data = [], 
        index = df_past.index.tolist(),
        columns = cols)
    
    changes_past_future_indicator[[changes_past_future_indicator.columns[0]]]=df_past[[df_past.columns[5]]]
    
    changes_past_future_indicator[[changes_past_future_indicator.columns[1]]]=(((df_futur[[df_futur.columns[5]]].values-df_past[[df_past.columns[5]]].values)/df_past[[df_past.columns[5]]].values)*100).reshape(len(df_past[[df_past.columns[5]]].values,),1)
    
    changes_past_future_indicator[[changes_past_future_indicator.columns[2]]]=df_past[[df_past.columns[4]]]
    
    changes_past_future_indicator[[changes_past_future_indicator.columns[3]]]=(((df_futur[[df_futur.columns[4]]].values-df_past[[df_past.columns[4]]].values)/df_past[[df_past.columns[4]]].values)*100).reshape(len(df_past[[df_past.columns[4]]].values,),1)
    
    changes_past_future_indicator[[changes_past_future_indicator.columns[4]]]=df_past[[df_past.columns[6]]]
    
    changes_past_future_indicator[[changes_past_future_indicator.columns[5]]]=(((df_futur[[df_futur.columns[6]]].values-df_past[[df_past.columns[6]]].values)/df_past[[df_past.columns[6]]].values)*100).reshape(len(df_past[[df_past.columns[6]]].values,),1)
    
    return changes_past_future_indicator


# # Level of exposure

# In[ ]:


# inverse columns and rows for final df

def level_exposure(df):
    # level of exposure by climate variable
    
    # create empty dataframe
    #midx = pd.MultiIndex.from_product([df_years_avg_2041_2060_distribution.index.tolist(),precipitation_2021_2060_copy.index.levels[1].tolist(),models],names=['Name project','Experiment', 'Model'])
    cols = pd.MultiIndex.from_product([('Exposure level',),df.columns.levels[0].tolist()]) # df.columns.levels[0].tolist() is liste of climate variable
    ExposureLevel = pd.DataFrame(data = [],
                            index = df.index.tolist(),
                            columns = cols)
    
    for name_p in ExposureLevel.index.tolist():
        for climate_variable in df.columns.levels[0].tolist():
            list_indicators=list(set([ind[1] for ind in df.columns if climate_variable  in ind]))
            for indicator in list_indicators:
                print('For project '+name_p+', climate variable '+climate_variable)
                if ExposureLevel.loc[name_p,('Exposure level',climate_variable)] != 'High':
                    # for the moment, no other indicator for the climate variable made the exposure high (big changes with bug uncertainty)

                    # select the columns of interest in the list of columns
                    col_interest_med= [cols for cols in df.columns.tolist() if climate_variable in cols and indicator in cols and 'Change in the median in %' in cols]
                    col_interest_p10= [cols for cols in df.columns.tolist() if climate_variable in cols and indicator in cols and 'Change in 10-th percentile %' in cols]
                    col_interest_p90= [cols for cols in df.columns.tolist() if climate_variable in cols and indicator in cols and 'Change in 90-th percentile %' in cols]
                    print(col_interest_p10)
                    if ExposureLevel.loc[name_p,('Exposure level',climate_variable)] != 'Medium':
                        # for the moment, no other indicator for the climate variable made the exposure medium (medium changes with bug uncertainty)
                        print(df.loc[name_p,col_interest_p10][0])
                        print(type(df.loc[name_p,col_interest_p10][0]))
                        if type(df.loc[name_p,col_interest_p10][0])==str:
                            print('string')
                            if (df.loc[name_p,col_interest_p10].values == 'Low') or (df.loc[name_p,col_interest_p90].values == 'Low'):
                                # test if there are any True, if any value is under the threshold indicated
                                ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'No' # attribute value to exposure level

                            if (df.loc[name_p,col_interest_p10].values == 'Medium') or (df.loc[name_p,col_interest_p90].values == 'Medium'):
                            # test if there are any True, if any value is over the threshold indicated
                                ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'Medium' # attribute value to exposure level

                        else:
                            if (df.loc[(name_p),col_interest_p10][abs(df.loc[(name_p),col_interest_p10])<20].notnull().values.any() or df.loc[(name_p),col_interest_p90][abs(df.loc[(name_p),col_interest_p90])<20].notnull().values.any()):
                                # test if there are any True, if any value is under the threshold indicated
                                ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'No' # attribute value to exposure level

                            if (df.loc[(name_p),col_interest_p10][abs(df.loc[(name_p),col_interest_p10])>20].notnull().values.any() or df.loc[(name_p),col_interest_p90][abs(df.loc[(name_p),col_interest_p90])>20].notnull().values.any()):
                            # test if there are any True, if any value is over the threshold indicated
                                ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'Medium' # attribute value to exposure level
                    if type(df.loc[name_p,col_interest_p10][0])==str:
                        if  (df.loc[name_p,col_interest_p10][0] == 'High') or (df.loc[name_p,col_interest_p90][0] == 'High'):
                            ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'High' # attribute value to exposure level
                    else:
                        if (df.loc[(name_p),col_interest_med][abs(df.loc[(name_p),col_interest_med])>20].notnull().values.any()) or (df.loc[(name_p),col_interest_p10][abs(df.loc[(name_p),col_interest_p10])>50].notnull().values.any() or df.loc[(name_p),col_interest_p90][abs(df.loc[(name_p),col_interest_p90])>50].notnull().values.any()):
                            # test if there are any True, if any value is over the threshold indicated
                            ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'High' # attribute value to exposure level

    # those 2 next lines are meant to put colors for Exposure, but prevent from using it as a dataframe after, so give up for the moement
    #ExposureLevel=ExposureLevel.style.apply(exposureColor) # apply color depending on value of Exposure
    #ExposureLevel=ExposureLevel.set_table_styles([{'selector': 'th.col_heading', 'props': 'text-align: left;'}],[{'selector': 'td', 'props': 'text-align: center;'}],overwrite = True) # place first level column to the left
    # meant to place element in dataframe, but do not work very well
    return ExposureLevel


# In[ ]:


# function use in function 'level_exposure' to color result depending on the result
# function not in use for the moment
def exposureColor(series):
    green = 'background-color: lightgreen'
    orange = 'background-color: orange'
    red = 'background-color: red'
    return [red if value == 'High' else orange if value == 'Medium' else green for value in series]


# # Vulnerability

# In[ ]:


# this function only makes sense if data are in the past or the future
# input of this function is a dataframe with no multi evel index. If the dataframe is with multilevel index, should .reset_index()
# Should just put a datfarme of 2 columns as df. The 2 columns sould be ['Name project'] and the colummn of interest
def df_stat_distr(df):
    df = df.groupby(['Name project']).describe(percentiles=[.1, .5, .9])
    # if describe() does not return al wanted statistics, it is maybe because the elements in it are not recognized as int
    # add astype(int) as in following example; df.astype(int).groupby(['Name project']).describe(percentiles=[.1, .5, .9])
    return df


# permit to have the foloowing matrix
# ![image.png](attachment:image.png)

# In[ ]:


def vulnerability(df_sensitivity,df_exposure):
    
    if len(df_sensitivity.columns.levels[1])!=len(df_exposure.columns.levels[1]): # check if both dataframe have the same numbers of indicators we are checking in
        print('The number of climate variables is the sensitivity and in exposure is different')
        return
    
    df_vulnerability = df_sensitivity.copy(deep=True)
        
    df_vulnerability.loc[:,:]='No' # default value
    df_vulnerability=df_vulnerability.rename(columns={'Sensitivity level':'Vulnerability level'}) # chnage name of the column
    
    for name_p in list(df_exposure.index):# go throught projects
        print(name_p)
        for k in np.arange(0,len(df_vulnerability.index.levels[1])): # go through project elements
            for i in np.arange(0,len(df_exposure.columns.levels[1])): # go through climate variables
                #print('i = '+str(i))
                if df_exposure.loc[name_p,('Exposure level',df_exposure.columns.levels[1][i])]=='No':
                    print('No exposure')
                    if df_sensitivity.loc[name_p,('Sensitivity level',df_sensitivity.columns.levels[1][i])][k]=='High':
                        # assign vulnerability to 'Medium'
                        df_vulnerability.loc[name_p,('Vulnerability level',df_vulnerability.columns.levels[1][i])][k]='Medium'
                if df_exposure.loc[name_p,('Exposure level',df_exposure.columns.levels[1][i])]=='Medium':
                    print('Medium exposure')
                    if df_sensitivity.loc[name_p,('Sensitivity level',df_exposure.columns.levels[1][i])][k]=='Medium':
                        # assign vulnerability to 'Medium'
                        df_vulnerability.loc[name_p,('Vulnerability level',df_vulnerability.columns.levels[1][i])][k]='Medium'
                    if df_sensitivity.loc[name_p,('Sensitivity level',df_exposure.columns.levels[1][i])][k]=='High':
                        # assign vulnerability to 'High'
                        df_vulnerability.loc[name_p,('Vulnerability level',df_vulnerability.columns.levels[1][i])][k]='High'
                if df_exposure.loc[name_p,('Exposure level',df_exposure.columns.levels[1][i])]=='High':
                    print('High exposure')
                    if df_sensitivity.loc[name_p,('Sensitivity level',df_exposure.columns.levels[1][i])][k]=='No':
                        # assign vulnerability to 'Medium'
                        df_vulnerability.loc[name_p,('Vulnerability level',df_vulnerability.columns.levels[1][i])][k]='Medium'
                    if df_sensitivity.loc[name_p,('Sensitivity level',df_exposure.columns.levels[1][i])][k]=='Medium':
                        # assign vulnerability to 'High'
                        df_vulnerability.loc[name_p,('Vulnerability level',df_vulnerability.columns.levels[1][i])][k]='High'
                    if df_sensitivity.loc[name_p,('Sensitivity level',df_exposure.columns.levels[1][i])][k]=='High':
                        # assign vulnerability to 'High'
                        df_vulnerability.loc[name_p,('Vulnerability level',df_vulnerability.columns.levels[1][i])][k]='High'

    return df_vulnerability


# In[ ]:





# In[ ]:




