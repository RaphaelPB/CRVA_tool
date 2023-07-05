#!/usr/bin/env python
# coding: utf-8

# This notebook aims to contain all functions for indicators.

# In[1]:


from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gumbel_r
from scipy.stats import gumbel_l
import os
import os.path
import math


# In[2]:


# Treat data


# In[3]:


## Add Year, Month and Season to df


# In[4]:


def add_year_month_season(df,column_date):
    # add Year, month and season columns for graphs
    Year = df[[column_date]].values.reshape(len(df[[column_date]].values),)
    Month = df[[column_date]].values.reshape(len(df[[column_date]].values),)
    Season = df[[column_date]].values.reshape(len(df[[column_date]].values),)
    
    if str(Year[1]).find('-')==2:
        for i in np.arange(0,len(df[[column_date]].values)):
            Year[i]=int(Year[i][6:10])
            Month[i]=int(Month[i][3:5])
            if Month[i]>3 and Month[i]<10: # dry season in Mozambique is between April and September
                Season[i]='Dry'
            else:# humid season is between October and March
                Season[i]='Humid'
            
            Month[i]=str_month(Month[i])
            
    if str(Year[1]).find('-')==4:
        for i in np.arange(0,len(df[[column_date]].values)):
            Year[i]=int(Year[i][0:4])
            Month[i]=int(Month[i][5:7])
            if Month[i]>3 and Month[i]<10: # dry season in Mozambique is between April and September
                Season[i]='Dry'
            else:# humid season is between October and March
                Season[i]='Humid'
            
            Month[i]=str_month(Month[i])
                
    df['Year'] = Year
    df['Month'] = Month
    df['Season'] = Season
    return df


# In[5]:


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


# In[6]:


# this function is meant to filter the data wnated 
def filter_dataframe(df,name_location,list_model_to_kill,start_y=1950,stop_y=2100):
    df = df.reset_index() # to take out multiindex that may exist and complicate filtering process
    
    for name_model in list_model_to_kill:
        df = df[df['Model']!=name_model]
    
    df = df[df['Name project']==name_location] # select only data of interest
    df=add_year_month_season(df,'Date') # add column 'Year', 'Month', 'Season'
    
    if start_y!=1950 or stop_y!=2100:
        df = df[df['Year'].between(start_y,stop_y)] # select only the years of interest
    
    return df


# In[7]:


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


# # Precipitation

# ### Return period

# In[8]:


# return period for each project, model, scenario


# In[9]:


# function value for return period


# In[10]:


def threshold_coresponding_to_return_period(loc,scale,T):
    p_non_exceedance = 1 - (1/T)
    try:
        threshold_coresponding = round(gumbel_r.ppf(p_non_exceedance,loc,scale))
    except OverflowError: # the result is not finite
        if math.isinf(gumbel_r.ppf(p_non_exceedance,loc,scale)) and gumbel_r.ppf(p_non_exceedance,loc,scale)<0:
            # ppf is the inverse of cdf
            # the result is -inf
            threshold_coresponding = 0 # the value of wero is imposed
    return threshold_coresponding
    # ppf: Percent point function
    #print('Threshold '+str(threshold_coresponding)+' mm/day will be exceeded at least once in '+str(n)+' year, with a probability of '+str(round(p_exceedance*100))+ ' %')
    #print('This threshold corresponds to a return period of '+str(round(return_period))+ ' year event over a '+str(n)+' year period')


# In[11]:


def dataframe_threshold_coresponding_to_return_period(df):
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
                Z=df_max.loc[(name_p,ssp,model)].values.reshape(len(df_max.index.levels[3]),)
                (loc1,scale)=stats.gumbel_r.fit(Z) # return the function necessary to establish the continous function
                # choice of gumbel because suits to extreme precipitation
                return_period.loc[(name_p,ssp,model),('Value for return period 50 years mm/day')] = threshold_coresponding_to_return_period(loc1,scale,50)
                return_period.loc[(name_p,ssp,model),('Value for return period 100 years mm/day')] = threshold_coresponding_to_return_period(loc1,scale,100)
                
    return return_period


# In[12]:


def return_period_coresponding_to_threshold(Z):
    (loc,scale)=stats.gumbel_r.fit(Z) # return the function necessary to establish the continous function
    # gumbel_r is chosen because
    #try:
    p_non_exceedance = round(gumbel_r.cdf(max(Z),loc,scale))
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.ppf.html
    #except OverflowError: # the result is not finite
        
   #     if math.isinf(gumbel_r.cdf(threshold,loc,scale)) and gumbel_r.cdf(max(Z),loc,scale)<0:
   #         # the result is -inf
    #        threshold_coresponding = 0 # the value of wero is imposed
    return_period_coresponding = 1/(1-p_non_exceedance)
    return return_period_coresponding


# In[13]:


def dataframe_future_return_period_of_1_day_event(df):
    df_copy=df.copy(deep=True)
    df_copy=df_copy.drop(labels='Date',axis=1)
    df_max = df_copy.groupby(['Name project','Experiment','Model','Year']).max() # maximum    
    midx = pd.MultiIndex.from_product([list(set(df_copy[df_copy.columns[0]])),list(set(df_copy[df_copy.columns[1]])),list(set(df_copy[df_copy.columns[2]]))],names=['Name project','Experiment', 'Model'])
    cols = ['Return period 50 years mm/day','Return period 100 years mm/day']
    return_period = pd.DataFrame(data = [], 
                                index = midx,
                                columns = cols)
    for name_p in return_period.index.levels[0].tolist():
        for ssp in return_period.index.levels[1].tolist():
            for model in return_period.index.levels[2].tolist():
                print('Name project '+name_p+ ' ssp '+ssp+ ' model '+model)
                Z=df_max.loc[(name_p,ssp,model)].values.reshape(len(df_max.index.levels[3]),)
                return_period.loc[(name_p,ssp,model),('Return period 1 day event')] = return_period_coresponding_to_threshold(Z)
    return return_period


# In[14]:


r'''
#Z = test['Precipitation mm/day'].values
Z.sort()
(loc,scale)=stats.gumbel_r.fit(Z) # return the function necessary to establish the continous function
# xaxis is precipitation and yaxis is densiy of probability
myHist = plt.hist(Z,density=True) # If ``True``, draw and return a probability density: each bin 
# will display the bin's raw count divided by the total number of counts *and the bin width*
h = plt.plot(Z,gumbel_r.pdf(Z,loc,scale))
plt.xlabel('Precipitation value mm/day')
plt.ylabel('Density of probability' )
plt.title('Histogram and probability density function of precipitation values\nfor year 2021 for one project ,\n one scenario and one model',fontdict={'fontsize': 10})
plt.legend(['Probability density function','Histogramm'])
title_png = 'test_density.png'
path_figure = os.path.join(out_path,'figures')
if not os.path.isdir(path_figure):
    os.makedirs(path_figure)
#plt.savefig(os.path.join(path_figure,title_png),format ='png')
plt.show()'''


# In[15]:


# accross models and scenarios


# In[16]:


r'''
test=precipitation_2021_2060_copy.loc[(precipitation_2021_2060_copy.index.levels[0][0])]
test=test[[('Longitude','36.875')]]
test=test.droplevel(level=3) # drop latitude index
test.columns = test.columns.droplevel(0) # drop first level of column name
test=test.rename(columns={test.columns[0]:'Precipitation mm/day'})
test=test.swaplevel(0,2,axis=0)
test = test.filter(like = str(2021), axis=0) # select only data for one year
#test#['Precipitation mm'].values
test.drop(labels='NESM3',level=1,inplace=True)
'''


# In[17]:


r'''
Z = test['Precipitation mm/day'].values
Z.sort()
(loc,scale)=stats.gumbel_r.fit(Z) # return the function necessary to establish the continous function
# xaxis is precipitation and yaxis is densiy of probability
myHist = plt.hist(Z,density=True) # If ``True``, draw and return a probability density: each bin 
# will display the bin's raw count divided by the total number of counts *and the bin width*
h = plt.plot(Z,gumbel_r.pdf(Z,loc,scale))
plt.xlabel('Precipitation value mm/day')
plt.ylabel('Density of probability' )
plt.title('Histogram and probability density function of precipitation values\nfor year 2021 for one project ,\n with all scenarios and models',fontdict={'fontsize': 10})
plt.legend(['Probability density function','Histogramm'])
title_png = 'test_density.png'
path_figure = os.path.join(out_path,'figures')
if not os.path.isdir(path_figure):
    os.makedirs(path_figure)
#plt.savefig(os.path.join(path_figure,title_png),format ='png')
plt.show()
'''


# In[18]:


# questions Temps retour :
#      tjs avec maximum ? oui, proba sur un an
#      besoin de caler mieux distribution ? package pour le faire automatiquement ? si on fiat pas avec maxima, mais on va faire que avec maxima pour le moment


# In[ ]:





# ### N-day event 

# In[19]:


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


# In[20]:


# this functions aims to calculate the n_day_event
def n_day_maximum_rainfall(number_day,df):
    df1=df.copy(deep=True)
    # df.use function rolling(n).sum() to calculate cumulative precipitation over n days
    df1[['Precipitation mm']]=df1[['Precipitation mm']].rolling(number_day).sum()
    time=df1.index.tolist()
    for k in np.arange(len(time)-number_day,-1,-1):
        time[number_day-1+k] = time[k] + ' to '+time[number_day-1+k]
    df1.drop(df1.index[np.arange(0,number_day-1)], inplace=True) # delete first elements which are NaNs
    del time[0:number_day-1] # delete firsts elements, which have no value associated with
    #midx = pd.MultiIndex.from_product([ time],names=['Date'])
    name_col = ['Precipitation mm']
    Dataframe_n_day_event = pd.DataFrame(data = df1.values, 
                                index = [time],
                                columns = name_col)
    return Dataframe_n_day_event


# In[21]:


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


# In[22]:


# function dataframe_n_day_event produce a dataframe, with the n_day event precipitation for the period, models and scenarios asked
# this function use the function : 'delete_NaN_model', 'n_day_maximum_rainfall' and 'fill_dataframe'


def dataframe_n_day_event(df,number_day):
    df_copy=df.copy(deep=True) # copy the original dataframe, not to modify the original one    
    df_n_day_event = pd.DataFrame() # create empty dataframe, that will be filled later
    # extract years of the period of interest, make a vector containing all the years of interest
    years = np.arange(int(df.index.levels[3].tolist()[0][6:10]),int(df.index.levels[3].tolist()[len(df.index.levels[3].tolist())-1][6:10])+1)
    #models_index=delete_NaN_model(df_copy) # use function 'delete_NaN_model' to know which models have no Nan values
    models_index = df_copy.index.levels[2].tolist()
    models_index.remove('NESM3')
    df_copy=df_copy.droplevel(level=4) # drop latitude index
    df_copy.columns = df_copy.columns.droplevel(0) # drop first level of column name
    for project in df_copy.index.levels[0].tolist(): # projects
        for scenario in df_copy.index.levels[1].tolist(): # scenarios
            for model in models_index: # models
                print('Project '+ project+', scenario '+scenario+', model '+model)
                # select on project, one scenario, one model and drop Latitude index
                df_temp_all_years = df_copy.loc[(project,scenario,model)]
                # find which columns does not only have NaN
                for j in np.arange(0,len(df_temp_all_years.columns)): # for loop to have number of the column
                    if ~df_temp_all_years[[df_temp_all_years.columns[j]]].isnull().values.all():
                        # the column does not only have Nan values
                        df_temp_all_years=df_temp_all_years[[df_temp_all_years.columns[j]]] # register only column with values, and not the NaN values
                        df_temp_all_years=df_temp_all_years.rename(columns={df_temp_all_years.columns[0]:'Precipitation mm'})
                        # rename the column
                        break # stop the for loop with the number of columns, because values were found
                        # go to line if df_temp_all_years.columns.nlevels!=1:
                if df_temp_all_years.columns.nlevels!=1:
                    # the dataframe still has two levels of columns, so the precedent if condition was never fullfilled
                    print('The model '+model+' has no data')
                    continue # try with the next model
                else:
                    # the dataframe still has one level of columns, there was one column not containing only NaN values
                    for year in years:
                        print(year)
                        df_temp_one_year = df_temp_all_years.filter(like = str(year), axis=0) # select only data for one year
                        #return df_temp_one_year
                        df_temp_one_year_n_event=n_day_maximum_rainfall(number_day,df_temp_one_year) # use function to calculate cumulative precipitation
                        #return df_temp_one_year_n_event
                        # format time vector differently
                        time = [df_temp_one_year_n_event.index.tolist()[i][0] for i in np.arange(0,len(df_temp_one_year_n_event.index.tolist()))]
                        # fill dataframe
                        df_temp_one_year_n_event = fill_dataframe((project,),(scenario,),(model,),time,df_temp_one_year_n_event.values,'Maximum '+str(number_day)+' days rainfall mm')
                        df_n_day_event = pd.concat([df_n_day_event,df_temp_one_year_n_event])
    return df_n_day_event # return a dataframe, with all the projects, scenarios, models and period of n day


# In[23]:


def dataframe_1_day_event(df):
    df_copy = df.copy(deep=True)
    df_max = df_copy.groupby(['Name project','Experiment','Model','Year']).max() # maximum
    df_max=df_max.drop(labels='Date',axis=1)# drop columns Date
    df_max=df_max.rename(columns={df_max.columns[0]:'Maximum 1 day rainfall mm '+str(df_max.index.levels[3][0])+'-'+str(df_max.index.levels[3][len(df_max.index.levels[3])-1])})
    return df_max


# In[24]:


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


# ### Yearly average precipitation

# In[32]:


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


# In[26]:


def yearly_avg_distr(df_yearly_avg):
    df_years_avg_distribution = df_yearly_avg.groupby(['Name project']).describe(percentiles=[.1, .5, .9])
    # if describe() does not return al wanted statistics, it is maybe because the elements in it are not recognized as int
# add astype(int) as in following example; df.astype(int).groupby(['Name project']).describe(percentiles=[.1, .5, .9])
    return df_years_avg_distribution


# ### Seasonal average precipitation

# In[27]:


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

# In[28]:


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

# In[29]:


## Functions not finished

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
            print('For project '+name_p+', climate variable '+climate_variable)
            if ExposureLevel.loc[name_p,('Exposure level',climate_variable)] != 'High':
                # for the moemnt, no other indicator for the climate variable made the exposure high (big changes with bug uncertainty)
                
                # select the columns of interest in the list of columns
                col_interest_med= [cols for cols in df.columns.tolist() if climate_variable in cols and 'Change in the median in %' in cols]
                col_interest_p10= [cols for cols in df.columns.tolist() if climate_variable in cols and 'Change in 10-th percentile %' in cols]
                col_interest_p90= [cols for cols in df.columns.tolist() if climate_variable in cols and 'Change in 90-th percentile %' in cols]
                
                if ExposureLevel.loc[name_p,('Exposure level',climate_variable)] != 'Medium':
                    if (df.loc[(name_p),col_interest_p10][abs(df.loc[(name_p),col_interest_p10])<20].notnull().values.any() or df.loc[(name_p),col_interest_p90][abs(df.loc[(name_p),col_interest_p90])<20].notnull().values.any()):
                        # test if there are any True, if any value is under the threshold indicated
                        ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'No' # attribute value to exposure level

                    if (df.loc[(name_p),col_interest_p10][abs(df.loc[(name_p),col_interest_p10])>20].notnull().values.any() or df.loc[(name_p),col_interest_p90][abs(df.loc[(name_p),col_interest_p90])>20].notnull().values.any()):
                    # test if there are any True, if any value is over the threshold indicated
                        ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'Medium' # attribute value to exposure level


                if (df.loc[(name_p),col_interest_med][abs(df.loc[(name_p),col_interest_med])>20].notnull().values.any()) or (df.loc[(name_p),col_interest_p10][abs(df.loc[(name_p),col_interest_p10])>50].notnull().values.any() or df.loc[(name_p),col_interest_p90][abs(df.loc[(name_p),col_interest_p90])>50].notnull().values.any()):
                    # test if there are any True, if any value is over the threshold indicated
                    ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'High' # attribute value to exposure level
    
    ExposureLevel=ExposureLevel.style.apply(exposureColor) # apply color depending on value of Exposure
    ExposureLevel=ExposureLevel.set_table_styles([{'selector': 'th.col_heading', 'props': 'text-align: left;'}],[{'selector': 'td', 'props': 'text-align: center;'}],overwrite = True) # place first level column to the left
    # meant to place element in dataframe, but do not work very well
    return ExposureLevel


# In[30]:


# function use in function 'level_exposure' to color result depending on the result
def exposureColor(series):
    green = 'background-color: lightgreen'
    orange = 'background-color: orange'
    red = 'background-color: red'
    return [red if value == 'High' else orange if value == 'Medium' else green for value in series]


# In[ ]:





# In[ ]:


# graphs


# In[34]:


# data_1 : first set of data to be used, should only contains the location of interest
# source_1 : source of the first set of data
# data_2 : second set of dat to be used, should only contains the location of interest
# source_2 : source of the second set of data

def trends_month(climate_var,data_1,source_1,data_2,source_2,stats,location,temporal_resolution='Month',start_year_line=1970,stop_year_line=2014,start_year_boxplot=2015,stop_year_boxplot=2100):
    
    (climate_var_longName,unit)= infos_str(climate_var,temporal_resolution)
    
    # define the new common name, that will be used as y_axis for boxplots and line
    new_name_col = temporal_resolution+'ly '+climate_var_longName+' '+unit
    
    if 'NEX-GDDP-CMIP6' in source_1:
        if (start_year_boxplot!=2014) or (stop_year_boxplot!=2100):
            data_1=data_1[data_1['Year'].between(start_year_boxplot,stop_year_boxplot)]
        data_boxplot=prepare_NEX_GDDP_CMIP6(data_1,climate_var_longName,stats,temporal_resolution,new_name_col)
        source_boxplot=source_1
    if 'NEX-GDDP-CMIP6' in source_2:
        if (start_year_boxplot!=2014) or (stop_year_boxplot!=2100):
            data_2=data_2[data_2['Year'].between(start_year_boxplot,stop_year_boxplot)]
        data_boxplot=prepare_NEX_GDDP_CMIP6(data_2,climate_var_longName,stats,temporal_resolution,new_name_col)
        source_boxplot=source_2
    if 'NOAA' in source_1:
        if (start_year_line!=1970) or (stop_year_line!=2014):
            data_1=data_1[data_1['Year'].between(start_year_line,stop_year_line)]
        title_column=title_column_NOAA_obs(source_1,climate_var)
        data_line=prepare_NOAA(data_1,title_column,temporal_resolution,new_name_col)
        source_line=source_1
    if 'NOAA' in source_2:
        if (start_year_line!=1970) or (stop_year_line!=2014):
            data_2=data_2[data_2['Year'].between(start_year_line,stop_year_line)]
        title_column=title_column_NOAA_obs(source_2,climate_var)
        data_line=prepare_NOAA(data_2,title_column,temporal_resolution,new_name_col)
        source_line=source_2
    
    if temporal_resolution == 'Month': # to plot the data in the chronological order of the months
        month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        data_boxplot=data_boxplot.reset_index().set_index(temporal_resolution).loc[month_order].reset_index()
        data_line=data_line.reset_index().set_index(temporal_resolution).loc[month_order].reset_index()    
    
    if stats == 'Sum':
        title_plot = climate_var_longName+' '+unit+', modeled by '+source_boxplot+',\nbetween '+str(start_year_boxplot)+' and '+str(stop_year_boxplot)+' at '+location+' compared with '+source_line+'\nobservation data, between '+str(start_year_line)+' and '+str(stop_year_line)
    else:
        title_plot = stats+' '+climate_var_longName+' '+unit+', modeled by '+source_boxplot+',\nbetween '+str(start_year_boxplot)+' and '+str(stop_year_boxplot)+' at '+location+' compared with '+source_line+'\nobservation data, between '+str(start_year_line)+' and '+str(stop_year_line)
        
    boxplots_line(data_boxplot,data_line,temporal_resolution,new_name_col,source_line,title_plot)


# In[35]:


# data_boxplot : dataframr that will be used to do the boxplots
# data_line : dataframe that will be used to add a line
# x_axis : Name of the column that wil be used for the x_axis
# y_axis : Name of the column that wil be used for the y_axis
# ----> x_axis and y_axis are both a str, and both should be used as name of colum in the dataframes for the boxplots and 
#       the line
# source_line : name of the source of the data plot in the line
# title_plot : title for this plot. Should be defined in the function before
# categories : default parameters, will be used for the hue of the boxplot. The hue is a third dimension along a depth axis, 
#              where different levels are plotted with different colors

#stats+' monthly precipitation mm/month between '+start_year+' and '+stop_year+'\n with '+source_obs+' observed data and '+source_model+' modeled data, at '+location

def boxplots_line(data_boxplot,data_line,x_axis,y_axis,source_line,title_plot,categories='Experiment'):
    fig,ax=plt.subplots()
    sns.boxplot(data=data_boxplot, x=x_axis, y=y_axis, hue=categories,ax=ax)
    ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
    sns.lineplot(data=data_line,x=x_axis, y=y_axis,ax=ax,label=source_line)
    
    # display the common legend for the line and boxplots
    handles, labels=ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.2, 0.5),title='Legend')
    ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
    plt.title(title_plot)
    path_figure=os.path.join(r'C:\Users\CLMRX\OneDrive - COWI\Documents\GitHub\CRVA_tool\outputs\figures','trend_month.png')
    plt.savefig(path_figure,format ='png') # savefig or save text must be before plt.show. for savefig, format should be explicity written

    plt.show()


# In[36]:


def prepare_NOAA(df_NOAA,title_column,temporal_resolution,new_name_col):
    df_NOAA = df_NOAA.reset_index()
    df = df_NOAA[[title_column,temporal_resolution]].groupby(temporal_resolution).mean().rename(columns={title_column:new_name_col}).reset_index()
    
    print('title_column '+title_column)
    print('temporal_resolution '+temporal_resolution)
    
    
    if 'PR' in title_column and temporal_resolution=='Month':
        print('pr and month, multiplication by 30')
        df[new_name_col] = df[[new_name_col]].values*30
    
    return df


# In[37]:


def prepare_NEX_GDDP_CMIP6(df,climate_var_longName,stats,temporal_resolution,new_name_col):
    try:
        try:
            title_column=df.filter(like=climate_var_longName, axis=1).columns[0]
        except:
            title_column=df.filter(like=climate_var_longName.capitalize(), axis=1).columns[0]
    except:
        title_column=df.filter(like=climate_var_longName.upper(), axis=1).columns[0]
        
    if stats == 'Average':
        data_NEXGDDPCMIP6=df[['Experiment','Model',temporal_resolution,title_column]].groupby(['Experiment','Model',temporal_resolution]).mean().rename(columns={title_column:new_name_col}).reset_index()
    if stats == 'Sum':
        data_NEXGDDPCMIP6=df[['Experiment','Model',temporal_resolution,title_column]].groupby(['Experiment','Model',temporal_resolution]).sum().rename(columns={title_column:new_name_col}).reset_index()
    if stats == 'Median':
        data_NEXGDDPCMIP6=df[['Experiment','Model',temporal_resolution,title_column]].groupby(['Experiment','Model',temporal_resolution]).median().rename(columns={title_column:new_name_col}).reset_index()
    
    if 'pr' in climate_var_longName and temporal_resolution =='Month':
        data_NEXGDDPCMIP6[new_name_col] = data_NEXGDDPCMIP6[[new_name_col]].values*30
    
    return data_NEXGDDPCMIP6


# In[40]:


def infos_str(climate_var,temporal_resolution):
    if 'pr' in climate_var.lower():
        climate_var_longName = 'precipitation'
        unit='mm/'+temporal_resolution[0].lower()+temporal_resolution[1:len(temporal_resolution)]
    if 'tas' in climate_var.lower():
        unit=u'\N{DEGREE SIGN}C'
        climate_var_longName = 'temperature'
    if climate_var=='tasmax':
        climate_var_longName = 'Daily Maximum Near-Surface Air Temperature '
    if climate_var=='tasmin':
        climate_var_longName = 'minimum '+climate_var_longName
    return climate_var_longName,unit


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




