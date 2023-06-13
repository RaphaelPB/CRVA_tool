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


# # Precipitation

# ### Return period

# In[2]:


# return period for each project, model, scenario


# In[3]:


# function value for return period


# In[4]:


def threshold_coresponding_to_return_period(Z,T):
    (loc,scale)=stats.gumbel_r.fit(Z) # return the function necessary to establish the continous function
    # gumbel_r is chosen because
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


# In[5]:


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
                return_period.loc[(name_p,ssp,model),('Value for return period 50 years mm/day')] = threshold_coresponding_to_return_period(Z,50)
                return_period.loc[(name_p,ssp,model),('Value for return period 100 years mm/day')] = threshold_coresponding_to_return_period(Z,100)
    return return_period


# In[6]:


def return_period_coresponding_to_threshold(Z):
    (loc,scale)=stats.gumbel_r.fit(Z) # return the function necessary to establish the continous function
    # gumbel_r is chosen because
    #try:
    p_non_exceedance = round(gumbel_r.cdf(max(Z),loc,scale))
    #except OverflowError: # the result is not finite
        
   #     if math.isinf(gumbel_r.cdf(threshold,loc,scale)) and gumbel_r.cdf(max(Z),loc,scale)<0:
   #         # the result is -inf
    #        threshold_coresponding = 0 # the value of wero is imposed
    return_period_coresponding = 1/(1-p_non_exceedance)
    return return_period_coresponding


# In[7]:


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


# In[8]:


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


# In[9]:


# accross models and scenarios


# In[10]:


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


# In[11]:


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


# In[12]:


# questions Temps retour :
#      tjs avec maximum ? oui, proba sur un an
#      besoin de caler mieux distribution ? package pour le faire automatiquement ? si on fiat pas avec maxima, mais on va faire que avec maxima pour le moment


# In[ ]:





# ### N-day event 

# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


def dataframe_1_day_event(df):
    df_copy = df.copy(deep=True)
    df_max = df_copy.groupby(['Name project','Experiment','Model','Year']).max() # maximum
    df_max=df_max.drop(labels='Date',axis=1)# drop columns Date
    df_max=df_max.rename(columns={df_max.columns[0]:'Maximum 1 day rainfall mm '+str(df_max.index.levels[3][0])+'-'+str(df_max.index.levels[3][len(df_max.index.levels[3])-1])})
    return df_max


# ### Yearly average precipitation

# In[18]:


def yearly_avg_pr(df,title_column):
    df_yearly_avg = df.copy(deep =True)
    df_yearly_avg=df_yearly_avg.drop(labels='Date',axis=1)
    df_yearly_avg=df_yearly_avg.rename(columns={df_yearly_avg.columns[3]:title_column})
    df_yearly_avg = df_yearly_avg.groupby(['Name project','Experiment','Model','Year']).sum() # sum per year
    df_yearly_avg = df_yearly_avg.groupby(['Name project','Experiment','Model']).mean()
    df_years_avg_distribution = df_yearly_avg.groupby(['Name project']).describe(percentiles=[.1, .5, .9])
# if describe() does not return al wanted statistics, it is maybe because the elements in it are not recognized as int
# add astype(int) as in following example; df.astype(int).groupby(['Name project']).describe(percentiles=[.1, .5, .9])
    return df_years_avg_distribution


# In[ ]:





# ### Seasonal average precipitation

# In[19]:


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

# In[20]:


def changes_in_indicators(df_past,df_futur,title_indicator,climate_var):
    # create empty dataframe
    #midx = pd.MultiIndex.from_product([df_years_avg_2041_2060_distribution.index.tolist(),precipitation_2021_2060_copy.index.levels[1].tolist(),models],names=['Name project','Experiment', 'Model'])
    cols = pd.MultiIndex.from_product([(climate_var,),(title_indicator,),('Change in the median in %', 'Change in 10-th percentile %','Change in 90-th percentile %')])
    changes_past_future_indicator = pd.DataFrame(data = [], 
        index = df_past.index.tolist(),
        columns = cols)
    
    changes_past_future_indicator[[changes_past_future_indicator.columns[0]]]=(((df_futur[[df_futur.columns[5]]].values-df_past[[df_past.columns[5]]].values)/df_past[[df_past.columns[5]]].values)*100).reshape(len(df_past[[df_past.columns[5]]].values,),1)
    changes_past_future_indicator[[changes_past_future_indicator.columns[1]]]=(((df_futur[[df_futur.columns[4]]].values-df_past[[df_past.columns[4]]].values)/df_past[[df_past.columns[4]]].values)*100).reshape(len(df_past[[df_past.columns[4]]].values,),1)
    changes_past_future_indicator[[changes_past_future_indicator.columns[2]]]=(((df_futur[[df_futur.columns[6]]].values-df_past[[df_past.columns[6]]].values)/df_past[[df_past.columns[6]]].values)*100).reshape(len(df_past[[df_past.columns[6]]].values,),1)
    
    return changes_past_future_indicator


# # Level of exposure

# In[21]:


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
        for climate_variable in changes_past_future_indicator.columns.levels[0].tolist():
            print('For project '+name_p+', climate variable '+climate_variable)
            # select the columns of interest in the list of columns
            col_interest_med= [cols for cols in changes_past_future_indicator.columns.tolist() if climate_variable in cols and changes_past_future_indicator.columns.levels[2][2] in cols]
            col_interest_p10= [cols for cols in changes_past_future_indicator.columns.tolist() if climate_variable in cols and changes_past_future_indicator.columns.levels[2][0] in cols]
            col_interest_p90= [cols for cols in changes_past_future_indicator.columns.tolist() if climate_variable in cols and changes_past_future_indicator.columns.levels[2][1] in cols]

            if (changes_past_future_indicator.loc[(name_p),col_interest_med][abs(changes_past_future_indicator.loc[(name_p),col_interest_med])>20].notnull().values.any()) or (changes_past_future_indicator.loc[(name_p),col_interest_p10][abs(changes_past_future_indicator.loc[(name_p),col_interest_p10])>50].notnull().values.any() or changes_past_future_indicator.loc[(name_p),col_interest_p90][abs(changes_past_future_indicator.loc[(name_p),col_interest_p90])>50].notnull().values.any()):
                # test if there are any True, if any value is over the threshold indicated
                ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'High' # attribute value to exposure level
            if (changes_past_future_indicator.loc[(name_p),col_interest_p10][abs(changes_past_future_indicator.loc[(name_p),col_interest_p10])>20].notnull().values.any() or changes_past_future_indicator.loc[(name_p),col_interest_p90][abs(changes_past_future_indicator.loc[(name_p),col_interest_p90])>20].notnull().values.any()):
                # test if there are any True, if any value is over the threshold indicated
                ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'Medium' # attribute value to exposure level
            if (changes_past_future_indicator.loc[(name_p),col_interest_p10][abs(changes_past_future_indicator.loc[(name_p),col_interest_p10])<20].notnull().values.any() or changes_past_future_indicator.loc[(name_p),col_interest_p90][abs(changes_past_future_indicator.loc[(name_p),col_interest_p90])<20].notnull().values.any()):
                # test if there are any True, if any value is under the threshold indicated
                ExposureLevel.loc[name_p,('Exposure level',climate_variable)] = 'Low' # attribute value to exposure level
    
    ExposureLevel.style.apply(exposureColor) # apply color depending on value of Exposure
    return ExposureLevel


# In[22]:


def exposureColor(series):
    green = 'background-color: lightgreen'
    orange = 'background-color: orange'
    red = 'background-color: red'
    return [red if value == 'High' else orange if value == 'Medium' else green for value in series]


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




