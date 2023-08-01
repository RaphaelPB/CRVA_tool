#!/usr/bin/env python
# coding: utf-8

# This notebook aims to contain all the functions that will permit to apply bias correction.
# 
# TO DO :
# 
# NEED TO CHECK FUNCTIONALITY OF THE CODE for other methods than bscd precipitation
# apply cdf for quantile ..
# impose a version for the past and for the future (without y test, not used for BC, just for presentation of results)

# In[1]:


# function to calculte return period

from scipy import stats
from scipy.stats import gumbel_r
from scipy.stats import gumbel_l

# function qui marche pour precipitation return period


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

from Functions_Indicators import add_year_month_season # need to add conversion of time


# # BIAS CORRECTION - POINT WISE METHOD
# 
# [Scikit-downscale](https://github.com/pangeo-data/scikit-downscale/tree/main)
# [Detailed process here](https://github.com/pangeo-data/scikit-downscale/blob/main/examples/2020ECAHM-scikit-downscale.ipynb)

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


import numpy as np
import pandas as pd
import scipy
import xarray as xr
import os
import os.path

import warnings
warnings.filterwarnings("ignore")  # sklearn

import matplotlib.pyplot as plt
import seaborn as sns



# exploratory data analysis for arrm model
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
# train_test_split Quick utility that wraps input validation and
#    ``next(ShuffleSplit().split(X, y))`` and application to input data
#    into a single call for splitting (and optionally subsampling) data in a
#    oneliner.
#    Returns
#    -------
#    splitting : list, length=2 * len(arrays)
#        List containing train-test split of inputs.

#        .. versionadded:: 0.16
#            If the input is sparse, the output will be a
#            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
#            input type.

#from utils import get_sample_data

from sklearn.preprocessing import KBinsDiscretizer
# use for discretization

sns.set(style='darkgrid')


# In[3]:


# function to prepare data for the fitting
def treat_data_for_test(df_obs,name_col_obs,df_model_past,name_col_model,name_station,model):
    #from sklearn.preprocessing import StandardScaler
    if 'pr' in name_col_model.lower():
        new_name = 'pcp'
    if 'temp' in name_col_model.lower():
        if 'maximum' in name_col_model.lower():
            new_name = 'temp_max'
        if 'minimum' in name_col_model.lower():
            new_name = 'temp_min'
        if 'maximum' not in name_col_model.lower() and 'minimum' not in name_col_model.lower():
            new_name = 'temp'
    
    # prepare training data
    df_model_past_BC=df_model_past[df_model_past['Name station']==name_station].drop(['Name station','Year','Month','Season'],axis =1)
    df_model_past_BC = df_model_past_BC[df_model_past_BC['Model'] ==model].drop(['Model','Experiment','Latitude','Longitude'],axis=1)
    training = df_model_past_BC.rename(columns = {'Date':'time',name_col_model:new_name}).reset_index(drop=True)
    

    # Scale your dataset to avoid “ValueError: Input contains NaN, infinity or a value too large for dtype(‘float64’)”
    #scaler = StandardScaler()
    #training[new_name].values = scaler.fit_transform(training[new_name].values)
    
    Date1 = training['time'].values
    for i in np.arange(0,len(training)):
        training['time'][i] = Date1[i][6:10]+'-'+Date1[i][3:5]+'-'+Date1[i][0:2]#datetime.strptime(, '%Y-%M-%d').date()
        #print(training['time'][i])
    # .date() to avoid having the hours in the datetime
    training=training.set_index('time')
    
    # prepare targets data
    targets = df_obs[['NAME','DATE',name_col_obs]] # select only 3 columns of interest
    targets = targets[targets['NAME']==name_station].rename(columns = {'DATE':'time',name_col_obs:new_name}).set_index('time').drop(['NAME'],axis=1) # the targets data is meant to represent our "observations"
    
    if len(targets)>len(training):
        targets = targets.dropna() # drop rows with NaN
        training = training[training.index.isin(list(targets.index))]
    if len(targets)<len(training):
        training = training.dropna() # drop rows with NaN
        targets = targets[targets.index.isin(list(training.index))]
    
    # concat training and target data in one dataframe
    df=pd.concat({'training': training, 'targets': targets}, axis=1)
    df=df.dropna()
    
    return df


# In[4]:


# df_obs and df_model should be under a dataframe format, with no nan values, with a common timelaps, with the data as a string format '%Y-%m-%d', and as index

# Method could be :
#        piecewise_regressor
#        Quantile_Linear_Regression

def BC(df,name_col,method,name_station,name_project,name_model):
    # set title and xaxis
    if name_col == 'pcp':
        climate_var = 'Precipitation '
        unit = '[mm/day]'
    if name_col == 'temp':
        climate_var = 'Temperature '
        unit = u'\{°}C'
    if name_col == 'temp_max':
        climate_var = 'Maximum temperature '
        unit = u'\{°}C'
    if name_col == 'temp_min':
        climate_var = 'Minimum temperature '
        unit = u'\{°}C'
    
    if method == 'piecewise_regressor':
        (X_train, X_test, y_train, y_test,pred)=piecewise_regressor(df,name_col)
        #(X_train, X_test, y_train, y_test,name_strat,score_strat)=piecewise_regressor(df,name_col)
        #return X_train, X_test, y_train, y_test,name_strat,score_strat
        plot_train_test(X_train, X_test, y_train, y_test,name_station,name_col)
        plot_train_test_pred(X_train, X_test, y_train, y_test,pred,name_station,name_project,name_model,name_col)
        plot_test_pred(X_test,y_test, y_train, pred,name_station,name_project,name_model,name_col)
        # plot CDF
        #plot_cdfs(X_test,y_test,pred,name_station,name_project,name_model,name_col)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_cdf(ax=ax,X=X_test, y=y_test, out=pred)
        ax.set_xlabel('Cumulative distribution function')
        ax.set_ylabel(climate_var+unit)
        fig.suptitle(climate_var+'cumulative distribution function with observed data from '+name_station+' and modelled data from '+name_project)
        
    if method == 'Quantile_Linear_Regression':
        (X_train, X_test, y_train, y_test,pred)=Quantile_Linear_Regression(df,name_col) 
        plot_train_test(X_train, X_test, y_train, y_test,name_station,name_col)
        plot_train_test_pred(X_train, X_test, y_train, y_test,pred,name_station,name_project,name_model,name_col)
        plot_test_pred(X_test,y_test, y_train, pred,name_station,name_project,name_model,name_col)
        # plot CDF
        #plot_cdfs(X_test,y_test,pred,name_station,name_project,name_model,name_col)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_cdf(ax=ax,X=X_test, y=y_test, out=pred)
        ax.set_xlabel('Cumulative distribution function')
        ax.set_ylabel(climate_var+unit)
        fig.suptitle(climate_var+'cumulative distribution function with observed data from '+name_station+' and modelled data from '+name_project)
        
    if method == 'Quantile_MLP_Regressor':
        (X_train, X_test, y_train, y_test,pred)=Quantile_MLP_Regressor(df,name_col)
        plot_train_test(X_train, X_test, y_train, y_test,name_station,name_col)
        plot_train_test_pred(X_train, X_test, y_train, y_test,pred,name_station,name_project,name_model,name_col)
        plot_test_pred(X_test,y_test, y_train, pred,name_station,name_project,name_model,name_col)
        # plot CDF
        #plot_cdfs(X_test,y_test,pred,name_station,name_project,name_model,name_col)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_cdf(ax=ax,X=X_test, y=y_test, out=pred)
        ax.set_xlabel('Cumulative distribution function')
        ax.set_ylabel(climate_var+unit)
        fig.suptitle(climate_var+'cumulative distribution function with observed data from '+name_station+' and modelled data from '+name_project)
    
    if method == 'Bcsd_Precipitation':
        (X_train, X_test, y_train, y_test,pred)=BCSD_Precipitation(df)
        plot_time_series(X_test,y_test,pred,name_model)
        plot_train_test_pred(X_train.values, X_test.values, y_train.values, y_test.values,pred.values,name_station,name_project,name_model,name_col)
        plot_test_pred(X_test.values,y_test.values, y_train.values, pred.values,name_station,name_project,name_model,name_col)
        plot_cdfs(X_test,y_test,pred,name_station,name_project,name_model,name_col)
        
        
    if method == 'BCSD_Precipitation_without_multi':
        (X_train, X_test, y_train, y_test,pred)=BCSD_Precipitation_without_multi(df)
        plot_time_series(X_test,y_test,pred,name_model)
        plot_train_test_pred(X_train.values, X_test.values, y_train.values, y_test.values,pred.values,name_station,name_project,name_model,name_col)
        plot_test_pred(X_test.values,y_test.values, y_train.values, pred.values,name_station,name_project,name_model,name_col)
        plot_cdfs(X_test,y_test,pred,name_station,name_project,name_model,name_col)
        
        
    if method == 'Bcsd_Temperature':
        (X_train, X_test, y_train, y_test,pred)=BCSD_Temperature(df)
        plot_time_series(X_test,y_test,pred,name_model)
        plot_train_test_pred(X_train.values, X_test.values, y_train.values, y_test.values,pred.values,name_station,name_project,name_model,name_col)
        plot_test_pred(X_test.values,y_test.values, y_train.values, pred.values,name_station,name_project,name_model,name_col)
        plot_cdfs(X_test,y_test,pred,name_station,name_project,name_model,name_col)
    
    if method == 'BCSD_Temperature_without_addition':
        (X_train, X_test, y_train, y_test,pred)=BCSD_Temperature_without_addition(df)
        plot_time_series(X_test,y_test,pred,name_model)
        plot_train_test_pred(X_train.values, X_test.values, y_train.values, y_test.values,pred.values,name_station,name_project,name_model,name_col)
        plot_test_pred(X_test.values,y_test.values, y_train.values, pred.values,name_station,name_project,name_model,name_col)
        plot_cdfs(X_test,y_test,pred,name_station,name_project,name_model,name_col)

    return pred 


# In[5]:


def piecewise_regressor(df,name_col):
    from mlinsights.mlmodel import PiecewiseRegressor # in piecewise estimator
    #     Uses a :epkg:`decision tree` to split the space of features
    #    into buckets and trains a linear regression (default) on each of them.
    #    The second estimator is usually a :epkg:`sklearn:linear_model:LinearRegression`.
    #    It can also be :epkg:`sklearn:dummy:DummyRegressor` to just get
    #    the average on each bucket.
    
    X = df[[('training',name_col)]][min(df.index)[0:4]: max(df.index)[0:4]].values#training[[name_col]]['1980': '2000'].values
    y = df[[('targets',name_col)]][min(df.index)[0:4]: max(df.index)[0:4]].values#targets[[name_col]]['1980': '2000'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)# splits data
    
    # parameters for Quantile transforms
    qqwargs = {'n_quantiles': int(1e6), 'copy': True, 'subsample': int(1e6)} # add int for n_quantiles and subsample to avoid
    # following problem:  InvalidParameterError: The 'n_quantiles' parameter of QuantileTransformer must be an int in the range [1, inf). Got 1000000.0 instead.
    n_bins = 7

    
    score_strat =[]
    name_strat = ['kmeans', 'uniform', 'quantile']
    print('R2 score')
    for strat in name_strat:
        model = PiecewiseRegressor(binner=KBinsDiscretizer(n_bins=n_bins, strategy=strat))
        #model.fit(X_train, y_train)
        model.fit(X_train.reshape((len(X_train),1)), y_train.reshape((len(y_train),)))
        #pred = model.predict(X_test)
        pred = model.predict(X_test.reshape((len(X_test),1)))#*X_test.reshape((len(X_test),))
        #print(model.score(X_test, y_test))
        print(model.score(X_test.reshape((len(X_test),1)), y_test.reshape((len(y_test),))))
        score_strat.append(model.score(X_test.reshape((len(X_test),1)), y_test.reshape((len(y_test),))))
        # how is the score calculated ? r2 score
    #return X_train, X_test, y_train, y_test,name_strat,score_strat
    #model = PiecewiseRegressor(binner=KBinsDiscretizer(n_bins=n_bins, strategy=name_strat[int(np.where(score_strat==max(score_strat))[0])]))
    #model.fit(X_train, y_train)
    model = PiecewiseRegressor(binner=KBinsDiscretizer(n_bins=n_bins, strategy=name_strat[int(np.where(score_strat==max(score_strat))[0])]))
    model.fit(X_train.reshape((len(X_train),1)), y_train.reshape((len(y_train),)))
    #if name_col=='pcp':
    #    pred = model.predict(X_test.reshape((len(X_test),1)))*X_test.reshape((len(X_test),))
    #    print('Applying correction for precipitation')
    #if 'temp' in name_col.lower():
    #    pred = model.predict(X_test.reshape((len(X_test),1)))+X_test.reshape((len(X_test),))
    #    print('Applying correction for temperature')
    #if name_col!='pcp' and 'temp' not in name_col.lower():
    #else:
    pred = model.predict(X_test)
    #pred = model.predict(X_test.reshape((len(X_test),1)))
    #    print('Applying correction for other climate variable')
    print('Strategy chosen is '+name_strat[int(np.where(score_strat==max(score_strat))[0])])
    
    return X_train, X_test, y_train, y_test,pred


# In[6]:


def Quantile_Linear_Regression(df,name_col):
    from mlinsights.mlmodel import QuantileLinearRegression # in quantile_regression
    
    X = df[[('training',name_col)]][min(df.index)[0:4]: max(df.index)[0:4]].values#training[[name_col]]['1980': '2000'].values
    y = df[[('targets',name_col)]][min(df.index)[0:4]: max(df.index)[0:4]].values#targets[[name_col]]['1980': '2000'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)# splits data
    
    #y_train = y_train[:, 0]
    
    model = QuantileLinearRegression()

    model.fit(X_train.reshape((len(X_train),1)), y_train.reshape((len(y_train),)))
    #if name_col=='pcp':
       # pred = model.predict(X_test.reshape((len(X_test),1)))*X_test.reshape((len(X_test),))
    #if 'temp' in name_col.lower():
        #pred = model.predict(X_test.reshape((len(X_test),1)))+X_test.reshape((len(X_test),))
    #if name_col!='pcp' and 'temp' not in name_col.lower():
    #else:
    pred = model.predict(X_test.reshape((len(X_test),1)))
    print('mean absolute error')
    print(model.score(X_test.reshape((len(X_test),1)), y_test.reshape((len(y_test),))))# mean absolute error
    

    return (X_train, X_test, y_train, y_test,pred)


# In[7]:


def Quantile_MLP_Regressor(df,name_col):
    from mlinsights.mlmodel import QuantileMLPRegressor # in qunatile_mlpregressor
    
    X = df[[('training',name_col)]][min(df.index)[0:4]: max(df.index)[0:4]].values#training[[name_col]]['1980': '2000'].values
    y = df[[('targets',name_col)]][min(df.index)[0:4]: max(df.index)[0:4]].values#targets[[name_col]]['1980': '2000'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)# splits data
    
    model = QuantileMLPRegressor()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print('mean absolute error')
    print(model.score(X_test, y_test)) # mean absolute error
    
    return (X_train, X_test, y_train, y_test,pred)


# In[8]:


def BCSD_Precipitation(df):
    from skdownscale.pointwise_models import BcsdPrecipitation

    training = df['training']
    targets = df['targets']
    training.index = pd.to_datetime(training.index)
    targets.index = pd.to_datetime(targets.index)
    X_pcp = training[["pcp"]].resample("MS").sum()#MS
    y_pcp = targets[["pcp"]].resample("MS").sum()
    # Fit/predict the BCSD Temperature model
    bcsd_temp = BcsdPrecipitation()
    bcsd_temp.fit(X_pcp, y_pcp)
    out = bcsd_temp.predict(X_pcp) * X_pcp # additive for temperature, multiplicative for precipitation
    
    return (X_pcp,X_pcp,y_pcp,y_pcp,out)


# In[1]:


def BCSD_Precipitation_return_anoms(df):
    from skdownscale.pointwise_models import BcsdPrecipitation

    training = df['training']
    targets = df['targets']
    training.index = pd.to_datetime(training.index)
    targets.index = pd.to_datetime(targets.index)
    X_pcp = training[["pcp"]]#.resample("MS").sum()#MS
    y_pcp = targets[["pcp"]]#.resample("MS").sum()
    # Fit/predict the BCSD Temperature model
    bcsd_temp = BcsdPrecipitation(return_anoms=False)
    bcsd_temp.fit(X_pcp, y_pcp)
    out = bcsd_temp.predict(X_pcp) * X_pcp # additive for temperature, multiplicative for precipitation
    
    return (X_pcp,X_pcp,y_pcp,y_pcp,out)


# In[9]:


def BCSD_Precipitation_without_multi(df):
    from skdownscale.pointwise_models import BcsdPrecipitation

    training = df['training']
    targets = df['targets']
    training.index = pd.to_datetime(training.index)
    targets.index = pd.to_datetime(targets.index)
    X_pcp = training[["pcp"]].resample("MS").sum()#MS
    y_pcp = targets[["pcp"]].resample("MS").sum()
    # Fit/predict the BCSD Temperature model
    bcsd_temp = BcsdPrecipitation()
    bcsd_temp.fit(X_pcp, y_pcp)
    out = bcsd_temp.predict(X_pcp)# * X_pcp # additive for temperature, multiplicative for precipitation
    
    return (X_pcp,X_pcp,y_pcp,y_pcp,out)


# In[10]:


def BCSD_Precipitation_one_more_time(df,out):
    
    df=df.loc[out.index]
    df['training'] = out['pcp']
    print(df)
    out = BCSD_Precipitation(df)
    
    return out


# In[11]:


# missing graphs

def BCSD_Temperature(df):
    from skdownscale.pointwise_models import BcsdTemperature
    training = df['training']
    targets = df['targets']
    training.index = pd.to_datetime(training.index)
    targets.index = pd.to_datetime(targets.index)
    X_temp = training[[training.columns[0]]].resample("MS").mean()#MS
    y_temp = targets[[training.columns[0]]].resample("MS").mean()
    
    X_temp = X_temp.dropna()
    y_temp = y_temp.dropna()
    
    if len(X_temp) != len(y_temp):
        if len(X_temp) <= len(y_temp):
            y_temp[y_temp.index.isin(list(X_temp.index))]
        if len(X_temp) >= len(y_temp):
            X_temp[X_temp.index.isin(list(y_temp.index))]
            
    print('Check for nan values')
    print(X_temp.isnull().sum())
    print(y_temp.isnull().sum())
    print('Check for infinity values')
    print(np.isinf(y_temp).sum())
    print(y_temp.isnull().sum())
    # Fit/predict the BCSD Temperature model
    bcsd_temp = BcsdTemperature()
    bcsd_temp.fit(X_temp, y_temp)
    out = bcsd_temp.predict(X_temp) + X_temp # additive for temperature, multiplicative for precipitation
    return (X_temp,X_temp,y_temp,y_temp,out)


# In[12]:


# missing graphs

def BCSD_Temperature_without_addition(df):
    from skdownscale.pointwise_models import BcsdTemperature
    training = df['training']
    targets = df['targets']
    training.index = pd.to_datetime(training.index)
    targets.index = pd.to_datetime(targets.index)
    X_temp = training[[training.columns[0]]].resample("MS").mean()#MS
    y_temp = targets[[training.columns[0]]].resample("MS").mean()
    
    X_temp = X_temp.dropna()
    y_temp = y_temp.dropna()
    
    if len(X_temp) != len(y_temp):
        if len(X_temp) <= len(y_temp):
            y_temp[y_temp.index.isin(list(X_temp.index))]
        if len(X_temp) >= len(y_temp):
            X_temp[X_temp.index.isin(list(y_temp.index))]
            
    print('Check for nan values')
    print(X_temp.isnull().sum())
    print(y_temp.isnull().sum())
    print('Check for infinity values')
    print(np.isinf(y_temp).sum())
    print(y_temp.isnull().sum())
    # Fit/predict the BCSD Temperature model
    bcsd_temp = BcsdTemperature()
    bcsd_temp.fit(X_temp, y_temp)
    out = bcsd_temp.predict(X_temp)# + X_temp # additive for temperature, multiplicative for precipitation
    return (X_temp,X_temp,y_temp,y_temp,out)


# In[13]:


# plot results

def plot_time_series(X_test_to_copy,y_test_to_copy,out_to_copy,name_model):
    
    #out.plot()
    #plt.title('')
    
    X_test=X_test_to_copy.copy(deep=True)
    y_test=y_test_to_copy.copy(deep=True)
    out=out_to_copy.copy(deep=True)
    
    
    #out=out.rename(columns={'training':'pcp'})
    
    out.index = out.index.strftime('%Y-%m-%d')
    X_test.index = X_test.index.strftime('%Y-%m-%d')
    y_test.index = y_test.index.strftime('%Y-%m-%d')
    
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(8, 9), sharex=True)
    time_slice = slice(min(X_test.index), max(X_test.index))

    # plot-temperature
    #training[time_slice]['pcp'].plot(ax=axes[0], label='training')
    X_test[time_slice][X_test.columns[0]].plot(ax=axes[0], label='training')
    #X_test[time_slice][[('training','pcp')]].plot(ax=axes[0], label='training')
    axes[0].legend()
    if X_test.columns[0]=='pcp':
        axes[0].set_ylabel('Precipitation [mm/day]')
        climate_var = 'precipitation'
    if 'temp' in X_test.columns[0]:
        axes[0].set_ylabel(u'Temperature ['+u'\{°}'+'C]')
        climate_var = 'temperature'
    axes[0].set_ylim(0,max(X_test.values))


    # plot-precipitation
    #targets[time_slice]['pcp'].plot(ax=axes[1], label='target')
    y_test[time_slice][y_test.columns[0]].plot(ax=axes[1], label='target')
    #y_test[time_slice][[('targets','pcp')]].plot(ax=axes[1], label='target')
    axes[1].legend()
    if y_test.columns[0]=='pcp':
        str_ylabel='Precipitation [mm/day]'
    if 'temp' in y_test.columns[0]:
        if X_test.columns[0]=='temp':
            str_ylabel='Temperature [°C]'
        if X_test.columns[0]=='temp_max':
            str_ylabel='Maximum temperature [°C]'
        if X_test.columns[0]=='temp_min':
            str_ylabel='Minimum temperature [°C]'
    _ = axes[1].set_ylabel(str_ylabel)
    axes[1].set_ylim(0,max(y_test.values))

    # plot-precipitation
    out[time_slice][out.columns[0]].plot(ax=axes[2], label='out')
    #out[time_slice][[('training','pcp')]].plot(ax=axes[2], label='out')
    axes[2].legend()
    if X_test.columns[0]=='pcp':
        str_ylabel='Precipitation [mm/day]'
    if 'temp' in X_test.columns[0]:
        if X_test.columns[0]=='temp':
            str_ylabel='Temperature [°C]'
        if X_test.columns[0]=='temp_max':
            str_ylabel='Maximum temperature [°C]'
        if X_test.columns[0]=='temp_min':
            str_ylabel='Minimum temperature [°C]'
    _ = axes[2].set_ylabel(str_ylabel)
    axes[2].set_ylim(0,max(out.values))
    
    fig.suptitle('Comparison of observed and modeled data ('+name_model+') with '+climate_var+' bias corrected time serie')
    return

def plot_train_test(X_train, X_test, y_train, y_test,name_station,name_col):
    if name_col == 'pcp':
        climate_var = 'Precipitation'
    if name_col == 'temp':
        climate_var = 'Temperature'
    if name_col == 'temp_max':
        climate_var = 'Maximum temperature'
    if name_col == 'temp_min':
        climate_var = 'Minimum temperature'
        
    sns.set(style='whitegrid')
    c = {'train': 'black', 'predict': 'blue', 'test': 'grey'}
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.scatter(X_train, y_train, c=c['train'], s=5, label='train')
    plt.scatter(X_test, y_test, c=c['test'], s=5, label='test')
    plt.title(climate_var+' train and test data from '+name_station)
    plt.xlabel('modeled data')
    plt.ylabel('observed data')
    ax.legend()
    return

def plot_train_test_pred(X_train, X_test, y_train, y_test,pred,name_station,name_project,name_model,name_col):
    import pyspark
    from pyspark.sql import DataFrame
    
    if name_col == 'pcp':
        climate_var = 'Precipitation'
    if name_col == 'temp':
        climate_var = 'Temperature'
    if name_col == 'temp_max':
        climate_var = 'Maximum temperature'
    if name_col == 'temp_min':
        climate_var = 'Minimum temperature'
        
    sns.set(style='whitegrid')
    c = {'train': 'black', 'predict': 'blue', 'test': 'grey'}
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.scatter(np.sort(X_train, axis=0), np.sort(y_train, axis=0), c=c['train'], s=5, label='train')
    if not isinstance(X_train, DataFrame): # test if it is a DataFrame
        #if not sum(X_test - X_train)[0]==0: # test if it is the dataframe are the similar
        plt.scatter(np.sort(X_test, axis=0), np.sort(y_test, axis=0), c=c['test'], s=5, label='test')
    plt.plot(np.sort(X_test, axis=0), np.sort(pred, axis=0), c=c['predict'], lw=2, label='predictions')
    plt.title(climate_var+' sorted train and test data from '+name_station+' and prediction for '+name_project+' modelled data with '+name_model)
    plt.xlabel('modeled data')
    plt.ylabel('observed data and prediction')
    ax.legend()
    return
    
def plot_test_pred(X_test,y_test, y_train, pred,name_station,name_project,name_model,name_col):
    
    if name_col == 'pcp':
        climate_var = 'Precipitation'
    if name_col == 'temp':
        climate_var = 'Temperature'
    if name_col == 'temp_max':
        climate_var = 'Maximum temperature'
    if name_col == 'temp_min':
        climate_var = 'Minimum temperature'
        
    sns.set(style='whitegrid')
    c = {'train': 'black', 'predict': 'blue', 'test': 'grey'}

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    #ax.plot(X_test[:, 0], y_test, ".", label='data', c=c['test'])
    #ax.plot(X_test[:, 0], pred, ".", label="predictions", c=c['predict'])
    ax.plot(X_test, y_test, ".", label='data', c=c['test'])
    ax.plot(X_test, pred, ".", label="predictions", c=c['predict'])
    #ax.set_title(f"Piecewise Linear Regression\n{n_bins} buckets")
    plt.title(climate_var+' test data '+name_station+' and prediction  for '+name_project+' data modelled with '+name_model)
    plt.xlabel('modeled data')
    plt.ylabel('observed data and prediction')
    ax.legend()
    return

def plot_cdfs(X_test,y_test,out,name_station,name_project,name_model,name_col):
    if name_col == 'pcp':
        climate_var = 'Precipitation'
        unit = '[mm/day]'
    if name_col == 'temp':
        climate_var = 'Temperature'
        unit = '°C'
    if name_col == 'temp_max':
        climate_var = 'Maximum temperature'
        unit = '°C'
    if name_col == 'temp_min':
        climate_var = 'Minimum temperature'
        unit = '°C'
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_cdf(ax=ax,X=X_test, y=y_test, out=out)
    # set title and xaxis
    ax.set_xlabel('Cumulative distribution function')
    ax.set_ylabel(climate_var+unit)
    fig.suptitle(climate_var+'cumulative distribution function with observed data from '+name_station+' and modelled data with '+name_model+' from '+name_project)
    
    #plot_cdf_by_month(X=X_test, y=y_test.loc[list(X_test.index)], out=out)
    fig=plot_cdf_by_month(X=X_test, y=y_test, out=out)
    fig.suptitle(climate_var+'cumulative distribution function with observed data from '+name_station+' and modelled data with '+name_model+' from '+name_project+' for each month')
    
    return

# utilities for plotting cdfs
def plot_cdf(ax=None, **kwargs):
    if ax:
        plt.sca(ax)
    else:
        ax = plt.gca()
    LW = 4
    for label, X in kwargs.items():
        vals = np.sort(X, axis=0)
        pp = scipy.stats.mstats.plotting_positions(vals)
        ax.plot(pp, vals, label=label,linewidth=LW)
        LW -= 1
    ax.legend()
    return ax


def plot_cdf_by_month(ax=None, **kwargs):
    fig, axes = plt.subplots(4, 3, sharex=True, sharey=False, figsize=(12, 8))
    for label, X in kwargs.items():
        for month, ax in zip(range(1, 13), axes.flat):

            vals = np.sort(X[X.index.month == month], axis=0)
            pp = scipy.stats.mstats.plotting_positions(vals)
            ax.plot(pp, vals, label=label)
            ax.set_title(month)
    ax.legend()
    # set title and xaxis
    if X.columns[0] == 'pcp':
        climate_var = 'Precipitation '
        unit = '[mm/day]'
    if X.columns[0] == 'temp':
        climate_var = 'Temperature '
        unit = '°C'
    if X.columns[0] == 'temp_max':
        climate_var = 'Maximum temperature '
        unit = '°C'
    if X.columns[0] == 'temp_min':
        climate_var = 'Minimum temperature '
        unit = '°C'
    
    fig.supxlabel('Cumulative distribution function')
    fig.supylabel(climate_var+unit)
    return fig


# In[ ]:





# In[ ]:





# In[14]:


# comment on fait pour savoir chronologie de donnees corrigees ?


# In[ ]:





# In[ ]:





# In[15]:


# test bcsd one more time


# In[ ]:


r'''
climate_var_NEX_GDDP_CMIP6_EmplacementStation_BC=climate_var_NEX_GDDP_CMIP6_EmplacementStation[climate_var_NEX_GDDP_CMIP6_EmplacementStation['Name project']==name_station].drop(['Name project','Year','Month','Season'],axis =1)
climate_var_NEX_GDDP_CMIP6_EmplacementStation_BC_model = climate_var_NEX_GDDP_CMIP6_EmplacementStation_BC[climate_var_NEX_GDDP_CMIP6_EmplacementStation_BC['Model'] =='ACCESS-CM2'].drop(['Model'],axis=1)
training = climate_var_NEX_GDDP_CMIP6_EmplacementStation_BC_model.rename(columns = {'Date':'time','Mean of the daily precipitation rate mm/day':'pcp'}).reset_index()

# changing format of Date for training
Date1 = training['time'].values
for i in np.arange(0,len(training)):
    training['time'][i] = Date1[i][6:10]+'-'+Date1[i][3:5]+'-'+Date1[i][0:2]#datetime.strptime(, '%Y-%M-%d').date()
    print(training['time'][i])
# .date() to avoid having the hours in the datetime
training=training.set_index('time').drop(['index'],axis=1)


# targets
targets = data_obs_NOAA[['NAME','DATE','PRCP']] # select only 3 columns of interest
targets = targets[targets['NAME']==name_station].rename(columns = {'DATE':'time','PRCP':'pcp'}).set_index('time').drop(['NAME'],axis=1) # the targets data is meant to represent our "observations"


# to have the same size of vectors
targets = targets.dropna() # drop rows with NaN
training = training[training.index.isin(list(targets.index))]'''


# In[ ]:


r'''df=pd.concat({'training': training, 'targets': targets}, axis=1)

df=df.dropna()'''


# In[ ]:


r'''df = df.droplevel(1,axis=1)
df'''


# In[ ]:


#out=BCSD_Precipitation(df)


# In[ ]:


#out


# In[ ]:


# return period before first BC
#out3 = add_year_month_season(out.reset_index(),'time')
#Z=out3.groupby('Year')[['pcp']].max()
#(loc1,scale)=stats.gumbel_r.fit(Z) # return the function necessary to establish the continous function
# choice of gumbel because suits to extreme precipitation
#return_period.loc[(name_p,ssp,model),('Value for return period 50 years mm/day')] = 
#threshold_coresponding_to_return_period(loc1,scale,50) ## 113
#return_period.loc[(name_p,ssp,model),('Value for return period 100 years mm/day')] = 
#threshold_coresponding_to_return_period(loc1,scale,100) # 124


# In[ ]:


#out=BCSD_Precipitation_one_more_time(df,out) # second time


# In[ ]:


#out


# In[ ]:


#out=BCSD_Precipitation_one_more_time(df,out) # third time


# In[ ]:


#out


# In[ ]:


#out=BCSD_Precipitation_one_more_time(df,out) # fourth time


# In[ ]:


#out3 = add_year_month_season(out.reset_index(),'time')


# In[ ]:


#out3


# In[ ]:


#Z=out3.groupby('Year')[['pcp']].max()


# In[ ]:


#out


# In[ ]:


#(loc1,scale)=stats.gumbel_r.fit(Z) # return the function necessary to establish the continous function
# choice of gumbel because suits to extreme precipitation
#return_period.loc[(name_p,ssp,model),('Value for return period 50 years mm/day')] = 
#threshold_coresponding_to_return_period(loc1,scale,50) ## 220
#return_period.loc[(name_p,ssp,model),('Value for return period 100 years mm/day')] = 
#threshold_coresponding_to_return_period(loc1,scale,100) # 245


# In[ ]:


#out=BCSD_Precipitation_one_more_time(df,out) # fifth time


# In[ ]:


#out


# In[ ]:


#out = add_year_month_season(out.reset_index(),'time')


# In[ ]:


#Z=out.groupby('Year')[['pcp']].max()


# In[ ]:


#(loc1,scale)=stats.gumbel_r.fit(Z) # return the function necessary to establish the continous function
# choice of gumbel because suits to extreme precipitation
#return_period.loc[(name_p,ssp,model),('Value for return period 50 years mm/day')] = 
#threshold_coresponding_to_return_period(loc1,scale,50) ## 245
#return_period.loc[(name_p,ssp,model),('Value for return period 100 years mm/day')] = 
#threshold_coresponding_to_return_period(loc1,scale,100) # 274


# In[ ]:





# In[ ]:




