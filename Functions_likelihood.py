#!/usr/bin/env python
# coding: utf-8

# In this notebook, all the calculation for the likelihood are gathered.
# 
# The main function looks for a pdf distribution, that will then permit to determine the likelihood of an event, thanks to the cdf of the function.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt
import statistics


# In[2]:


# for tests
from Functions_ImportData import import_treat_modeled_NEX_GDDP_CMIP6
from Functions_Indicators import filter_dataframe


# In[3]:


# Create models from data
def best_fit_distribution(data, bins=5000, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

        #print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0)) # error calculated here with the mean squared error
                # y ; observed values from histogram
                # pdf ; predicted values from distribution function
                
                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))
        
        except Exception:
            pass

    
    return sorted(best_distributions, key=lambda x:x[2])


# In[4]:


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


# In[5]:


def look_best_distr(data,climate_var,unit):
    matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
    matplotlib.style.use('ggplot')

    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

    # Save plot limits
    dataYLim = ax.get_ylim()

    # Find best fit distribution
    best_distibutions = best_fit_distribution(data, 200, ax)
    best_dist = best_distibutions[0]

    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(u'Histogram data\n All Fitted Distributions')
    ax.set_xlabel(climate_var+' '+unit)
    ax.set_ylabel('Frequency')

    # Make PDF with best params 
    pdf = make_pdf(best_dist[0], best_dist[1])

    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

    param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
    dist_str = '{}({})'.format(best_dist[0].name, param_str)

    ax.set_title(u'Data with best fit distribution \n' + dist_str)
    ax.set_xlabel(climate_var+' '+unit)
    ax.set_ylabel('Frequency')
    print('best distribution '+str(best_dist[0].name))
    return best_dist[0], param_str # return the function of distribution attributes, return the parameters as strings


# In[6]:


# before appling this function, need to import the best ditribution function attributed to the distribution, otherwise, it will not work
# idea of function that gives you direclty the likelihood of an event

# rice is a distribution function
#rice.ppf(0.95,1.8,loc=36.01,scale=2.61)
#rice.cdf(45.512922702726385,1.8,loc=36.01,scale=2.61)
#rice.ppf(0.95,1.8,loc=36.01,scale=2.61)
#rice.ppf(0.05,1.8,loc=36.01,scale=2.61)


# In[7]:


def probability(event, name_distr, params):
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    
    # find the function of the distribution based on the name of the function
    # The getattr() method returns the value of the named attribute of an object. 
    # If not found, it returns the default value provided to the function.
    # https://www.programiz.com/python-programming/methods/built-in/getattr
    distribution = getattr(st, name_distr) # st is the short name for the module 'scipy.stats'
    
    pdf_ = distribution.pdf(x, loc=loc, scale=scale, *arg)
    cdf_ = distribution.cdf(event,loc=loc, scale=scale, *arg)
    
    return


# In[8]:


def range_likelihood(event,unit, distribution, params):
    
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    
    #distribution = getattr(st, name_distr) # st is the short name for the module 'scipy.stats'
    cdf_event = distribution.ppf(event,loc=loc, scale=scale, *arg)
    
    if cdf_event>=0.05 and cdf_event<=0.95:
        if cdf_event>=0.17 and cdf_event<=0.83:
            if cdf_event>=0.25 and cdf_event<=0.75:
                likelihood = str(round(distribution.ppf(0.25,loc=loc, scale=scale, *arg),2))+'-'+str(round(distribution.ppf(0.75,loc=loc, scale=scale, *arg),2))+unit+', probable range' # more likely than not
                return likelihood
            likelihood = str(round(distribution.ppf(0.17,loc=loc, scale=scale, *arg),2))+'-'+str(round(distribution.ppf(0.83,loc=loc, scale=scale, *arg),2))+unit+', likely range'
            return likelihood
        if cdf_event<0.17 and cdf_event>0.83:
            likelihood = 'unlikely range'
            return likelihood
        likelihood = str(round(distribution.ppf(0.05,loc=loc, scale=scale, *arg),2))+'-'+str(round(distribution.ppf(0.95,loc=loc, scale=scale, *arg),2))+unit+', very likely range'
        return likelihood
    else:
        likelihood = 'very unlikely range'
        return likelihood
    print('Problem, no likelihood was found')


# In[9]:


def likelihood(event,type_event,unit, distribution, params):
    
    proba_event=type_event_f(event,type_event,distribution,params)
    
    likelihood = define_likelihood(proba_event)
    
    return proba_event,likelihood


# In[10]:


def type_event_f(event,type_event,distribution,params):
    
    params1=params.split(',')
    arg=[]
    for p in params1:
        if 'loc' in p:
            temp=p.split('=')
            loc = float(temp[len(temp)-1])
            continue
        if 'scale' in p:
            temp=p.split('=')
            scale = float(temp[len(temp)-1]  )   
            continue
        else:
            temp=p.split('=')
            arg.append(float(temp[len(temp)-1]))
            continue

    #distribution = getattr(st, distribution) # st is the short name for the module 'scipy.stats'
    if type_event == '=':
        proba_event = distribution.pdf(event,loc=loc, scale=scale, *arg)
        
    if type_event == '>': # likelihood variable over a threshold
        proba_event = 1-distribution.cdf(event,loc=loc, scale=scale, *arg)
    
    if type_event == '<': # likelihood variable under a threshold
        proba_event = distribution.cdf(event,loc=loc, scale=scale, *arg)
    return proba_event


# In[11]:


def define_likelihood(proba_event):
    if proba_event<0.05:
        likelihood = 'Rare'
        return likelihood
    if proba_event<0.2:
        likelihood = 'Unlikely'
        return likelihood
        
    if proba_event<0.5:
        likelihood = 'Moderate'
        return likelihood
        
    if proba_event<0.8:
        likelihood = 'Likely'
        return likelihood
        
    if proba_event<0.95:
        likelihood = 'Almost certain'
        return likelihood
    else: # over 0.95
        likelihood = 'Certain'
        return likelihood


# In[12]:


def likelihood_accross_models(df,climate_var,unit,name_column,event,type_event):
    
    proba_event=[]
    model_not_in_final_calculation = []
    for model in list(set(df['Model'])):
        (distribution,params)=look_best_distr(df[df['Model']==model][[name_column]],climate_var,unit)
        print('params '+str(params))
        proba_event_model = type_event_f(event,type_event,distribution,params)
        print('proba_event_model '+str(proba_event_model))
        if not np.isnan(proba_event_model): # apparently, when the parameters are too small, can cause trouble to calculate the probability
            proba_event.append(proba_event_model)
            print('proba_event_model registered')
        else:
            model_not_in_final_calculation.append(distribution.name)
        
    proba_event_accross_model=statistics.mean(proba_event)
    
    likelihood=define_likelihood(proba_event_accross_model)
    
    return proba_event_accross_model,likelihood


# In[13]:


# for tests


# In[14]:


#df_tasmax_NEXGDDPCMIP6=import_treat_modeled_NEX_GDDP_CMIP6('tasmax','Celsius','day',1950,2100)


# In[15]:


#df_tasmax_NEXGDDPCMIP6_mutua = filter_dataframe(df_tasmax_NEXGDDPCMIP6,'WTP_Mutua_EIB',['TAIEMS1','CMCC-CM2-SR5'])


# In[16]:


#df_tasmax_NEXGDDPCMIP6_mutua_ssp370 = df_tasmax_NEXGDDPCMIP6_mutua[df_tasmax_NEXGDDPCMIP6_mutua['Experiment']=='ssp370']


# In[17]:


#df_tasmax_NEXGDDPCMIP6_mutua_ssp370[['Model','Daily Maximum Near-Surface Air Temperature °C']]


# In[18]:


#(proba_event_accross_model,likelihood)=likelihood_accross_models(df_tasmax_NEXGDDPCMIP6_mutua_ssp370[['Model','Daily Maximum Near-Surface Air Temperature °C']].dropna(),'maximum temperature','°C','Daily Maximum Near-Surface Air Temperature °C',40,'>')


# In[19]:


#likelihood


# In[20]:


#proba_event_accross_model


# In[21]:


#df_tasmax_NEXGDDPCMIP6_mutua_future = df_tasmax_NEXGDDPCMIP6_mutua[df_tasmax_NEXGDDPCMIP6_mutua['Experiment']!='historical']


# In[22]:


#(proba_event_accross_model_2,likelihood_2)=likelihood_accross_models(df_tasmax_NEXGDDPCMIP6_mutua_future[['Model','Daily Maximum Near-Surface Air Temperature °C']].dropna(),'maximum temperature','°C','Daily Maximum Near-Surface Air Temperature °C',40,'>')


# In[23]:


#proba_event_accross_model_2


# In[24]:


#likelihood_2


# In[ ]:




