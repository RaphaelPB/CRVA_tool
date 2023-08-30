#!/usr/bin/env python
# coding: utf-8

# This file contains function to be used to produce plots.

# In[1]:


# Packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# # Plot
# 
# the function produces a graph with 
# 
# several set of data
# the x ticks can be rotate
# x label
# y label
# a title
# a legend
# the possibility to save the image

# In[ ]:


# this function is to plot the statistics of the evolution of the climate variable of interest for a certain station
def plot_(data_obs,data_model,stats,climate_var,title_column_obs,title_column_modeled,source_obs,source_modeled,name_station,start_year,stop_year):
    if stats == 'sum':
        for model in list_models_NEX_GDDP_CMIP6:
            yearly_climate_var_NEX_GDDP_CMIP6 = data_model[data_model['Model']==model].groupby('Year')[[title_column_modeled]].sum().rename(columns = {title_column_modeled:'Yearly '+climate_var+' mm/year'})
            plt.plot(yearly_climate_var_NEX_GDDP_CMIP6.index,yearly_climate_var_NEX_GDDP_CMIP6,label=model)
        climate_var_yearly_obs=data_obs.groupby('Year')[[title_column_obs]].sum()
        plt.plot(climate_var_yearly_obs.index,climate_var_yearly_obs,'k',label='observation')
        plt.xlabel('Years')
        plt.ylabel('Yearly '+climate_var+' mm/year')
        plt.title('Yearly '+climate_var+' mm accross models from '+source_modeled+', with observation\nfrom '+source_obs+', at station '+name_station+', between '+str(start_year)+' and '+str(stop_year))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    if stats == 'average':
        for model in list_models_NEX_GDDP_CMIP6:
            yearly_climate_var_NEX_GDDP_CMIP6 = data_model[data_model['Model']==model].groupby('Year')[[title_column_modeled]].mean().rename(columns = {title_column_modeled:'Average yearly '+climate_var+' mm/day'})
            plt.plot(yearly_climate_var_NEX_GDDP_CMIP6.index,yearly_climate_var_NEX_GDDP_CMIP6,label=model)

        climate_var_yearly_obs=data_obs.groupby('Year')[[title_column_obs]].mean()
        plt.plot(climate_var_yearly_obs.index,climate_var_yearly_obs,'k',label='observation')    
        #plt.ylim(0,1.5)
        plt.xlabel('Years')
        plt.ylabel('Average yearly '+climate_var+' mm/day')
        plt.title('Average yearly '+climate_var+' mm accross models from '+source_modeled+', with observation\nfrom '+source_obs+', at name station '+name_station+', between '+str(start_year)+' and '+str(stop_year))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    if stats == 'median':
        for model in list_models_NEX_GDDP_CMIP6:
            yearly_climate_var_NEX_GDDP_CMIP6 = data_model[data_model['Model']==model].groupby('Year')[[title_column_modeled]].median().rename(columns = {title_column_modeled:'Median yearly '+climate_var+' mm/day'})
            plt.plot(yearly_climate_var_NEX_GDDP_CMIP6.index,yearly_climate_var_NEX_GDDP_CMIP6,label=model)

        climate_var_yearly_obs=data_obs.groupby('Year')[[title_column_obs]].median()
        plt.plot(climate_var_yearly_obs.index,climate_var_yearly_obs,'k',label='observation')    
        plt.ylim(0,1.5)
        plt.xlabel('Years')
        plt.ylabel('Median yearly '+climate_var+' mm/day')
        plt.title('Median yearly '+climate_var+' mm accross models from '+source_modeled+', with observation\nfrom '+source_obs+', at name station '+name_station+', between '+str(start_year)+' and '+str(stop_year))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


# # Boxplot
# 
# several set of data
# the x ticks can be rotate
# x label
# y label
# a title
# a legend
# possibility to color the boxes
# the possibility to save the image

# In[ ]:


# box_plot is a function to plot one boxplot in a graph
# the inputs are
#    data to put in boxplots
#    label of the data
#    the color of the box
# the function returns the properties of the boxplot graphe with bp
def box_plot(data, label, fill_color):
    
    bp = plt.boxplot(data,labels = [label],notch=True, whis =(10,90), patch_artist=True,showfliers=False)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       
        
    return bp

# the function several_boxplot is a function to plot several boxplots in one graph (to compare them)
# the inputs:
#    the data in a certainn format, the length should be the same as the text_label
#    text_label contains the name of each set of data to be presented in boxplots
#   All the following inputs are used for titles or labels
#    climate_var is the climate variable of interest (example:'precipitation')
#    source_obs is the source of the observation data
#    source_modeled is the source of the modeled data
#    full_name_climate_var is the complete name of the climate variable of interest (example:'Mean of the daily precipitation rate mm/day')
#    y_label_text is the label for the y axis (example:'Observational data vs Models')
#    path is the out_path where to register data
def several_boxplot(data_boxplot,text_label,climate_var,source_obs,source_modeled,full_name_climate_var,y_label_text,path_figure):
    fig, ax = plt.subplots()
    colors = []
    bp=plt.boxplot(data_boxplot,labels = text_label,notch=True, whis =(10,90),patch_artist = True,showfliers=False)
    # showfliers=False permits to have the boxplot without outliers
    # documentation about boxplot
    # ... present boxplot over the period for each models
    # this functions returns varius parameters of the boxplot in the dict_boxplot. This funcitons also returns an image of it
    # here, numpy_array is a vector. But can also include array with several columns. Each columns will have a boxplot
    # 'notch' is true to enhance part where the median is
    # 'whis' is the percentile value for the whiskers, every data out of the range indicted by those 2 floats are represented as points
    # 'widths' determine width of the boxes
    # 'patch_artist' colors the boxplots
    # 'labels' gives a name to every column included in the data part

    # prepare color depending on content of labels
    for i in np.arange(0,len(text_label)):
        if ('obs' in text_label[i]) or ('Obs' in text_label[i]):
            colors.append('lightpink')
        else:
            colors.append('lightblue')
    # fill colors with vector just prepared
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xticks(rotation=90) # to have the labels vertical
    # label axes and figure
    plt.xlabel(y_label_text)
    plt.ylabel(full_name_climate_var)
    plt.title('Boxplot presenting ditribution of '+climate_var+' data of the\n'+source_obs+' observation data vs '+source_modeled+' modeled data')
    # add legend
    ax.legend([bp['boxes'][0],bp['boxes'][1]], ['Observed', 'Modeled'])
    #title_png = climate_var+'_'+source_obs+'_'+source_modeled+'.png'
    #plt.savefig(os.path.join(path_figure,'figures','Boxplots',title_png),format ='png') # savefig or save text must be before plt.show. for savefig, format should be explicity written

    plt.show()


# In[2]:


path_file_SIPA = r'C:\Users\CLMRX\COWI\A248363 - Climate analysis - Documents\General\CRVA_tool\Master_thesis\Project\3 - Implementation\1 - Data\1-BC\DirecltyfromMoz\Dados_e_grafico_P_812.xls'
obs_SIPA=pd.read_excel(path_file_SIPA)
obs_SIPA


# In[ ]:




