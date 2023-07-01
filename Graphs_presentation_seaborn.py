#!/usr/bin/env python
# coding: utf-8

# This notebook was meant to contain the graphs to be used for the presentation of the 3 July 2023, with people from the sub project from Gorongosa.
# 
# Plots to do, with only temperature and precipitation for the moment:
# 
# >Compare historic observed v historic model​
# 
# >Compare WB v downscaled/bc data​
# 
# >Create overview of trends (monthly, annual evolution)​
# 
# >Compare historic model v historic projection​ (some analysis require separation of SSPs / model uncertainty)
# 
# >Taste of indicators:​
# 
# >SSP3: days above 40C (over time)​
# 
# >100yr precipitation​

# In[1]:


emplacement_of_int = 'Gorongosa'


# In[2]:


# packages
import pandas as pd
import numpy as np
import os
import os.path
import seaborn as sns
import matplotlib
import geopy.distance
from matplotlib import pyplot as plt


# In[3]:


# functions

def import_treat_modeled_NEX_GDDP_CMIP6(climate_var, unit):
    # import data
    
    path_NEX_GDDP_CMIP6_EmplacementStation=os.path.join(r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file',climate_var,climate_var+'_'+unit+'_day_1980-2060',climate_var+'_1980-2060_projectsMoz_wrong_emplacement.csv')
    
    data_NEX_GDDP_CMIP6_EmplacementStation = pd.read_csv(path_NEX_GDDP_CMIP6_EmplacementStation)
    
    data_NEX_GDDP_CMIP6_EmplacementStation = add_year_month_season(data_NEX_GDDP_CMIP6_EmplacementStation,'Date')
    
    return data_NEX_GDDP_CMIP6_EmplacementStation


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


# In[105]:


# this function is meant to find which meteo stations are the closest to the projects of interest
# find which stations are of interest, which one are the closest to the coordinates of the projects
def find_closest_meteo_station_to_projects(data_obs_NOAA,name_projects,lat_projects,lon_projects):
    # save in a dataframe name, latitudes and longitudes informations for each station
    df_station_NOAA=data_obs_NOAA.loc[:, ["NAME", "LATITUDE","LONGITUDE"]]
    df_station_NOAA.drop_duplicates(inplace = True) # drop duplicates to only have name of the towns and latitudes and longitudes
    df_station_NOAA.reset_index(drop=True,inplace = True)  # drop = true avoids to keep the former index
    # inplace = True modifies the original dataframe
    
    name_closest_station_to_project = [] # create an empty list to contain the name of the closest station to each project
    index_closest_station_to_project = []
    for (i,name_project) in zip(np.arange(0,len(name_projects)),name_projects):
        # calculate difference between the different coordinates
        df_station_NOAA['Diff latitude project '+str(i)] = abs(abs(df_station_NOAA['LATITUDE']) - abs(lat_projects[i]))
        df_station_NOAA['Diff longitude project '+str(i)] = abs(abs(df_station_NOAA['LONGITUDE']) - abs(lon_projects[i]))
        df_station_NOAA['Diff coordinates project '+str(i)] = df_station_NOAA['Diff latitude project '+str(i)]+df_station_NOAA['Diff longitude project '+str(i)]
        # register the name of the stations that are the closest to the projects and the index in df_station_NOAA corresponding to those closest stations
        name_closest_station = df_station_NOAA['NAME'].iloc[np.where(df_station_NOAA['Diff coordinates project '+str(i)]==min(df_station_NOAA['Diff coordinates project '+str(i)]))[0][0]]
        name_closest_station_to_project.append(name_closest_station)
        index_closest_station_to_project.append(np.where(df_station_NOAA['Diff coordinates project '+str(i)]==min(df_station_NOAA['Diff coordinates project '+str(i)]))[0][0])
        print('The closest meteorological station to the project '+name_project+' is the one located in '+name_closest_station)

        #coords_1 = (df_station_NOAA['LATITUDE'][index_closest_station_to_project], df_station_NOAA['LONGITUDE'][index_closest_station_to_project])
        #coords_2 = (lat_projects[i], lon_projects[i])
        #str_dist = str(geopy.distance.geodesic(coords_1, coords_2).km)
        #print('The distance between the station '+ df_station_NOAA['NAME'][index_closest_station_to_project] +' and the emplacement of interest '+name_projects[i]+' is '+str_dist+ ' km.')

    # take off the duplicates from the list of name of station which are the closest to our projects and the indexes in the dataframe of those corresponding stations
    name_closest_station_to_project_without_duplicates=list(set(name_closest_station_to_project))
    index_closest_station_to_project_without_duplicates=list(set(index_closest_station_to_project))
    print('\n')
    print('The coordinates for the meteorological stations which are the closest to the project of interest are :')
    print('\n')
    for k in np.arange(len(index_closest_station_to_project_without_duplicates)):
        print('Name '+df_station_NOAA['NAME'][index_closest_station_to_project_without_duplicates[k]])
        print('Longitude '+str(df_station_NOAA['LONGITUDE'][index_closest_station_to_project_without_duplicates[k]]))
        print('Latitude '+str(df_station_NOAA['LATITUDE'][index_closest_station_to_project_without_duplicates[k]]))
        print('\n')


# In[106]:


coords_1 = (2,3)
coords_2 = (4,5)
str(geopy.distance.geodesic(coords_1, coords_2).km)


# # Import data

# ## Observations data
# precipitation: NOAA, gorongosa
# 
# temperature: NOAA

# ### Precipitation: NOAA

# In[8]:


path = r'C:\Users\CLMRX\COWI\A248363 - Climate analysis - Documents\General\CRVA_tool\Master_thesis\Project\3 - Implementation\1 - Data\1-BC\NOAA-ClimateDataOnline\3370204.csv'
daily_sum_obs_from_NOAA = pd.read_csv(path)


# In[9]:


# find closest station to the station of interest
name_projects_data = np.array(['Gorongosa'])
name_projects = pd.Series(name_projects_data)

lon_projects_data = np.array([34.07824286310398])
lon_projects = pd.Series(lon_projects_data)

lat_projects_data = np.array([-18.68063728746643])
lat_projects = pd.Series(lat_projects_data)

#find_closest_meteo_station_to_projects(daily_sum_obs_from_NOAA,name_projects,lat_projects,lon_projects)


# In[10]:


daily_sum_obs_from_NOAA_gorongosa = daily_sum_obs_from_NOAA[daily_sum_obs_from_NOAA['NAME']=='CHIMOIO, MZ']
daily_sum_obs_from_NOAA_gorongosa = add_year_month_season(daily_sum_obs_from_NOAA_gorongosa,'DATE')
daily_sum_obs_from_NOAA_gorongosa


# In[11]:


daily_sum_obs_from_NOAA_gorongosa


# ### Precipitation : observation from Gorongosa
# 
# Observation precipitation data given by André Görgens (Cosnultant, Water resources Management, Zutari) in an email, on the 20th of June 2023.

# In[12]:


path = r'C:\Users\CLMRX\COWI\A248363 - Climate analysis - Documents\General\CRVA_tool\Master_thesis\Project\3 - Implementation\1 - Data\1-BC\DirecltyfromMoz\Precipitation_Gorongosa_reformat.csv'
pr_obs_gorongosa_from_gorongosa = pd.read_csv(path)


# In[13]:


daily_sum_obs_gorongosa_from_NOAA = daily_sum_obs_from_NOAA[daily_sum_obs_from_NOAA['NAME']=='CHIMOIO, MZ']


# ## Modeled data
# precipitation :  WB, NEX GDDP CMIP6, (copernicus)
# 
# temperature : WB, NEX GDDP CMIP6, (Copernicus)

# ### Precipitation NEX GDDP CMIP6

# In[14]:


# at the emplacement of our sub projects
path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file\pr\pr_mm_per_day_day_1980-2060\pr_1980-2060_projectsMoz_wrong_emplacement.csv'
pr_wrong_modeled_NEXGDDPCMIP6 = import_treat_modeled_NEX_GDDP_CMIP6('pr', 'mm_per_day')
pr_wrong_historic_modeled_NEXGDDPCMIP6 = pr_wrong_modeled_NEXGDDPCMIP6[pr_wrong_modeled_NEXGDDPCMIP6['Experiment']=='historical']
pr_wrong_future_modeled_NEXGDDPCMIP6 = pr_wrong_modeled_NEXGDDPCMIP6[pr_wrong_modeled_NEXGDDPCMIP6['Experiment']!='historical']
pr_futur_model_NEXGDDPCMIP6_gorongosa=pr_wrong_future_modeled_NEXGDDPCMIP6[pr_wrong_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB']


# In[15]:


pr_wrong_future_modeled_NEXGDDPCMIP6[pr_wrong_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB']


# In[16]:


# to compare with NOAA observation data
path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file\pr\pr_mm_per_day_day_1970-2014_CLosest_to_NOAA\NEXGDDPCMIP6_at_same_emplacement_as_NOAA_stationPembaChimoioBeira_pr_1970-2014_projectsMoz.csv'
pr_model_NEX_GDDPCMIP6_to_comp_NOAA = pd.read_csv(path)
pr_model_NEX_GDDPCMIP6_to_comp_NOAA_gorongosa = pr_model_NEX_GDDPCMIP6_to_comp_NOAA[pr_model_NEX_GDDPCMIP6_to_comp_NOAA['Name station']=='CHIMOIO, MZ']


# ### Temperature NEX-GDDP-CMIP6

# In[17]:


# at the emplacement of our sub projects
path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file\tas\tas_Celsius_day_1950-2100\tas_1950-2100_projectsMoz_wrong_emplacement.csv'
tas_wrong_modeled_NEXGDDPCMIP6 = import_treat_modeled_NEX_GDDP_CMIP6('tas', 'Celsius')
tas_wrong_historic_modeled_NEXGDDPCMIP6 = tas_wrong_modeled_NEXGDDPCMIP6[tas_wrong_modeled_NEXGDDPCMIP6['Experiment']=='historical']
tas_wrong_future_modeled_NEXGDDPCMIP6 = tas_wrong_modeled_NEXGDDPCMIP6[tas_wrong_modeled_NEXGDDPCMIP6['Experiment']!='historical']


# In[18]:


# to compare with NOAA observation data
path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file\tas\tas_Celsius_day_1970-2014_closest_point_to_NOAA\NEXGDDPCMIP6_at_same_emplacement_as_NOAA_stationPembaChimoioBeira_tas_1970-2014_projectsMoz.csv'
tas_model_NEX_GDDPCMIP6_to_comp_NOAA = pd.read_csv(path)
tas_model_NEX_GDDPCMIP6_to_comp_NOAA_gorongosa = tas_model_NEX_GDDPCMIP6_to_comp_NOAA[tas_model_NEX_GDDPCMIP6_to_comp_NOAA['Name station']=='CHIMOIO, MZ']
tas_model_NEX_GDDPCMIP6_to_comp_NOAA_gorongosa


# # Compare historic observed vs historic model

# In[19]:


# deja fait, reproduire ce qui a ete fait sans seaborn


# In[ ]:





# # Create overview of trends (monthly, annual evolution)

# In[54]:


# data_1 : first set of data to be used, should only contains the location of interest
# source_1 : source of the first set of data
# data_2 : second set of dat to be used, should only contains the location of interest
# source_2 : source of the second set of data

def trends_month(climate_var,data_1,source_1,data_2,source_2,stats,location,temporal_resolution='Month',start_year_line=1970,stop_year_line=2014,start_year_boxplot=2015,stop_year_boxplot=2060):
    
    (climate_var_longName,unit)= infos_str(climate_var,temporal_resolution)
    
    # define the new common name, that will be used as y_axis for boxplots and line
    new_name_col = temporal_resolution+'ly '+climate_var_longName+' '+unit
    
    if 'NEX-GDDP-CMIP6' in source_1:
        if (start_year_boxplot!=2014) or (stop_year_boxplot!=2060):
            data_1=data_1[data_1['Year'].between(start_year_boxplot,stop_year_boxplot)]
        data_boxplot=prepare_NEX_GDDP_CMIP6(data_1,climate_var_longName,stats,temporal_resolution,new_name_col)
        source_boxplot=source_1
    if 'NEX-GDDP-CMIP6' in source_2:
        if (start_year_boxplot!=2014) or (stop_year_boxplot!=2060):
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


# In[21]:


# select years because impossible to read
def trends_year(climate_var,data_1,source_1,stats,location,start_year,stop_year,temporal_resolution='Year'):
    (climate_var_longName,unit)= infos_str(climate_var,temporal_resolution)
    
    # define the new common name, that will be used as y_axis for boxplots and line
    new_name_col = temporal_resolution+'ly '+climate_var_longName+' '+unit
    
    data_boxplot=prepare_NEX_GDDP_CMIP6(data_1,climate_var_longName,stats,temporal_resolution,new_name_col)
    if stats =='Sum':
        stats = ''
    
    if (stop_year-start_year+1)>10:
        for i in np.arange(0,round(((stop_year-start_year+1)/10))):
            df_filter=data_boxplot[data_boxplot['Year'].between(start_year+i*10,start_year+i*10+10)]
            title_plot = stats+' '+ climate_var_longName+' '+unit+' between '+str(start_year)+' and '+str(stop_year)+' at '+location
            boxplots(df_filter,temporal_resolution,new_name_col,title_plot)
    else:
        df_filter=data_boxplot[data_boxplot['Year'].between(start_year,stop_year)]
        title_plot = stats+' '+climate_var_longName+' '+unit+' between '+str(start_year)+' and '+str(stop_year)+' at '+location
        boxplots(df_filter,temporal_resolution,new_name_col,title_plot)


# In[22]:


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
    return data_NEXGDDPCMIP6


# In[23]:


def prepare_NOAA(df_NOAA,title_column,temporal_resolution,new_name_col):
    df_NOAA = df_NOAA.reset_index()
    df = df_NOAA[[title_column,temporal_resolution]].groupby(temporal_resolution).mean().rename(columns={title_column:new_name_col}).reset_index()
    return df


# In[24]:


def infos_str(climate_var,temporal_resolution):
    if climate_var=='pr':
        climate_var_longName = 'precipitation'
        unit='mm/'+temporal_resolution[0].lower()+temporal_resolution[1:len(temporal_resolution)]
    if 'tas' in climate_var:
        unit=u'\N{DEGREE SIGN}C'
        climate_var_longName = 'temperature'
    if climate_var=='tasmax':
        climate_var_longName = 'maximum '+climate_var_longName
    if climate_var=='tasmin':
        climate_var_longName = 'minimum '+climate_var_longName
    return climate_var_longName,unit


# In[25]:


def title_column_NOAA_obs(source,climate_var):
    if source == 'NOAA':
        if climate_var=='pr':
            title_column='PRCP'
        if climate_var=='tas':
            title_column='TAVG'
        if climate_var=='tasmax':
            title_column='TMAX'
        if climate_var=='tasmin':
            title_column='TMIN'
        return title_column


# In[29]:


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
    plt.show()
    
# data_boxplot : dataframr that will be used to do the boxplots
# x_axis : Name of the column that wil be used for the x_axis
# y_axis : Name of the column that wil be used for the y_axis
# ----> x_axis and y_axis are both a str
# title_plot : title for this plot. Should be defined in the function before
# categories : default parameters, will be used for the hue of the boxplot. The hue is a third dimension along a depth axis, 
#              where different levels are plotted with different colors

#stats+' monthly precipitation mm/month between '+start_year+' and '+stop_year+'\n with '+source_obs+' observed data and '+source_model+' modeled data, at '+location

def boxplots(data_boxplot,x_axis,y_axis,title_plot,categories='Experiment'):
    fig,ax=plt.subplots()
    sns.boxplot(data=data_boxplot, x=x_axis, y=y_axis, hue=categories,ax=ax)
    
    # display the legend
    handles, labels=ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.1, 0.5),title='Legend')
    ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
    plt.title(title_plot)
    plt.show()


# In[66]:


def plot_spaghetti(data,x_axis,y_axis):
    fig,ax=plt.subplots()
    sns.lineplot(data=data,x=x_axis, y=y_axis,ax=ax)
        # display the legend
    handles, labels=ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.1, 0.5),title='Legend')
    #ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
    #plt.title(title_plot)
    plt.show()


# In[27]:


path = r'C:\Users\CLMRX\COWI\A248363 - Climate analysis - Documents\General\CRVA_tool\Master_thesis\Project\3 - Implementation\1 - Data\1-BC\NOAA-ClimateDataOnline\3370204.csv'
daily_sum_obs_from_NOAA = pd.read_csv(path)
daily_sum_obs_from_NOAA_gorongosa = daily_sum_obs_from_NOAA[daily_sum_obs_from_NOAA['NAME']=='CHIMOIO, MZ']
daily_sum_obs_from_NOAA_gorongosa = add_year_month_season(daily_sum_obs_from_NOAA_gorongosa,'DATE')
daily_sum_obs_from_NOAA_gorongosa


# In[57]:


trends_month('pr',pr_futur_model_NEXGDDPCMIP6_gorongosa,'NEX-GDDP-CMIP6',daily_sum_obs_from_NOAA_gorongosa,'NOAA','Median','Gorongosa')


# In[39]:


trends_year('pr',pr_futur_model_NEXGDDPCMIP6_gorongosa,'NEX-GDDP-CMIP6','Average','Gorongosa',2020,2040)


# In[62]:


trends_month('tas',tas_wrong_future_modeled_NEXGDDPCMIP6[tas_wrong_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB'],'NEX-GDDP-CMIP6',daily_sum_obs_from_NOAA_gorongosa,'NOAA','Average','Gorongosa')


# In[61]:


trends_year('tas',tas_wrong_future_modeled_NEXGDDPCMIP6[tas_wrong_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB'],'NEX-GDDP-CMIP6','Average','Gorongosa',2020,2040)


# In[67]:


plot_spaghetti(tas_wrong_future_modeled_NEXGDDPCMIP6[tas_wrong_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB'],'Date','Daily Near-Surface Air Temperature °C')


# In[96]:


tas_wrong_future_modeled_NEXGDDPCMIP6_gorongosa=tas_wrong_future_modeled_NEXGDDPCMIP6[tas_wrong_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB'].drop(['Name project','Latitude','Longitude'],axis=1).groupby(['Experiment','Model','Year'])[['Daily Near-Surface Air Temperature °C']].mean().reset_index()


# In[97]:


tas_wrong_future_modeled_NEXGDDPCMIP6_gorongosa


# In[99]:


g = sns.FacetGrid(tas_wrong_future_modeled_NEXGDDPCMIP6_gorongosa, col="Experiment", height=4, aspect=.5)
g.map(sns.lineplot, "Year",'Daily Near-Surface Air Temperature °C')


# In[101]:


tas_wrong_future_modeled_NEXGDDPCMIP6_gorongosa_overMandS=tas_wrong_future_modeled_NEXGDDPCMIP6[tas_wrong_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB'].drop(['Name project','Latitude','Longitude'],axis=1).groupby(['Year'])[['Daily Near-Surface Air Temperature °C']].mean().reset_index()


# In[104]:


sns.lineplot(tas_wrong_future_modeled_NEXGDDPCMIP6,x='Year',y='Daily Near-Surface Air Temperature °C')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




