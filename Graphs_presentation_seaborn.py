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

start_year_hist = 1980
stop_year_hist = 2014

tuple_error_bar = ('pi',80) # default one is confidence interval of 95%


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
    
    path_NEX_GDDP_CMIP6_EmplacementStation=os.path.join(r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file',climate_var,climate_var+'_'+unit+'_day_1950-2100',climate_var+'_1950-2100_projectsMoz.csv')
    
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


# In[6]:


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


# In[7]:


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


# In[138]:


daily_sum_obs_from_NOAA_gorongosa = daily_sum_obs_from_NOAA[daily_sum_obs_from_NOAA['NAME']=='CHIMOIO, MZ']
daily_sum_obs_from_NOAA_gorongosa = add_year_month_season(daily_sum_obs_from_NOAA_gorongosa,'DATE')
daily_sum_obs_from_NOAA_gorongosa


# In[11]:


daily_sum_obs_from_NOAA_gorongosa


# How much precipitation data are we missing in those data coming from NOAA ?

# In[12]:


def countna(data):
    return data.isna().sum()


# In[13]:


na_values_daily_sum_obs_from_NOAA_gorongosa_PRCP=daily_sum_obs_from_NOAA_gorongosa[['PRCP','Year']].groupby(['Year']).agg(countna).reset_index()


# In[14]:


len(daily_sum_obs_from_NOAA_gorongosa['PRCP'])


# In[15]:


print('The missing value represent '+str((daily_sum_obs_from_NOAA_gorongosa['PRCP'].isna().sum()/len(daily_sum_obs_from_NOAA_gorongosa['PRCP']))*100)+' % of the total')


# ### Precipitation : observation from Gorongosa
# 
# Observation precipitation data given by André Görgens (Cosnultant, Water resources Management, Zutari) in an email, on the 20th of June 2023.

# In[16]:


path = r'C:\Users\CLMRX\COWI\A248363 - Climate analysis - Documents\General\CRVA_tool\Master_thesis\Project\3 - Implementation\1 - Data\1-BC\DirecltyfromMoz\Precipitation_Gorongosa_reformat.csv'
pr_obs_gorongosa_from_gorongosa = pd.read_csv(path)


# In[17]:


pr_obs_gorongosa_from_gorongosa=add_year_month_season(pr_obs_gorongosa_from_gorongosa,'time')


# In[18]:


pr_obs_gorongosa_from_gorongosa['pr'].iloc[0]


# In[19]:


pr_obs_gorongosa_from_gorongosa['pr'][pr_obs_gorongosa_from_gorongosa['pr']=='s/i'] = pr_obs_gorongosa_from_gorongosa['pr'].iloc[0]


# In[20]:


pr_obs_gorongosa_from_gorongosa['pr'] = pr_obs_gorongosa_from_gorongosa['pr'].astype(float)


# In[21]:


pr_obs_gorongosa_from_gorongosa['pr'].iloc[14971]


# How much precipitation data are we missing in those data coming from the measuring station ?

# In[22]:


pr_obs_gorongosa_from_gorongosa['pr'].isna().sum()


# In[23]:


len(pr_obs_gorongosa_from_gorongosa['pr'])


# In[24]:


print('The missing value represent '+str((pr_obs_gorongosa_from_gorongosa['pr'].isna().sum()/len(pr_obs_gorongosa_from_gorongosa['pr']))*100)+' % of the total')


# In[25]:


na_values_pr_obs_gorongosa_from_gorongosa=pr_obs_gorongosa_from_gorongosa.groupby(['Year']).agg(countna).reset_index()


# ### Temperature : NOAA

# How much average temperature data are we missing in those data coming from the measuring station ?

# In[26]:


daily_sum_obs_from_NOAA_gorongosa['TAVG'].isna().sum()


# In[27]:


len(daily_sum_obs_from_NOAA_gorongosa['TAVG'])


# In[28]:


print('The missing value represent '+str((daily_sum_obs_from_NOAA_gorongosa['TAVG'].isna().sum()/len(daily_sum_obs_from_NOAA_gorongosa['TAVG']))*100)+' % of the total')


# In[29]:


na_values_daily_sum_obs_from_NOAA_gorongosa_TAVG=daily_sum_obs_from_NOAA_gorongosa[['TAVG','Year']].groupby(['Year']).agg(countna).reset_index()


# How much maximum temperature data are we missing in those data coming from the measuring station ?

# In[30]:


daily_sum_obs_from_NOAA_gorongosa['TMAX'].isna().sum()


# In[31]:


len(daily_sum_obs_from_NOAA_gorongosa['TMAX'])


# In[32]:


print('The missing value represent '+str((daily_sum_obs_from_NOAA_gorongosa['TMAX'].isna().sum()/len(daily_sum_obs_from_NOAA_gorongosa['TAVG']))*100)+' % of the total')


# In[33]:


na_values_daily_sum_obs_from_NOAA_gorongosa_TMAX=daily_sum_obs_from_NOAA_gorongosa[['TMAX','Year']].groupby(['Year']).agg(countna).reset_index()


# How much minimum temperature data are we missing in those data coming from the measuring station ?

# In[34]:


daily_sum_obs_from_NOAA_gorongosa['TMIN'].isna().sum()


# In[35]:


len(daily_sum_obs_from_NOAA_gorongosa['TMIN'])


# In[36]:


print('The missing value represent '+str((daily_sum_obs_from_NOAA_gorongosa['TMIN'].isna().sum()/len(daily_sum_obs_from_NOAA_gorongosa['TMIN']))*100)+' % of the total')


# In[37]:


na_values_daily_sum_obs_from_NOAA_gorongosa_TMIN=daily_sum_obs_from_NOAA_gorongosa[['TMIN','Year']].groupby(['Year']).agg(countna).reset_index()


# ## Modeled data
# precipitation :  WB, NEX GDDP CMIP6, (copernicus)
# 
# temperature : WB, NEX GDDP CMIP6, (Copernicus)

# ### Precipitation NEX GDDP CMIP6

# In[38]:


# at the emplacement of our sub projects
path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file\pr\pr_mm_per_day_day_1950-2100\pr_1950-2100_projectsMoz.csv'
pr_modeled_NEXGDDPCMIP6 = import_treat_modeled_NEX_GDDP_CMIP6('pr', 'mm_per_day')
pr_historic_modeled_NEXGDDPCMIP6 = pr_modeled_NEXGDDPCMIP6[pr_modeled_NEXGDDPCMIP6['Experiment']=='historical']
pr_future_modeled_NEXGDDPCMIP6 = pr_modeled_NEXGDDPCMIP6[pr_modeled_NEXGDDPCMIP6['Experiment']!='historical']
pr_historic_modeled_NEXGDDPCMIP6_gorongosa = pr_historic_modeled_NEXGDDPCMIP6[pr_historic_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB']
pr_futur_model_NEXGDDPCMIP6_gorongosa=pr_future_modeled_NEXGDDPCMIP6[pr_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB']


# In[111]:


pr_modeled_NEXGDDPCMIP6_gorongosa = pr_modeled_NEXGDDPCMIP6[pr_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB']


# In[183]:


list(set(pr_modeled_NEXGDDPCMIP6['Name project']))


# In[184]:


pr_modeled_NEXGDDPCMIP6_mutua = pr_modeled_NEXGDDPCMIP6[pr_modeled_NEXGDDPCMIP6['Name project']=='WTP_Mutua_EIB']


# In[39]:


# to compare with NOAA observation data
path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file\pr\pr_mm_per_day_day_1970-2014_CLosest_to_NOAA\NEXGDDPCMIP6_at_same_emplacement_as_NOAA_stationPembaChimoioBeira_pr_1970-2014_projectsMoz.csv'
pr_model_NEX_GDDPCMIP6_to_comp_NOAA = pd.read_csv(path)
pr_model_NEX_GDDPCMIP6_to_comp_NOAA_gorongosa = pr_model_NEX_GDDPCMIP6_to_comp_NOAA[pr_model_NEX_GDDPCMIP6_to_comp_NOAA['Name station']=='CHIMOIO, MZ']


# ### Temperature NEX-GDDP-CMIP6

# In[118]:


# at the emplacement of our sub projects
path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file\tas\tas_Celsius_day_1950-2100\tas_1950-2100_projectsMoz.csv'
tas_modeled_NEXGDDPCMIP6 = import_treat_modeled_NEX_GDDP_CMIP6('tas', 'Celsius')
tas_modeled_NEXGDDPCMIP6_gorongosa = tas_modeled_NEXGDDPCMIP6[tas_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB']
tas_historic_modeled_NEXGDDPCMIP6 = tas_modeled_NEXGDDPCMIP6[tas_modeled_NEXGDDPCMIP6['Experiment']=='historical']
tas_future_modeled_NEXGDDPCMIP6 = tas_modeled_NEXGDDPCMIP6[tas_modeled_NEXGDDPCMIP6['Experiment']!='historical']


# In[41]:


# to compare with NOAA observation data
path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file\tas\tas_Celsius_day_1970-2014_closest_point_to_NOAA\NEXGDDPCMIP6_at_same_emplacement_as_NOAA_stationPembaChimoioBeira_tas_1970-2014_projectsMoz.csv'
tas_model_NEX_GDDPCMIP6_to_comp_NOAA = pd.read_csv(path)
tas_model_NEX_GDDPCMIP6_to_comp_NOAA_gorongosa = tas_model_NEX_GDDPCMIP6_to_comp_NOAA[tas_model_NEX_GDDPCMIP6_to_comp_NOAA['Name station']=='CHIMOIO, MZ']
tas_model_NEX_GDDPCMIP6_to_comp_NOAA_gorongosa


# ### Maximum temperature NEX-GDDPCMIP6

# In[121]:


# at the emplacement of our sub projects
path = r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6-AllMoz\csv_file\tasmax\tasmax_Celsius_day_1950-2100\tasmax_1950-2100_projectsMoz.csv'
tasmax_modeled_NEXGDDPCMIP6 = import_treat_modeled_NEX_GDDP_CMIP6('tasmax', 'Celsius')
tasmax_modeled_NEXGDDPCMIP6_gorongosa = tasmax_modeled_NEXGDDPCMIP6[tasmax_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB']
tasmax_historic_modeled_NEXGDDPCMIP6 = tasmax_modeled_NEXGDDPCMIP6[tasmax_modeled_NEXGDDPCMIP6['Experiment']=='historical']
tasmax_future_modeled_NEXGDDPCMIP6 = tasmax_modeled_NEXGDDPCMIP6[tasmax_modeled_NEXGDDPCMIP6['Experiment']!='historical']


# ### Minimum temperature NEX-GDDPCMIP6

# # Compare historic observed vs historic model

# ## Precipitation

# In[42]:


pr_historic_modeled_NEXGDDPCMIP6_gorongosa=pr_historic_modeled_NEXGDDPCMIP6_gorongosa[pr_historic_modeled_NEXGDDPCMIP6_gorongosa['Year'].between(start_year_hist,stop_year_hist)]


# In[43]:


pr_obs_gorongosa_from_gorongosa = pr_obs_gorongosa_from_gorongosa[pr_obs_gorongosa_from_gorongosa['Year'].between(start_year_hist,stop_year_hist)]


# In[53]:


pr_obs_gorongosa_from_gorongosa


# In[62]:


pr_historic_modeled_NEXGDDPCMIP6_gorongosa


# In[63]:


pr_historic_modeled_NEXGDDPCMIP6_gorongosa.groupby(['Experiment','Model','Year'])[['Mean of the daily precipitation rate mm_per_day']].mean()


# In[180]:


fig,ax=plt.subplots()
plt.tight_layout() # Adjust the padding between and around subplots.

sns.lineplot(data=pr_historic_modeled_NEXGDDPCMIP6_gorongosa.groupby(['Experiment','Model','Year'])[['Mean of the daily precipitation rate mm_per_day']].mean()*365.25,x='Year', y='Mean of the daily precipitation rate mm_per_day',hue='Model',errorbar=tuple_error_bar,ax=ax)
sns.lineplot(data=pr_obs_gorongosa_from_gorongosa.groupby('Year')[['Mean of the daily precipitation rate mm_per_day']].mean()*365.25,x='Year', y='Mean of the daily precipitation rate mm_per_day',color='black',label='Observation from Gorongosa',errorbar=tuple_error_bar,ax=ax)

# display the legend
handles, labels=ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.38, 0.88),title='Legend')
ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
plt.ylabel('Yearly average precipitation mm_per_year')
plt.title('Modeled NEX-GDDP-CMIP6 yearly average precipitation accross time at Gorongosa,\n compared to observed yearly average temperature from gorongosa, between '+str(start_year_hist)+' and '+str(stop_year_hist))
path_figure=r'C:\Users\CLMRX\OneDrive - COWI\Documents\GitHub\CRVA_tool\outputs\figures\Comp_hist_m_o_pr.png'
plt.savefig(path_figure,format ='png') # savefig or save text must be before plt.show. for savefig, format should be explicity written
plt.show()


# In[176]:


pr_historic_modeled_NEXGDDPCMIP6_gorongosa['Mean of the daily precipitation rate mm_per_year']=pr_historic_modeled_NEXGDDPCMIP6_gorongosa[['Mean of the daily precipitation rate mm_per_day']].values*365.25


# In[179]:


fig,ax=plt.subplots()
plt.tight_layout() # Adjust the padding between and around subplots.

sns.lineplot(data=pr_historic_modeled_NEXGDDPCMIP6_gorongosa.groupby(['Experiment','Model','Year'])[['Mean of the daily precipitation rate mm_per_year']].mean(),x='Year', y='Mean of the daily precipitation rate mm_per_year',hue='Model',errorbar=tuple_error_bar,ax=ax)
data_line=daily_sum_obs_from_NOAA_gorongosa[daily_sum_obs_from_NOAA_gorongosa['Year'].between(1980,2014)]
data_line = data_line.groupby('Year')[['PRCP']].mean()*365.25
sns.lineplot(data=data_line,x='Year', y='PRCP',color='black',label='Observation from NOAA',errorbar=tuple_error_bar,ax=ax)

# display the legend
handles, labels=ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.38, 0.88),title='Legend')
ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
plt.title('Modeled NEX-GDDP-CMIP6 yearly average precipitation accross time at Gorongosa,\n compared to observed yearly average temperature from NOAA, between '+str(start_year_hist)+' and '+str(stop_year_hist))
path_figure=r'C:\Users\CLMRX\OneDrive - COWI\Documents\GitHub\CRVA_tool\outputs\figures\Comp_hist_m_o_pr.png'
plt.savefig(path_figure,format ='png') # savefig or save text must be before plt.show. for savefig, format should be explicity written
plt.show()


# In[140]:


daily_sum_obs_from_NOAA


# In[137]:


daily_sum_obs_from_NOAA


# In[77]:


fig,ax=plt.subplots()
plt.tight_layout() # Adjust the padding between and around subplots.

sns.lineplot(data=pr_historic_modeled_NEXGDDPCMIP6_gorongosa.groupby(['Experiment','Model','Year'])[['Mean of the daily precipitation rate mm_per_day']].mean(),x='Year', y='Mean of the daily precipitation rate mm_per_day',label='Variation accross models',errorbar=tuple_error_bar,ax=ax)
sns.lineplot(data=pr_obs_gorongosa_from_gorongosa.groupby('Year')[['Mean of the daily precipitation rate mm_per_day']].mean(),x='Year', y='Mean of the daily precipitation rate mm_per_day',color='black',label='Observation from Gorongosa',errorbar=tuple_error_bar,ax=ax)
ax2 = plt.twinx()
sns.lineplot(data=na_values_pr_obs_gorongosa_from_gorongosa,x='Year',y='pr',label='Missing value',ax=ax2)

# display the legend
handles, labels=ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.38, 0.88),title='Legend')
ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
plt.title('Modeled NEX-GDDP-CMIP6 yearly average precipitation accross time at Gorongosa,\n compared to observed yearly average temperature from gorongosa, between '+str(start_year_hist)+' and '+str(stop_year_hist))
path_figure=r'C:\Users\CLMRX\OneDrive - COWI\Documents\GitHub\CRVA_tool\outputs\figures\Comp_hist_m_o_pr2.png'
plt.savefig(path_figure,format ='png') # savefig or save text must be before plt.show. for savefig, format should be explicity written
plt.show()


# In[73]:


na_values_pr_obs_gorongosa_from_gorongosa


# In[75]:


sns.lineplot(data=na_values_pr_obs_gorongosa_from_gorongosa,x='Year',y='pr')


# In[78]:


pr_obs_gorongosa_from_gorongosa['Model']='Observation from Gorongosa'
pr_obs_gorongosa_from_gorongosa=pr_obs_gorongosa_from_gorongosa.rename(columns={'pr':'Mean of the daily precipitation rate mm_per_day'})
pr_obs_gorongosa_from_gorongosa


# In[79]:


df_boxplot=pd.concat([pr_obs_gorongosa_from_gorongosa,pr_historic_modeled_NEXGDDPCMIP6_gorongosa])
df_boxplot


# In[80]:


fig,ax=plt.subplots()
plt.tight_layout() # Adjust the padding between and around subplots.

cols = ['pink' if (x =='Observation from Gorongosa') else 'skyblue' for x in df_boxplot.Model.drop_duplicates().values]
sns.boxplot(data=df_boxplot,x=df_boxplot.Model, y='Mean of the daily precipitation rate mm_per_day',palette=cols,ax=ax)

# display the legend
#handles, labels=ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.3, 1),title='Legend')
#ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.title('Compare observation for precipitation average from Gorongosa,\n with modeled data by NEX-GDDP-CMIP6, between '+str(start_year_hist)+' and '+str(stop_year_hist))
path_figure=r'C:\Users\CLMRX\OneDrive - COWI\Documents\GitHub\CRVA_tool\outputs\figures\BoxplotsComp_hist_m_o_pr.png'
plt.savefig(path_figure,format ='png') # savefig or save text must be before plt.show. for savefig, format should be explicity written

plt.show()


# In[81]:


# without outliers

fig,ax=plt.subplots()
plt.tight_layout() # Adjust the padding between and around subplots.

cols = ['pink' if (x =='Observation from Gorongosa') else 'skyblue' for x in df_boxplot.Model.drop_duplicates().values]
sns.boxplot(data=df_boxplot,x=df_boxplot.Model, y='Mean of the daily precipitation rate mm_per_day', fliersize=0,palette=cols,ax=ax)

# display the legend
#handles, labels=ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.3, 1),title='Legend')
#ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.ylim((0,10))
plt.title('Compare observation for precipitation average from Gorongosa,\n with modeled data by NEX-GDDP-CMIP6, between '+str(start_year_hist)+' and '+str(stop_year_hist))
path_figure=r'C:\Users\CLMRX\OneDrive - COWI\Documents\GitHub\CRVA_tool\outputs\figures\Boxplots_without_outliers_Comp_hist_m_o_pr.png'
plt.savefig(path_figure,format ='png') # savefig or save text must be before plt.show. for savefig, format should be explicity written

plt.show()


# ## Temperature

# In[82]:


tas_historic_modeled_NEXGDDPCMIP6_gorongosa=tas_historic_modeled_NEXGDDPCMIP6[tas_historic_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB'].drop('Name project',axis=1)
tas_historic_modeled_NEXGDDPCMIP6_gorongosa=add_year_month_season(tas_historic_modeled_NEXGDDPCMIP6_gorongosa,'Date')
tas_historic_modeled_NEXGDDPCMIP6_gorongosa = tas_historic_modeled_NEXGDDPCMIP6_gorongosa[tas_historic_modeled_NEXGDDPCMIP6_gorongosa['Year'].between(start_year_hist,stop_year_hist)]


# In[83]:


daily_sum_obs_from_NOAA_gorongosa = daily_sum_obs_from_NOAA_gorongosa[daily_sum_obs_from_NOAA_gorongosa['Year'].between(start_year_hist,stop_year_hist)]


# In[84]:


# with confidence interval of 95 %
fig,ax=plt.subplots()
plt.tight_layout() # Adjust the padding between and around subplots.

sns.lineplot(data=tas_historic_modeled_NEXGDDPCMIP6_gorongosa,x='Year', y='Daily Near-Surface Air Temperature °C',hue='Model',ax=ax)
sns.lineplot(data=daily_sum_obs_from_NOAA_gorongosa,x='Year', y='TAVG',color='black',label='Observation NOAA',ax=ax)

# display the legend
handles, labels=ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.21, 0.87),title='Legend')
ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
plt.title('Modeled NEX-GDDP-CMIP6 yearly average temperature accross time,\n compared to observed yearly average temperature from NOAA, between '+str(start_year_hist)+' and '+str(stop_year_hist))
path_figure=r'C:\Users\CLMRX\OneDrive - COWI\Documents\GitHub\CRVA_tool\outputs\figures\Graphs_ci95_Comp_hist_m_o_tas.png'
plt.savefig(path_figure,format ='png') # savefig or save text must be before plt.show. for savefig, format should be explicity written


plt.show()


# In[135]:


fig,ax=plt.subplots()
plt.tight_layout() # Adjust the padding between and around subplots.
data1 = tas_historic_modeled_NEXGDDPCMIP6_gorongosa.groupby(['Experiment','Model','Year'])[['Daily Near-Surface Air Temperature °C']].mean().reset_index()
sns.lineplot(data=data1,x='Year', y='Daily Near-Surface Air Temperature °C',hue='Model',errorbar=tuple_error_bar,ax=ax)
sns.lineplot(data=daily_sum_obs_from_NOAA_gorongosa[['Year','TAVG']].groupby('Year')[['TAVG']].mean(),x='Year', y='TAVG',color='black',label='Observation NOAA',errorbar=tuple_error_bar,ax=ax)

# display the legend
handles, labels=ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.3, 0.87),title='Legend')
ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
plt.title('Modeled NEX-GDDP-CMIP6 yearly average temperature accross time and scenarios,\n compared to observed yearly average temperature from NOAA, between '+str(start_year_hist)+' and '+str(stop_year_hist))
path_figure=r'C:\Users\CLMRX\OneDrive - COWI\Documents\GitHub\CRVA_tool\outputs\figures\Graphs_pi80_Comp_hist_m_o_tas.png'
plt.savefig(path_figure,format ='png') # savefig or save text must be before plt.show. for savefig, format should be explicity written


plt.show()


# In[133]:


daily_sum_obs_from_NOAA_gorongosa


# In[130]:


tas_historic_modeled_NEXGDDPCMIP6_gorongosa.groupby(['Experiment','Model','Year'])[['Daily Near-Surface Air Temperature °C']].mean().reset_index()


# In[131]:


tas_historic_modeled_NEXGDDPCMIP6_gorongosa.Model.unique()


# In[86]:


daily_sum_obs_from_NOAA_gorongosa['Model']='Observation NOAA'
daily_sum_obs_from_NOAA_gorongosa=daily_sum_obs_from_NOAA_gorongosa.rename(columns={'TAVG':'Daily Near-Surface Air Temperature °C'})
daily_sum_obs_from_NOAA_gorongosa


# In[87]:


df_boxplot=pd.concat([daily_sum_obs_from_NOAA_gorongosa,tas_historic_modeled_NEXGDDPCMIP6_gorongosa])
df_boxplot


# In[88]:


fig,ax=plt.subplots()
plt.tight_layout() # Adjust the padding between and around subplots.
cols = ['pink' if (x =='Observation NOAA') else 'skyblue' for x in df_boxplot.Model.drop_duplicates().values]
sns.boxplot(data=df_boxplot,x=df_boxplot.Model, y='Daily Near-Surface Air Temperature °C',palette=cols,whis=[10,90],ax=ax)

# display the legend
#handles, labels=ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.3, 1),title='Legend')
#ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.title('Compare observation for temperature average from NOAA between 1970 and 2014,\n with modeled data by NEX-GDDP-CMIP6 between 1950 to 2014')

path_figure=r'C:\Users\CLMRX\OneDrive - COWI\Documents\GitHub\CRVA_tool\outputs\figures\Boxplots_whis10_90_Comp_hist_m_o_tas.png'
plt.savefig(path_figure,format ='png') # savefig or save text must be before plt.show. for savefig, format should be explicity written

plt.show()


# We can see that 'CMCC-CM2-SR5' and 'TaiESM1' do not have the same behaviour as the other models. We will take them off for the following analysis of the temperature

# In[89]:


tas_historic_modeled_NEXGDDPCMIP6_gorongosa


# In[90]:


tas_historic_modeled_NEXGDDPCMIP6_gorongosa=tas_historic_modeled_NEXGDDPCMIP6_gorongosa[tas_historic_modeled_NEXGDDPCMIP6_gorongosa['Model']!='TaiESM1']


# In[91]:


tas_historic_modeled_NEXGDDPCMIP6_gorongosa=tas_historic_modeled_NEXGDDPCMIP6_gorongosa[tas_historic_modeled_NEXGDDPCMIP6_gorongosa['Model']!='CMCC-CM2-SR5']


# In[92]:


tas_historic_modeled_NEXGDDPCMIP6_gorongosa


# # Create overview of trends (monthly, annual evolution)

# In[167]:


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


# In[168]:


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


# In[169]:


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


# In[174]:


def prepare_NOAA(df_NOAA,title_column,temporal_resolution,new_name_col):
    df_NOAA = df_NOAA.reset_index()
    df = df_NOAA[[title_column,temporal_resolution]].groupby(temporal_resolution).mean().rename(columns={title_column:new_name_col}).reset_index()
    
    print('title_column '+title_column)
    print('temporal_resolution '+temporal_resolution)
    
    
    if 'PR' in title_column and temporal_resolution=='Month':
        print('pr and month, multiplication by 30')
        df[new_name_col] = df[[new_name_col]].values*30
    
    return df


# In[126]:


def infos_str(climate_var,temporal_resolution):
    if climate_var=='pr':
        climate_var_longName = 'precipitation'
        unit='mm/'+temporal_resolution[0].lower()+temporal_resolution[1:len(temporal_resolution)]
    if 'tas' in climate_var:
        unit=u'\N{DEGREE SIGN}C'
        climate_var_longName = 'temperature'
    if climate_var=='tasmax':
        climate_var_longName = 'Daily Maximum Near-Surface Air Temperature '
    if climate_var=='tasmin':
        climate_var_longName = 'minimum '+climate_var_longName
    return climate_var_longName,unit


# In[98]:


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


# In[99]:


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
    plt.tight_layout() # Adjust the padding between and around subplots.
    sns.boxplot(data=data_boxplot, x=x_axis, y=y_axis, hue=categories,whis=[10,90],ax=ax)
    
    # display the legend
    handles, labels=ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.1, 0.5),title='Legend')
    ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
    plt.title(title_plot)
    plt.show()


# In[100]:


def plot_spaghetti(data,x_axis,y_axis,title_plot,have_legend):
    fig,ax=plt.subplots()
    sns.lineplot(data=data,x=x_axis, y=y_axis,ax=ax)
        # display the legend
    if have_legend == 'yes':
        handles, labels=ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.1, 0.5),title='Legend')
    #ax.get_legend().remove() # this line permits to have a common legend for the boxplots and the line
    plt.title(title_plot)
    plt.show()


# In[101]:


path = r'C:\Users\CLMRX\COWI\A248363 - Climate analysis - Documents\General\CRVA_tool\Master_thesis\Project\3 - Implementation\1 - Data\1-BC\NOAA-ClimateDataOnline\3370204.csv'
daily_sum_obs_from_NOAA = pd.read_csv(path)
daily_sum_obs_from_NOAA_gorongosa = daily_sum_obs_from_NOAA[daily_sum_obs_from_NOAA['NAME']=='CHIMOIO, MZ']
daily_sum_obs_from_NOAA_gorongosa = add_year_month_season(daily_sum_obs_from_NOAA_gorongosa,'DATE')
daily_sum_obs_from_NOAA_gorongosa = daily_sum_obs_from_NOAA_gorongosa[daily_sum_obs_from_NOAA_gorongosa['Year'].between(start_year_hist,stop_year_hist)]


# In[145]:


pr_modeled_NEXGDDPCMIP6_gorongosa


# In[175]:


trends_month('pr',pr_modeled_NEXGDDPCMIP6_gorongosa,'NEX-GDDP-CMIP6',daily_sum_obs_from_NOAA_gorongosa,'NOAA','Average','Gorongosa',start_year_line=start_year_hist,stop_year_line=stop_year_hist,start_year_boxplot=1970,stop_year_boxplot=2100)


# In[159]:


df['New col'] = df[['Monthly precipitation mm/month']].values*30
df


# In[113]:


trends_month('pr',pr_modeled_NEXGDDPCMIP6_gorongosa,'NEX-GDDP-CMIP6',daily_sum_obs_from_NOAA_gorongosa,'NOAA','Average','Gorongosa',start_year_line=start_year_hist,stop_year_line=stop_year_hist)


# In[120]:


trends_month('tas',tas_modeled_NEXGDDPCMIP6_gorongosa,'NEX-GDDP-CMIP6',daily_sum_obs_from_NOAA_gorongosa,'NOAA','Average','Gorongosa',start_year_line=start_year_hist,stop_year_line=stop_year_hist,start_year_boxplot=1970,stop_year_boxplot=2100)


# In[127]:


trends_month('tasmax',tasmax_modeled_NEXGDDPCMIP6_gorongosa,'NEX-GDDP-CMIP6',daily_sum_obs_from_NOAA_gorongosa,'NOAA','Average','Gorongosa',start_year_line=start_year_hist,stop_year_line=stop_year_hist,start_year_boxplot=1970,stop_year_boxplot=2100)


# In[123]:


tasmax_modeled_NEXGDDPCMIP6_gorongosa


# In[107]:


trends_year('tas',tas_future_modeled_NEXGDDPCMIP6[tas_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB'],'NEX-GDDP-CMIP6','Average','Gorongosa',2020,2040)


# In[108]:


plot_spaghetti(tas_future_modeled_NEXGDDPCMIP6[tas_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB'],'Date','Daily Near-Surface Air Temperature °C','title','Yes')


# In[109]:


tas_future_modeled_NEXGDDPCMIP6_gorongosa=tas_future_modeled_NEXGDDPCMIP6[tas_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB'].drop(['Name project','Latitude','Longitude'],axis=1).groupby(['Experiment','Model','Year'])[['Daily Near-Surface Air Temperature °C']].mean().reset_index()


# In[ ]:


tas_future_modeled_NEXGDDPCMIP6_gorongosa


# In[ ]:


g = sns.FacetGrid(tas_future_modeled_NEXGDDPCMIP6_gorongosa, col="Experiment", height=4, aspect=.5)
g.map(sns.lineplot, "Year",'Daily Near-Surface Air Temperature °C')


# In[ ]:


sns.lineplot(data=tas_future_modeled_NEXGDDPCMIP6_gorongosa, x="Year",y='Daily Near-Surface Air Temperature °C',hue='Experiment')


# In[ ]:


tas_future_modeled_NEXGDDPCMIP6_gorongosa_overMandS=tas_future_modeled_NEXGDDPCMIP6[tas_future_modeled_NEXGDDPCMIP6['Name project']=='Gorongosa_EIB'].drop(['Name project','Latitude','Longitude'],axis=1).groupby(['Year'])[['Daily Near-Surface Air Temperature °C']].mean().reset_index()


# In[ ]:


sns.lineplot(tas_future_modeled_NEXGDDPCMIP6,x='Year',y='Daily Near-Surface Air Temperature °C')


# In[ ]:


plot_spaghetti(tas_future_modeled_NEXGDDPCMIP6,'Year','Daily Near-Surface Air Temperature °C','title','No')


# In[ ]:


tas_historic_modeled_NEXGDDPCMIP6_gorongosa


# In[ ]:


fig,ax=plt.subplots() # sans aire
sns.lineplot(data=tas_historic_modeled_NEXGDDPCMIP6_gorongosa,x= 'Month',y='Daily Near-Surface Air Temperature °C',hue='Model',ax=ax)#,errorbar=('pi',50) )
sns.lineplot(data=daily_sum_obs_from_NOAA_gorongosa,x='Month',y='TAVG',color='black',ax=ax)


# In[ ]:


fig,ax=plt.subplots() # sans aire
sns.lineplot(data=tas_historic_modeled_NEXGDDPCMIP6_gorongosa,x= 'Year',y='Daily Near-Surface Air Temperature °C',hue='Model',ax=ax)#,errorbar=('pi',50) )
sns.lineplot(data=daily_sum_obs_from_NOAA_gorongosa,x='Year',y='TAVG',color='black',ax=ax)


# In[ ]:


sns.lineplot(data=na_values,y='TAVG',x='Year')


# In[ ]:


daily_sum_obs_from_NOAA_gorongosa['TAVG']


# In[ ]:





# In[ ]:





# In[ ]:




