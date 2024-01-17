#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd

# from Functions_ImportData import import_treat_modeled_NEX_GDDP_CMIP6
# from Functions_ImportData import import_BC_NOAA_NEX_GDDP_CMIP6
# from Functions_ImportData import import_BC_Gorongosa_NEX_GDDP_CMIP6
from Functions_ImportData import add_year_month_season
from Evaporation_code import ET0 #FAO 
#from Potential_evapostranspiration_function import PET
#from Evaporation_E0 import E0

def filter_data(df,loc,mod_excl): # Filter data by location, models, and time period
    df = df[df['Name project']==loc]
    df = df[~df['Model'].isin(mod_excl)]
    df=add_year_month_season(df,'Date')
    df=df.set_index(['Name project','Experiment','Model','Date'])
    return df


# Source of image : https://link.springer.com/article/10.1007/s10584-021-03122-z, section 3.2
# 
# Indications are misleading the results : RH_mean should be [-] (divide the value in percent by 100). The equation used here divide by 100 the number of RH_mean placed in the equation

# In[2]:


# ctrl * for at udkommentere tekst!!!! 


# Objectif: find the number of cumulative days per year with positive net precipitation (Pr - E)

# # Project information that could be useful

# ![image.png](attachment:image.png)

# # Defined by user 

# In[12]:


# CHANGE ACCORDING TO PURPOSE 
loc='Vemasse'
mod_excl=[]
yr_past=np.array([1980, 1914])
yr_future=np.array([2030,2080])
#	WTP_Mutua_EIB	latitude:-19.495080	elevation: 15
#	Gorongosa_EIB	latitdue: -18.680637	elevtaion: 383
# 	Chimoio_WTP_EIB	latitude:-19.125095 elevation:	723
#	Pemba_EIB	latitude:	-12.973943	 elevation: 47


#Timor


#station information
latitude = -8.504763	 # 
z_station= 16#mutua m 

#PATHS DATA INPUT
#precipitation 
path_pr=r'\\cowi.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6\csv_timor\pr\pr_mm\day_day_1980-2080\pr_1980-2080timor_gr2.csv'
#tas: daily mean temperature
path_tas=r'\\cowi.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6\csv_timor\tas\tas_Celsius_day_1980-2080\tas_1980-2080timor_gr2.csv'
#tasmin: minimum daily temperature 
path_tasmin=r'\\cowi.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6\csv_timor\tasmin\tasmin_Celsius_day_1980-2080\tasmin_1980-2080timor_gr2.csv'
#tasmax: maximum daily temperature 
path_tasmax=r'\\cowi.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6\csv_timor\tasmax\tasmax_Celsius_day_1980-2080\tasmax_1980-2080timor_gr2.csv'
#rs: radiaion (MJ.m-2.day-1)
path_rs=r'\\cowi.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6\csv_timor\rsds\rsds_MJ.m-2.day-1_day_1980-2080\rsds_1980-2080timor_gr2.csv'
# near surface relative humidity RH 
path_RH=r'\\cowi.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6\csv_timor\hurs\hurs_%_day_1980-2080\hurs_1980-2080timor_gr2.csv'
# Daily-Mean Near-Surface Wind Speed [m_s-1]
path_wind10=r'\\cowi.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6\csv_timor\sfcWind\sfcWind_m_s-1_day_1980-2080\sfcWind_1980-2080timor_gr2.csv'


# # Evapotranspiration at Gorongosa

# #### needed parameters
# 
# Air temperature in degrees, normal, max and min
# 
# Downwelling short Rs in MJ/(m^2.day)
# 
# For calculation of Rnl (net longwave radiaiton, used for calculation of net radiation), need to know lat and month for the calculation to calculate Ra
# 
# Wind at 2 m [m/s], height where measurement taken is approx 10 m
# 
# Mean relative humidity in %

# ## Read all data sheets 

# In[13]:


#READ PRECIPITATION 
df_pr=pd.read_csv(path_pr)
df_pr=filter_data(df_pr,loc,mod_excl)
df_pr_future=df_pr[df_pr['Year'].between(yr_future[0],yr_future[1])]

#READ TAS (DAILY MEAN TEMPERATURE)
df_tas=pd.read_csv(path_tas)
df_tas=filter_data(df_tas,loc,mod_excl)
df_tas_future=df_tas[df_tas['Year'].between(yr_future[0],yr_future[1])]

#READ TASMIN (DAILY MINIMUM TEMPERATURE)
df_tasmin=pd.read_csv(path_tasmin)
df_tasmin=filter_data(df_tasmin,loc,mod_excl)
df_tasmin_future=df_tasmin[df_tasmin['Year'].between(yr_future[0],yr_future[1])]

#READ TASMAX (DAILY MAXIMUM TEMPERATURE)
df_tasmax=pd.read_csv(path_tasmax)
df_tasmax=filter_data(df_tasmax,loc,mod_excl)
df_tasmax_future=df_tasmax[df_tasmax['Year'].between(yr_future[0],yr_future[1])]

#READ RADIATION DATA  (MJ.m-2.day-1)
df_rs=pd.read_csv(path_rs)
# evt. udkommentér  # camille laver noget med drop (se Netprecipitation file), men tror ikke det er aktuelt her
df_rs=filter_data(df_rs,loc,mod_excl)
df_rs_future=df_rs[df_rs['Year'].between(yr_future[0],yr_future[1])]

#READ NEAR SURFACE RELATIVE HUMIDITY PERCENTAGE 
df_RH=pd.read_csv(path_RH)
df_RH=filter_data(df_RH,loc,mod_excl)
df_RH_future=df_RH[df_RH['Year'].between(yr_future[0],yr_future[1])]

#READ WIND SPEED DATA: Daily-Mean Near-Surface Wind Speed [m_s-1]
df_wind10=pd.read_csv(path_wind10)
df_wind10=filter_data(df_wind10,loc,mod_excl)
df_wind2=df_wind10.copy(deep=True) #Convert to 2 meters above ground instead of initial 10 meters
z=10 #height above ground where the wind was measures 
df_wind2[['Daily-Mean Near-Surface Wind Speedm_s-1']]=df_wind2[['Daily-Mean Near-Surface Wind Speedm_s-1']]*(4.87/(np.log((67.8*z)-5.42)))
df_wind2_future=df_wind2[df_wind2['Year'].between(yr_future[0],yr_future[1])]

#READ sheet for radiation data acoording to month and latitude
path_Ra = r'\\cowi.net\projects\A245000\A248363\CRVA\Datasets\calculate_PET\Ra\Table2-Ra.csv'
Ra=pd.read_csv(path_Ra)


# ##  Dataframe to record the PET
# Duplicate the one with pr, put nan in it.
# Need to take in account changing month and lat
df_PET = df_pr.copy(deep=True)
df_PET.rename(columns={'Mean of the daily precipitation rate mm_per_day':'ET0 mm'},inplace=True)
df_PET[['ET0 mm']] = np.nan
# dataframe for net precipitation
df_Net_Pr = df_PET.copy(deep=True)
df_Net_Pr.rename(columns={'ET0 mm':'Net precipitation mm'},inplace=True)



# In[15]:



# def filtr(df,date,model,ssp,col):
#     dfa=df.set_index(['Model','Experiment','Date'])
#     if (model,ssp,date) in dfa.index:
#         return dfa.loc[(model,ssp,date),col]
#     else:
#         return np.nan
    
def dataframe_net_pr(df_pr,df_tas,df_tasmax,df_tasmin,df_Rs,df_RH,df_wind2,latitude,z_station,Ra):   
    # prepare dataframe to fill
    # dataframe for potential evapotranspiration
    df_PET = df_pr.copy(deep=True)
    df_PET.rename(columns={'Mean of the daily precipitation rate mm_per_day':'ET0 mm'},inplace=True)
    # ssps=list(df_PET.Experiment.unique())
    # models=list(df_PET.Model.unique())
    # dates=list(df_PET.Date.unique())
    # df_PET=df_PET.set_index(['Model','Experiment','Date'])
    df_PET[['ET0 mm']] = np.nan
    # dataframe for net precipitation
    df_Net_Pr = df_PET.copy(deep=True)
    df_Net_Pr.rename(columns={'ET0 mm':'Net precipitation mm'},inplace=True)


                
    # what are the values of the parameters for the ssp, model and date precised ?
    T=df_tas['Daily Near-Surface Air Temperature °C ']
    #print('T=',T)
    T_max=df_tasmax['Daily Maximum Near-Surface Air Temperature Celsius']
    #print('Tmax=',T_max)
    T_min=df_tasmin['SDaily Minimum Near-Surface Air TemperatureCelsius']
    #print('Tmin=',T_min)
    Rs_=df_Rs['Surface Downwelling Shortwave RadiationMJ.m-2.day-1']
    #print('Rs=',Rs_)
    RH_mean=df_RH['Daily-Mean Near-Surface Wind Speed%']
    wind2=df_wind2['Daily-Mean Near-Surface Wind Speedm_s-1']
     
    month=df_pr['Month']
    df_PET['ET0 mm']=ET0(T,T_max,T_min,Rs_,RH_mean,wind2,z_station,latitude,month,Ra) # Open water evaporation from reservoirs may be estimated by multiplying PET by a factor of 1.2. source: 3.Evapotranspiration -FAO
    df_Net_Pr['Net precipitation mm']=df_pr['Mean of the daily precipitation ratemm/day']-df_PET['ET0 mm'] #filtr(df_PET, date, model, ssp, 'Potential evapotranspiration mm')
    # filtr(df_PET,date,model,ssp,'Potential evapotranspiration mm')=PET(T,T_max,T_min,Rs_,RH_mean,wind2,z_station_elevation,lat,month)
    # filtr(df_Net_pr,date,model,ssp,'Net precipitation mm') = filtr(df_pr,date,model, ssp,'Mean of the daily precipitation rate mm_per_day')- filtr(df_PET, date, model, ssp, 'Potential evapotranspiration mm')


    return df_PET, df_Net_Pr


# In[16]:

# (df_PET_future, df_netPr_future)=dataframe_net_pr(df_pr_future,df_tas_future,df_tasmax_future,df_tasmin_future,
#                                                   df_rs_future,df_RH_future,df_wind2_future,latitude,z_station,Ra)

(df_ET0, df_netPr)=dataframe_net_pr(df_pr,df_tas,df_tasmax,df_tasmin,
                                     df_rs,df_RH,df_wind2,latitude,z_station,Ra)

#(df_ET0_future, df_netPr_future)=dataframe_net_pr(df_pr_future,df_tas_future,df_tasmax_future,df_tasmin_future,
  #                                  df_rs_future,df_RH_future,df_wind2_future,latitude,z_station,Ra)
# In[18]:

# df_netPr_future.to_csv("df_netPr_future_2030_2060.csv")
# df_PET_future.to_csv("df_PET_future_2030_2060.csv")

# df_netPr.to_csv("df_netPr_future2_2030_2060.csv")
df_ET0.to_csv("df_ET0_FAOequation_Vemasse_albedo.csv")

# df_ET0=_=df_ET0.copy()
# df_ET0= df_ET0.groupby(['Month']).mean()*30
# df_ET0['ET0 mm']