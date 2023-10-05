#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)
# 
# Source of image : https://link.springer.com/article/10.1007/s10584-021-03122-z, section 3.2
# 
# Indications are misleading the results : RH_mean should be [-] (divide the value in percent by 100). The equation used here divide by 100 the number of RH_mean placed in the equation

# Units of PET is kg.m^(-2).day^(-1)

# # Needed parameters
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

# # Import packages

# In[1]:


import math
import xarray as xr
import numpy as np
import pandas as pd


# # Slope of vapor pressure
# ![image-4.png](attachment:image-4.png)
# ![image.png](attachment:image.png)
# ![image-5.png](attachment:image-5.png)
# ![image-2.png](attachment:image-2.png)
# ![image-3.png](attachment:image-3.png)
# 
# Source : https://www.researchgate.net/publication/269337754_Quantifying_water_savings_with_greenhouse_farming section II B

# # Net radiation
# 
# Source for all information for radiation : https://www.fao.org/3/x0490e/x0490e07.htm#radiation
# ![image-3.png](attachment:image-3.png)

# ### Net shortwave radiation Rns
# ![image.png](attachment:image.png)

# ### Net longwave radiation Rnl
# ![image.png](attachment:image.png)

# #### e_a : actual vapour pressure
# ![image.png](attachment:image.png)
# Calculated e_a from RH and e_0(T)

# #### Ra :  Daily extraterrestrial radiation (Ra) for different latitudes for the 15th day of the month
# 
# Source : https://www.fao.org/3/x0490e/x0490e0j.htm#TopOfPage

# In[2]:


def R_a_determination(lat,month,R_a_df):
    if pd.isnull(month):
        R_a=np.nan
    else:
    # month should be one of the elements of the following list ['Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'July', 'Jun', 'Mar', 'May', 'Nov','Oct', 'Sep']
        #R_a_df = dataframe_Ra(Ra)
        # determine if Northern or Southern hemisphere
        if lat >0:
            Hemisphere_of_interest = R_a_df.columns.levels[0][0]
        if lat <0:
            Hemisphere_of_interest = R_a_df.columns.levels[0][1]

        # find closest lat to the ones in the dataframe
        # convert lat to numpy.float64 to compare it ? numpy.float64 is the type of thelatitude in R_a_df
        # no need to convert
        diff = diff=abs(lat-R_a_df.index.levels[0].values) # calculate the absolute difference
        lat_of_interest = R_a_df.index.levels[0].values[np.where(diff==min(diff))[0]]
        R_a=R_a_df[Hemisphere_of_interest,month].loc[lat_of_interest].values
        R_a=float(R_a[0])
    return R_a


# In[1]:


# this function shapes the Ra dataframe with all the information in it
def dataframe_Ra(Ra):
    #path_Ra = r'\\cowi.net\projects\A245000\A248363\CRVA\Datasets\calculate_PET\Ra\Table2-Ra.csv'
    #Ra = pd.read_csv(path_Ra)
    # Create the MultiIndex
    midx = pd.MultiIndex.from_product([Ra[Ra.columns[0]][1:len(Ra[Ra.columns[0]])].values],names=[Ra.columns[0]])
    # multiindex to name the columns
    cols = pd.MultiIndex.from_product([(Ra.columns[1],),Ra.iloc[0][1:13]])
    # Create the Dataframe
    Northern_dataframe = pd.DataFrame(data = Ra.iloc[1:len(Ra[Ra.columns[0]])][Ra.columns[1:13]].values, 
                                index = midx,
                                columns = cols)
    # Concatenate former and new dataframe

    midx = pd.MultiIndex.from_product([Ra[Ra.columns[0]][1:len(Ra[Ra.columns[0]])].values],names=[Ra.columns[0]])
    # multiindex to name the columns
    cols = pd.MultiIndex.from_product([(Ra.columns[13],),Ra.iloc[0][1:13]])
    # Create the Dataframe
    Southern_dataframe = pd.DataFrame(data = Ra.iloc[1:len(Ra[Ra.columns[0]])][Ra.columns[13:len(Ra.columns)]].values, 
                                index = midx,
                                columns = cols)
    Ra = pd.concat([Northern_dataframe,Southern_dataframe],axis=1)# register information for project
    return Ra


# # Latent heat of vaporization of water 
# 
# ![image.png](attachment:image.png)
# Source : https://www.researchgate.net/publication/267803552_Climate_parameters_used_to_evaluate_the_evapotranspiration_in_delta_central_zone_of_Egypt, section 3
# 
# ![image-2.png](attachment:image-2.png)

# # Psychrometric constant [kPa/°C]
# 
# ![image.png](attachment:image.png)
# Source : https://www.researchgate.net/publication/267803552_Climate_parameters_used_to_evaluate_the_evapotranspiration_in_delta_central_zone_of_Egypt, section 3

# # Wind ![image.png](attachment:image.png)
# 
# Source of the image : https://link.springer.com/article/10.1007/s10584-021-03122-z section 3.2

# # (Mean) saturation vapor pressure
# 
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)
# ![image-3.png](attachment:image-3.png)
# 
# source : https://www.researchgate.net/publication/269337754_Quantifying_water_savings_with_greenhouse_farming section II B

# # Evapotranspiration [kg.m^(-2).day^(-1)]
# 
# ![image.png](attachment:image.png)

# In[4]:


# values for test
#T = 20
#T_max = 40
#T_min = 10
#Rs = 10 
#RH_mean = 1
#U_z =2


# In[1]:


# Function PET is to determine potential evapotranspiration

def PET(T,T_max,T_min,Rs,RH_mean,U_2,z_station_elevation,lat,month,Ra):
    
    #### Slope of vapor pressure
    _delta = 4098*e_0(T)/(T+237.3)**2
    
    #### Net radiation
    # Net shortwave radiation Rns
    # from DANIEL A. VALLERO, in Fundamentals of Air Pollution (Fourth Edition), 2008
    # https://www.sciencedirect.com/science/article/abs/pii/B9780123736154500066?via%3Dihub
    # https://edisciplinas.usp.br/pluginfile.php/5464081/mod_book/chapter/23386/Fundamentals%20of%20Air%20Pollution.pdf
    albedo = 0.035 # for the surface of water

    Rns = (1 - albedo)*Rs
    
    # Net longwave radiation Rnl
    Boltzman_constant = 4.903*10**(-9)
    #NEW
    Ra = dataframe_Ra(Ra)
    R_a = month.apply(lambda x: R_a_determination(lat,x,Ra))
    #R_a = R_a_determination(lat,month,Ra)
    Rs0 = (0.75 + 2*10**(-5)*z_station_elevation)*R_a# depens on the station elevation above sea level [m] z_station_elevation and
    # Ra, which depends on the month and latitude
    e_a = RH_mean*e_0(T)/100
    Rnl = Boltzman_constant*((T_max+273.16+T_min+273.16)/2)*(0.34-(0.14*math.sqrt(e_a)))*(1.35*(Rs/Rs0)-0.35)
    
    # Final calculation of net radiation
    R_n = Rns - Rnl
    
    #### Latent heat of vaporization of water 
    _lambda = 2.501 - (2.361*10**(-3))*T
    
    # Psychrometric constant [kPa/°C]
    _gamma = (1.005*101.3)/(0.622*_lambda*10**3)
    
    #### Wind, conversion already done out of the function
    #z = 10 # m
    #U_2 = U_z*(4.87)/(np.log(67.8*z-5.42)) # np.log is the neperian logarithm ln
    
    #### (Mean) saturation vapor pressure
    e_s = (e_0(T_max)+e_0(T_min))/2
    
    
    ##### potential evapotranspiration calculation
    PET_value = (_delta*R_n + (6.43*_gamma*(1+0.536*U_2)*(1-(RH_mean/100))*e_s))/((_delta+_gamma)*_lambda)
    
#     print('_delta '+str(_delta))
#     print('R_n '+str(R_n))
#     print('_gamma '+str(_gamma))
#     print('U_2 '+str(U_2))
#     print('RH_mean '+str(RH_mean))
#     print('_lambda '+str(_lambda))
    
    # if PET_value < 0:
    #     print('The potential evapotranspiration is negative, there is a problem with input data')
    
    return PET_value


# In[6]:


# saturation vapor pressure at the air
def e_0(T):
    #NEW
    e_0_result = 0.6108*(17.27*T/(T+237.3)).apply(math.exp)
    #0.6108*math.exp(17.27*T/(T+237.3))
    return e_0_result


# In[7]:


# test
#pet = PET(T,T_max,T_min,Rs,RH_mean,U_z,3.5,45,'Jan')


# In[8]:


#pet


# In[ ]:




