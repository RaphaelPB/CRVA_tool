#!/usr/bin/env python
# coding: utf-8

# This file aims to regroup all function involved in file management

# In[7]:


import os
import os.path
import numpy as np
from netCDF4 import Dataset


# In[8]:


# Gros bug sur cette function

def download_extract(path_file,path_for_file):
    #if not os.path.isdir(path_for_file): # path_for_file does not exists, need to ensure that is is created
    #    os.makedirs(path_for_file) # to ensure the creation of the path
    # unzip the downloaded file
    from zipfile import ZipFile
  
    # loading the temp.zip and creating a zip object
    os.chdir(path_file)
    with ZipFile(path_for_file, 'r') as zObject:
      
    # Extracting all the members of the zip 
    # into a specific location.
        print(zObject)
        zObject.extractall()
    
    print('\n ----------------------------- The downloaded file is extracted in the indicated file -----------------------------')
    return


# In[9]:


def path_length(str1):
    if len(str1)>250:
        path = os.path.abspath(str1) # normalize path
        if path.startswith(u"\\\\"):
            path=u"\\\\?\\UNC\\"+path[2:]
        else:
            path=u"\\\\?\\"+path
        return path
    else:
        return str1


# In[10]:


def read_nc_file(path):
    name_variable = find_column_name(path)
    
    #df=Dataset(path)
    
    lat=np.ma.getdata(Dataset(path).variables['lat']).data
    lon=np.ma.getdata(Dataset(path).variables['lon']).data
    time=np.ma.getdata(Dataset(path).variables['time']).data
    variable=np.ma.getdata(Dataset(path).variables[name_variable]).data
    
    return lat, lon, time, variable


# In[12]:


# function to return column name in the netCDF file
# all netCDF file form copernicus have this format for their variables names
# ['time', 'time_bnds', 'lat', 'lat_bnds', 'lon', 'lon_bnds', Name of climate variable of interest]
# take of 'time', 'time_bnds', 'lat', 'lat_bnds', 'lon', 'lon_bnds'
def find_column_name(path):
    # make a list with every variables of the netCDF file of interest
    climate_variable_variables=list(Dataset(path).variables)
    # variables that are not the column name of interest 
    elements_not_climate_var =['time', 'time_bnds', 'bnds','lat', 'lat_bnds', 'lon', 'lon_bnds','time_bounds','bounds','lat_bounds','lon_bounds']
    for str in elements_not_climate_var:
        if str in climate_variable_variables:
            climate_variable_variables.remove(str)
    return climate_variable_variables[0]


# In[ ]:




