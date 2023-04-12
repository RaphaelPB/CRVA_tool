#!/usr/bin/env python
# coding: utf-8

# ### Import python packages

# In[1]:


#Import python packages
import os.path
from rasterstats import zonal_stats
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import show
import rioxarray #used when calling ncdata.rio.write_crs
import xarray as xr
import os
import os.path
import matplotlib.pyplot as plt
import netCDF4 as nc#not directly used but needs to be imported for some nc4 files manipulations, use for nc files
from netCDF4 import Dataset
import csv #REMOVE ? not in use ?
import numpy as np
import numpy.ma as ma
from mpl_toolkits.basemap import Basemap
import shutil # to move folders
import warnings
warnings.filterwarnings('ignore') # to ignore the warnings
import cdsapi # needed package to download from the website


# ## Copernicus function

# In[2]:


##################################################### Copernicus function ######################################################
# Aim of the function : read nc data found on copernicus CMIP6 projections (https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6?tab=overview )
# Actions of this function
#     1) check which parameters are asked or not in the variables dictionnary, and modify the last depend on the parameters chosen byt the user before
#     2) thanks to c.retrieve function and the variables dictionnary, the chosen data are download in zip format
#     3) The downloaded file (always in zip format) is dezipped and registered in a specific folder
#     4) the function looks in the specific folder for a nc format file, and once found, return the path of this nc format file

# Parameters of the function
# temporal_resolution : daily or monthly or fixed
# SSP : sscenario that is studied "Historical", "SSP1-1.9", "SSP1-2.6" ...
# name_variable : variable to be studied
# model: model of projection to choose
# year: year(s) of study to choose
# area: area of study, if not specific, area should be an empty array area=[]
# path_for_file: path where the file must be unzipped

def copernicus_data(temporal_resolution,SSP,name_variable,model,year,area,path_for_file,out_path): 
    
    # creat a path to register data
    if not os.path.isdir(path_for_file):
        
        start_path = os.path.join(out_path,'Data_download_zip')

        if len(year)==1:
            file_download = os.path.join(start_path,name_variable,SSP,model,year)
        elif len(year)>1:
            period=year[0]+'-'+year[len(year)-1]
            file_download = os.path.join(start_path,name_variable,SSP,model,period)
        elif temporal_resolution == 'fixed':
            file_download = os.path.join(start_path,name_variable,SSP,model,'fixed_period')

        if not os.path.isdir(file_download):
            
            c = cdsapi.Client()# function to use the c.retrieve
            # basic needed dictionnary to give to the c.retrieve function the parameters asked by the user
            variables = {
                        'format': 'zip', # this function is only designed to download and unzip zip files
                        'temporal_resolution': temporal_resolution,
                        'experiment': SSP,
                        'variable': name_variable,
                        'model': model,
            }

            if area != []: # the user is interested by a sub region and not the whole region 
                variables.update({'area':area}) 

            if name_variable == 'air_temperature':
                variables['level'] = '1000' # [hPa], value of the standard pressure at sea level is 1013.25 [hPa], so 1000 [hPa] is the neareste value. Other pressure value are available but there is no interest for the aim of this project

            if temporal_resolution != 'fixed':# if 'fixed', no year, month, date to choose
                variables['year']=year
                variables['month']= [ 
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    ]  # be default, all the months are given
                if temporal_resolution == 'daily':
                    variables['day']= [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                        ] # be default, all the days are given
            # c.retrieve download the data from the website
            try:
                c.retrieve(
                    'projections-cmip6',
                    variables,
                    'download.zip') # the file in a zip format is registered in the current directory
            except:
                print('Some parameters are not matching')
                return # stop the function, because some data the user entered are not matching
            
            os.makedirs (path_for_file) # to ensure the creation of the path
            # unzip the downloaded file
            from zipfile import ZipFile
            zf = ZipFile('download.zip', 'r')
            zf.extractall(path_for_file)
            zf.close()
            
            os.makedirs (file_download) # to ensure the creation of the path
            # moving download to appropriate place
            #file_download = os.path.join(file_download,'download.zip')
            shutil.move('download.zip',file_download) # no need to delete 'download.zip' from inital place

        else: # if the path already exist, the data should also exists
            pass

        # look for nc file types in path_for_file. There should only be one nc files for every downloading
        for file in os.listdir(path_for_file):
            if file.endswith(".nc"):
                final_path=os.path.join(path_for_file, file)
                print('The path exists')
                return final_path # the function returns the path of the nc file of interest
                break # stop the function if a nc file was found 
            else:
                pass
        
        print('Problem : No nc file was found')
        
    else:
        for file in os.listdir(path_for_file):
            if file.endswith(".nc"):
                final_path=os.path.join(path_for_file, file)
                print('The path exists')
                return final_path # the function returns the path of the nc file of interest
                break # stop the function if a nc file was found 
            else:
                pass
    print('Problem : No nc file was found')


# In[ ]:




