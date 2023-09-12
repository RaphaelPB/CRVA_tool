#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:56:23 2023

@author: NADB
"""


#Import python packages
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
import cdsapi # for copernicus function
import datetime # to have actual date
from netCDF4 import Dataset



#def read cckp (world bank) nc files
#reads data from world bank climate knowledge portal, nc files, with a single band
#assigns projection and exports to tif since zonal_stats seems to have issues with it otherwise (not ideal solution)
def read_cckp_ncdata(nc_path,output='tempfile.tif'):
    with rioxarray.open_rasterio(nc_path,decode_times=False)[0] as ncdata:
        ncdata.rio.write_crs('EPSG:4326', inplace=True)
        ncdata=ncdata.isel(time=0)
        if os.path.exists(output):
            os.remove(output) 
        ncdata.rio.to_raster(output)
        ncdata.close()
        
       # output=output #here
   # else: 
      #  print(nc_path,"not found") # in this case, the data printed in the table will apply to the previous print.. 
       # output=0 #here
    return output       


#get filename from cckp based on ssp, period and gcm
def get_cckp_file_name(var,ssp='ssp245',period='2010-2039',gcm='median',data_folder=r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets'):
    cwd=os.getcwd()
    main=os.path.abspath(os.path.join(cwd, os.pardir))
    #data_folder=os.path.join(main,'dataset') 
    #data_folder=r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets'
    if period in ['1991-2020']:
 #cru/era
    #Precipitation   
        if var in ['climatology-r50mm-annual-mean_era_annual','climatology-rx1day-monthly-mean_era_monthly','climatology-rx1day-annual-mean_era_annual','climatology-pr-annual-mean_era_annual','climatology-pr-monthly-mean_era_monthly']:
            filename='precipitation/wb_cckp/climatology-rx5day-annual-mean_era_annual_era5-0.5x0.5-climatology_mean_1991-2020.nc'
            filename=filename.replace('climatology-rx5day-annual-mean_era_annual',var)
        elif var in ['climatology-pr-annual-mean_cru']:
            filename='precipitation/wb_cckp/climatology-pr-annual-mean_cru_annual_cru-ts4.06-climatology_mean_1991-2020.nc'
    #Temperature
        elif var in ['climatology-tasmax-annual-mean_era','climatology-hd35-annual-mean_era','climatology-tas-annual-mean_era','climatology-hd40-annual-mean_era']:
            filename='temperature/wb_cckp/climatology-tasmax-annual-mean_era_annual_era5-0.5x0.5-climatology_mean_1991-2020.nc'
            filename=filename.replace('climatology-tasmax-annual-mean_era',var)                                                                                                                                 
        elif var in ['climatology-tasmax-annual-mean_cru']: 
            filename='temperature/wb_cckp/climatology-tasmax-annual-mean_cru_annual_cru-ts4.06-climatology_mean_1991-2020.nc' 
 #Realtime             
    elif period not in ['1991-2020']:
    #Precipitation     
        if var in ['frp100yr-rx1day-period-mean_cmip6_period','climatology-rx1day-annual-mean_cmip6_annual','frp50yr-rx1day-period-mean_cmip6_period','climatology-pr-monthly-mean_cmip6_monthly','climatology-pr-annual-mean_cmip6_annual','climatology-pr-seasonal-mean_cmip6_seasonal','changefactorfaep100yr-rx1day-period-mean_cmip6_period','anomaly-pr-monthly-mean_cmip6_monthly','climatology-rx5day-annual-mean_cmip6_annual','anomaly-pr-annual-mean_cmip6_annual']: 
            filename='precipitation/wb_cckp/frp100yr-rx1day-period-mean_cmip6_period_all-regridded-bct-ssp245-climatology_median_2010-2039.nc'   
            filename=filename.replace('2010-2039',period)
            filename=filename.replace('frp100yr-rx1day-period-mean_cmip6_period',var)                      
    #Temperature
        elif var in ['climatology-hd40','anomaly-hd40','anomaly-hd35','anomaly-tasmax','climatology-tasmax','anomaly-txx','climatology-txx','anomaly-tas','climatology-tas']: 
            filename='temperature/wb_cckp/climatology-hd40-annual-mean_cmip6_annual_all-regridded-bct-ssp245-climatology_median_2020-2039.nc'
            filename=filename.replace('2020-2039',period)    
            filename=filename.replace('climatology-hd40',var)
        filename=filename.replace('ssp245',ssp)
        filename=filename.replace('median',gcm)
    data_path=os.path.join(data_folder,filename)
    return data_path
#import data from copernicus

