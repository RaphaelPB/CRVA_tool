#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd
from multiprocessing.pool import ThreadPool
import os
import xarray as xr
import os.path
import netCDF4
from netCDF4 import Dataset
# import numpy as np
# import numpy.ma as ma

#%% Options
#options
TRYCORRUPT=0 #1 to open each existing file to see if corrupt (slow), 0 to ignore existing files (faster)
ADDLOCATION=1 #1 if specify coordinate, 0 to download all
NBCORES=5 #parallel cores
#Years
y_start = 1950 
y_end = 1980
#Variables
list_var=['tasmin']#'tasmin','pr']#'sfcWind', 'rsds', 'hurs', 'tas','tasmax'] 
#location (max, min cooordinate south, noth, west and east)
north=-10
south=-30
west=30
east=45
#experiment
experiment='r1i1p1f1_gn'


#%%
# register information from csv file
all_urls = pd.read_csv('gddp-cmip6-thredds-fileserver.csv')

### make all elements of the csv into a readable list

# transpose csv
temp_list = all_urls[[' fileUrl']].T
temp_list=temp_list.values.tolist()
temp_list=temp_list[0]
url_list=[s.replace(' ', '') for s in temp_list]

#URL EXAMPLE paths:
#https://ds.nccs.nasa.gov/thredds2/fileServer/AMES/NEX/GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/hurs/hurs_day_ACCESS-CM2_historical_r1i1p1f1_gn_1950.nc
#https://ds.nccs.nasa.gov/thredds/ncss/AMES/NEX/GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/pr/pr_day_ACCESS-CM2_historical_r1i1p1f1_gn_2014.nc

# download only precipitation files
#for file in url_list_precipitation: # for loop to download the file in each url
def download_file(file):
    # find the name of the file
    index_before_name=file.rfind('/') # returns the highest index where the last character '/' was found, which is just before the name of the file    
    f_name = file[index_before_name+1:len(file)-3] # return the name of the file as a string
    # check if the file was already downloaded
    test = os.path.join('//COWI.net/projects/A245000/A248363/CRVA/Datasets/NEX-GDDP-CMIP6-AllMoz/',f_name+'.nc')
    if os.path.isfile(test): # if the file was already download
        if TRYCORRUPT==1: #tries to open file to see if it can read it (time consuming)
            try: # test if the file is corrupted
                ds = xr.open_dataset(test)
                ds.close
                return
            except:
                print('The file is corrupted')
        else:
            return
    # the file was not downloaded or is corrupted, the following code will permit to download it in the servor for dataset
    if ADDLOCATION==1: #add corrdinates to download only selected geography:
        year=file[-7:-3]
        var = file.split('/')[11]
        add_info='?var=pr&north=-10&west=30&east=45&south=-30&disableProjSubset=on&horizStride=1&time_start=2014-01-01T12%3A00%3A00Z&time_end=2014-12-31T12%3A00%3A00Z&timeStride=1'
        add_info=add_info.replace('north=-10','north='+str(north))
        add_info=add_info.replace('south=-30','south='+str(south))
        add_info=add_info.replace('west=30','west='+str(west))
        add_info=add_info.replace('east=45','east='+str(east))
        add_info=add_info.replace('2014',year)
        add_info=add_info.replace('var=pr','var='+var)
        file=file.replace('thredds2/fileServer','thredds/ncss')
        file=file+add_info

    # this will get the url and retry 10 times in case of requests.exceptions.ConnectionError
    # backoff_factor will help to apply delays between attempts to avoid failing again in case of periodic request quota
    session = requests.Session()
    retry = Retry(connect=10, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)


    with session.get(file) as r:
        # return the url were data need to be downloaded
        # download data in the servor for datasets
        with open(f'//COWI.net/projects/A245000/A248363/CRVA/Datasets/NEX-GDDP-CMIP6-AllMoz/{f_name}.nc', 'wb') as f:
            f.write(r.content)
    return file

   
## download  data
for var in list_var: #loop through variables
    url_list_climate_variable = [url for url in url_list if var+'_' in url 
                                 and int(url[len(url)-7:len(url)-3])>=y_start 
                                 and int(url[len(url)-7:len(url)-3])<=y_end 
                                 and experiment in url and 'day' in url]
    
    #launch pool download    
    results = ThreadPool(NBCORES).imap_unordered(download_file, url_list_climate_variable)
    for r in results: #I don't understand why, ut without this step it does not work
        print(r)
