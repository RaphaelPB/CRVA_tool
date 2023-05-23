#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import pandas as pd
from multiprocessing.pool import ThreadPool
import os
import os.path
import netCDF4 as nc#not directly used but needs to be imported for some nc4 files manipulations, use for nc files
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


# In[ ]:

NBCORES=5
# register information from csv file
all_urls = pd.read_csv('gddp-cmip6-thredds-fileserver.csv')


# In[ ]:


### make all elements of the csv into a readable list

# transpose csv
temp_list = all_urls[[' fileUrl']].T
temp_list=temp_list.values.tolist()
temp_list=temp_list[0]
url_list=[s.replace(' ', '') for s in temp_list]


# In[ ]:


## download only precipitation data
# select only precipitation files, between 2040 and 2080
url_list_precipitation = [url for url in url_list if 'pr_day_' in url and int(url[len(url)-7:len(url)-3])>2020 and int(url[len(url)-7:len(url)-3])<2061]


# In[ ]:


# download only precipitation files
#for file in url_list_precipitation: # for loop to download the file in each url
def download_file(file):
    # find the name of the file
    index_before_name=file.rfind('/') # returns the highest index where the last character '/' was found, which is just before the name of the file    
    f_name = file[index_before_name+1:len(file)-3] # return the name of the file as a string
    # check if the file was already downloaded
    test = os.path.join('//COWI.net/projects/A245000/A248363/CRVA/Datasets/NEX-GDDP-CMIP6/',f_name+'.nc')
    if os.path.isfile(test): # if the file was aleready download
        #continue # continue the for loop without executing the code after this line. The code follonwing download the file
        return# if we entered the if, the file was already dowloaded, no need to downloaded it again
    # the file was not downloaded, the following code will permit to download it in the servor for dataset
    r = requests.get(file) # return the url were data need to be downloaded
    # download data in the servor for datasets
    with open(f'//COWI.net/projects/A245000/A248363/CRVA/Datasets/NEX-GDDP-CMIP6/{f_name}.nc', 'wb') as f:
        f.write(r.content)
    return file
results = ThreadPool(NBCORES).imap_unordered(download_file, url_list_precipitation)
for r in results: #I don't understand why, ut without this step it does not work
     print(r)
# In[ ]:


## read data, _FillValue should be NaN
# solar_1950=Dataset(r'\\COWI.net\projects\A245000\A248363\CRVA\Datasets\NEX-GDDP-CMIP6\hurs_day_ACCESS-CM2_historical_r1i1p1f1_gn_1950.nc')
# hurs=solar_1950.variables['hurs']
# hurs
# hurs._FillValue
# solar_dataframe = np.ma.getdata(solar_1950.variables['hurs']).data
# solar_dataframe
# len(solar_dataframe[solar_dataframe!=1.e+20])
# len(solar_dataframe[solar_dataframe==1.e+20])
# type(solar_dataframe)
# solar_dataframe.size
# solar_dataframe.size-len(solar_dataframe[solar_dataframe!=1.e+20])-len(solar_dataframe[solar_dataframe==1.e+20])


# In[ ]:




