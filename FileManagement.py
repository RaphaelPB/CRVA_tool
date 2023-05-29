#!/usr/bin/env python
# coding: utf-8

# This file aims to regroup all function involved in file management

# In[1]:


import os
import os.path
import numpy as np
from netCDF4 import Dataset


# In[2]:


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


# In[3]:


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


# In[4]:


def read_nc_file(path):
    name_variable = find_column_name(path)
    
    #df=Dataset(path)
    
    lat=get_data_nc(path,'lat')
    lon=get_data_nc(path,'lon')
    time=get_data_nc(path,'time')
    variable=return_NaN(path,name_variable)
    
    return lat, lon, time, variable


# In[5]:


def get_data_nc(path,name_variable):
    variable = np.ma.getdata(Dataset(path).variables[name_variable]).data
    return variable


# In[6]:


# function to return column name in the netCDF file
# all netCDF file form copernicus have this format for their variables names
# ['time', 'time_bnds', 'lat', 'lat_bnds', 'lon', 'lon_bnds', Name of climate variable of interest]
# take of 'time', 'time_bnds', 'lat', 'lat_bnds', 'lon', 'lon_bnds'
def find_column_name(path):
    # make a list with every variables of the netCDF file of interest
    climate_variable_variables=list(Dataset(path).variables)
    # variables that are not the column name of interest 
    elements_not_climate_var =['time', 'time_bnds', 'bnds','lat', 'lat_bnds', 'lon', 'lon_bnds','time_bounds','bounds','lat_bounds','lon_bounds','height']
    for str in elements_not_climate_var:
        if str in climate_variable_variables:
            climate_variable_variables.remove(str)
    return climate_variable_variables[0]


# In[7]:


# Cette fonction ne fonctionne pas

def return_NaN(path,name_variable):
    variable = get_data_nc(path,name_variable)
    value_NaN = Dataset(path).variables[name_variable]._FillValue
    import math
    #variable[variable==value_NaN] = math.nan#float('NaN')
    return variable


# In[8]:


# this function aims to convert the vector of the time vector from Unix time to the format '%Y-%m-%d'
# the function us the 2 following functions 'extract_start_date' and 'time_conversion'
def time_vector_conversion(path,resolution):
    from datetime import date # package to work with date and time
    (year,month,day) = extract_start_date(path) # next function, returning the year, month and day of the start date
    start = date(year,month,day) # the function date, with some number in the format int as input
    # the function date return the date in format 'YYYY-MM-DD'
    time=get_data_nc(path,'time') # get data from the nc file
    time_converted = [] # create an empty list, to register the converted time vector
    # for loop to convert the date from Unix to the format '%Y-%m-%d'
    for day in time:
        # add each converted time in the list 'time_converted'
        time_converted.append(time_conversion(day,start,resolution)) # convert time with the function 'time_conversion'
        # the function the function 'time_conversion'
    return time_converted # return the list with time converted in format '%Y-%m-%d'

# this function aims to extract the year, month and day of the start date
# the input is the path leading to the file of interest
def extract_start_date(path):
    # start date is after the str 'days since ' in units in the path indicated as input of the function
    start_date=Dataset(path).variables['time'].units.replace('days since ','') # the start_date in string format
    # next step is to extract and convert in int format the information in te str start_date
    # the year is always the 4 first elements of the str start_date
    year = int(start_date[0:4])
    # the month is always between the first '-' and the second '-' the str start_date
    month = int(start_date[start_date.find('-')+1:start_date.rfind('-')])
    # the day is always after the second '-' and before the end the str start_date
    day = int(start_date[start_date.rfind('-')+1:start_date.rfind('-')+3])
    return year,month,day # return year, mont and day in int format

# this function convert time from unix in str format
# input : 
##### days is the number representing a time in Unix
##### start is the date in format 'YYYY-MM-DD'
##### resolution : in str form. can be 'monthly' or 'daily'
def time_conversion(days,start,resolution):
    from datetime import timedelta # import the function timedelta
    if not days.dtype == int:
        # the days is not in int format
        days = int(days)
    # use the function timedelta, with an int as imput
    delta = timedelta(days) # Create a time delta object from the number of days
    offset = start + delta # add the delta to the start date
    # depending on the resolution, converted the offset in str format with strftime function
    if resolution == 'monthly':
        offset = offset.strftime('%Y-%m')
    if resolution == 'daily':
        offset = offset.strftime('%Y-%m-%d')
    return offset


# In[ ]:


def create_xr_array(data,coordonates):
    data_structure = xr.DataArray(data, coords=[times, locs], dims=["time", "space"])


# In[3]:


# the dataframe_copernicus functions aims to test if the data with the specific parameters exists (with copernicus_data)
# and then produce a csv file if the data exists

def create_dataframe1(temporal_resolution,year_str,experiments,models,out_path, name_variable, name_project,area,period,index_dates,dates,title_file):    
    path_for_csv = os.path.join(out_path,'csv_file',name_variable+period)
    df = pd.DataFrame() # create an empty dataframe
    for year in year_str:
        for SSP in experiments:
            experiment = (SSP,) # create tuple for iteration of dataframe
            print('Test with scenario '+SSP)
            for model_simulation in models:
                model =(model_simulation,) # create tuple for iteration of dataframe
                print('Test with model '+model_simulation)
                # path were the futur downloaded file is registered
                #path_for_file= os.path.join(out_path,'Datasets','NEX-GDDP-CMIP6',name_variable,name_project,SSP,model_simulation,period)
                # existence of path_for_file tested in copernicus function
                # climate_variable_path=copernicus_data(temporal_resolution,SSP,name_variable,model_simulation,year_str,area,path_for_file,out_path,name_project,source)
                climate_variable_path = find_path_file(out_path,url_list,variable,model,scenario,int(year))
                # area is determined in the "Load shapefiles and plot" part
                if (climate_variable_path is not None):
                    # register data concerning each project under the form of a csv, with the model, scenario, period, latitude and longitude
                    df=register_data(climate_variable_path,name_project,index_dates,dates,experiment,model,df)
                    print('\nValue were found for the period and the project tested\n')
                else:
                    print('\nNo value were found for the period and the project tested\n')
                    continue # do the next for loop
        # test if dataframe is empty, if values exist for this period
    if not df.empty: # if dataframe is not empty, value were registered, the first part is run : a path to register the csv file is created, and the dataframe is registered in a csv file
        full_name = os.path.join(path_for_csv,title_file)
        print(full_name)
        df.to_csv(full_name) # register dataframe in csv file
        return #df,period 
    else: # if the dataframe is empty, no value were found, there is no value to register or to return
        #os.remove(path_for_file)# remove path
        return #df,period# there is no dataframe to return


# In[11]:


def find_path_file(out_path,name_file_list,variable,model,scenario,year):
    name_found = [name for name in name_file_list if scenario in name and model in name and year in name]
    print(name_found)
    if name_found == []:
        return name_found
    path = os.path.join(out_path,name_found[0])
    return path


# In[ ]:




