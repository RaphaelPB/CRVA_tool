#!/usr/bin/env python
# coding: utf-8

# This file aims to regroup all function involved in file management

# In[19]:


import os
import os.path
import numpy as np
import numpy.ma as ma
import pandas as pd
from netCDF4 import Dataset


# In[20]:


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


# In[21]:


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


# In[22]:


def read_nc_file(path):
    name_variable = find_column_name(path)
    
    #df=Dataset(path)
    
    lat=get_data_nc(path,'lat')
    lon=get_data_nc(path,'lon')
    time=get_data_nc(path,'time')
    variable=return_NaN(path,name_variable)
    
    return lat, lon, time, variable


# In[23]:


def get_data_nc(path,name_variable):
    variable = np.ma.getdata(Dataset(path).variables[name_variable]).data
    return variable


# In[24]:


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


# In[25]:


# Cette fonction ne fonctionne pas

def return_NaN(path,name_variable):
    variable = get_data_nc(path,name_variable)
    value_NaN = Dataset(path).variables[name_variable]._FillValue
    import math
    #variable[variable==value_NaN] = math.nan#float('NaN')
    return variable


# In[26]:


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


# In[27]:


def create_xr_array(data,coordonates):
    data_structure = xr.DataArray(data, coords=[times, locs], dims=["time", "space"])


# In[28]:


# the dataframe_copernicus functions aims to test if the data with the specific parameters exists (with copernicus_data)
# and then produce a csv file if the data exists

def create_dataframe(url_list,temporal_resolution,year_str,experiments,models,out_path, name_variable, name_project,area_projects,period,title_file):    
    path_for_csv = os.path.join(out_path,'csv_file',name_variable+period)
    df = pd.DataFrame() # create an empty dataframe
    for year in year_str:
        print('Year '+year)
        for SSP in experiments:
            #experiment = (SSP,) # create tuple for iteration of dataframe
            print('Test with scenario '+SSP)
            for model_simulation in models:
                #model =(model_simulation,) # create tuple for iteration of dataframe
                print('Test with model '+model_simulation)
                # path were the futur downloaded file is registered
                #path_for_file= os.path.join(out_path,'Datasets','NEX-GDDP-CMIP6',name_variable,name_project,SSP,model_simulation,period)
                # existence of path_for_file tested in copernicus function
                # climate_variable_path=copernicus_data(temporal_resolution,SSP,name_variable,model_simulation,year_str,area,path_for_file,out_path,name_project,source)
                climate_variable_path = find_path_file(out_path,url_list,name_variable,model_simulation,SSP,year,'r1i1p1f1_gn')
                #df=register_data(climate_variable_path,temporal_resolution,name_project,area_projects,SSP,model_simulation,df)
                # area is determined in the "Load shapefiles and plot" part
                if climate_variable_path != []:
                    # register data concerning each project under the form of a csv, with the model, scenario, period, latitude and longitude
                    df=register_data(climate_variable_path,temporal_resolution,name_project,area_projects,SSP,model_simulation,df)
                    
                    #print('\nValue were found for the period and the project tested\n')
                    #return
                else:
                    print('\nNo value were found for the period and the project tested\n')
                    continue # do the next for loop
        # test if dataframe is empty, if values exist for this period
    if not df.empty: # if dataframe is not empty, value were registered, the first part is run : a path to register the csv file is created, and the dataframe is registered in a csv file
        full_name = os.path.join(path_for_csv,title_file)
        print(full_name)
        df.to_csv(full_name) # register dataframe in csv file
        #return #df,period 
    #else: # if the dataframe is empty, no value were found, there is no value to register or to return
        #os.remove(path_for_file)# remove path
        #return #df,period# there is no dataframe to return


# In[34]:


# register data concerning each project under the form of a csv, with the model, scenario, period, latitude and longitude
def register_data(climate_variable_path,temporal_resolution,name_project,area_projects,experiment,model,df):
    print('Registering the data in a dataframe')
    experiment=(experiment,)
    model=(model,)
    Open_path = Dataset(climate_variable_path) # open netcdf file
    lat_dataframe = np.ma.getdata(Open_path.variables['lat']).data
    lon_dataframe = np.ma.getdata(Open_path.variables['lon']).data
    time_dataframe = time_vector_conversion(climate_variable_path,temporal_resolution)
    #return lat_dataframe,lon_dataframe
    column_name = find_column_name(climate_variable_path)
    data_with_all = ma.getdata(Open_path.variables[column_name]).data
    
    index_lat_project=find_index_project(lat_dataframe,area_projects[0],area_projects[1])
    index_lon_project=find_index_project(lon_dataframe,area_projects[2],area_projects[3])
    lat_dataframe =lat_dataframe[index_lat_project]
    lon_dataframe =lon_dataframe[index_lon_project]

    data_with_all=data_with_all[:,index_lat_project,:]
    data_with_all=data_with_all[:,:,index_lon_project]
    print(data_with_all)
    
    for (moment,index_moment) in zip(time_dataframe,np.arange(0,len(time_dataframe)-1)): # case if temporal resolution is daily
        data_dataframe = data_with_all[index_moment,:,:]
        #print(data_dataframe)
        Date = (moment,) # create tuple for iteration of dataframe
        Name_Project = (name_project,)

        # Create the MultiIndex
        midx = pd.MultiIndex.from_product([Name_Project,experiment, model, Date, lat_dataframe],names=['Name project','Experiment', 'Model', 'Date', 'Latitude'])
        # multiindex to name the columns
        lon_str = ('Longitude',)
        cols = pd.MultiIndex.from_product([lon_str,lon_dataframe])
        # Create the Dataframe
        Variable_dataframe = pd.DataFrame(data = data_dataframe, 
                                    index = midx,
                                    columns = cols)
        # Concatenate former and new dataframe
        df = pd.concat([df,Variable_dataframe])# register information for project

    Open_path.close # to spare memory
    return df


# In[30]:


def find_index_project(vector,min_vector,max_vector):
    vector_project =[element for element in vector if element > min_vector and element < max_vector]
    index_item =[]
    for item in vector_project:
        index_item.append(np.where(vector == item)[0][0])
    return index_item


# In[31]:


def find_path_file(out_path,name_file_list,variable,model,scenario,year,ensemble):
    name_found = [name for name in name_file_list if scenario in name and model in name and year in name and ensemble in name]
    print(name_found)
    if name_found == []:
        return name_found
    path = os.path.join(out_path,name_found[0])
    return path


# In[32]:


# function to extract the name of the file from its url
# the imput is an url
def extract_name_file(url):
    index_before_name=url.rfind('/') # returns the highest index where the last character '/' was found, which is just before the name of the file    
    name = url[index_before_name+1:len(url)] # return the name of the file as a string, with the suffix '.nc'
    return name

# function 'produce_name_list' produce a list of files' name, with the suffix '.nc'
# 'produce_name_list' use the function 'extract_name_file' to have the name of a file from its url
# the input is a list of url, from which we want to extract the corresponding names of files
def produce_name_list(url_list):
    name_list=[] # create empty list
    for file in url_list:
        f_name = extract_name_file(file) # return the name of the file as a string, with the suffix '.nc'
        name_list.append(f_name) # add extracted name in the list
    return name_list # return the list of names in the url_list


# In[33]:


## those three function are used to have the information concerning a file
## information are in the name of the file, so the name of the file is used to find its related information
## information mean variable, time_aggregation, model, scenario, year of the file

### this function permit to extract the word before the first character '_' in the input 'name'
### the input name is in format str
### returning the new_name, without the word found, will permit to re-use the function to find all 
#     the information concerning the studied file
def name_next_boundary(name):
    index_before_name=name.find('_') # returns the lowest index where the character '_' was found
    word = name[0:index_before_name] # first word in the string 'name', before the first character '_'
    new_name = name.replace(word+'_','') # delete the word found from the string 'name'
    return word, new_name # return, in string format, the word found (which is an information of the studied file), 
                    # and the string 'new_name', which is 'name' without the word found

# this function permit to extract the year of the studied file
# the year is always writen at the end of the name's file
# the input name is in format str
def find_year(name):
    index_before_name=name.rfind('_') # returns the highest index where the character '_' was found
    # the last character '_' is just before the year in the string 'name'
    # determine if the string 'name' ends with '.nc'
    if name.endswith('.nc'):
        # 'name' ends with '.nc'
        name_end = 3 # the three last character of the string name will be removed to find the year of the studied file
    else:
        # 'name' does not end with '.nc'
        name_end = 0 # no character will be removed at the end of 'name' to find the year of the studied file
    year = name[index_before_name+1:len(name)-name_end] # the year is extracted from the name of the file studied
    # based on the index_before_name (highest index where the character '_' was found) and the suffix of 'name'
    return year # the year in string format is returned

# This function use the functions 'name_next_boundary' and 'find_year' to extract the information of the file studied
# the input name is in format str, the name of the file from which we want information
def data_information(name):
    #### use of the function 'name_next_boundary': each time it is used, 
    # returns an information, and the name of the studied file without this information
    (variable, shorten_name) = name_next_boundary(name)
    (time_aggregation, shorten_name) = name_next_boundary(shorten_name)
    (model, shorten_name) = name_next_boundary(shorten_name)
    (scenario, shorten_name) = name_next_boundary(shorten_name)
    #### use the function 'find_year' to extract the information 'year' from the string 'shorten_name'
    year = find_year(shorten_name)
    # the function returns all the information of the studied file
    return variable, time_aggregation, model, scenario, year


# In[ ]:





# In[ ]:




