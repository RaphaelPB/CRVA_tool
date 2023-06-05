#!/usr/bin/env python
# coding: utf-8

# This file aims to regroup all function involved in file management

# In[1]:


import os
import os.path
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import time # to measure elasped time
from netCDF4 import Dataset
from timeit import default_timer as timer


# # Download files

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


# # Manage path

# In[3]:


# This functions aims to check if the path is too long, and if yes to deal with it
# this function was created because a bug exist when using python on windows. When the path is too long (more than 250 characters), 
# '\\\\?\\' should be added before the path in order for Windows to understand it 
# (source: https://stackoverflow.com/questions/29557760/long-paths-in-python-on-windows)

# the input is a path in a string format
# the output is the path in a string format
def path_length(str1):
    if len(str1)>250:
        # the path has more than 250 characters
        path = os.path.abspath(str1) # normalize path
        if path.startswith(u"\\\\"):
            path=u"\\\\?\\UNC\\"+path[2:]
        else:
            path=u"\\\\?\\"+path
        return path
    else:
        # the path has less than 250 characters, the path is not too long
        return str1


# # Manage with url of files to download

# In[4]:


# function to extract the name of the file from its url
# the input is an url
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


# In[5]:


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


# # Reading of nc files

# In[6]:


# this function aims to read from a netCDF file the information of interest
# the input is a ....
# the ouputs are vector arrays;
#     first one with the latitudes of the nc file
#     second one with the longitudes of the nc file
#     third one with the time of the nc file
#     fourth one with the variable of interest (for example precipitation) of the nc file

def read_nc_file(Open_path):
    start = timer()
    # use the function find_column_name to find the name of the variable of interest
    start2 = timer()
    name_variable = find_column_name(Open_path) # name_variable is the name of the variable of interest in a string format
    end2 = timer()
    if (end2 - start2) > 1.0 :
        print('Time to execute the function find_column_name: ' + str(end2 - start2)+' seconds')
        print('\n')
    # use the function 'get_data_nc' to have the data from the nc file
    # the function 'get_data_nc' return an array containing the values of interest of the variable indicated as second input
    lat=get_data_nc(Open_path,'lat')
    lon=get_data_nc(Open_path,'lon')
    time=get_data_nc(Open_path,'time')
    #variable=return_NaN(path,name_variable) # this function does not work for the moment
    variable=get_data_nc(Open_path,name_variable)
    end = timer()
    if (end - start) > 1.0 :
        print('Time to execute the function read_nc_file: ' + str(end - start)+' seconds')
        print('\n')
    return lat, lon, time, variable # return arrays


# In[7]:


# this function 'get_data_nc' aims to acces the masked data of the nc files
def get_data_nc(Open_path,name_variable):
    start = timer()
    variable = np.ma.getdata(Open_path.variables[name_variable]).data
    end = timer()
    if (end - start) > 1.0 :
        print('\n')
        print('Time to execute the function get_data_nc for the variable '+ name_variable +': ' + str(end - start)+' seconds')
        print('\n')
    return variable


# In[8]:


# function to return column name in the netCDF file
# all netCDF file form copernicus have this format for their variables names
# ['time', 'time_bnds', 'lat', 'lat_bnds', 'lon', 'lon_bnds', Name of climate variable of interest]
# take of 'time', 'time_bnds', 'lat', 'lat_bnds', 'lon', 'lon_bnds'
def find_column_name(Open_path):
    # make a list with every variables of the netCDF file of interest
    climate_variable_variables=list(Open_path.variables)
    # variables that are not the column name of interest 
    elements_not_climate_var =['time', 'time_bnds', 'bnds','lat', 'lat_bnds', 'lon', 'lon_bnds','time_bounds','bounds','lat_bounds','lon_bounds','height']
    for str in elements_not_climate_var:
        if str in climate_variable_variables:
            climate_variable_variables.remove(str)
    return climate_variable_variables[0]


# In[9]:


# Cette fonction ne fonctionne pas

def return_NaN(Open_path,name_variable):
    variable = get_data_nc(Open_path,name_variable)
    value_NaN = Open_path.variables[name_variable]._FillValue
    import math
    #variable[variable==value_NaN] = math.nan#float('NaN')
    return variable


# ## Reading of nc file: convert vector time from unix to string

# In[10]:


# this function aims to convert the vector of the time vector from Unix time to the format '%Y-%m-%d'
# the function us the 2 following functions 'extract_start_date' and 'time_conversion'
def time_vector_conversion(Open_path,resolution):
    from datetime import date # package to work with date and time
    (year,month,day) = extract_start_date(Open_path) # next function, returning the year, month and day of the start date
    start = date(year,month,day) # the function date, with some number in the format int as input
    # the function date return the date in format 'YYYY-MM-DD'
    time=get_data_nc(Open_path,'time') # get data from the nc file
    time_converted = [] # create an empty list, to register the converted time vector
    # for loop to convert the date from Unix to the format '%Y-%m-%d'
    for day in time:
        # add each converted time in the list 'time_converted'
        time_converted.append(time_conversion(day,start,resolution)) # convert time with the function 'time_conversion'
        # the function the function 'time_conversion'
    return time_converted # return the list with time converted in format '%Y-%m-%d'

# this function aims to extract the year, month and day of the start date
# the input is the path leading to the file of interest
def extract_start_date(Open_path):
    # start date is after the str 'days since ' in units in the path indicated as input of the function
    start_date=Open_path.variables['time'].units.replace('days since ','') # the start_date in string format
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


# In[11]:


def create_xr_array(data,coordonates):
    data_structure = xr.DataArray(data, coords=[times, locs], dims=["time", "space"])


# # Register information from nc files

# In[12]:


# the register_data_in_dataframe function aims to test if data with the specific parameters exist in the folder of concern
# As inputs :
#      the list of urls of the files of interest. The name of the file will be extracted from them
#      temporal_resolution: the temporal resolution of the climate variable in question in string format
#      year_str: a vector containing the year of the period of interest in a string format
#      scenarios: a list of the scenorios of interest in string format
#      models: a list of the models of interest in string format
#      out_path: the out_path in a string format
#      name_variable: the name of the variable of interest (example: 'pr' for precipitation)
#      name_project: the list of names of the project of interest
#      index_closest_lat: array containing an index for each project, 
#                           corresponding to the index of the value in latitude vector which is the closest to 
#                           the project latitude
#      index_closest_lat: array containing an index for each project, 
#                           corresponding to the index of the value in longitude vector which is the closest to 
#                           the project longitude
#      closest_value_lat: array containing a value for each project, corresponding to the value in the 
#                           latitude vector which is the closest to the project's latitude
#      closest_value_lon: array containing a value for each project, corresponding to the value in the 
#                           longitude vector which is the closest to the project's longitude
#      df : empty dataframe to fill

# Outputs are:
#      df: the filled dataframe with the values of interest
#      path_file_not_found: the list of files that were not found with the parameters asked
#      ds_did_not_open: the list of files that could not be read

def register_data_in_dataframe(name_list,temporal_resolution,year_str,scenarios,models,out_path, name_variable, name_project,index_closest_lat,index_closest_lon,closest_value_lat,closest_value_lon,df):    
    path_file_not_found = [] # create empty list to register names of files that were not found with the corresponding parameters
    ds_did_not_open = [] # create empty list to register names of files that couldn't be opened
    for year in year_str:
        for SSP in scenarios:
            for model_simulation in models:
                # for each year, each scenarios and each models, test if there is a corresponding file existing
                # with function 'find_path_file'
                climate_variable_path = find_path_file(out_path,name_list,name_variable,temporal_resolution,model_simulation,SSP,year,'r1i1p1f1_gn')
                if climate_variable_path!= []:
                    # a file with the corresponding parameters were found
                    ds =  xr.open_dataset(climate_variable_path) # open the file corresponding to the parameters
                    time = ds.indexes['time'].strftime('%d-%m-%Y').values # register the time in the file
                    for i in np.arange(0,len(name_project)):
                        print('For the year '+year+' and project '+name_project[i]+', test with scenario '+SSP+', with model '+model_simulation)
                        try: # to register information from the dataset ds in the dataframe df
                            # for each year, scenarios, models and each project, the values of the opened dataset ds
                            # are registered in the empty dataframe df, to a specific place corresponding to the parameters of the loop
                            df.loc[(name_project[i],SSP,model_simulation,time,closest_value_lat[i]),('Longitude',closest_value_lon[i])] = ds.pr.isel(lat=index_closest_lat[i],lon=index_closest_lon[i]).values
                        except: # the dataset ds can not be read
                            # add information of the dataset that can't be read in the empty list ds_did_not_open
                            ds_did_not_open.append(climate_variable_path+'_'+'project_'+name_project[i])
                            continue # try with next project
                    ds.close() # the opened dataset is closed to spare memory
                else:
                    # NO file with the corresponding parameters were found
                    # add information of the missing file in the empty list path_file_not_found
                    path_file_not_found.append(name_variable+'_'+temporal_resolution+'_'+model_simulation+'_'+SSP+'_'+year+'_'+'r1i1p1f1_gn')
                    continue # try another file
    return df,path_file_not_found,ds_did_not_open

# the function df_to_csv aims to return the filled dataframe in a csv format
# Inputs are:
#       df: the dataframe that should be register in a csv file
#      path_for_csv: this is the path where the csv file should be registered, in a string format
#      title_file: this is the name of the csv file to be created in a string format
#                  CAREFUL --> title_file MUST have the extension of the file in the string (.csv for example)
# Output is:
#      in the case where the dataframe is not empty, the ouput is the full path to the created csv file
#      in the case where the dataframe is empty, the output is an empty list
def df_to_csv(df,path_for_csv,title_file):
    # test if dataframe is empty, if values exist for this period
    if not df.empty: 
        # if dataframe is not empty, value were registered, the first part is run : 
        # a path to register the csv file is created, .....
        if not os.path.isdir(path_for_csv):
            # the path to the file does not exist
            os.makedirs(path_for_csv) # to ensure creation of the folder
            # creation of the path for the csv file, in a string format
        full_name = os.path.join(path_for_csv,title_file)
        # ..... and the dataframe is registered in a csv file
        df.to_csv(full_name) # register dataframe in csv file
        print('Path for csv file is: ' + full_name)
        return full_name # return the full path that leads to the created csv file
    else: # if the dataframe is empty, no value were found, there is no value to register or to return
        print('The dataframe is empty')
        return []


# In[13]:


## this function is used in register_data. The function aims to select in a vector, elements between the values min_vector 
# and max_vector
# in the register_data function, the function 'find_index_project' is used to find the index of the latitudes and longitude 
# vectors elements which are between min_vector and max_vector values. this allows to select the latitudes and longitudes around
# the projects
# the input of the function are:
# a vector; in register_data, it is the latitude or the longitude vector of the file studied
# the minimal value of the interval of interest
# the maximal value of the interval of interest
# the output is;
# the indexes of the vector elements, which values are between min_vector and max_vector
def find_index_project(vector,min_vector,max_vector):
    # values of vector between min_vector and max_vector
    vector_project =[element for element in vector if element > min_vector and element < max_vector]
    index_item =[] # prepare empty list to register the indexes of the values of vector whice are between min_vector
    # and max_vector
    # for loop, to register the indexes of the values of vector whice are between min_vector and max_vector
    for item in vector_project:
        # values in vector are compared to vector_project values 
        index_item.append(np.where(vector == item)[0][0])
        # the function np.where return the indexes of vector, where the values of vector are equal to item
    return index_item # a vector containing the indexes of the vectors elements, which are between min_vector and max_vector


# In[14]:


# this function is used in 'create_dataframe'. The function aims to return the path of the file of interest
# The function looks into a list of name which name in the list has every input 
# The inputs are:
#    out_path: a general file path where the files are registered, 
#    name_file_list: a list of files' names
#    variable: the name of the variable of interest
#    model: the model of interest (example: ACCESS-CM2)
#    scenario: the scenario of interest (example:ssp245)
#    year: the year of interest
#    ensemble: the ensemble of interest (example: r1i1p1f1_gn)
# the output is:
#    the path of the file corresponding to all the parameters indicated in input

def find_path_file(out_path,name_file_list,variable,temporal_resolution,model,scenario,year,ensemble):
    # look into the list of names if find a name with every parameter indicated in inputs
    name_found = [name for name in name_file_list if scenario in name and model in name and year in name and ensemble in name and temporal_resolution in name]
    print('The name of the file is ' + name_found[0])
    if name_found == []:
        # no name with all the parameters indicated as inputs was found
        return name_found # return an empty element instead of a path, the function does not run the following lines
    # the name was found, so prepare the path of the file of interest
    path = os.path.join(out_path,name_found[0])
    return path # return the path of the file of interest


# In[15]:


# this function aims to select the closest point to the geographical point of the project
# the function takes as input 
#     location_project, which is a numpy.float64
#     vector, which is a numpy.ndarray
# the function returns
#     closest_value[0], a numpy.float64

def closest_lat_lon_to_proj(location_project,vector):
    # the function any() returns a boolean value. Here, the function test if there are elements in the array 
    # containing the difference between the vector and the location_project, equal to the minimum of the absolute 
    # value of the difference between the vector and the location_project
    if any(np.where((vector - location_project) == min(abs(vector - location_project)))[0]):
        # the function any() returned True
        # there is an element in the vector that is equal to the minimum of the absolute value of the difference 
        # between the vector and the location_project
        
        # the function np.where() returns the index for which (vector - location_project) == min(abs(vector - location_project))
        index_closest = np.where((vector - location_project) == min(abs(vector - location_project)))[0]
        closest_value = vector[index_closest]
    else:
        # the function any() returned False
        # there is NO element in the vector that is equal to the minimum of the absolute value of the difference 
        # between the vector and the location_project
        
        # the function np.where() returns the index for which (vector - location_project) == -min(abs(vector - location_project))
        index_closest = np.where((vector - location_project) == -min(abs(vector - location_project)))[0]
        closest_value = vector[index_closest]
    return index_closest, closest_value 
    # the function returns
    #     first, the value of the index of the element of vector, that is the closest to location_project    
    #     second, the array containing the element of vector, that is the closest to location_project


# In[16]:


# this function aims to create the empty dataframe that will be filled

def create_empty_dataframe(name_project,scenarios,models,time,closest_value_lat,closest_value_lon):
    df = pd.DataFrame()
    for i in np.arange(0,len(name_project)):
        midx = pd.MultiIndex.from_product([(name_project[i],),scenarios, models, time, (closest_value_lat[i],)],names=['Name project','Experiment', 'Model', 'Date', 'Latitude'])
        cols = pd.MultiIndex.from_product([('Longitude',),(closest_value_lon[i],)])
        Variable_dataframe = pd.DataFrame(data = [], 
                                    index = midx,
                                    columns = cols)
        df = pd.concat([df,Variable_dataframe])
    return df


# In[17]:


# this functions aims to regroup all the scenarios, models, time_aggregation and variables in vectors
# the function use the function 'data_information'

def information_files_in_vectors(name_list):
    variables= []
    time_aggregations= []
    models= []
    scenarios= []
    for file_name in name_list:
        (variable, time_aggregation, model, scenario, year) = data_information(file_name) 
        # use function data_information to find information concerning the file_name
        if variable not in variables:
            variables.append(variable)
        if time_aggregation not in time_aggregations:
            time_aggregations.append(time_aggregation)
        if model not in models:
            models.append(model)
        if scenario not in scenarios:
            scenarios.append(scenario)
    return variables, time_aggregations,models,scenarios


# In[18]:


# this functions aims to return the time, latitudes and longitudes of the files of concern
def _lat_lon(path,lat_projects,lon_projects):
    ds =  xr.open_dataset(path) 
    # ds.indexes['time'] gives back CFTimeIndex format, with hours. The strftime('%d-%m-%Y') permits to have time 
    # as an index, with format '%d-%m-%Y'. The .values permits to have an array
    lat  = ds.lat.values
    lon  = ds.lon.values
    ds.close() # to spare memory
    # preallocate space for the future vectors
    index_closest_lat = []
    index_closest_lon = []
    closest_value_lat = []
    closest_value_lon = []
    for j in np.arange(0,len(lat_projects)):
        (A,B)=closest_lat_lon_to_proj(lat_projects[j],lat)
        index_closest_lat.append(A[0])
        closest_value_lat.append(B[0])
        (C,D)=closest_lat_lon_to_proj(lon_projects[j],lon)
        index_closest_lon.append(C[0])
        closest_value_lon.append(D[0])
    return index_closest_lat,index_closest_lon,closest_value_lat,closest_value_lon


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




