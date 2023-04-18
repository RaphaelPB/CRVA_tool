#!/usr/bin/env python
# coding: utf-8

# ### Import python packages

# In[1]:


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


# # Class

# ### Time class

# In[2]:


# class to define parameter of time that remain constant durinf the whole script
class time:
    default_month = [ 
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                ]
    default_day = [
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
                ]
    actual_date = datetime.date.today()
    actual_year = actual_date.year


# ### Map class

# In[11]:


class map_elements:
    parallels = np.arange(-360,360,10) # make latitude lines ever 10 degrees
    meridians = np.arange(-360,360,10) # make longitude lines every 10 degrees


# ### Copernicus class

# In[3]:


## Definition of tuples that will be useful to search which data are available or not
# make it tuples to make unchangeable
class copernicus_elements:
    models =('access_cm2','awi_cm_1_1_mr','bcc_csm2_mr','cams_csm1_0','canesm5_canoe','cesm2_fv2','cesm2_waccm_fv2','cmcc_cm2_hr4','cmcc_esm2','cnrm_cm6_1_hr','e3sm_1_0','e3sm_1_1_eca','ec_earth3_aerchem','ec_earth3_veg','fgoals_f3_l','fio_esm_2_0','giss_e2_1_g','hadgem3_gc31_ll','iitm_esm','inm_cm5_0','ipsl_cm6a_lr','kiost_esm','miroc6','miroc_es2l','mpi_esm1_2_hr','mri_esm2_0','norcpm1','noresm2_mm','taiesm1','access_esm1_5','awi_esm_1_1_lr','bcc_esm1','canesm5','cesm2','cesm2_waccm','ciesm','cmcc_cm2_sr5','cnrm_cm6_1','cnrm_esm2_1','e3sm_1_1','ec_earth3','ec_earth3_cc','ec_earth3_veg_lr','fgoals_g3','gfdl_esm4','giss_e2_1_h','hadgem3_gc31_mm','inm_cm4_8','ipsl_cm5a2_inca','kace_1_0_g','mcm_ua_1_0','miroc_es2h','mpi_esm_1_2_ham','mpi_esm1_2_lr','nesm3','noresm2_lm','sam0_unicon','ukesm1_0_ll')
    experiments = ('ssp1_1_9','ssp1_2_6','ssp4_3_4','ssp5_3_4os','ssp2_4_5','ssp4_6_0','ssp3_7_0','ssp5_8_5')
    experiments_historical=('historical',)


# # Functions

# ### Period for the copernicus function

# In[7]:


################################################ Period for copernicus function ################################################
# Aim of the function: by giving it a first and last year of the period that must analyzed, this function produce several 
# vectors,containing time informations, useful to download and treat data from CMIP6 projections (https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6?tab=overview )
# Those time vectors are used in the copernicus_data and the dataframe_csv_copernicus functions

def year_copernicus(first_year,last_year):
    year = np.arange(first_year,(last_year+1),1) # create vector of years
    year_str = [0]*len(year) # create initiale empty vector to convert years in int
    index = np.arange(0,len(year)) # create vector of index for year
    i = 0 # initialize index
    for i in index: # convert all the date in string format
        year_str[i]=str(year[i])
    return (year, year_str, index)

def date_copernicus(temporal_resolution,year_str):
    if temporal_resolution =='daily':
        start_date = "01-01-"+year_str[0] # string start date based on start year
        stop_date = "31-12-"+year_str[len(year_str)-1] # string stop date based on stop year
        dates = pd.date_range(start_date,stop_date) # vector of dates between start date and stop date
        index_dates = np.arange(0,len(dates)) # vector containning index o dates vector
    if temporal_resolution =='monthly':
        date = np.arange(0,len(time.default_month))
        k=0
        for j in year_str:
            for i in time.default_month:
                dates[k] = i + '-' + j # vector of dates between start date and stop date
        index_dates = np.arange(0,len(dates)) # vector containning index o dates vector
    #if temporal_resolution =='fixed': trouver donnees pour gerer cela
    return (dates, index_dates)


# ### Copernicus function
# Some data comes from copernicus and can be directly taken form the website thans to CDS. The following functions serves this purpose
# #### Parameters of the function :
# projections-cmip6 : name of the web page, in this case, 'projections-cmip6'
# format : zip or tar.gz
# temporal_resolution : daily or monthly or fixed
# SSP : sscenario that is studied "Historical", "SSP1-1.9", "SSP1-2.6" ...
# Variable : variable to be studied
# model: model of projection to choose
# year: year of study to choose
# area: area of study
# month: month to be studied

# In[5]:


################################################### Copernicus data function ###################################################
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
                variables['year']=year # period chosen by the user
                variables['month']= time.default_month  # be default, all the months are given; defined in class time
                if temporal_resolution == 'daily':
                    variables['day']= time.default_day # be default, all the days are given; defined in class time
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


# ### Registering data in dataframe and csv form copernicus CMIP6

# In[6]:


########################################### Register data from nc file of Copernicus ############################################
# Aim of the function: this function aims to register in a dataframe and a csv file the data from the nc file downloaded with
# the function copernicus_data
# Actions of this function
#     1) Create the string indicating the period of interest
#     2) Creating path and file name to register dataframe in csv file
#     3) Register data, with its corresponding experiments and models, in dataframe and csv file
#        3 a) Test if path does not exists (if dataframe is not registered) : 
#                1 . Thanks to copernicus_data, download nc fils from copernicus CMIP6 website for each experiment and each model
#                2 . Open the dowloaded nc file in the jupyter notebook if it exists
#                3 . In a dataframe, register the value in the nc file, for each experiment, model and day
#                4 . if there no value for each experiments and models tested, the datfram is empty and the user is informed
#        3 b) Test if path exists (dataframe is registered) : no need to register again, return in dataframe the existing 
#             csv file in a dataframe

# Parameters of the function

def dataframe_csv_copernicus(temporal_resolution,year_str,experiments,models,out_path,global_variable, name_variable,area):    
    # create string for name of folder depending on type of period
    if temporal_resolution == 'fixed':
        period = 'fixed'
    else:
        period=year_str[0]+'-'+year_str[len(year_str)-1]
        
    (dates, index_dates)=date_copernicus(temporal_resolution,year_str) # create time vector depending on temporal resolution

    title_file = period + '_' + temporal_resolution + '_' +name_variable
    
    path_for_csv = os.path.join('outputs','csv','data',period,name_variable) # create path for csv file
    
    if not os.path.isdir(path_for_csv): # test if the data were already downloaded; if not, first part if the if is applied
        df = pd.DataFrame() # create an empty dataframe

        for SSP in experiments:
            experiment = (SSP,) # create tuple for iteration of dataframe
            print(SSP)
            for model_simulation in models:
                model =(model_simulation,) # create tuple for iteration of dataframe
                print(model)
                # path were the futur downloaded file is registered
                path_for_file= os.path.join(out_path,'Datasets', global_variable, name_variable, SSP, model_simulation,period)#,'')
                # existence of path_for_file tested in copernicus function
                wind_path=copernicus_data(temporal_resolution,SSP,name_variable,model_simulation,year_str,area,path_for_file,out_path)
                # area is determined in the "Load shapefiles and plot" part
                if (wind_path is not None):
                    Open_path = Dataset(wind_path) # open netcdf file
                    lat_dataframe = np.ma.getdata(Open_path.variables['lat']).data
                    lon_dataframe = np.ma.getdata(Open_path.variables['lon']).data
                    data_with_all = ma.getdata(Open_path.variables['sfcWind']).data

                    for moment in index_dates: # case if temporal resolution is daily
                        print('FINAAAAL')
                        data_dataframe = data_with_all[moment,:,:]
                        time = (dates[moment],) # create tuple for iteration of dataframe
                        # Create the MultiIndex
                        midx = pd.MultiIndex.from_product([experiment, model, time, lat_dataframe],names=['Experiment', 'Model', 'Date', 'Latitude'])
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
                else:
                    print("Path does not exist")
                    pass
        # test if dataframe is empty, if values exist for this period
        if not df.empty: # if dataframe is not empty, value were registered, the first part is run : a path to register the csv file is created, and the dataframe is registered in a csv file
            os.makedirs(path_for_csv) # to ensure the creation of the path
            df.to_csv(path_for_csv) # register dataframe in csv file
            return df 
        else: # if the dataframe is empty, no value were found, there is no value to register or to return
            print('No value were found for the period tested')
            return # there is no dataframe to return
    else:# test if the data were already downloaded; if yes, this part of the if is applied
        print('The file was already downloaded')
        df = pd.read_csv(path_for_csv) # read the downloaded data for the analysis
        return df


# ### Display map

# In[12]:


# function to display a map
def Display_map(indexes_lat,indexes_lon,lat,lon,lat_min_wanted,lat_max_wanted,lon_min_wanted,lon_max_wanted,data,title_png,title_to_adapt,label,parallels,meridians):#,projects):

    lon_moz, lat_moz = np.meshgrid(lon, lat) # this is necessary to have a map
    
    # create Map for Mozambique coast
    fig = plt.figure()
    plt.title(title_to_adapt) # title of the map # automatized with year
    map = Basemap(projection ='merc',llcrnrlon=lon_min_wanted+5,llcrnrlat=lat_min_wanted+2,urcrnrlon=lon_max_wanted-5,urcrnrlat=lat_max_wanted-2,resolution='i', epsg = 4326) # projection, lat/lon extents an
    # adding and substracting a quantity to the lon and lat to have a bit of margin when presenting it
    # substracting more to longitude because the range of longitude is -180 to 180. The range of latitude is -90 to 90
    map.drawcountries()
    map.drawcoastlines()
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

    temp = map.contourf(lon_moz,lat_moz,data)
    #projects.plot(ax=ax) # project in projection EPSG:4326
    cb = map.colorbar(temp,"right", size="5%", pad="2%") # color scale, second parameter can be locationNone or {'left', 'right', 'top', 'bottom'}
    cb.set_label(label) # name for color scale
    plt.savefig(os.path.join(out_path,'figures',title_png),format ='png') # savefig or save text must be before plt.show. for savefig, format should be explicity written
    plt.show()


# In[ ]:




