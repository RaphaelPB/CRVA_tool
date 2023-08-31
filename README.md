# CRVA_tool
 tool to perform Climate Risk and Vulnerability Analyses

Gitignore keeps from other viewers every file in folder 'outputs', tempfile and cdsapirc
Quick intro. Put European mthod link
# Summary of ReadME
- [Functions](#Function)
- [Download and format data](#DownloadAndFormatData)
- [Bias correction](#BiasCorrection)
- [Application for the CRVA tool](#ApplicationCRVA)
	- [Import data](#ImportData)
	- [Indicators](#Indicator)
	- [Vulnerability](#Vulnerability)
		- [Sensitivity](#Sensitivity)
		- [Exposure](#Exposure)
		- [Determination of vulnerability](#DetVulnerability)
	- [Risk](#Risk)
- [StudyCase-Gorongosa_Mozambique](#StudyCaseGorongosa)
	- [Observed data](#StudyCaseObservedData)
	- [Validation of data](#ValidationData)
	- [Bias correction results](#BCResults)
- [What to still implement](#ToImplement)
- [*Commands* and explanations](#Commands)
	- [Packages installed](#Packages)
	- [Resolving errors](#ResolvingErrors)
<a id='Function'></a>
# Functions
This folder contains the functions in the different codes.
<a id=DownloadAndFormatData></a>
# Download and format data
## Modelled data
NEX-GDDP-CMIP6 dataset is [produced by NASA](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6), by bias correcting and downscaling data from [CMIP6](https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6?tab=overview). The dataset contains 9 climate variables, for several experiments and models ([Detailed paper about the dataset](https://www.nature.com/articles/s41597-022-01393-4)).

For this tool, NEX-GDDP-CMIP6 data were downloaded with [Download_NEX-GDDP-CMIP6.py](https://github.com/RaphaelPB/CRVA_tool/blob/main/1-DownloadAndFormatData/Download_NEX-GDDP-CMIP6.py), using the csv file made available by NASA [here page 16](https://www.nccs.nasa.gov/sites/default/files/NEX-GDDP-CMIP6-Tech_Note.pdf).

Once the data are downloaded, need to reformate them in csv files with [CSV_NEX-GDDP-CMIP6_one_lat_lon] https://github.com/RaphaelPB/CRVA_tool/blob/main/1-DownloadAndFormatData/CSV_NEX-GDDP-CMIP6_one_lat_lon.py).
<a id=BiasCorrection></a>
# Bias correction
For this tool, bias correction applied is BCSD method ([Wood 2004](https://link.springer.com/article/10.1023/B:CLIM.0000013685.99609.9e)), implemented thanks to [scikit-downscale package](https://github.com/pangeo-data/scikit-downscale). In the end, only the quantile mapping step is performed, and not the downscaling. The [BC_NEX-GDDP-CMIP6](https://github.com/RaphaelPB/CRVA_tool/blob/main/2-BiasCorrection/BC_NEX-GDDP-CMIP6.ipynb) ([folder 2-BiasCorrection](https://github.com/RaphaelPB/CRVA_tool/tree/main/2-BiasCorrection)) applies functions BCSD_Temperature_return_anoms_to_apply and BCSD_Precipitation_return_anoms_to_apply from [Bias_correction_function](https://github.com/RaphaelPB/CRVA_tool/blob/main/0-Functions/Bias_correction_function.ipynb) ([folder 0-Functions](https://github.com/RaphaelPB/CRVA_tool/tree/main/0-Functions)) on respectively temperature and precipitation NEX-GDDP-CMIP6 dataset.

In folder [Archives-BiasCorrectionTests](https://github.com/RaphaelPB/CRVA_tool/tree/main/Archives-BiasCorrectionTests), other bias correction tests are gathered.
<a id=ApplicationCRVA></a>
# Application for the CRVA tool
Application of the method is all in [CRVA_data_analyst](https://github.com/RaphaelPB/CRVA_tool/blob/main/3-Tool/CRVA_data_analyst.ipynb). The application of the tool was done with data from the Gorongosa study.
<a id=ImportData></a>
## Import data
Before performing any indicators, data are imported thanks to functions in [Functions_ImportData](https://github.com/RaphaelPB/CRVA_tool/blob/main/0-Functions/Functions_ImportData.ipynb) ([folder 0-Functions](https://github.com/RaphaelPB/CRVA_tool/tree/main/0-Functions)).
<a id=Indicator></a>
## Indicators
In folder [0-Functions](https://github.com/RaphaelPB/CRVA_tool/tree/main/0-Functions), [Functions_Indicators](https://github.com/RaphaelPB/CRVA_tool/blob/main/0-Functions/Functions_Indicators.ipynb) contains the indicators that can be used.
Evaluate the evolution of Net Precipitation would be useful for some infrastructure. the work was started and is in the folder [InProcess-NetPrecipitation](https://github.com/RaphaelPB/CRVA_tool/tree/main/InProcess-NetPrecipitation).
<a id=Vulnerability></a>
## Vulnerability
<a id=Sensitivity></a>
### Sensitivity
Sensitivity is based often on expert judgement. It is evaluated outside of the tool. It should be summarized in a matrix format (as below), and added to the tool with the function ... .
<a id=Exposure></a>
### Exposure
Two sets of data are compared; one from the past and form the future.
- Calculate statistics of the period
- Change between past and future statistics
- Based on change between past and future, categorization in low, medium, or high Exposure
<a id=DetVulnerability></a>
### Determination of vulnerability
Crossing information from sensitivity and exposure with function , the final vulnerability to climate variables can be known with function ().
<a id=Risk></a>
## Risk
Severity not applied in the tool. But likelihood yes.
<a id=StudyCaseGorongosa></a>
# StudyCase-Gorongosa_Mozambique
<a id=StudyCaseObservedData></a>
## Observed data
For the study case in Gorongosa, the tool uses some observation data from NOAA. Some of them were absurd and were therefore treated with [Treat Data tas NOAA Station and pr meteorological station Gorongosa](https://github.com/RaphaelPB/CRVA_tool/blob/main/4-StudyCase-Gorongosa_Mozambique/Treat%20Data%20tas%20NOAA%20Station%20and%20pr%20meteorological%20station%20Gorongosa.ipynb).
<a id=ValidationData></a>
## Validation of data
To confirm the use of NEX-GDDP-CMIP6 dataset as representative of the location of interest, modelled temperature and precipitation were compared to observation data in [Compare NOAA station and NEXGDDP CMIP6 data in Chimoio](https://github.com/RaphaelPB/CRVA_tool/blob/main/4-StudyCase-Gorongosa_Mozambique/Compare%20NOAA%20station%20and%20NEXGDDP%20CMIP6%20data%20in%20Chimoio.ipynb).
<a id=BCResults></a>
## Bias correction results
BCSD is a bias correction method ([Wood 2004](https://link.springer.com/article/10.1023/B:CLIM.0000013685.99609.9e)), performing first quantile mapping, and then downscaling. It was performed on NEX-GDDP-CMIP6 at Gorongosa location with the package [scikit-downscale](https://github.com/pangeo-data/scikit-downscale), with its functions BcsdPrecipitation and BcsdTemperature. Results were compared with observation data from NOAA website in [Compare NOAA station and BC NEXGDDP CMIP6 data](https://github.com/RaphaelPB/CRVA_tool/blob/main/4-StudyCase-Gorongosa_Mozambique/Compare%20NOAA%20station%20and%20BC%20NEXGDDP%20CMIP6%20data.ipynb).
<a id=ToImplement></a>
# What to still implement
ouverture du rapport + risk concomitant
<a id=Commands></a>
# *Commands* and explanations
Environment initially created as following .*conda create -n geodata -c conda-forge python=3.10.6 geopandas=0.12.1 pandas=1.5. pysheds=0.3.3 rasterstats=0.17.0 rasterio=1.3.3 numpy=1.23.4 seaborn matplotlib netcdf4=1.6.1*
<a id=Packages></a>
## Packages installed: 
- *pip install jupyter notebook*: to use this environment in jupyter notebook
- *pip install numpy matplotlib*: to use those 2 packages on the environment. Use for many different purposes (numbers and plots)
- *pip3 install -U matplotlib* [upgrade of matplotlib to have access to colors for matplotlib](https://stackoverflow.com/questions/47497097/module-matplotlib-has-no-attribute-colors)
- *python -m pip install rioxarray*
- *conda install cdsapi*: to download copernicus data
- *conda install basemap*, then pip install basemap-data: to map nc files (more infos on [stackoverflow 1](https://stackoverflow.com/questions/33020202/how-to-install-matplotlibs-basemap) and [stackoverflow 2](https://stackoverflow.com/questions/47587670/how-to-install-basemap-in-jupyter-notebook))
- *pip install bias-correction*: to perform bias correction, infos on module [here](https://pankajkarman.github.io/bias_correction/index.html#bias_correction.gamma_correction) 
- *pip install python-cmethods*: other package to perform Bias correction ([python-cmethods Github here](https://github.com/btschwertfeger/python-cmethods#installation))
- *conda install --channel conda-forge pysal*: to map projects on a map ([Pysal library](http://pysal.org/pysal/) and [Installation Pysal](http://pysal.org/pysal/installation.html))
- *conda install -c conda-forge r-nasaaccess*: to have access to have access to climate and earth observation data, but not use in the end ([Github scikit downscale](https://github.com/pangeo-data/scikit-downscale))
- Attempt to install scikit learn and scikit-downscale, install all the following dependencies, but package scikit-downscale still not working : scipy, scikit-learn, dask, docopt, zarr   , ipython, sphinx, numpydoc, sphinx_rtd_theme, pangeo-notebook, mpl-probscale, pydap, gcsfs, pwlf, sphinx-gallery, mlinsights. In the end, clone scikit downscale repository from the [website of the package](https://github.com/pangeo-data/scikit-downscale). This worked
- *pip install h5netcdf*: to manage those types of files ([Gitbud h5netcdf](https://github.com/h5netcdf/h5netcdf))
- *pip install geopy*: to calculate distance between two geographical points ([stackoverflow explanation of geopy](https://github.com/h5netcdf/h5netcdf) and [python library geopy](https://pypi.org/project/geopy/))
- *pip install Nio* :, to increase speed of reading NEtCDF files ([stackoverflow explanation](https://stackoverflow.com/questions/34159747/efficient-reading-of-netcdf-variable-in-python))
<a id=ResolvingErrors></a>
## Resolving errors: 
- Data attribute error: Upgrade pandas to deal with it ([stackoverflow explanations](https://stackoverflow.com/questions/67165659/how-to-update-pandas-for-jupyter-notebook))
- Problem ValueError: did not find a match in any of xarray's currently installed IO backends. To resolve that:
*python -m pip install xarray*
*python -m pip install "xarray[io]"*
*python -m pip install git+https://github.com/pydata/xarray.git*
- Raising error RuntimeWarning: Engine 'cfgrib' loading failed:
try to install *conda install -c conda-forge python-eccodes*, did not work, conflicting packages apparently
try *pip install ecmwflibs*, with import ecmwflibs import eccodes in the script  installed, worked

