# CRVA_tool
 tool to perform Climate Risk and Vulnerability Analyses

Gitignore keeps from other viewers every file in folder 'outputs', tempfile and cdsapirc
# *Commands* and explanations
Environment initially created as following .*conda create -n geodata -c conda-forge python=3.10.6 geopandas=0.12.1 pandas=1.5. pysheds=0.3.3 rasterstats=0.17.0 rasterio=1.3.3 numpy=1.23.4 seaborn matplotlib netcdf4=1.6.1*
## Packages installed: 
- *pip install jupyter notebook*: to use this environment in jupyter notebook
- *pip install numpy matplotlib*: to use those 2 packages on the environment. Use for many different purposes (numbers and plots)
- *pip3 install -U matplotlib* [upgrade of matplotlib to have access to colors for matplotlib](https://stackoverflow.com/questions/47497097/module-matplotlib-has-no-attribute-colors)
- *python -m pip install rioxarray*
- *conda install cdsapi*: to download copernicus data
- *conda install basemap*, then pip install basemap-data: to map nc files (more infos on [stackoverflow 1](https://stackoverflow.com/questions/33020202/how-to-install-matplotlibs-basemap) and [stackoverflow 2](https://stackoverflow.com/questions/47587670/how-to-install-basemap-in-jupyter-notebook))
- *pip install bias-correction*: to perform bias correction, infos on module [here](https://pankajkarman.github.io/bias_correction/index.html#bias_correction.gamma_correction) 
- *pip install python-cmethods*: other package to perform Bias correction. ([python-cmethods Github here](https://github.com/btschwertfeger/python-cmethods#installation))
- *conda install --channel conda-forge pysal*: to map projects on a map ([Pysal library](http://pysal.org/pysal/) and [Installation Pysal](http://pysal.org/pysal/installation.html))
- *conda install -c conda-forge r-nasaaccess*: to have access to have access to climate and earth observation data, but not use in the end ([Github scikit downscale](https://github.com/pangeo-data/scikit-downscale))
- Attempt to install scikit learn and scikit-downscale, install all the following dependencies, but package scikit-downscale still not working : scipy, scikit-learn, dask, docopt, zarr   , ipython, sphinx, numpydoc, sphinx_rtd_theme, pangeo-notebook, mpl-probscale, pydap, gcsfs, pwlf, sphinx-gallery, mlinsights. In the end, clone scikit downscale repository from the [website of the package](https://github.com/pangeo-data/scikit-downscale). This worked
- *pip install h5netcdf*: to manage those types of files ([Gitbud h5netcdf](https://github.com/h5netcdf/h5netcdf))
- *pip install geopy*: to calculate distance between two geographical points ([stackoverflow explanation of geopy](https://github.com/h5netcdf/h5netcdf) and [python library geopy](https://pypi.org/project/geopy/))
- *pip install Nio* :, to increase speed of reading NEtCDF files ([stackoverflow explanation](https://stackoverflow.com/questions/34159747/efficient-reading-of-netcdf-variable-in-python))
## Resolving errors: 
- Data attribute error: Upgrade pandas to deal with it ([stackoverflow explanations](https://stackoverflow.com/questions/67165659/how-to-update-pandas-for-jupyter-notebook))
- Problem ValueError: did not find a match in any of xarray's currently installed IO backends. To resolve that:
*python -m pip install xarray*
*python -m pip install "xarray[io]"*
*python -m pip install git+https://github.com/pydata/xarray.git*
- Raising error RuntimeWarning: Engine 'cfgrib' loading failed:
try to install *conda install -c conda-forge python-eccodes*, did not work, conflicting packages apparently
try *pip install ecmwflibs*, with import ecmwflibs import eccodes in the script  installed, worked
