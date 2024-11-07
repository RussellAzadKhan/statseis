This module is a work in progress.
This module contains functions to statistically analyse seismicity (source parameters e.g. time, location, and magnitude).
Many functions require the renaming of earthquake catalog dataframe columns to: ID, MAGNITUDE, DATETIME, LON, LAT, DEPTH.
This module also includes the code for the mc_Lilliefors method, created and published by Marcus Herman and Warner Marzocchi (https://zenodo.org/records/4162497).
We integrate this code into our workflow, we do not change it in any way. 