"""
Module with functions to anaylse seismicity
"""

import datetime as dt
import math
import random
from decimal import Decimal
from math import asin, cos, radians, sin, sqrt
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.special as special
import scipy.stats as stats
import statsmodels.api as sm
# import utm
from scipy.special import factorial
from scipy.stats import gamma, ks_2samp, kstest, nbinom, poisson, truncnorm
import pyproj
import glob
from IPython.display import clear_output, Image
from functools import reduce
from sklearn import preprocessing

def convert_extent_to_epsg3857(extent):
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x_min, y_min = transformer.transform(extent[0], extent[2])
    x_max, y_max = transformer.transform(extent[1], extent[3])
    return [x_min, x_max, y_min, y_max]

def find_event_in_catalog(ID, catalog):
    return catalog.loc[catalog['ID']==ID]

def estimate_axis_labels(array, n_labels=5):
    """
    Failed attempt to automaticallly generate better axis labels than matplotlib
    """
    min, max = np.nanmin(array), np.nanmax(array) 
    range = round(max - min)
    str_num = str(range)
    str_len = len(str_num)
    step = int(str_num[0] + '0'*(str_len-1))/2
    min, max = math.floor(min/step)*step, math.ceil(max/step)*step
    print(min, max, step)
    return np.arange(min, max+step, step)

def get_bins(numbers, nearest=10):
    """
    Returns optimal bins for plotting a histogram.
    """
    numbers = np.array(numbers)
    min = math.ceil(np.nanmin(numbers)/nearest)*nearest
    max = math.floor(np.nanmax(numbers)/nearest)*nearest
    bins = np.arange(min-nearest, max+(nearest*2), nearest)    
    return bins


def magnitude_to_moment(magnitude):
    """
    Covert moment magnitude to seismic moment
    """
    moment = 10**(1.5*magnitude+9.05)
    return moment


def string_to_datetime(list_of_datetimes):
    """
    Turn datetimes from string into datetime objects
    """
    Datetime = pd.to_datetime(list_of_datetimes,
                              format = '%Y-%m-%d %H:%M:%S')
    return Datetime


def string_to_datetime_df(dataframe):
    """
    Find DATETIME column in df and change to datetime objects
    """
    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'],
                                           format = '%Y-%m-%d %H:%M:%S')
    
def string_to_datetime_return(dataframe):
    """
    Find DATETIME column in df and change to datetime objects
    Returns dataframe so function can be mapped
    """
    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'],
                                           format = '%Y-%m-%d %H:%M:%S')
    return dataframe


def datetime_to_decimal_days(DATETIMES):
    """
    Durn datetime objects to decimal days
    """
    decimal_days = (DATETIMES - DATETIMES.iloc[0]).apply(lambda d: (d.total_seconds()/(24*3600)))
    return decimal_days


def datetime_to_decimal_year(timestamps):
    """
    Turn datetime objects to decimal years
    """
    decimal_years = timestamps.apply(lambda x: x.year + (x - dt.datetime(year=x.year, month=1, day=1)).total_seconds()/24/3600/365.25)
    return decimal_years


def reformat_catalogue(df):
    """
    standardise column names, must feed in dataframe with columns: ['ID', 'MAGNITUDE', 'DATETIME', 'DEPTH', 'LON', 'LAT']
    """
    df.columns = ['ID', 'MAGNITUDE', 'DATETIME', 'DEPTH', 'LON', 'LAT'] 
    return df


# chat GPT function
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r


def restrict_catalogue_geographically(df, region):
        """
        Returns catalogue within LON/LAT region of interest
        """
        df = df[(df.LON > region[0]) & (df.LON <  region[1]) &\
                (df.LAT > region[2]) & (df.LAT <  region[3])].copy()
        return df


def get_catalogue_extent(catalogue, buffer=None):
    """
    Returns the min/max of the Lon/Lat of an earthquake catalogue
    """
    if buffer==None:
        extent = np.array([min(catalogue['LON']), max(catalogue['LON']), min(catalogue['LAT']), max(catalogue['LAT'])])
    else:
        extent = np.array([min(catalogue['LON'])-buffer, max(catalogue['LON'])+buffer, min(catalogue['LAT'])-buffer, max(catalogue['LAT'])+buffer])
    return extent 


def min_max_median_mean(numbers):
    """
    Returns the min, max, median, and mean of a list of numbers
    """
    numbers = np.array(numbers)
    min = np.nanmin(numbers)
    max = np.nanmax(numbers)
    median = np.nanmedian(numbers)
    mean = np.nanmean(numbers)
    
    return min, max, median, mean


# from stack overflow
def find_nearest(array, value):
    """
    Returns the nearest value in an array to its argument.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# Chat GPT?
def calculate_distance_pyproj_vectorized(lon1, lat1, lon2_array, lat2_array, ellipsoid="WGS84"):
    """
    Returns the distance (km) from a point to an array of points using the Pyproj module
    """
    geod = pyproj.Geod(ellps=ellipsoid)
    _, _, distance_m = geod.inv(lons1=np.full_like(lon2_array, lon1), lats1=np.full_like(lat2_array, lat1), lons2=np.array(lon2_array), lats2=np.array(lat2_array))
    distance_km = distance_m / 1000
    return distance_km


def haversine_vectorised(lon1, lat1, lon2, lat2):
    """
    Returns the distance (km) from a point to an array of points using the haversine method
    """
    lon2, lat2 = np.array(lon2), np.array(lat2)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c 
    return km


def add_distance_to_position_haversine(lon, lat, distance_km_horizontal, distance_km_vertical):
    """
    Returns the a point shifted in km by the value of its arguments using the haversine method
    """
    delta_lon = (distance_km_horizontal / haversine(lon, lat, lon + 1, lat)) * 1
    delta_lat = (distance_km_vertical / haversine(lon, lat, lon, lat + 1)) * 1
    new_lon = lon + delta_lon
    new_lat = lat + delta_lat
    return new_lon, new_lat


def add_distance_to_position_pyproj(lon, lat, distance_km_horizontal, distance_km_vertical):
    """
    Returns the a point shifted in km by the value of its arguments using the Pyproj module
    """
    geod = pyproj.Geod(ellps="WGS84")
    new_lon_horizontal, new_lat_horizontal, _ = geod.fwd(lon, lat, 90, distance_km_horizontal * 1000)
    new_lon, new_lat, _ = geod.fwd(new_lon_horizontal, new_lat_horizontal, 0, distance_km_vertical * 1000)
    return new_lon, new_lat


def gamma_law_MLE(t):
    """
    Calculate background seismicity rate based on the interevent time distribution. From CORSSA (originally in MATLAB), changed to Python (by me).
    """
    dt = np.diff(t)
    dt = dt[dt>0]
    T = sum(dt)
    N = len(dt)
    S = sum(np.log(dt))
    dg = 10**-4
    gam = np.arange(dg, 1-dg, dg) # increment from dg to 1-dg with a step of dg (dg:dg:dg-1 in matlab)
    ell = N*gam*(1-np.log(N)+np.log(T)-np.log(gam))+N*special.loggamma(gam)-gam*S # scipy gamma funcion
    ell_min = np.amin(ell)
    i = np.where(ell == ell_min)
    gam=gam[i]
    mu=N/T*gam
    return mu[0]


def freq_mag_dist(mag, mbin):
    """
    A basic frequency magnitude distribution analysis that requires an array of magnitudes (mag) and a chosen
    binning (mbin, we use 0.1). It returns magnitude bins, no. of events per bin, and cum. mag. distribution.
    [CORSSA, Lapins] - modified
    """
    mag = np.array(mag)
    minmag = math.floor(min(mag/mbin)) * mbin # Lowest bin
    maxmag = math.ceil(max(mag/mbin)) * mbin # Highest bin bin
    mi = np.arange(minmag, maxmag + mbin, mbin) # Make array of bins
    nbm = len(mi)
    cumnbmag = np.zeros(nbm) # Array for cumulative no. of events
    for i in range(nbm): # cumulative no. of events
        cumnbmag[i] = np.where(mag > mi[i] - mbin/2)[0].shape[0]
    nbmag = abs(np.diff(np.append(cumnbmag, 0))) # no. of events
    return mi, nbmag, cumnbmag


def b_val_max_likelihood(mag, mc, mbin=0.1):
    """
    This code calculates b values by maximum likelihood estimate. It takes in an array of magnitude (mag), a
    binning (mbin, we use 0.1) and a completeness magnitude (mc). It provides returns productivity (a), b value
    (b), and two estimates of uncertainty (aki_unc, shibolt_unc). [Aki 1965, Bender 1983, CORSSA, Lapins] - modified
    """
    mag = np.array(mag) # [me]
    mag_above_mc = mag[np.where(mag > round(mc,1)-mbin/2)[0]]# Magnitudes for events larger than cut-off magnitude mc
    n = mag_above_mc.shape[0] # No of. events larger than cut-off magnitude mc
    mbar = np.mean(mag_above_mc) # Mean magnitude for events larger than cut-off magnitude mc
    b = math.log10(math.exp(1)) / (mbar - (mc - mbin/2)) # b-value from Eq 2
    a = math.log10(n) + b * mc # 'a-value' for Eq 1
    aki_unc = b / math.sqrt(n) # Uncertainty estimate from Eq 3
    shibolt_unc = 2.3 * b**2 * math.sqrt(sum((mag_above_mc - mbar)**2) / (n * (n-1))) # Uncertainty estimate from Eq 4
#     return a, b, aki_unc, shibolt_unc # Return b-value and estimates of uncertainty
    return b
 
def Mc_by_maximum_curvature(mag, mbin=0.1):
    """
    This code returns the magnitude of completeness estimates using the maximum curvature method. It takes a magnitude
    array (mag) and binning (mbin). [Wiemer & Wyss (2000), Lapins, CORSSA] - modified
    """
    mag = np.array(mag)
    this_fmd = freq_mag_dist(mag, mbin) # uses the fmd distribution (a previous function)
    maxc = this_fmd[0][np.argmax(this_fmd[1])] # Mag bin with highest no. of events
    return maxc
 
 
def Mc_by_goodness_of_fit(mag, mbin=0.1):
    """
    This code returns the magnitude of completeness estimates using a goodness of fit method. It takes a magnitude
    array (mag) and binning (mbin, we use 0.1). It returns the estimate (mc), the fmd (this_fmd[0]) and confidence level (R).
    The equation numbers refer to those in the CORSSA documentation(*). It defaults to maxc if confidence levels
    are not met. [Wiemer & Wyss (2000), Lapins, CORSSA] - modified
    """
    mag = np.array(mag)
    this_fmd = freq_mag_dist(mag, mbin) # FMD
    this_maxc = Mc_by_maximum_curvature(mag, mbin) # Runs the previous max curvature method first
    # Zeros to accommodate synthetic GR distributions for each magnitude bin
    a = np.zeros(this_fmd[0].shape[0]) # Pre-allocate array to accommodate a values from Eq 1
    b = np.zeros(this_fmd[0].shape[0]) # Pre-allocate array to accommodate b values from Eq 1 & 2
    R = np.zeros(this_fmd[0].shape[0]) # Pre-allocate array to accommodate R values from Eq 5
    for i in range(this_fmd[0].shape[0]): # Loop through each magnitude bin, using it as cut-off magnitude
        mi = round(this_fmd[0][i], 1) # Cut-off magnitude
        a[i], b[i], tmp1, tmp2 = b_val_max_likelihood(mag, mbin, mi) # a and b-values for this cut-off magnitude
        synthetic_gr = 10**(a[i] - b[i]*this_fmd[0]) # Synthetic GR for a and b
        Bi = this_fmd[2][i:] # B_i in Eq 5
        Si = synthetic_gr[i:] # S_i in Eq 5
        R[i] = (sum(abs(Bi - Si)) / sum(Bi)) * 100 # Eq 5
    R_to_test = [95, 90] # Confidence levels to test (95% and 90% conf levels)
    GFT_test = [np.where(R <= (100 - conf_level)) for conf_level in R_to_test] # Test whether R within confidence level
    for i in range(len(R_to_test)+1): # Loop through and check first cut-off mag within confidence level
        # If no GR distribution fits within confidence levels then use MAXC instead
        if i == (len(R_to_test) + 1):
            mc = this_maxc
            print("No fits within confidence levels, using MAXC estimate")
            break
        else:
            if len(GFT_test[i][0]) > 0:
                mc = round(this_fmd[0][GFT_test[i][0][0]], 1) # Use first cut-off magnitude within confidence level
                break
#     return mc, this_fmd[0], R
    return mc

 
def Mc_by_b_value_stability(mag, mbin=0.1, dM = 0.4, min_mc = -3, return_b=False):
    """
    This code returns the magnitude of completeness estimates using a b value stability method. It takes a magnitude
    array (mag), binning (mbin, we use 0.1), number of magnitude units to calculate a rolling average b value over (dM,
    we use 0.4) and a minimum mc to test (min_mc). The outputs are a completeness magnitude (mc), frequency magnitude
    distribution (this_fmd[0]), the b value calculated for this mc and average b(*) (b and b_average) and b value uncertainty
    estimate (shibolt_unc). The equation numbers refer to those in the CORSSA documentation(*). It defaults to maxc if
    confidence levels are not met.[ Cao & Gao (2002), Lapins, CORSSA]. - modified
    """
    mag = np.array(mag)
    this_fmd = freq_mag_dist(mag, mbin) # FMD
    this_maxc = Mc_by_maximum_curvature(mag, mbin) # Needed further down
    # Zeros to accommodate synthetic GR distributions for each magnitude bin
    a = np.zeros(this_fmd[0].shape[0]) # Pre-allocate array to accommodate a values from Eq 1
    b = np.zeros(this_fmd[0].shape[0]) # Pre-allocate array to accommodate b values from Eq 1 & 2
    b_avg = np.zeros(this_fmd[0].shape[0]) # Pre-allocate array to accommodate b values from Eq 1 & 2
    shibolt_unc = np.zeros(this_fmd[0].shape[0]) # Pre-allocate array to accommodate uncertainty values from Eq 4
    for i in range(this_fmd[0].shape[0]): # Loop through each magnitude bin, using it as cut-off magnitude
        mi = round(this_fmd[0][i], 1) # Cut-off magnitude
        if this_fmd[2][i] > 1:
            a[i], b[i], tmp1, shibolt_unc[i] = b_val_max_likelihood(mag, mbin, mi) # a and b-values for this cut-off magnitude
        else:
            a[i] = np.nan
            b[i] = np.nan
            shibolt_unc[i] = np.nan
    no_bins = round(dM/mbin)
    check_bval_stability = []
    for i in range(this_fmd[0].shape[0]): # Loop through again, calculating rolling average b-value over following dM magnitude units
        if i >= this_fmd[0].shape[0] - (no_bins + 1):
            b_avg[i] = np.nan
            next
        if any(np.isnan(b[i:(i+no_bins+1)])):
            b_avg[i] = np.nan
            check_bval_stability.append(False)
        else:
            b_avg[i] = np.mean(b[i:(i+no_bins+1)])
            check_bval_stability.append(abs(b_avg[i] - b[i]) <= shibolt_unc[i])
    if any(check_bval_stability):
        bval_stable_points = this_fmd[0][np.array(check_bval_stability)]
        mc = round(min(bval_stable_points[np.where(bval_stable_points > min_mc)[0]]), 1) # Completeness mag is first mag bin that satisfies Eq 6
    else:
        mc = this_maxc # If no stability point, use MAXC
#     return mc, this_fmd[0], b, b_avg, shibolt_unc
    return mc, b#, this_fmd[0], b, b_avg, shibolt_unc


def select_mainshocks(earthquake_catalogue,
                      catalogue_name='_',
                      search_style='radius',
                      search_distance_km=10,
                      mainshock_magnitude_threshold = 4,
                      minimum_exclusion_distance = 20,
                      scaling_exclusion_distance = 5,
                      minimum_exclusion_time = 50,
                      scaling_exclusion_time = 25
                      ):
    """
    Return mainshocks from an earthquake catalogue selected using both the methods from Trugman & Ross (2019) and Moutote et al. (2021).

    """
    
    # print(catalogue_name)

    earthquakes_above_magnitude_threshold = earthquake_catalogue.loc[earthquake_catalogue['MAGNITUDE'] >= mainshock_magnitude_threshold].copy()
    N_earthquakes_above_magnitude_threshold = len(earthquakes_above_magnitude_threshold)
    # print(f"Mw4+ earthquakes: {N_earthquakes_above_magnitude_threshold}")
    
    exclusion_criteria_results = []
    TR_mainshocks_to_exclude = []
    for mainshock in earthquakes_above_magnitude_threshold.itertuples():
        min_box_lon, min_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, -search_distance_km, -search_distance_km)
        max_box_lon, max_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, search_distance_km, search_distance_km)

        local_catalogue = earthquake_catalogue.loc[
                                        (earthquake_catalogue['LON']>= min_box_lon) &\
                                        (earthquake_catalogue['LON']<= max_box_lon) &\
                                        (earthquake_catalogue['LAT']>= min_box_lat) &\
                                        (earthquake_catalogue['LAT']<= max_box_lat)
                                        ].copy()
        
        if search_style=='radius':
            local_catalogue['DISTANCE_TO_MAINSHOCK'] = calculate_distance_pyproj_vectorized(mainshock.LON, mainshock.LAT, local_catalogue['LON'],  local_catalogue['LAT'])
            local_catalogue = local_catalogue[(local_catalogue['DISTANCE_TO_MAINSHOCK'] < search_distance_km)].copy()    

        elif search_style=='box':
            print(f"A box has been chosen, even though a box allows a distance of 14 km between mainshock epicentre and box corner.")

        else:
            print(f"Invalid search style - we are going to craaaaash")

        n_local_catalogue = len(local_catalogue)

        local_catalogue_1yr = local_catalogue[(local_catalogue.DATETIME <= mainshock.DATETIME) &\
                                        ((mainshock.DATETIME - local_catalogue.DATETIME) < dt.timedelta(days=365)) &\
                                        (local_catalogue['ID'] != mainshock.ID)
                                        ].copy()

        n_local_catalogue_1yr = len(local_catalogue_1yr)
        
        if n_local_catalogue_1yr > 0:
            max_magnitude = max(local_catalogue_1yr['MAGNITUDE'])
            Mc_1yr = Mc_by_maximum_curvature(local_catalogue_1yr['MAGNITUDE']) + 0.2
            if max_magnitude <= mainshock.MAGNITUDE:
                Moutote_method = 'Selected'
                Moutote_excluded_by=[]
            elif max_magnitude > mainshock.MAGNITUDE:
                Moutote_excluded_by = list(local_catalogue_1yr.loc[local_catalogue_1yr['MAGNITUDE'] > mainshock.MAGNITUDE, 'ID'])
                Moutote_method = 'Excluded'
        else:
            max_magnitude, Mc_1yr =[float('nan')]*2
            Moutote_method = 'Selected'
            Moutote_excluded_by = []

        if n_local_catalogue_1yr > 0:
            Mc = Mc_by_maximum_curvature(local_catalogue['MAGNITUDE']) + 0.2
        else:
            Mc = float('nan')
                    
        subsurface_rupture_length = 10**((mainshock.MAGNITUDE - 4.38)/1.49)
        distance_exclusion_threshold = minimum_exclusion_distance + scaling_exclusion_distance * subsurface_rupture_length
        time_exclusion_threshold = minimum_exclusion_time + scaling_exclusion_time * (mainshock.MAGNITUDE - mainshock_magnitude_threshold)

        distances_between_earthquakes = calculate_distance_pyproj_vectorized(mainshock.LON, mainshock.LAT, earthquakes_above_magnitude_threshold['LON'],  earthquakes_above_magnitude_threshold['LAT'])

        earthquakes_within_exclusion_criteria = earthquakes_above_magnitude_threshold.loc[
            (mainshock.ID != earthquakes_above_magnitude_threshold.ID) &\
            (distances_between_earthquakes <= distance_exclusion_threshold) &\
            (earthquakes_above_magnitude_threshold.MAGNITUDE < mainshock.MAGNITUDE) &\
            (((earthquakes_above_magnitude_threshold['DATETIME'] - mainshock.DATETIME).apply(lambda d: d.total_seconds()/(3600*24))) < time_exclusion_threshold) &\
            (((earthquakes_above_magnitude_threshold['DATETIME'] - mainshock.DATETIME).apply(lambda d: d.total_seconds()/(3600*24)) > 0))
            ]
        
        TR_mainshocks_to_exclude.extend(list(earthquakes_within_exclusion_criteria['ID']))
        
        results_dict = {'ID':mainshock.ID,
                        'DATETIME':mainshock.DATETIME,
                        'MAGNITUDE':mainshock.MAGNITUDE,
                        'LON':mainshock.LON,
                        'LAT':mainshock.LAT,
                        'DEPTH':mainshock.DEPTH,
                        'n_local_catalogue_1yr':n_local_catalogue_1yr,
                        'n_local_catalogue':n_local_catalogue,
                        'Mc':Mc,
                        'Mc_1yr':Mc_1yr,
                        'Largest_preceding':max_magnitude,
                        'Moutote_method':Moutote_method,
                        'Moutote_excluded_by':Moutote_excluded_by,
                        'subsurface_rupture_length':subsurface_rupture_length,
                        'distance_exclusion_threshold':distance_exclusion_threshold,
                        'time_exclusion_threshold':time_exclusion_threshold,
                        'TR_excludes':list(earthquakes_within_exclusion_criteria['ID'])}
        
        exclusion_criteria_results.append(results_dict)
    
    exclusion_criteria_results = pd.DataFrame.from_dict(exclusion_criteria_results)

    exclusion_criteria_results['TR_method'] = np.select([~exclusion_criteria_results['ID'].isin(TR_mainshocks_to_exclude),
                                                         exclusion_criteria_results['ID'].isin(TR_mainshocks_to_exclude)],
                                                         ['Selected', 'Excluded'],
                                                         default='error')

    TR_excluded_by = []
    for mainshock in exclusion_criteria_results.itertuples():
        excluded_by_list = []
        for mainshock_2 in exclusion_criteria_results.itertuples():
            if mainshock.ID in mainshock_2.TR_excludes:
                excluded_by_list.append(mainshock_2.ID)
                # print(mainshock.ID, mainshock_2.ID)
        TR_excluded_by.append(excluded_by_list)

    exclusion_criteria_results['TR_excluded_by'] = TR_excluded_by

    selection_list = []
    for mainshock in exclusion_criteria_results.itertuples():
        if (mainshock.Moutote_method=='Selected') & (mainshock.TR_method=='Selected'):
            selection='Both'
        elif (mainshock.Moutote_method=='Selected') & (mainshock.TR_method=='Excluded'):
            selection='FET'
        elif (mainshock.Moutote_method=='Excluded') & (mainshock.TR_method=='Selected'):
            selection='MDET'
        elif (mainshock.Moutote_method=='Excluded') & (mainshock.TR_method=='Excluded'):
            selection='Neither'
        selection_list.append(selection)

    exclusion_criteria_results['Selection'] = selection_list

    # print(f"TR_mainshocks: {len(exclusion_criteria_results.loc[exclusion_criteria_results['TR_method']=='Selected'])}")
    # print(f"Moutote_mainshocks: {len(exclusion_criteria_results.loc[exclusion_criteria_results['Moutote_method']=='Selected'])}")

    return exclusion_criteria_results


def ESR_model(mainshock, earthquake_catalogue, local_catalogue,
              local_catalogue_radius = 10, foreshock_window = 20, modelling_time_period=365):
    
    """
    Calculate signals prior to mainshocks in a sliding window: seismicity rates, distances to mainshock.
    """
    
    mainshock_ID = mainshock.ID
    mainshock_LON = mainshock.LON
    mainshock_LAT = mainshock.LAT
    mainshock_DATETIME = mainshock.DATETIME
    mainshock_Mc = mainshock.Mc
    mainshock_MAG = mainshock.MAGNITUDE

    local_catalogue = local_catalogue[(local_catalogue['DATETIME'] < mainshock_DATETIME) &\
                                        (local_catalogue['DAYS_TO_MAINSHOCK'] < modelling_time_period+foreshock_window) &\
                                        (local_catalogue['DAYS_TO_MAINSHOCK'] > 0)  &\
                                        (local_catalogue['DISTANCE_TO_MAINSHOCK'] < local_catalogue_radius) &\
                                        (local_catalogue['ID'] != mainshock_ID)
                                        ].copy()

    regular_seismicity_period = local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] >= foreshock_window)].copy()
    foreshocks = local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] < foreshock_window)].copy()

    n_local_catalogue = len(local_catalogue)
    n_regular_seismicity_events = len(regular_seismicity_period)
    n_events_in_foreshock_window = len(foreshocks)
    foreshock_distance = np.median(foreshocks['DISTANCE_TO_MAINSHOCK'])

    catalogue_start_date = earthquake_catalogue['DATETIME'].iloc[0]
    time_since_catalogue_start = (mainshock_DATETIME - catalogue_start_date).total_seconds()/3600/24
    cut_off_day = math.floor(time_since_catalogue_start)
    if cut_off_day > 365:
        cut_off_day = 365
    range_scaler = 100    

    sliding_window_points = np.array(np.arange((-cut_off_day+foreshock_window)*range_scaler, -foreshock_window*range_scaler, 1))/range_scaler*-1
    sliding_window_counts = np.array([len(regular_seismicity_period.loc[(regular_seismicity_period['DAYS_TO_MAINSHOCK'] > point) &\
                                                                    (regular_seismicity_period['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]) for point in sliding_window_points])
    sliding_window_distances = np.array([np.median(regular_seismicity_period.loc[(regular_seismicity_period['DAYS_TO_MAINSHOCK'] > point) &\
                                                                    (regular_seismicity_period['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window)), 'DISTANCE_TO_MAINSHOCK']) for point in sliding_window_points])

    try:
        distance_probability = len(sliding_window_distances[sliding_window_distances >= foreshock_distance])/len(sliding_window_distances)
    except:
        distance_probability = float('nan')

    try:
        max_window = max(sliding_window_counts)
    except:
        max_window = float('nan')

    if n_events_in_foreshock_window > max_window:
        max_window_method = 0.0
    elif n_events_in_foreshock_window <= max_window:
        max_window_method = 1.0
    else:
        max_window_method = float('nan')

    if (len(sliding_window_counts)==0) & (n_events_in_foreshock_window > 0):
        sliding_window_probability = 0.00
        sliding_window_99CI = float('nan')
    elif (len(sliding_window_counts)==0) & (n_events_in_foreshock_window == 0):    
        sliding_window_probability = 1.00
        sliding_window_99CI = float('nan')
    else:
        sliding_window_probability = len(sliding_window_counts[sliding_window_counts >= n_events_in_foreshock_window])/len(sliding_window_counts)
    # sliding_window_probability = len(list(filter(lambda c: c >= n_events_in_foreshock_window, sliding_window_counts)))/len(sliding_window_counts)
        sliding_window_99CI = np.percentile(sliding_window_counts,99)
                
    results_dict = {'ID':mainshock_ID,
                    'MAGNITUDE':mainshock_MAG,
                    'LON':mainshock_LON,
                    'LAT':mainshock_LAT,
                    'DATETIME':mainshock_DATETIME,
                    'DEPTH':mainshock.DEPTH,
                    'Mc':mainshock_Mc,
                    'time_since_catalogue_start':time_since_catalogue_start,
                    'n_regular_seismicity_events':n_regular_seismicity_events,
                    'n_events_in_foreshock_window':n_events_in_foreshock_window,
                    'max_20day_rate':max_window,
                    'ESR':sliding_window_probability,
                    'ESR_99CI':sliding_window_99CI,
                    'ESD':distance_probability,
                    'cut_off_day':cut_off_day
                    }
    
    file_dict = {'local_catalogue':local_catalogue,
                #  'local_catalogue_pre_Mc_cutoff':local_catalogue_pre_Mc_cutoff,
                #  'local_catalogue_below_Mc':local_catalogue_below_Mc,
                 'foreshocks':foreshocks,
                #  'foreshocks_below_Mc':foreshocks_below_Mc,
                 'sliding_window_points':sliding_window_points,
                 'sliding_window_counts':sliding_window_counts,
                 'sliding_window_distances':sliding_window_distances
                 }
    
    return results_dict, file_dict

def select_mainshocks_magnitude_dependent_thresholds(earthquake_catalogue,
                                catalogue_name,
                                search_style='radius',
                                search_distance_km=10,
                                mainshock_magnitude_threshold = 4,
                                minimum_exclusion_distance = 20,
                                scaling_exclusion_distance = 5,
                                minimum_exclusion_time = 50,
                                scaling_exclusion_time = 25,
                                detail=False
                                ):
    """
    Select mainshocks from an earthquake catalogue using the method from Trugman and Ross (2019)
    """
    
    print("Magnitude dependent exclusion threshold method")
    print(catalogue_name)

    earthquakes_above_magnitude_threshold = earthquake_catalogue.loc[earthquake_catalogue['MAGNITUDE'] >= mainshock_magnitude_threshold].copy()
    N_potential_mainshocks = len(earthquakes_above_magnitude_threshold)
    print(str(N_potential_mainshocks) + ' potential mainshocks')
    
    exclusion_criteria_results = []
    mainshocks_to_exclude = []
    for mainshock in earthquakes_above_magnitude_threshold.itertuples():
        min_box_lon, min_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, -search_distance_km, -search_distance_km)
        max_box_lon, max_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, search_distance_km, search_distance_km)

        local_catalogue = earthquake_catalogue.loc[
                                        (earthquake_catalogue['LON']>= min_box_lon) &\
                                        (earthquake_catalogue['LON']<= max_box_lon) &\
                                        (earthquake_catalogue['LAT']>= min_box_lat) &\
                                        (earthquake_catalogue['LAT']<= max_box_lat)
                                        ].copy()
        
        if search_style=='radius':
            local_catalogue['DISTANCE_TO_MAINSHOCK'] = calculate_distance_pyproj_vectorized(mainshock.LON, mainshock.LAT, local_catalogue['LON'],  local_catalogue['LAT'])
            local_catalogue = local_catalogue[(local_catalogue['DISTANCE_TO_MAINSHOCK'] < search_distance_km)].copy()    

        elif search_style=='box':
            print(f"A box has been chosen, even though a box allows a distance of 14 km between mainshock epicentre and box corner.")

        else:
            print(f"Invalid search style - we are going to craaaaash")

        local_catalogue_1yr = local_catalogue[(local_catalogue.DATETIME <= mainshock.DATETIME) &\
                                        ((mainshock.DATETIME - local_catalogue.DATETIME) < dt.timedelta(days=365)) &\
                                        (local_catalogue['ID'] != mainshock.ID)
                                        ].copy()

        len_local_catalogue = len(local_catalogue_1yr) # if want at least 10 events in 1 year before mainshocks
#         len_local_catalogue = len(local_catalogue) # 10 events within whole catalogue before mainshocks
        
        if len_local_catalogue>0:
            max_magnitude = max(local_catalogue_1yr['MAGNITUDE'])
            Mc = Mc_by_maximum_curvature(local_catalogue_1yr['MAGNITUDE'])
        else:
            max_magnitude, Mc =[float('nan')]*2
                    
        subsurface_rupture_length = 10**((mainshock.MAGNITUDE - 4.38)/1.49)
        distance_exclusion_threshold = minimum_exclusion_distance + scaling_exclusion_distance * subsurface_rupture_length
        time_exclusion_threshold = minimum_exclusion_time + scaling_exclusion_time * (mainshock.MAGNITUDE - mainshock_magnitude_threshold)
#         print('    time_exclusion_threshold: ' + str(time_exclusion_threshold))

        distances_between_earthquakes = calculate_distance_pyproj_vectorized(mainshock.LON, mainshock.LAT, earthquakes_above_magnitude_threshold['LON'],  earthquakes_above_magnitude_threshold['LAT'])

        earthquakes_within_exclusion_criteria = earthquakes_above_magnitude_threshold.loc[
            (mainshock.ID != earthquakes_above_magnitude_threshold.ID) &\
            (distances_between_earthquakes <= distance_exclusion_threshold) &\
            (earthquakes_above_magnitude_threshold.MAGNITUDE < mainshock.MAGNITUDE) &\
            (((earthquakes_above_magnitude_threshold['DATETIME'] - mainshock.DATETIME).apply(lambda d: d.total_seconds()/(3600*24))) < time_exclusion_threshold) &\
#             ((((mainshock.DATETIME - earthquakes_above_magnitude_threshold['DATETIME']).apply(lambda d: d.total_seconds()/(3600*24)))**2)**0.5 <= time_exclusion_threshold)
            (((earthquakes_above_magnitude_threshold['DATETIME'] - mainshock.DATETIME).apply(lambda d: d.total_seconds()/(3600*24)) > 0))
                                                                                         ]
#         print(earthquakes_within_exclusion_criteria)
#         print('    earthquakes_within_exclusion_criteria: ' + str(len(earthquakes_within_exclusion_criteria)))
#         print(' ')
        
        mainshocks_to_exclude.extend(list(earthquakes_within_exclusion_criteria['ID']))
        
        results_dict = {'ID':mainshock.ID,
                        'DATETIME':mainshock.DATETIME,
                        'MAGNITUDE':mainshock.MAGNITUDE,
                        'LON':mainshock.LON,
                        'LAT':mainshock.LAT,
                        'local_catalogue_length':len_local_catalogue,
                        'Mc':Mc,
                        'Largest_preceding':max_magnitude,
                        'subsurface_rupture_length':subsurface_rupture_length,
                        'distance_exclusion_threshold':distance_exclusion_threshold,
                        'time_exclusion_threshold':time_exclusion_threshold,
                        'TR_excludes':list(earthquakes_within_exclusion_criteria['ID'])}
        
        exclusion_criteria_results.append(results_dict)
    
    mainshocks_not_near_other_mainshocks = earthquakes_above_magnitude_threshold.loc[~earthquakes_above_magnitude_threshold['ID'].isin(mainshocks_to_exclude)]
    
    N_mainshocks_selected = len(mainshocks_not_near_other_mainshocks)
    
    output_path = '../outputs/' + catalogue_name + '/TR_method/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    mainshocks_not_near_other_mainshocks.to_csv(output_path + 'mainshocks.csv', index=False)
    
    exclusion_criteria_results = pd.DataFrame.from_dict(exclusion_criteria_results)

    exclusion_criteria_results['state'] = np.select([exclusion_criteria_results['ID'].isin(mainshocks_not_near_other_mainshocks['ID']),
                                                      ~exclusion_criteria_results['ID'].isin(mainshocks_not_near_other_mainshocks['ID'])],
                                                       ['selected', 'excluded'],
                                                       default='error')

    excluded_by_column = []
    for mainshock in exclusion_criteria_results.itertuples():
        excluded_by_list = []
        for mainshock_2 in exclusion_criteria_results.itertuples():
            if mainshock.ID in mainshock_2.TR_excludes:
                excluded_by_list.append(mainshock_2.ID)
                # print(mainshock.ID, mainshock_2.ID)
        excluded_by_column.append(excluded_by_list)

    print(len(excluded_by_column))
    exclusion_criteria_results['excluded_by'] = excluded_by_column
    
    print(str(N_mainshocks_selected) + ' mainshocks selected')
    print(" ")

    if detail==True:
        return exclusion_criteria_results
    else:
        return mainshocks_not_near_other_mainshocks, N_potential_mainshocks, N_mainshocks_selected #, exclusion_criteria_results, mainshocks_to_exclude


def select_mainshocks_Moutote_method_2(earthquake_catalogue,
                                        catalogue_name,
                                        mainshock_magnitude_threshold = 4,
                                        box_halfwidth_km=10,
                                        local_catalogue_time=365,
                                        more_detail=False,
                                        ):
    """
    Select mainshocks from an earthquake catalogue using the method from Moutote et al. (2021). 
    Modified to not require 10 earthquakes within 10 km in the year prior for a mainshock to be selected.
    """
        
    print("Modified fixed exclusion threshold method")
    print(catalogue_name)
    
    earthquakes_above_magnitude_threshold = earthquake_catalogue.loc[earthquake_catalogue['MAGNITUDE'] >= mainshock_magnitude_threshold].copy()
    
    N_potential_mainshocks = len(earthquakes_above_magnitude_threshold)
    print(str(N_potential_mainshocks) + ' potential mainshocks')
    
    results = []
    exclusion_dict = {}
    for mainshock in earthquakes_above_magnitude_threshold.itertuples():

        min_box_lon, min_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, -box_halfwidth_km, -box_halfwidth_km)
        max_box_lon, max_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, box_halfwidth_km, box_halfwidth_km)

        local_catalogue = earthquake_catalogue.loc[
                                        (earthquake_catalogue['LON']>= min_box_lon) &\
                                        (earthquake_catalogue['LON']<= max_box_lon) &\
                                        (earthquake_catalogue['LAT']>= min_box_lat) &\
                                        (earthquake_catalogue['LAT']<= max_box_lat)
                                        ].copy()
        
        local_catalogue_1yr = local_catalogue[(local_catalogue.DATETIME <= mainshock.DATETIME) &\
                                                ((mainshock.DATETIME - local_catalogue.DATETIME) < dt.timedelta(days=local_catalogue_time)) &\
                                                (local_catalogue['ID'] != mainshock.ID)
                                                ].copy()
        
        len_local_catalogue = len(local_catalogue_1yr) # if want at least 10 events in 1 year before mainshocks

        if len_local_catalogue > 0:
            max_magnitude = max(local_catalogue_1yr['MAGNITUDE'])
            Mc = Mc_by_maximum_curvature(local_catalogue_1yr['MAGNITUDE'])
        else:
            max_magnitude, Mc = [float('nan')]*2

        if (max_magnitude <= mainshock.MAGNITUDE):
            state = 'Selected'

        elif max_magnitude > mainshock.MAGNITUDE:
            exclusion_IDs = list(local_catalogue.loc[local_catalogue['MAGNITUDE'] > mainshock.MAGNITUDE, 'ID'])
            exclusion_dict.update({mainshock.ID: exclusion_IDs})
            state = 'Larger event'

        else:
            state = 'Error'
            
        results_dict = {'state':state,
                        'len_local_catalogue':len_local_catalogue,
                        'Mc':Mc
                        }
        
        results.append(results_dict)

    results = pd.DataFrame.from_dict(results)

    earthquakes_above_magnitude_threshold.reset_index(inplace=True)
    earthquakes_above_magnitude_threshold = pd.concat([earthquakes_above_magnitude_threshold, results], axis=1)
    
    mainshocks_selected = earthquakes_above_magnitude_threshold.loc[earthquakes_above_magnitude_threshold['state']=='Selected'].copy()

    N_mainshocks_selected = len(mainshocks_selected)
        
    output_path = '../outputs/' + catalogue_name + '/Moutote_method_2/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    mainshocks_selected.to_csv(output_path + 'mainshocks.csv', index=False)
        
    print(str(N_mainshocks_selected) + ' mainshocks selected')
    print(" ")

    if more_detail==True:
        return earthquakes_above_magnitude_threshold#, mainshocks_selected, N_potential_mainshocks, N_mainshocks_selected
    else:
        return mainshocks_selected, N_potential_mainshocks, N_mainshocks_selected, #, exclusion_dict


def select_mainshocks_fixed_thresholds_mod(earthquake_catalogue,
                                           catalogue_name,
                                           mainshock_magnitude_threshold = 4,
                                           search_style = 'box',
                                           search_distance_km = 10,
                                           more_detail=False,
                                           ):
    """
    Select mainshocks from an earthquake catalogue using the method from Moutote et al. (2021). 
    Modified to not require 10 earthquakes within 10 km in the year prior for a mainshock to be selected. 
    (Newer than select_mainshocks_Moutote_method_2).
    """
    
    print("Fixed exclusion threshold method - modified")
    print(catalogue_name)
    
    earthquakes_above_magnitude_threshold = earthquake_catalogue.loc[earthquake_catalogue['MAGNITUDE'] >= mainshock_magnitude_threshold].copy()
    
    N_potential_mainshocks = len(earthquakes_above_magnitude_threshold)
    print(str(N_potential_mainshocks) + ' potential mainshocks')

    states = []
    local_catalogue_lengths = []
    completeness_magnitudes = []
    exclusion_dict = {}
    Moutote_excluded_by_list = []
    for mainshock in earthquakes_above_magnitude_threshold.itertuples():
        if search_style=='box':
            min_box_lon, min_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, -search_distance_km, -search_distance_km)
            max_box_lon, max_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, search_distance_km, search_distance_km)

            local_catalogue = earthquake_catalogue.loc[
                                            (earthquake_catalogue['LON']>= min_box_lon) &\
                                            (earthquake_catalogue['LON']<= max_box_lon) &\
                                            (earthquake_catalogue['LAT']>= min_box_lat) &\
                                            (earthquake_catalogue['LAT']<= max_box_lat)
                                            ].copy()
        elif search_style=='radius':
            earthquake_catalogue['DISTANCE_TO_MAINSHOCK'] = calculate_distance_pyproj_vectorized(mainshock.LON, mainshock.LAT, earthquake_catalogue['LON'],  earthquake_catalogue['LAT'])
            local_catalogue = earthquake_catalogue[(earthquake_catalogue['DISTANCE_TO_MAINSHOCK'] < search_distance_km)].copy()     

        local_catalogue_1yr = local_catalogue[(local_catalogue.DATETIME <= mainshock.DATETIME) &\
                                        ((mainshock.DATETIME - local_catalogue.DATETIME) < dt.timedelta(days=365)) &\
                                        (local_catalogue['ID'] != mainshock.ID)
                                        ].copy()

        len_local_catalogue = len(local_catalogue_1yr) # if want at least 10 events in 1 year before mainshocks
#         len_local_catalogue = len(local_catalogue) # 10 events within whole catalogue before mainshocks
        local_catalogue_lengths.append(len_local_catalogue)
        
        if len_local_catalogue>0:
            max_magnitude = max(local_catalogue_1yr['MAGNITUDE'])
            Mc = Mc_by_maximum_curvature(local_catalogue_1yr['MAGNITUDE'])
            if max_magnitude <= mainshock.MAGNITUDE:
                Moutote_method = 'Selected'
                Moutote_excluded_by=[]
            elif max_magnitude > mainshock.MAGNITUDE:
                Moutote_excluded_by = list(local_catalogue.loc[local_catalogue['MAGNITUDE'] > mainshock.MAGNITUDE, 'ID'])
                Moutote_method = 'Excluded'
        else:
            max_magnitude, Mc =[float('nan')]*2
            Moutote_method = 'Selected'
            Moutote_excluded_by = []
            
        completeness_magnitudes.append(Mc)    
        states.append(Moutote_method)
        Moutote_excluded_by_list.append(Moutote_excluded_by)

    earthquakes_above_magnitude_threshold['state'] = states
    earthquakes_above_magnitude_threshold['len_local_catalogue'] = local_catalogue_lengths
    earthquakes_above_magnitude_threshold['Mc'] = completeness_magnitudes
    earthquakes_above_magnitude_threshold['Moutote_excluded_by'] = Moutote_excluded_by_list
    
    mainshocks_selected = earthquakes_above_magnitude_threshold.loc[earthquakes_above_magnitude_threshold['state']=='Selected'].copy()

    N_mainshocks_selected = len(mainshocks_selected)
        
    output_path = '../outputs/' + catalogue_name + '/Moutote_method/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    mainshocks_selected.to_csv(output_path + 'mainshocks.csv', index=False)
        
    print(str(N_mainshocks_selected) + ' mainshocks selected')
    print(" ")

    if more_detail==True:
        return earthquakes_above_magnitude_threshold
    else:
        return mainshocks_selected, N_potential_mainshocks, N_mainshocks_selected


def select_mainshocks_fixed_thresholds(earthquake_catalogue,
                                     catalogue_name,
                                     mainshock_magnitude_threshold = 4,
                                     search_distance_km = 10,
                                     event_threshold=10,
                                     more_detail=False,
                                     start_date=dt.datetime(2009,1,1),
                                     end_date=dt.datetime(2016,12,31),
                                     local_catalogue_time=365
                                    ):
    """
    Select mainshocks from an earthquake catalogue using the method from Moutote et al. (2021).
    Attempted exact recreation.
    """
    print("Fixed exclusion threshold method")
    print(catalogue_name)
    
    earthquakes_above_magnitude_threshold = earthquake_catalogue.loc[earthquake_catalogue['MAGNITUDE'] >= mainshock_magnitude_threshold].copy()
    
    N_potential_mainshocks = len(earthquakes_above_magnitude_threshold)
    print(str(N_potential_mainshocks) + ' potential mainshocks')

    states = []
    local_catalogue_lengths = []
    completeness_magnitudes = []
    exclusion_dict = {}
    for mainshock in earthquakes_above_magnitude_threshold.itertuples():

        min_box_lon, min_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, -search_distance_km, -search_distance_km)
        max_box_lon, max_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, search_distance_km, search_distance_km)

        local_catalogue = earthquake_catalogue.loc[
                                        (earthquake_catalogue['LON']>= min_box_lon) &\
                                        (earthquake_catalogue['LON']<= max_box_lon) &\
                                        (earthquake_catalogue['LAT']>= min_box_lat) &\
                                        (earthquake_catalogue['LAT']<= max_box_lat)
                                        ].copy()

        local_catalogue_1yr = local_catalogue[(local_catalogue.DATETIME <= mainshock.DATETIME) &\
                                        ((mainshock.DATETIME - local_catalogue.DATETIME) < dt.timedelta(days=local_catalogue_time)) &\
                                        (local_catalogue['ID'] != mainshock.ID)
                                        ].copy()

        len_local_catalogue = len(local_catalogue_1yr) # if want at least 10 events in 1 year before mainshocks
#         len_local_catalogue = len(local_catalogue) # 10 events within whole catalogue before mainshocks
        local_catalogue_lengths.append(len_local_catalogue)
        
        if len_local_catalogue > event_threshold:
            max_magnitude = max(local_catalogue_1yr['MAGNITUDE'])
            Mc = Mc_by_maximum_curvature(local_catalogue_1yr['MAGNITUDE'])

            if (max_magnitude <= mainshock.MAGNITUDE) & (mainshock.DATETIME >= start_date) & (mainshock.DATETIME <= end_date):
                state = 'Selected'

            elif (max_magnitude <= mainshock.MAGNITUDE) & ((mainshock.DATETIME < start_date) | (mainshock.DATETIME > end_date)):
                state = 'Outside timeframe'

            elif max_magnitude > mainshock.MAGNITUDE:
                exclusion_IDs = list(local_catalogue.loc[local_catalogue['MAGNITUDE'] > mainshock.MAGNITUDE, 'ID'])
                exclusion_dict.update({mainshock.ID: exclusion_IDs})
                state = 'Larger event'

        elif len_local_catalogue < event_threshold:
#             print('Mainshock not selected - not preceded by 10 or more events')
            state = 'Insufficient events'
            Mc = float('nan')

        else:
#             print('Something else has gone wrong')
            state = 'Error'
            Mc = float('nan')
            
        completeness_magnitudes.append(Mc)    
        states.append(state)

    earthquakes_above_magnitude_threshold['state'] = states
    earthquakes_above_magnitude_threshold['len_local_catalogue'] = local_catalogue_lengths
    earthquakes_above_magnitude_threshold['Mc'] = completeness_magnitudes
    
    mainshocks_selected = earthquakes_above_magnitude_threshold.loc[earthquakes_above_magnitude_threshold['state']=='Selected'].copy()

    N_mainshocks_selected = len(mainshocks_selected)
        
    output_path = '../outputs/' + catalogue_name + '/Moutote_method/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    mainshocks_selected.to_csv(output_path + 'mainshocks.csv', index=False)
        
    print(str(N_mainshocks_selected) + ' mainshocks selected')
    print(" ")

    if more_detail==True:
        return earthquakes_above_magnitude_threshold#, mainshocks_selected, N_potential_mainshocks, N_mainshocks_selected
    else:
        return mainshocks_selected, N_potential_mainshocks, N_mainshocks_selected, #, exclusion_dict


def identify_foreshocks(mainshock_file,
                        earthquake_catalogue,
                        catalogue_name, 
                        iterations=10000, 
                        local_catalogue_radius = 10, # km 
                        search_style='radius',
                        search_radius = 10, # km
                        foreshock_window = 20, # days
                        modelling_time_period=365, # days
                        significance_level = 0.01, # (1%)
                        plot=True,
                        show_plots=False,
                        erase_local_catalogues=False,
                        Mc_cutoff=True,
                        save=True,
                        observation_time_scalar=4,
                        Wetzler_cutoff=3
                       ):
    """
    Function to identify foreshocks prior to mainshocks using the methods from Trugman and Ross (2019) and van den Ende & Ampuero (2020). 
    Function creates local catalogues around mainshocks, then creates seismicity rate probability models with which to deem foreshocks as anomalous.
    Plots the local catalogue earthquake time series, and histograms of seismicity rate probability models.
    Returns (1) the mainshock file with seisimicity rate probability model thresholds, and 
    (2) a table of the foreshock rate for all mainshocks according to each model.
    """

    min_observation_period=foreshock_window*observation_time_scalar

    catalogue_start_date = earthquake_catalogue['DATETIME'].iloc[0]

    mainshock_Mw_threshold = math.floor(mainshock_file['MAGNITUDE'].min())
    results_name = f"Mw_{mainshock_Mw_threshold}_iter{iterations}_Mc_cut_{Mc_cutoff}_{search_radius}km_{foreshock_window}day_{modelling_time_period}day"

    if erase_local_catalogues==True:
        # print("New run, exciting! Deleting previous local catalogues")
        files = glob.glob('../data/' + catalogue_name + '/local_catalogues/*.csv')
        for f in files:
            os.remove(f)
    
    method_dict = {"ESR":'ESR',
                #    "VA_2nd_method":'ESR',
                   "VA_method":'G-IET',
                   "Max_window":'Max_rate',
                   "VA_half_method":'Random Inter-Event Times',
                   "TR_method":'Background Poisson',
                  }
    
    align = 'left'

    foreshocks_colour = 'red'
    regular_earthquakes_colour = 'black'
    mainshock_colour = 'gold'
    poisson_colour = 'green'
    gamma_colour = '#FFA500'

    # JOB - stop setting copy warning
    mainshock_file['observation_time'] = (mainshock_file['DATETIME'].copy() - catalogue_start_date).apply(lambda x: x.total_seconds()/3600/24 - foreshock_window)
    # mainshock_file['observation_time'] = [(x-catalogue_start_date).total_seconds()/3600/24 - foreshock_window for x in mainshock_file['DATETIME']]

    # print(catalogue_name)
    # print(f"Number of mainshocks: {len(mainshock_file)}")
    mainshock_file_min_obs_time = mainshock_file.loc[mainshock_file['observation_time']>=min_observation_period].copy()
    # print(f"Number of mainshocks with sufficeint observation time: {len(mainshock_file_min_obs_time)}")
    # print(" ")
        
    t_day = 3600 * 24.0 
    t_win = foreshock_window * t_day    
    count = 1
    
    results_dict = {}
    results = []
    for mainshock in mainshock_file_min_obs_time.itertuples():
        # print(f"{count} / '{len(mainshock_file_min_obs_time)} mainshocks")
        # count +=1
                
        # print(catalogue_name)
        print(f"\r {count} of {len(mainshock_file)}")
        print(f"\r    mainshock ID: {mainshock.ID}")
    
        try:
            local_catalogue = pd.read_csv('../data/' + catalogue_name + '/local_catalogues/' + str(mainshock.ID) + '.csv')
            string_to_datetime_df(local_catalogue)
            # print("    succesfully loaded in data")
        except:
            # print("    data not found - creating")

            box_halfwidth_km = search_radius
            min_box_lon, min_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, -box_halfwidth_km, -box_halfwidth_km)
            max_box_lon, max_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, box_halfwidth_km, box_halfwidth_km)

            local_catalogue = earthquake_catalogue.loc[
                                            (earthquake_catalogue['LON']>= min_box_lon) &\
                                            (earthquake_catalogue['LON']<= max_box_lon) &\
                                            (earthquake_catalogue['LAT']>= min_box_lat) &\
                                            (earthquake_catalogue['LAT']<= max_box_lat)
                                            ].copy()

            local_catalogue['DAYS_TO_MAINSHOCK'] = (mainshock.DATETIME - local_catalogue['DATETIME']).apply(lambda d: (d.total_seconds()/(24*3600)))

            local_catalogue['DISTANCE_TO_MAINSHOCK'] = calculate_distance_pyproj_vectorized(mainshock.LON, mainshock.LAT, local_catalogue['LON'],  local_catalogue['LAT'])
            # local_catalogue['DISTANCE_TO_MAINSHOCK'] = calculate_distances_haversine_vect(mainshock.LON, mainshock.LAT, local_catalogue['LON'],  local_catalogue['LAT'])

            if save==True:
                Path('../data/' + catalogue_name + '/local_catalogues/').mkdir(parents=True, exist_ok=True)
                local_catalogue.to_csv('../data/' + catalogue_name + '/local_catalogues/' + str(mainshock.ID) + '.csv', index=False)
                
        if search_style=='radius':
                local_catalogue = local_catalogue[(local_catalogue['DATETIME'] < mainshock.DATETIME) &\
                                                        (local_catalogue['DAYS_TO_MAINSHOCK'] < modelling_time_period+foreshock_window) &\
                                                        (local_catalogue['DAYS_TO_MAINSHOCK'] > 0)  &\
                                                        (local_catalogue['DISTANCE_TO_MAINSHOCK'] < local_catalogue_radius) &\
                                                        (local_catalogue['ID'] != mainshock.ID)
                                                        ].copy()
                
        # elif (search_style=='box') & (local_catalogue_radius!=search_radius):
        elif (search_style=='box'):
            box_halfwidth_km = search_radius
            min_box_lon, min_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, -box_halfwidth_km, -box_halfwidth_km)
            max_box_lon, max_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, box_halfwidth_km, box_halfwidth_km)

            local_catalogue = local_catalogue.loc[
                                            (local_catalogue['LON']>= min_box_lon) &\
                                            (local_catalogue['LON']<= max_box_lon) &\
                                            (local_catalogue['LAT']>= min_box_lat) &\
                                            (local_catalogue['LAT']<= max_box_lat) &\
                                            (local_catalogue['DATETIME'] < mainshock.DATETIME) &\
                                            (local_catalogue['DAYS_TO_MAINSHOCK'] < modelling_time_period+foreshock_window) &\
                                             (local_catalogue['DAYS_TO_MAINSHOCK'] > 0)  &\
                                            (local_catalogue['ID'] != mainshock.ID)
                                            ].copy()
            
        # print(f"    len(local_catalogue) {len(local_catalogue)}")
            
        local_catalogue_pre_Mc_cutoff = local_catalogue.copy()
        
        try:
            Mc = round(Mc_by_maximum_curvature(local_catalogue['MAGNITUDE']),2)
        except:
            Mc = float('nan')
        # print(f"    Mc {Mc}")
        
        local_catalogue_below_Mc = local_catalogue.loc[local_catalogue['MAGNITUDE']<Mc].copy()
        foreshocks_below_Mc = local_catalogue_below_Mc.loc[local_catalogue_below_Mc['DAYS_TO_MAINSHOCK']<foreshock_window]

        # local_catalogue_below_Mc = local_catalogue_below_Mc.loc[(local_catalogue_below_Mc['DAYS_TO_MAINSHOCK']) < modelling_time_period].copy()
        
        if Mc_cutoff==True:
            local_catalogue = local_catalogue.loc[local_catalogue['MAGNITUDE']>=Mc].copy()
        else:
            local_catalogue = local_catalogue_pre_Mc_cutoff.copy()
        
        regular_seismicity_period = local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] >= foreshock_window)]
        foreshocks = local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] < foreshock_window)]
        
        n_local_catalogue_pre_Mc_cutoff = len(local_catalogue_pre_Mc_cutoff)
        n_local_catalogue = len(local_catalogue)
        n_local_catalogue_below_Mc = len(local_catalogue_below_Mc)
        n_regular_seismicity_events = len(regular_seismicity_period)
        n_events_in_foreshock_window = len(foreshocks)
    
        # print(f"    local_catalogue_pre_Mc_cutoff {len(local_catalogue_pre_Mc_cutoff)}")
        # print(f"    n_local_catalogue ({search_radius} km) {n_local_catalogue}")
        # print(f"    n_local_catalogue_below_Mc {n_local_catalogue_below_Mc}")
        # print(f"    n_regular_seismicity_events {n_regular_seismicity_events}")
        # print(f"    n_events_in_foreshock_window {n_events_in_foreshock_window}")
        # print(f"    observation_time_period {mainshock.observation_time}")

        b_values = []
        for seismicity_period in [local_catalogue, foreshocks, regular_seismicity_period]:
            try:
                b_value = round(b_val_max_likelihood(seismicity_period['MAGNITUDE'], mc=Mc), 2)
            except:
                b_value = float('nan')
            b_values.append(b_value)
        overall_b_value, foreshock_b_value, regular_b_value = b_values
        # print(f"    overall_b_value {overall_b_value}, foreshock_b_value {foreshock_b_value}, regular_b_value {regular_b_value}")

        ##############################
        ### WETZLER WINDOW METHOD ###
        Wetzler_foreshocks = foreshocks.loc[foreshocks['MAGNITUDE']>Wetzler_cutoff].copy()
        N_Wetzler_foreshocks = len(Wetzler_foreshocks)
        ##############################

        ##############################
        ### MAX RATE /ESR 2.0 METHOD ###
        time_since_catalogue_start = (mainshock.DATETIME - catalogue_start_date).total_seconds()/3600/24
        cut_off_day = math.floor(time_since_catalogue_start)
        if cut_off_day > 365:
            cut_off_day = 365
        range_scaler = 100    
        
        sliding_window_points = np.array(range((-cut_off_day+foreshock_window)*range_scaler, -foreshock_window*range_scaler+1, 1))/range_scaler*-1

        sliding_window_counts = np.array([len(regular_seismicity_period[(regular_seismicity_period['DAYS_TO_MAINSHOCK'] > point) & (regular_seismicity_period['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]) for point in sliding_window_points])
            
        try:
            max_window = max(sliding_window_counts)
        except:
            max_window = float('nan')

        if n_events_in_foreshock_window > max_window:
            max_window_method = 0.0
        elif n_events_in_foreshock_window <= max_window:
            max_window_method = 1.0
        else:
            max_window_method = float('nan')

        sliding_window_probability = len(sliding_window_counts[sliding_window_counts >= n_events_in_foreshock_window])/len(sliding_window_counts)
        # sliding_window_probability = len(list(filter(lambda c: c >= n_events_in_foreshock_window, sliding_window_counts)))/len(sliding_window_counts)
        sliding_window_99CI = np.percentile(sliding_window_counts,99)

        ##############################
    
        ##############################
        # Creating time series for probability based models
        if not regular_seismicity_period.empty:
            time_series = np.array(regular_seismicity_period['DATETIME'].apply(lambda d: (d-regular_seismicity_period['DATETIME'].iloc[0]).total_seconds()/3600/24))
        else:
            time_series = np.array([])

        ###################################
        ### TR BACKGROUND POISSON MODEL ###
        if n_regular_seismicity_events >= 2:
            background_rate = gamma_law_MLE(time_series)
            TR_expected_events = background_rate*foreshock_window
            TR_probability = poisson.sf(n_events_in_foreshock_window, TR_expected_events)
            TR_99CI = poisson.ppf(0.99, TR_expected_events)
        elif n_regular_seismicity_events < 2:
            background_rate, TR_expected_events, TR_99CI = [float('nan')]*3
            if (n_events_in_foreshock_window==0):
                TR_probability = 1.00
            elif (n_events_in_foreshock_window > n_regular_seismicity_events):
                TR_probability = 0.00
            else:
                TR_probability = float('nan')
        else:
            background_rate, TR_expected_events, TR_probability, TR_99CI = [float('nan')]*4
        # print(f"    TR_expected_events: {TR_expected_events}")
        ##############################

        ########################################################
        ### MONTE-CARLO SAMPLING METHODS: G-IET MODELS ###
        if n_regular_seismicity_events > 2:
            IET = np.diff(time_series) ### V&As Gamma IET method
            IET = IET[IET>0]
            try:
                y_, loc_, mu_ = gamma.fit(IET, floc=0.0)
            except:
                y_, loc_, mu_ = gamma.fit(IET, loc=0.0)
            # print(f"y_ {y_}, loc_ {loc_}, mu_ {mu_}")
        
            # upper_time_limit = math.ceil(max(time_series)) - foreshock_window # wrong, went from 1st earthquake, not from 1 year prior to mainshock or start of catalogue (if less than 1 year) 
            upper_time_limit = cut_off_day - foreshock_window 
    
            if (np.isnan(y_)==False) & (np.isnan(mu_)==False):
                # print("     Creating ESR & G-IET models.")
                N_eq = np.zeros(iterations, dtype=int) # Buffer for the number of earthquakes observed in each random sample
                for i in range(0,iterations):
                        
                    ## V&A IET method
                    prev_size = 200 # Generate a random IET sample with 200 events
                    IET2 = gamma.rvs(a=y_, loc=0, scale=mu_, size=prev_size) * t_day # Sample from gamma distribution
                    t0 = np.random.rand() * IET2[0] # Random shift of timing of first event
                    t_sum = np.cumsum(IET2) - t0 # Cumulative sum of interevent times
                    inds = (t_sum > t_win) # Find the events that lie outside t_win
                    while (inds.sum() == 0):
                        prev_size *= 2 # If no events lie outside t_win, create a bigger sample and stack with previous sample
                        IET2 = np.hstack([IET2, gamma.rvs(a=y_, loc=0, scale=mu_, size=prev_size) * t_day])
                        t_sum = np.cumsum(IET2) # Cumulative sum of event times
                        inds = (t_sum > t_win) # Find the events that lie outside t_win
                    N_inside_t_win = (~inds).sum()
                    if N_inside_t_win == 0: 
                        N_eq[i] = 0 # No events inside t_win, seismicity rate = 0.
                    else:
                        N_eq[i] =  N_inside_t_win - 1 # Store the number of events that lie within t_win (excluding shifted event)

                try:
                    y_gam_IETs, loc_gam_IETs, mu_gam_IETs = gamma.fit(N_eq[N_eq > 0], floc=0.0)
                except:
                    y_gam_IETs, loc_gam_IETs, mu_gam_IETs = gamma.fit(N_eq[N_eq > 0], loc=0.0)
                
                # print(f"y_gam_IETs {y_gam_IETs}, loc_gam_IETs {loc_gam_IETs}, mu_gam_IETs {mu_gam_IETs}")
                VA_gamma_probability = gamma.sf(n_events_in_foreshock_window, y_gam_IETs, loc_gam_IETs, mu_gam_IETs)
                VA_gamma_99CI = gamma.ppf(0.99, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
                VA_IETs_probability = len(N_eq[N_eq>=n_events_in_foreshock_window])/iterations
                VA_IETs_99CI = np.percentile(N_eq,99)

        elif n_regular_seismicity_events <= 2:
            y_gam_IETs, loc_gam_IETs, mu_gam_IETs = [float('nan')]*3
            N_eq = np.array([])
            VA_gamma_99CI,  VA_IETs_99CI = [float('nan')]*2
            if (n_events_in_foreshock_window == 0):
                VA_gamma_probability, VA_IETs_probability = [1.00]*2
            elif (n_events_in_foreshock_window > n_regular_seismicity_events):
                VA_gamma_probability, VA_IETs_probability = [0.00]*2
            else:
                VA_gamma_probability, VA_IETs_probability, VA_gamma_99CI,  VA_IETs_99CI = [float('nan')]*4
        else:
            N_eq = np.array([])
            y_gam_IETs, loc_gam_IETs, mu_gam_IETs = [float('nan')]*3
            VA_gamma_probability, VA_IETs_probability, VA_gamma_99CI,  VA_IETs_99CI = [float('nan')]*4

        ########################################################
                
        results_dict = {'ID':mainshock.ID,
                        'observation_time_period':mainshock.observation_time,
                        'modelling_time':cut_off_day,
                        'Mc':Mc,
                        'n_regular_seismicity_events':n_regular_seismicity_events,
                        'n_events_in_foreshock_window':n_events_in_foreshock_window,
                        'n_Wetzler_foreshocks':N_Wetzler_foreshocks,
                        'max_20day_rate':max_window,
                        method_dict['Max_window']:max_window_method,
                        method_dict['ESR']:sliding_window_probability,
                        method_dict['VA_method']:VA_gamma_probability,
                        method_dict['VA_half_method']:VA_IETs_probability,
                        method_dict['TR_method']:TR_probability,
                        method_dict['ESR'] + '_99CI':sliding_window_99CI,
                        method_dict['VA_method'] + '_99CI':VA_gamma_99CI,
                        method_dict['VA_half_method'] + '_99CI':VA_IETs_99CI,
                        method_dict['TR_method'] + '_99CI':TR_99CI,
                        'overall_b_value':overall_b_value,
                        'regular_b_value':regular_b_value,
                        'foreshock_b_value':foreshock_b_value,
                        'y_gam_IETs':y_gam_IETs,
                        'loc_gam_IETs':loc_gam_IETs,
                        'mu_gam_IETs':mu_gam_IETs,
                        'background_rate':background_rate
                        }
        
        if plot == True:

            sliding_window_points_full = np.array(range((-cut_off_day+foreshock_window)*range_scaler, 0*range_scaler+1, 1))/range_scaler*-1
            sliding_window_counts_full = np.array([len(local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] > point) & (local_catalogue['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]) for point in sliding_window_points_full])

            time_series_plot, model_plot, CDF_plot, foreshock_window_plot, Mc_plot = 0, 1, 2, 3, 4
            panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)']
            fig, axs = plt.subplots(5,1, figsize=(10,15))

            title = 'Earthquake ID: ' + str(mainshock.ID)
            
            if len(local_catalogue_pre_Mc_cutoff)>0:
                bins = np.arange(math.floor(min(local_catalogue_pre_Mc_cutoff['MAGNITUDE'])), math.ceil(max(local_catalogue_pre_Mc_cutoff['MAGNITUDE'])), 0.1)
                values, base = np.histogram(local_catalogue_pre_Mc_cutoff['MAGNITUDE'], bins=bins)
                cumulative = np.cumsum(values)
                axs[Mc_plot].plot(base[:-1], len(local_catalogue_pre_Mc_cutoff)-cumulative, label='FMD')
                axs[Mc_plot].axvline(x=Mc, linestyle='--', color='red', label=r'$M_{c}$: ' + str(Mc))
            axs[Mc_plot].set_title(title, fontsize=20)
            axs[Mc_plot].set_title(panel_labels[Mc_plot], fontsize=20, fontweight='bold', loc='left')
            axs[Mc_plot].set_xlabel('M')
            axs[Mc_plot].set_ylabel('N')
            axs[Mc_plot].legend()
            
            axs[time_series_plot].set_title(panel_labels[time_series_plot], fontsize=20, fontweight='bold', loc='left')
            axs[time_series_plot].scatter(0, mainshock.MAGNITUDE, marker='*', s=400, color=mainshock_colour,
                                            label=r'$M_{w}$ ' + str(mainshock.MAGNITUDE) + ' Mainshock',  
                                            zorder=3)
            axs[time_series_plot].axvline(x=foreshock_window, color='red', linestyle='--', 
                                            label = f"{foreshock_window}-day foreshock window",
                                            zorder=4)
            axs[time_series_plot].set_xlabel('Days to mainshock', fontsize=20)
            axs[time_series_plot].set_ylabel('M', fontsize=20)
            axs[time_series_plot].set_xlim(0,cut_off_day+foreshock_window)
            axs[time_series_plot].invert_xaxis()

            if len(local_catalogue) >0:
                axs[time_series_plot].set_yticks(np.arange(math.floor(min(local_catalogue['MAGNITUDE'])), math.ceil(mainshock.MAGNITUDE), 1))
                axs[time_series_plot].scatter(local_catalogue['DAYS_TO_MAINSHOCK'], local_catalogue['MAGNITUDE'],
                                                label= str(n_regular_seismicity_events) + ' Earthquakes for modelling',
                                                color=regular_earthquakes_colour, alpha=0.5,  zorder=1)
                axs[time_series_plot].scatter(local_catalogue_below_Mc['DAYS_TO_MAINSHOCK'], local_catalogue_below_Mc['MAGNITUDE'], 
                                            label= str(len(local_catalogue_below_Mc)) + ' Earthquakes below Mc', 
                                            alpha=0.2, color='cyan')
            if len(foreshocks) > 0:
                axs[time_series_plot].scatter(foreshocks['DAYS_TO_MAINSHOCK'], foreshocks['MAGNITUDE'],
                                                label= str(n_events_in_foreshock_window) + ' Earthquakes in foreshock window (' + r'$N_{obs}$)',
                                                color=foreshocks_colour, alpha=0.5, 
                                                zorder=2)
                axs[foreshock_window_plot].scatter(foreshocks['DAYS_TO_MAINSHOCK'], foreshocks['MAGNITUDE'], color=foreshocks_colour, alpha=0.5,
                                                    label=r'$N_{obs}$: ' + str(n_events_in_foreshock_window))
                axs[foreshock_window_plot].scatter(foreshocks_below_Mc['DAYS_TO_MAINSHOCK'], foreshocks_below_Mc['MAGNITUDE'], 
                                                    label= str(len(foreshocks_below_Mc)) + ' Earthquakes below Mc', 
                                                    alpha=0.2, color='cyan')
                axs[foreshock_window_plot].set_yticks(np.arange(math.floor(min(foreshocks['MAGNITUDE'])), math.ceil(mainshock.MAGNITUDE), 1))
            if np.isnan(Mc)==False:
                axs[time_series_plot].axhline(y=Mc, color='cyan', linestyle='--', 
                                                label = r'$M_{c}$: ' + str(Mc),
                                                zorder=5)
                axs[foreshock_window_plot].axhline(y=Mc, color='cyan', linestyle='--', label = r'$M_{c}$: ' + str(Mc), zorder=5)

            
            ax2 = axs[time_series_plot].twinx()
            ax2.plot(sliding_window_points_full, sliding_window_counts_full, color='#1f77b4', label='Seismicity Rate (earthquakes per 20 days)')
            ax2.set_ylabel('Rate')
            lines, labels = axs[time_series_plot].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # axs[time_series_plot].legend(lines + lines2, labels + labels2, loc='upper left')

            axs[model_plot].set_title(panel_labels[model_plot], fontsize=20, fontweight='bold', loc='left')
            axs[model_plot].set_xlabel('Seismicity Rate (earthquakes per 20 days)', fontsize=20)
            axs[model_plot].set_ylabel('Probability', fontsize=20)
            axs[model_plot].axvline(x=n_events_in_foreshock_window, color='red', label=r'$N_{obs}$: ' + str(n_events_in_foreshock_window))      
            # axs[model_plot].set_xticks(range(0,20,2))

            if len(sliding_window_counts) > 0:
                event_counts_pdf = sliding_window_counts/sum(sliding_window_counts)
                event_counts_cdf = np.cumsum(event_counts_pdf)
                event_counts_sorted = np.sort(sliding_window_counts)

            if len(N_eq) > 0:
                axs[model_plot].hist(N_eq, bins=range(min(N_eq)-1, max(N_eq)+1), color='#ff7f0e',
                                    label='R-IETs' + ': ' + str(round(VA_IETs_probability,3)),
                                    density=True, rwidth=1.0, alpha=0.3, align=align)
                N_eq_pdf = N_eq/sum(N_eq)
                N_eq_cdf = np.cumsum(N_eq_pdf)
                N_eq_sorted = np.sort(N_eq)
                axs[CDF_plot].plot(N_eq_sorted, N_eq_cdf, label=method_dict['VA_method'], color='#ff7f0e', alpha=0.3)

            if (np.isnan(TR_expected_events)==False) & (TR_expected_events!=0):
                x_TR_Poisson = np.arange(poisson.ppf(0.001, TR_expected_events), poisson.ppf(0.999, TR_expected_events))
                y_TR_Poisson = poisson.pmf(x_TR_Poisson, TR_expected_events)
                axs[model_plot].plot(x_TR_Poisson, y_TR_Poisson, 
                        label= method_dict['TR_method'] + ': ' + str(round(TR_probability,3)),
                        alpha=0.9, color=poisson_colour)
                TR_poisson_cdf = poisson.cdf(x_TR_Poisson, TR_expected_events)
                axs[CDF_plot].plot(x_TR_Poisson, TR_poisson_cdf, label=method_dict['TR_method'], alpha=0.9, color=poisson_colour)

            if (np.isnan(y_gam_IETs)==False) & (np.isnan(mu_gam_IETs)==False):
                x_gam_IETs = np.arange(gamma.ppf(0.001, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs),
                                        gamma.ppf(0.999, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs))
                gamma_pdf = gamma.pdf(x_gam_IETs, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
                axs[model_plot].plot(x_gam_IETs, gamma_pdf, 
                                        label= method_dict['VA_method'] + ': ' + str(round(VA_gamma_probability,3)),
                                        alpha=0.9, color=gamma_colour)
                gamma_cdf = gamma.cdf(x_gam_IETs, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
                axs[CDF_plot].plot(x_gam_IETs, gamma_cdf, label= method_dict['VA_method'],
                            color=gamma_colour, alpha=0.9)
            if len(sliding_window_counts) > 0:
                axs[model_plot].hist(sliding_window_counts, bins=range(min(sliding_window_counts)-1, max(sliding_window_counts)+1), color='#1f77b4',
                        density=True, rwidth=1.0, alpha=0.3, align=align, label=method_dict['ESR'] + ': ' + str(round(sliding_window_probability,3)))
                window_counts_pdf = sliding_window_counts/sum(sliding_window_counts)
                window_counts_cdf = np.cumsum(window_counts_pdf)
                window_counts_sorted = np.sort(sliding_window_counts)
                axs[CDF_plot].plot(window_counts_sorted, window_counts_cdf, label=method_dict['ESR'], color='#1f77b4', alpha=0.3)

            axs[model_plot].legend()

    #         handles, labels = plt.gca().get_legend_handles_labels()       #specify order of items in legend
    #         order = range(0,len(handles))
    #         order = [0,1,5,2,4,3]
    #         plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

            axs[foreshock_window_plot].set_title(panel_labels[foreshock_window_plot], fontsize=20, fontweight='bold', loc='left')
            axs[foreshock_window_plot].scatter(0, mainshock.MAGNITUDE, marker='*', s=400, color=mainshock_colour, zorder=2)
            axs[foreshock_window_plot].axvline(x=foreshock_window, color='red', linestyle='--')

            axs[foreshock_window_plot].set_xlabel('Days to mainshock')
            axs[foreshock_window_plot].set_ylabel('M')
            axs[foreshock_window_plot].invert_xaxis()
            axs[foreshock_window_plot].set_xscale('log')
            axs[foreshock_window_plot].legend()

            axs[CDF_plot].set_title(panel_labels[CDF_plot], fontsize=20, fontweight='bold', loc='left')
            axs[CDF_plot].set_xlabel('Number of Earthquakes in 20 days')
            axs[CDF_plot].set_ylabel('CDF')
            axs[CDF_plot].axvline(x=n_events_in_foreshock_window, color='red',
                                  label=r'$N_{obs}$): ' + str(n_events_in_foreshock_window))
            axs[CDF_plot].legend()

            fig.tight_layout()

            if save==True:
                Path(f"../outputs/{catalogue_name}/{results_name}/plots/").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"../outputs/{catalogue_name}/{results_name}/plots/{str(mainshock.ID)}.png")

            if show_plots == False: 
                plt.close()

        # print("Processed " + str(count) + ' of ' + str(len(mainshock_file_min_obs_time)))
        count += 1
        # print(" ")
            
        results.append(results_dict)
        clear_output(wait=True)
        
    results_file = pd.DataFrame.from_dict(results)
    print(results_file.columns)

    results_file = pd.merge(mainshock_file, results_file, on='ID',  #suffixes=('x', ''),
                            how='left')
    print(results_file.columns)

    Path(f'../data/{catalogue_name}/foreshocks/').mkdir(parents=True, exist_ok=True)
    if save==True:
        results_file.to_csv(f'../data/{catalogue_name}/foreshocks/{results_name}.csv', index=False)
    
    # print(results_file)

    result_options_dict = {'FET':(results_file['Selection']=='FET') | (results_file['Selection']=='Both'),
                            'MDET':(results_file['Selection']=='MDET') | (results_file['Selection']=='Both'),
                            'Both':(results_file['Selection']=='Both'),
                            'FET_only':(results_file['Selection']=='FET'),
                            'MDET_only':(results_file['Selection']=='MDET')
                            }

    results_df_list = []
    for name, option in result_options_dict.items():
        results_file = pd.read_csv(f'../data/{catalogue_name}/foreshocks/{results_name}.csv')
        string_to_datetime_df(results_file)
        results_file = results_file.loc[option].copy()
        total_mainshocks = len(results_file)
        number_of_mainshocks_with_Wetzler_foreshocks = len(results_file.loc[results_file['n_Wetzler_foreshocks']>0])
        number_of_mainshocks_with_ge5_Wetzler_foreshocks = len(results_file.loc[results_file['n_Wetzler_foreshocks']>=5])

        all_results_dict = {'Count type':['Mainshocks selected',
                                        'Wetzler time window',
                                        'Wetzler time window ge5'],
                            f"{name}_count":[total_mainshocks,
                                            number_of_mainshocks_with_Wetzler_foreshocks,
                                            number_of_mainshocks_with_ge5_Wetzler_foreshocks]
                                            }

        results_table = []

        method_dict = {"Max_window":'Max_rate',
                    "ESR":'ESR',
                    #    "VA_2nd_method":'ESR',
                    "VA_method":'G-IET',
                    "VA_half_method":'Random Inter-Event Times',
                    "TR_method":'Background Poisson',
                    }
        significance_level = 0.01

        for model in method_dict.values():
            # print(model)
            number_of_mainshocks_with_foreshocks_significance_level = len(results_file[results_file[model] < significance_level])

            all_results_dict['Count type'].append(model)
            all_results_dict[f"{name}_count"].append(number_of_mainshocks_with_foreshocks_significance_level)
            
        results_table = pd.DataFrame.from_dict(all_results_dict)
        # print(name, total_mainshocks)
        results_table[f"{name}_perc"] = 100*results_table[f"{name}_count"]/total_mainshocks
        # JOB - make this work
        # results_table[f"{name}_perc"] = [round(x) for x in results_table[f"{name}_perc"]]
        results_df_list.append(results_table)

    results_table = reduce(lambda df1, df2: pd.merge(df1, df2, on='Count type'), results_df_list)
    results_table.to_csv(f"../outputs/{catalogue_name}/{results_name}/results_table.csv", index=False)
    
    # print('Total number of mainshocks: ' + str(len(mainshock_file)))
    
    return results_file#, results_table


def identify_foreshocks_old(mainshock_file,
                            mainshock_method,
                        earthquake_catalogue,
                        catalogue_name, 
                        iterations=10000, 
                        local_catalogue_radius = 10, # km 
                        search_style='radius',
                        search_radius = 10, # km
                        foreshock_window = 20, # days
                        modelling_time_period=365, # days
                        significance_level = 0.01, # (1%)
                        plot=True,
                        show_plots=False,
                        erase_local_catalogues=False,
                        Mc_cutoff=True,
                        save=True,
                        observation_time_scalar=4,
                        Wetzler_cutoff=3
                       ):
    """
    Function to identify foreshocks prior to mainshocks using the methods from Trugman and Ross (2019) and van den Ende & Ampuero (2020). 
    Function creates local catalogues around mainshocks, then creates seismicity rate probability models with which to deem foreshocks as anomalous.
    Plots the local catalogue earthquake time series, and histograms of seismicity rate probability models.
    Returns (1) the mainshock file with seisimicity rate probability model thresholds, and 
    (2) a table of the foreshock rate for all mainshocks according to each model. [Old version].
    """

    min_observation_period=foreshock_window*observation_time_scalar

    catalogue_start_date = earthquake_catalogue['DATETIME'].iloc[0]

    mainshock_Mw_threshold = math.floor(mainshock_file['MAGNITUDE'].min())
    results_file_name = f"Mw_{mainshock_Mw_threshold}_iter{iterations}_Mc_cut_{Mc_cutoff}_{search_radius}km_{foreshock_window}day_{modelling_time_period}mtp.csv"

    if erase_local_catalogues==True:
        print("New run, exciting! Deleting previous local catalogues")
        files = glob.glob('../outputs/' + catalogue_name + '/local_catalogues/*.csv')
        for f in files:
            os.remove(f)
    
    method_dict = {"ESR_2":'ESR_2',
                    "VA_2nd_method":'ESR',
                   "VA_method":'G-IET',
                   "Max_window":'Max_rate',
#                    "KW_Nbinom_method":'Empirical Negative Binomial',
                   "VA_half_method":'Random Inter-Event Times',
                   "TR_method":'Background Poisson',
                   "Low_rate_method":"Average rate Poisson"
                  }
    
    align = 'left'

    foreshocks_colour = 'red'
    regular_earthquakes_colour = 'black'
    mainshock_colour = 'gold'
    poisson_colour = 'green'
    gamma_colour = '#FFA500'

    mainshock_file['observation_time'] = (mainshock_file['DATETIME'].copy() - catalogue_start_date).apply(lambda x: x.total_seconds()/3600/24 - foreshock_window)

    print(catalogue_name)
    print(f"Number of mainshocks: {len(mainshock_file)}")
    mainshock_file_min_obs_time = mainshock_file.loc[mainshock_file['observation_time']>=min_observation_period].copy()
    print(f"Number of mainshocks with sufficeint observation time: {len(mainshock_file_min_obs_time)}")
    print(" ")
        
    t_day = 3600 * 24.0 
    t_win = foreshock_window * t_day    
    count = 1
    
    results_dict = {}
    results = []
    for mainshock in mainshock_file_min_obs_time.itertuples():
                
        print(catalogue_name, mainshock_method)
        print(f"\r {count} of {len(mainshock_file)}")
        print(f"\r    mainshock ID: {mainshock.ID}")
    
        try:
            local_catalogue = pd.read_csv('../outputs/' + catalogue_name + '/local_catalogues/' + str(mainshock.ID) + '.csv')
            string_to_datetime_df(local_catalogue)
            print("    succesfully loaded in data")
        except:
            print("    data not found - creating")

            box_halfwidth_km = search_radius
            min_box_lon, min_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, -box_halfwidth_km, -box_halfwidth_km)
            max_box_lon, max_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, box_halfwidth_km, box_halfwidth_km)

            local_catalogue = earthquake_catalogue.loc[
                                            (earthquake_catalogue['LON']>= min_box_lon) &\
                                            (earthquake_catalogue['LON']<= max_box_lon) &\
                                            (earthquake_catalogue['LAT']>= min_box_lat) &\
                                            (earthquake_catalogue['LAT']<= max_box_lat)
                                            ].copy()

            local_catalogue['DAYS_TO_MAINSHOCK'] = (mainshock.DATETIME - local_catalogue['DATETIME']).apply(lambda d: (d.total_seconds()/(24*3600)))

            local_catalogue['DISTANCE_TO_MAINSHOCK'] = calculate_distance_pyproj_vectorized(mainshock.LON, mainshock.LAT, local_catalogue['LON'],  local_catalogue['LAT'])
            # local_catalogue['DISTANCE_TO_MAINSHOCK'] = calculate_distances_haversine_vect(mainshock.LON, mainshock.LAT, local_catalogue['LON'],  local_catalogue['LAT'])

            if save==True:
                Path('../outputs/' + catalogue_name + '/local_catalogues/').mkdir(parents=True, exist_ok=True)
                local_catalogue.to_csv('../outputs/' + catalogue_name + '/local_catalogues/' + str(mainshock.ID) + '.csv', index=False)
                
        if search_style=='radius':
                local_catalogue = local_catalogue[(local_catalogue['DATETIME'] < mainshock.DATETIME) &\
                                                        (local_catalogue['DAYS_TO_MAINSHOCK'] < modelling_time_period+foreshock_window) &\
                                                        (local_catalogue['DAYS_TO_MAINSHOCK'] > 0)  &\
                                                        (local_catalogue['DISTANCE_TO_MAINSHOCK'] < local_catalogue_radius) &\
                                                        (local_catalogue['ID'] != mainshock.ID)
                                                        ].copy()
                
        # elif (search_style=='box') & (local_catalogue_radius!=search_radius):
        elif (search_style=='box'):
            box_halfwidth_km = search_radius
            min_box_lon, min_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, -box_halfwidth_km, -box_halfwidth_km)
            max_box_lon, max_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, box_halfwidth_km, box_halfwidth_km)

            local_catalogue = local_catalogue.loc[
                                            (local_catalogue['LON']>= min_box_lon) &\
                                            (local_catalogue['LON']<= max_box_lon) &\
                                            (local_catalogue['LAT']>= min_box_lat) &\
                                            (local_catalogue['LAT']<= max_box_lat) &\
                                            (local_catalogue['DATETIME'] < mainshock.DATETIME) &\
                                            (local_catalogue['DAYS_TO_MAINSHOCK'] < modelling_time_period+foreshock_window) &\
                                             (local_catalogue['DAYS_TO_MAINSHOCK'] > 0)  &\
                                            (local_catalogue['ID'] != mainshock.ID)
                                            ].copy()
            
        print(f"    len(local_catalogue) {len(local_catalogue)}")
            
        local_catalogue_pre_Mc_cutoff = local_catalogue.copy()
        
        try:
            Mc = round(Mc_by_maximum_curvature(local_catalogue['MAGNITUDE']),2)
        except:
            Mc = float('nan')
        print(f"    Mc {Mc}")
        
        local_catalogue_below_Mc = local_catalogue.loc[local_catalogue['MAGNITUDE']<Mc].copy()
        foreshocks_below_Mc = local_catalogue_below_Mc.loc[local_catalogue_below_Mc['DAYS_TO_MAINSHOCK']<foreshock_window]

        # local_catalogue_below_Mc = local_catalogue_below_Mc.loc[(local_catalogue_below_Mc['DAYS_TO_MAINSHOCK']) < modelling_time_period].copy()
        
        if Mc_cutoff==True:
            local_catalogue = local_catalogue.loc[local_catalogue['MAGNITUDE']>=Mc].copy()
        else:
            local_catalogue = local_catalogue_pre_Mc_cutoff.copy()
        
        regular_seismicity_period = local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] >= foreshock_window)]
        foreshocks = local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] < foreshock_window)]
        
        n_local_catalogue_pre_Mc_cutoff = len(local_catalogue_pre_Mc_cutoff)
        n_local_catalogue = len(local_catalogue)
        n_local_catalogue_below_Mc = len(local_catalogue_below_Mc)
        n_regular_seismicity_events = len(regular_seismicity_period)
        n_events_in_foreshock_window = len(foreshocks)
    
        print(f"    local_catalogue_pre_Mc_cutoff {len(local_catalogue_pre_Mc_cutoff)}")
        print(f"    n_local_catalogue ({search_radius} km) {n_local_catalogue}")
        print(f"    n_local_catalogue_below_Mc {n_local_catalogue_below_Mc}")
        print(f"    n_regular_seismicity_events {n_regular_seismicity_events}")
        print(f"    n_events_in_foreshock_window {n_events_in_foreshock_window}")
        print(f"    observation_time_period {mainshock.observation_time}")

        b_values = []
        for seismicity_period in [local_catalogue, foreshocks, regular_seismicity_period]:
            try:
                b_value = round(b_val_max_likelihood(seismicity_period['MAGNITUDE'], mc=Mc), 2)
            except:
                b_value = float('nan')
            b_values.append(b_value)
        overall_b_value, foreshock_b_value, regular_b_value = b_values
        print(f"    overall_b_value {overall_b_value}, foreshock_b_value {foreshock_b_value}, regular_b_value {regular_b_value}")

        ##############################
        ### AVERAGE POISSON MODEL ###
        if n_regular_seismicity_events>0:
            average_seismicity_rate = n_regular_seismicity_events/mainshock.observation_time
            average_Poisson_probability = poisson.sf(n_events_in_foreshock_window, average_seismicity_rate*foreshock_window)
            average_Poisson_99CI = poisson.ppf(0.99, average_seismicity_rate*foreshock_window)
        else:
            average_seismicity_rate = 0
            average_Poisson_99CI = 0
            if (n_events_in_foreshock_window > 0):
                average_Poisson_probability = 0.00
            elif (n_events_in_foreshock_window==0):
                average_Poisson_probability = 1.00
            else:
                average_Poisson_probability, average_Poisson_99CI = [float('nan')]*2
        ##############################

        ##############################
        ### WETZLER WINDOW METHOD ###
        Wetzler_foreshocks = foreshocks.loc[foreshocks['MAGNITUDE']>Wetzler_cutoff].copy()
        N_Wetzler_foreshocks = len(Wetzler_foreshocks)
        ##############################

        ##############################
        ### MAX RATE METHOD ###
        time_since_catalogue_start = (mainshock.DATETIME - catalogue_start_date).total_seconds()/3600/24
        cut_off_day = math.floor(time_since_catalogue_start)
        if cut_off_day > 365:
            cut_off_day = 365
        range_scaler = 100    
        
        sliding_window_points = np.array(range((-cut_off_day+foreshock_window)*range_scaler, -foreshock_window*range_scaler+1, 1))/range_scaler*-1

        sliding_window_counts = np.array([len(regular_seismicity_period[(regular_seismicity_period['DAYS_TO_MAINSHOCK'] > point) & (regular_seismicity_period['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]) for point in sliding_window_points])

        # sliding_window_counts = []
        # for point in sliding_window_points: # old loop, now use new list comp above
        #     windowed_data = regular_seismicity_period[(regular_seismicity_period['DAYS_TO_MAINSHOCK'] > point) & (regular_seismicity_period['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]
        #     window_count = len(windowed_data)
        #     sliding_window_counts.append(window_count)
        # sliding_window_counts = np.array(sliding_window_counts)
            
        try:
            max_window = max(sliding_window_counts)
        except:
            max_window = float('nan')

        if n_events_in_foreshock_window > max_window:
            max_window_method = 0.0
        elif n_events_in_foreshock_window <= max_window:
            max_window_method = 1.0
        else:
            max_window_method = float('nan')

        sliding_window_probability = len(sliding_window_counts[sliding_window_counts >= n_events_in_foreshock_window])/len(sliding_window_counts)
        # sliding_window_probability = len(list(filter(lambda c: c >= n_events_in_foreshock_window, sliding_window_counts)))/len(sliding_window_counts)
        sliding_window_99CI = np.percentile(sliding_window_counts,99)
    
        ##############################
        # Creating time series for probability based models
        if not regular_seismicity_period.empty:
            time_series = np.array(regular_seismicity_period['DATETIME'].apply(lambda d: (d-regular_seismicity_period['DATETIME'].iloc[0]).total_seconds()/3600/24))
        else:
            time_series = np.array([])

        ###################################
        ### TR BACKGROUND POISSON MODEL ###
        if n_regular_seismicity_events >= 2:
            background_rate = gamma_law_MLE(time_series)
            TR_expected_events = background_rate*foreshock_window
            TR_probability = poisson.sf(n_events_in_foreshock_window, TR_expected_events)
            TR_99CI = poisson.ppf(0.99, TR_expected_events)
        elif n_regular_seismicity_events==0:
            background_rate, TR_expected_events = [0]*2
            if (n_events_in_foreshock_window > 0):
                TR_probability = 0.00
                TR_99CI = 1
            elif (n_events_in_foreshock_window==0):
                TR_probability = 1.00
                TR_99CI = 1
        else:
            background_rate, TR_expected_events, TR_probability, TR_99CI = [float('nan')]*4
        print(f"    TR_expected_events: {TR_expected_events}")

        ###################################

        ########################################################
        ### MONTE-CARLO SAMPLING METHODS: ESR & G-IET MODELS ###
        
        if n_regular_seismicity_events > 0:
            IET = np.diff(time_series) ### V&As Gamma IET method
            IET = IET[IET>0]
            try:
                y_, loc_, mu_ = gamma.fit(IET, floc=0.0)
            except:
                y_, loc_, mu_ = gamma.fit(IET, loc=0.0)
            print(f"y_ {y_}, loc_ {loc_}, mu_ {mu_}")
        
            event_counts_in_x_days = []
            # upper_time_limit = math.ceil(max(time_series)) - foreshock_window # wrong, went from 1st earthquake, not from 1 year prior to mainshock or start of catalogue (if less than 1 year) 
            upper_time_limit = cut_off_day - foreshock_window 
    
            if (np.isnan(y_)==False) & (np.isnan(mu_)==False):
                print("     Creating ESR & G-IET models.")
                N_eq = np.zeros(iterations, dtype=int) # Buffer for the number of earthquakes observed in each random sample
                for i in range(0,iterations):
                    
                    random_point = dt.timedelta(days=random.random()*upper_time_limit)
                    
                    random_sample = regular_seismicity_period.loc[((mainshock.DATETIME - regular_seismicity_period['DATETIME']) < (random_point + dt.timedelta(days=foreshock_window))) &\
                                            ((mainshock.DATETIME - regular_seismicity_period['DATETIME']) > random_point)]
                    
                    event_counts_in_x_days.append(len(random_sample))
                        
                    ## V&A IET method
                    prev_size = 200 # Generate a random IET sample with 200 events
                    IET2 = gamma.rvs(a=y_, loc=0, scale=mu_, size=prev_size) * t_day # Sample from gamma distribution
                    t0 = np.random.rand() * IET2[0] # Random shift of timing of first event
                    t_sum = np.cumsum(IET2) - t0 # Cumulative sum of interevent times
                    inds = (t_sum > t_win) # Find the events that lie outside t_win
                    while (inds.sum() == 0):
                        prev_size *= 2 # If no events lie outside t_win, create a bigger sample and stack with previous sample
                        IET2 = np.hstack([IET2, gamma.rvs(a=y_, loc=0, scale=mu_, size=prev_size) * t_day])
                        t_sum = np.cumsum(IET2) # Cumulative sum of event times
                        inds = (t_sum > t_win) # Find the events that lie outside t_win
                    N_eq[i] = (~inds).sum() - 1 # Store the number of events that lie within t_win (excluding shifted event)
                
                try:
                    y_gam_IETs, loc_gam_IETs, mu_gam_IETs = gamma.fit(N_eq[N_eq > 0], floc=0.0)
                except:
                    y_gam_IETs, loc_gam_IETs, mu_gam_IETs = gamma.fit(N_eq[N_eq > 0], loc=0.0)
                
                print(f"y_gam_IETs {y_gam_IETs}, loc_gam_IETs {loc_gam_IETs}, mu_gam_IETs {mu_gam_IETs}")
                VA_gamma_probability = gamma.sf(n_events_in_foreshock_window, y_gam_IETs, loc_gam_IETs, mu_gam_IETs)
                VA_gamma_99CI = gamma.ppf(0.99, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
                VA_IETs_probability = len(N_eq[N_eq>=n_events_in_foreshock_window])/iterations
                VA_IETs_99CI = np.percentile(N_eq,99)

            else:
                print("     Creating ESR model but not G-IET model.")
                for i in range(0,iterations):

                    random_point = dt.timedelta(days=random.random()*upper_time_limit)
                    
                    random_sample = regular_seismicity_period.loc[((mainshock.DATETIME - regular_seismicity_period['DATETIME']) < (random_point + dt.timedelta(days=foreshock_window))) &\
                                            ((mainshock.DATETIME - regular_seismicity_period['DATETIME']) > random_point)]
                    
                    event_counts_in_x_days.append(len(random_sample))

                VA_gamma_probability, VA_gamma_99CI, y_gam_IETs, loc_gam_IETs, mu_gam_IETs, VA_IETs_probability, VA_IETs_99CI = [float('nan')]*7
                N_eq = np.array([])
                
            event_counts_in_x_days = np.array(event_counts_in_x_days)

            # Empirical_probability = len(list(filter(lambda c: c >= n_events_in_foreshock_window, event_counts_in_x_days)))/iterations
            Empirical_probability = len(event_counts_in_x_days[event_counts_in_x_days>=n_events_in_foreshock_window])/iterations
            Empirical_99CI = np.percentile(event_counts_in_x_days,99)
            
        else:
            print("    n=0: Not performing random sampling of regular seismicity")
            y_gam_IETs, loc_gam_IETs, mu_gam_IETs = [float('nan')]*3
            event_counts_in_x_days = np.array([])
            N_eq = np.array([])
            VA_gamma_99CI, Empirical_99CI,  VA_IETs_99CI = [1]*3
            if (n_events_in_foreshock_window > 0):
                print(f"(n_events_in_foreshock_window > 0)")
                VA_gamma_probability, Empirical_probability, VA_IETs_probability = [0.00]*3
            else:
                print(f"(n_events_in_foreshock_window==0)")
                VA_gamma_probability, Empirical_probability, VA_IETs_probability = [1.00]*3

        ########################################################
                
        results_dict = {'ID':mainshock.ID,
                        'LON':mainshock.LON,
                        'LAT':mainshock.LAT,
                        'DEPTH':mainshock.DEPTH,
                        'DATETIME':mainshock.DATETIME,
                        'observation_time_period':mainshock.observation_time,
                        'modelling_time':cut_off_day,
                        'MAGNITUDE':mainshock.MAGNITUDE,
                        'Mc':Mc,
                        'n_regular_seismicity_events':n_regular_seismicity_events,
                        'n_events_in_foreshock_window':n_events_in_foreshock_window,
                        'n_Wetzler_foreshocks':N_Wetzler_foreshocks,
                        'max_20day_rate':max_window,
                        method_dict['Max_window']:max_window_method,
                        method_dict['ESR_2']:sliding_window_probability,
                        method_dict['VA_2nd_method']:Empirical_probability,
                        method_dict['VA_method']:VA_gamma_probability,
                        method_dict['VA_half_method']:VA_IETs_probability,
                        method_dict['TR_method']:TR_probability,
                        method_dict['Low_rate_method']:average_Poisson_probability,
                        method_dict['ESR_2'] + '_99CI':sliding_window_99CI,
                        method_dict['VA_2nd_method'] + '_99CI':Empirical_99CI,
                        method_dict['VA_method'] + '_99CI':VA_gamma_99CI,
                        method_dict['VA_half_method'] + '_99CI':VA_IETs_99CI,
                        method_dict['TR_method'] + '_99CI':TR_99CI,
                        method_dict['Low_rate_method'] + '_99CI':average_Poisson_99CI,
                        'overall_b_value':overall_b_value,
                        'regular_b_value':regular_b_value,
                        'foreshock_b_value':foreshock_b_value,
                        'y_gam_IETs':y_gam_IETs,
                        'loc_gam_IETs':loc_gam_IETs,
                        'mu_gam_IETs':mu_gam_IETs,
                        'background_rate':background_rate,
                        'average_seismicity_rate':average_seismicity_rate
                    }
        
        if plot == True:

            sliding_window_points_full = np.array(range((-cut_off_day+foreshock_window)*range_scaler, 0*range_scaler+1, 1))/range_scaler*-1
            sliding_window_counts_full = np.array([len(local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] > point) & (local_catalogue['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]) for point in sliding_window_points_full])

            time_series_plot, model_plot, CDF_plot, foreshock_window_plot, Mc_plot = 0, 1, 2, 3, 4
            panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)']
            fig, axs = plt.subplots(5,1, figsize=(10,15))

            title = 'Earthquake ID: ' + str(mainshock.ID)
            
            if len(local_catalogue_pre_Mc_cutoff)>0:
                bins = np.arange(math.floor(min(local_catalogue_pre_Mc_cutoff['MAGNITUDE'])), math.ceil(max(local_catalogue_pre_Mc_cutoff['MAGNITUDE'])), 0.1)
                values, base = np.histogram(local_catalogue_pre_Mc_cutoff['MAGNITUDE'], bins=bins)
                cumulative = np.cumsum(values)
                axs[Mc_plot].plot(base[:-1], len(local_catalogue_pre_Mc_cutoff)-cumulative, label='FMD')
                axs[Mc_plot].axvline(x=Mc, linestyle='--', color='red', label=r'$M_{c}$: ' + str(Mc))
            axs[Mc_plot].set_title(title, fontsize=20)
            axs[Mc_plot].set_title(panel_labels[Mc_plot], fontsize=20, fontweight='bold', loc='left')
            axs[Mc_plot].set_xlabel('M')
            axs[Mc_plot].set_ylabel('N')
            axs[Mc_plot].legend()
            
            axs[time_series_plot].set_title(panel_labels[time_series_plot], fontsize=20, fontweight='bold', loc='left')
            axs[time_series_plot].scatter(0, mainshock.MAGNITUDE, marker='*', s=400, color=mainshock_colour,
                                            label=r'$M_{w}$ ' + str(mainshock.MAGNITUDE) + ' Mainshock',  
                                            zorder=3)
            axs[time_series_plot].axvline(x=foreshock_window, color='red', linestyle='--', 
                                            label = f"{foreshock_window}-day foreshock window",
                                            zorder=4)
            axs[time_series_plot].set_xlabel('Days to mainshock', fontsize=20)
            axs[time_series_plot].set_ylabel('M', fontsize=20)
            axs[time_series_plot].set_xlim(0,cut_off_day+foreshock_window)
            axs[time_series_plot].invert_xaxis()

            if len(local_catalogue) >0:
                axs[time_series_plot].set_yticks(np.arange(math.floor(min(local_catalogue['MAGNITUDE'])), math.ceil(mainshock.MAGNITUDE), 1))
                axs[time_series_plot].scatter(local_catalogue['DAYS_TO_MAINSHOCK'], local_catalogue['MAGNITUDE'],
                                                label= str(n_regular_seismicity_events) + ' Earthquakes for modelling',
                                                color=regular_earthquakes_colour, alpha=0.5,  zorder=1)
                axs[time_series_plot].scatter(local_catalogue_below_Mc['DAYS_TO_MAINSHOCK'], local_catalogue_below_Mc['MAGNITUDE'], 
                                            label= str(len(local_catalogue_below_Mc)) + ' Earthquakes below Mc', 
                                            alpha=0.2, color='cyan')
            if len(foreshocks) > 0:
                axs[time_series_plot].scatter(foreshocks['DAYS_TO_MAINSHOCK'], foreshocks['MAGNITUDE'],
                                                label= str(n_events_in_foreshock_window) + ' Earthquakes in foreshock window (' + r'$N_{obs}$)',
                                                color=foreshocks_colour, alpha=0.5, 
                                                zorder=2)
                axs[foreshock_window_plot].scatter(foreshocks['DAYS_TO_MAINSHOCK'], foreshocks['MAGNITUDE'], color=foreshocks_colour, alpha=0.5,
                                                    label=r'$N_{obs}$: ' + str(n_events_in_foreshock_window))
                axs[foreshock_window_plot].scatter(foreshocks_below_Mc['DAYS_TO_MAINSHOCK'], foreshocks_below_Mc['MAGNITUDE'], 
                                                    label= str(len(foreshocks_below_Mc)) + ' Earthquakes below Mc', 
                                                    alpha=0.2, color='cyan')
                axs[foreshock_window_plot].set_yticks(np.arange(math.floor(min(foreshocks['MAGNITUDE'])), math.ceil(mainshock.MAGNITUDE), 1))
            if np.isnan(Mc)==False:
                axs[time_series_plot].axhline(y=Mc, color='cyan', linestyle='--', 
                                                label = r'$M_{c}$: ' + str(Mc),
                                                zorder=5)
                axs[foreshock_window_plot].axhline(y=Mc, color='cyan', linestyle='--', label = r'$M_{c}$: ' + str(Mc), zorder=5)

            
            ax2 = axs[time_series_plot].twinx()
            ax2.plot(sliding_window_points_full, sliding_window_counts_full, color='#1f77b4', label='Seismicity Rate (earthquakes per 20 days)')
            ax2.set_ylabel('Rate')
            lines, labels = axs[time_series_plot].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # axs[time_series_plot].legend(lines + lines2, labels + labels2, loc='upper left')

            axs[model_plot].set_title(panel_labels[model_plot], fontsize=20, fontweight='bold', loc='left')
            axs[model_plot].set_xlabel('Seismicity Rate (earthquakes per 20 days)', fontsize=20)
            axs[model_plot].set_ylabel('Probability', fontsize=20)
            axs[model_plot].axvline(x=n_events_in_foreshock_window, color='red', label=r'$N_{obs}$: ' + str(n_events_in_foreshock_window))      
            # axs[model_plot].set_xticks(range(0,20,2))

            if len(event_counts_in_x_days) > 0:
                axs[model_plot].hist(event_counts_in_x_days, bins=range(min(event_counts_in_x_days)-1, max(event_counts_in_x_days)+1), color='#1f77b4',
                                        density=True, rwidth=1.0, alpha=0.3, align=align, label=method_dict['VA_2nd_method'] + ': ' + str(round(Empirical_probability,3)))
                event_counts_pdf = event_counts_in_x_days/sum(event_counts_in_x_days)
                event_counts_cdf = np.cumsum(event_counts_pdf)
                event_counts_sorted = np.sort(event_counts_in_x_days)
                axs[CDF_plot].plot(event_counts_sorted, event_counts_cdf, label=method_dict['VA_2nd_method'], color='#1f77b4', alpha=0.3)

            if len(N_eq) > 0:
                axs[model_plot].hist(N_eq, bins=range(min(N_eq)-1, max(N_eq)+1), color='#ff7f0e',
                                    label='R-IETs' + ': ' + str(round(VA_IETs_probability,3)),
                                    density=True, rwidth=1.0, alpha=0.3, align=align)
                N_eq_pdf = N_eq/sum(N_eq)
                N_eq_cdf = np.cumsum(N_eq_pdf)
                N_eq_sorted = np.sort(N_eq)
                axs[CDF_plot].plot(N_eq_sorted, N_eq_cdf, label=method_dict['VA_method'], color='#ff7f0e', alpha=0.3)

            if (np.isnan(TR_expected_events)==False) & (TR_expected_events!=0):
                x_TR_Poisson = np.arange(poisson.ppf(0.001, TR_expected_events), poisson.ppf(0.999, TR_expected_events))
                y_TR_Poisson = poisson.pmf(x_TR_Poisson, TR_expected_events)
                axs[model_plot].plot(x_TR_Poisson, y_TR_Poisson, 
                        label= method_dict['TR_method'] + ': ' + str(round(TR_probability,3)),
                        alpha=0.9, color=poisson_colour)
                TR_poisson_cdf = poisson.cdf(x_TR_Poisson, TR_expected_events)
                axs[CDF_plot].plot(x_TR_Poisson, TR_poisson_cdf, label=method_dict['TR_method'], alpha=0.9, color=poisson_colour)

            if (np.isnan(y_gam_IETs)==False) & (np.isnan(mu_gam_IETs)==False):
                x_gam_IETs = np.arange(gamma.ppf(0.001, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs),
                                        gamma.ppf(0.999, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs))
                gamma_pdf = gamma.pdf(x_gam_IETs, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
                axs[model_plot].plot(x_gam_IETs, gamma_pdf, 
                                        label= method_dict['VA_method'] + ': ' + str(round(VA_gamma_probability,3)),
                                        alpha=0.9, color=gamma_colour)
                gamma_cdf = gamma.cdf(x_gam_IETs, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
                axs[CDF_plot].plot(x_gam_IETs, gamma_cdf, label= method_dict['VA_method'],
                            color=gamma_colour, alpha=0.9)
            if len(sliding_window_counts) > 0:
                axs[model_plot].hist(sliding_window_counts, bins=range(min(sliding_window_counts)-1, max(sliding_window_counts)+1), color='yellow',
                        density=True, rwidth=1.0, alpha=0.3, align=align, label=method_dict['ESR'] + ': ' + str(round(sliding_window_probability,3)))
                window_counts_pdf = sliding_window_counts/sum(sliding_window_counts)
                window_counts_cdf = np.cumsum(window_counts_pdf)
                window_counts_sorted = np.sort(sliding_window_counts)
                axs[CDF_plot].plot(window_counts_sorted, window_counts_cdf, label=method_dict['ESR'], color='yellow', alpha=0.3)

            axs[model_plot].legend()

    #         handles, labels = plt.gca().get_legend_handles_labels()       #specify order of items in legend
    #         order = range(0,len(handles))
    #         order = [0,1,5,2,4,3]
    #         plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

            axs[foreshock_window_plot].set_title(panel_labels[foreshock_window_plot], fontsize=20, fontweight='bold', loc='left')
            axs[foreshock_window_plot].scatter(0, mainshock.MAGNITUDE, marker='*', s=400, color=mainshock_colour, zorder=2)
            axs[foreshock_window_plot].axvline(x=foreshock_window, color='red', linestyle='--')

            axs[foreshock_window_plot].set_xlabel('Days to mainshock')
            axs[foreshock_window_plot].set_ylabel('M')
            axs[foreshock_window_plot].invert_xaxis()
            axs[foreshock_window_plot].set_xscale('log')
            axs[foreshock_window_plot].legend()

            axs[CDF_plot].set_title(panel_labels[CDF_plot], fontsize=20, fontweight='bold', loc='left')
            axs[CDF_plot].set_xlabel('Number of Earthquakes in 20 days')
            axs[CDF_plot].set_ylabel('CDF')
            axs[CDF_plot].axvline(x=n_events_in_foreshock_window, color='red',
                                  label=r'$N_{obs}$): ' + str(n_events_in_foreshock_window))
            axs[CDF_plot].legend()

            fig.tight_layout()

            if save==True:
                results_plot_folder = f"Mw_{mainshock_Mw_threshold}_iter{iterations}_Mc_cut_{Mc_cutoff}_{search_radius}km_{foreshock_window}day_{modelling_time_period}mtp"
                Path(f"../outputs/{catalogue_name}/{mainshock_method}/results_plots/{results_plot_folder}").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"../outputs/{catalogue_name}/{mainshock_method}/results_plots/{results_plot_folder}/{str(mainshock.ID)}.png")

            if show_plots == False: 
                plt.close()

        print("Processed " + str(count) + ' of ' + str(len(mainshock_file_min_obs_time)))
        count += 1
        print(" ")
            
        results.append(results_dict)
        clear_output(wait=True)
        
    results_file = pd.DataFrame.from_dict(results)

    Path('../outputs/' + catalogue_name + '/' + mainshock_method + '/foreshocks').mkdir(parents=True, exist_ok=True)
    if save==True:
        results_file.to_csv('../outputs/' + catalogue_name + '/' + mainshock_method + '/foreshocks/' + results_file_name, index=False)
    
    print(results_file)

    number_of_mainshocks_with_Wetzler_foreshocks = len(results_file[results_file['n_Wetzler_foreshocks']>0])
    number_of_mainshocks_with_ge5_Wetzler_foreshocks = len(results_file[results_file['n_Wetzler_foreshocks']>=5])

    total_mainshocks = len(results_file)

    all_results_dict = {'Count type':['Total mainshocks',
                                        'Wetzler time window',
                                        'Wetzler time window ge5'],
                        'Count':[total_mainshocks,
                                 number_of_mainshocks_with_Wetzler_foreshocks,
                                 number_of_mainshocks_with_ge5_Wetzler_foreshocks]
                                 }

    results_table = []

    for model in method_dict.values():
        print(model)

        number_of_mainshocks_with_foreshocks_significance_level = len(results_file[results_file[model] < significance_level])

        all_results_dict['Count type'].append(model)
        all_results_dict['Count'].append(number_of_mainshocks_with_foreshocks_significance_level)
        
    results_table = pd.DataFrame.from_dict(all_results_dict)
    results_table['percentage'] = 100*results_table['Count']/total_mainshocks
    results_table['percentage'] = [round(x) for x in results_table['percentage']]

    Path('../outputs/' + catalogue_name + '/' + mainshock_method + '/results_tables').mkdir(parents=True, exist_ok=True)
    if save==True:
        results_table.to_csv('../outputs/'  + catalogue_name + '/' + mainshock_method + '/results_tables/' + results_file_name, index=False)
    
    print('Total number of mainshocks: ' + str(len(mainshock_file)))
    
    return results_file, results_table


def stack_mainshock_signals(catalogue_name,
                            mag_thresh = 4,
                            distance_threshold = 10,
                            time_threshold = 365,
                            bin_width = 20,
                           test=False):
    """
    Function to stack mainshock precursory signals.
    [TASK: update with new file paths, no seispy function calls, Pyproj not UTM, and probably other things].
    """
    
    earthquake_catalogue = pd.read_csv('../catalogues/reformatted_catalogues/' + catalogue_name + '_reformatted.csv')
    string_to_datetime_df(earthquake_catalogue)
    output_path = '../outputs/' + catalogue_name
    Path(output_path + '/data/modelling_data/mainshock_slices').mkdir(parents=True, exist_ok=True)
    earthquake_catalogue['DAY'] = (earthquake_catalogue['DATETIME'] - earthquake_catalogue.iloc[0].DATETIME).apply(lambda d: (d.total_seconds()/(24*3600)))
    
    Path(output_path + '/data/modelling_data/mainshock_slices/local_catalogues').mkdir(parents=True, exist_ok=True)
    Path(output_path + '/data/modelling_data/mainshock_slices/sampling_results').mkdir(parents=True, exist_ok=True)
    Path(output_path + '/images/mainshock_slices').mkdir(parents=True, exist_ok=True)
    Path(output_path + '/data/mainshocks/').mkdir(parents=True, exist_ok=True)
    Path(output_path + '/data/stacked_mainshocks_results').mkdir(parents=True, exist_ok=True)
    Path(output_path + '/images/big_picture_results').mkdir(parents=True, exist_ok=True)

    large_earthquakes = earthquake_catalogue[earthquake_catalogue['MAGNITUDE']>=mag_thresh].copy()
    if test==True:
            large_earthquakes = large_earthquakes.iloc[0:2].copy()
            
    large_earthquakes.sort_values(by='DATETIME', inplace=True)
    
    n_mainshocks = str(len(large_earthquakes))
    Mc = Mc_by_maximum_curvature(earthquake_catalogue['MAGNITUDE'], mbin=0.1)
    print("Events above Mw " + str(mag_thresh) + " : " + n_mainshocks)
    print("Mc: " + str(Mc))
    print(" ")
    IDs_to_skip = []
    for mainshock in large_earthquakes.itertuples():
        DAYS_TO_MAINSHOCK = (large_earthquakes['DATETIME'] - mainshock.DATETIME).apply(lambda d: (d.total_seconds()/(24*3600)))
        KM_TO_MAINSHOCK = ((((large_earthquakes['EASTING'] - mainshock.EASTING)**2) + ((large_earthquakes['NORTHING'] - mainshock.NORTHING)**2))**0.5)
        local_catalogue = large_earthquakes.loc[(KM_TO_MAINSHOCK < distance_threshold) & (DAYS_TO_MAINSHOCK >=0)  & (DAYS_TO_MAINSHOCK <= time_threshold)].copy()
        nearby_large_earthquakes = local_catalogue.loc[(local_catalogue['MAGNITUDE']>=mag_thresh) & (mainshock.ID!=local_catalogue['ID'])].copy()
        IDs_to_skip.extend(nearby_large_earthquakes['ID'])
    solo_large_earthquakes = large_earthquakes[~large_earthquakes['ID'].isin(IDs_to_skip)]
    n_mainshocks = str(len(solo_large_earthquakes))

    print("First events above threshold in sequences: " + n_mainshocks)
    
    window_points = np.array(range(-time_threshold+bin_width,1))
    list_of_sampling_results = []
    count = 0
    for mainshock in solo_large_earthquakes.itertuples():
        print('mainshock ' + str(mainshock.ID))
        print('    date: ' + str(mainshock.DATETIME))

        csv_file = str(mainshock.ID) + '.csv'

        if Path(output_path + '/data/modelling_data/mainshock_slices/local_catalogues/' + csv_file).exists() == True:
            events_within_100km = pd.read_csv(output_path + '/data/modelling_data/mainshock_slices/local_catalogues/' + csv_file)
            string_to_datetime(events_within_100km)
        else:
            km_to_mainshock = ((((earthquake_catalogue.EASTING - mainshock.EASTING)**2) + ((earthquake_catalogue.NORTHING - mainshock.NORTHING)**2))**0.5)
            events_within_100km = earthquake_catalogue.loc[(km_to_mainshock < 100)].copy()
            events_within_100km['DAYS_TO_MAINSHOCK'] = (events_within_100km['DATETIME'] - mainshock.DATETIME).apply(lambda d: (d.total_seconds()/(24*3600)))
            events_within_100km['KM_TO_MAINSHOCK'] = ((((events_within_100km.EASTING - mainshock.EASTING)**2) + ((earthquake_catalogue.NORTHING - mainshock.NORTHING)**2))**0.5)
            
            inter_event_distances = [] 
            for event in events_within_100km.itertuples():
                try:
                    point_1 = (event.EASTING, event.NORTHING)
                    point_2 = (events_within_100km.EASTING.iloc[event.Index+1], events_within_100km.NORTHING.iloc[event.Index+1])
            #         euclidean_distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    euclidean_distance = dist(point_1, point_2)
                except:
                    euclidean_distance = np.NAN

                inter_event_distances.append(euclidean_distance)

            events_within_100km['inter_event_distances'] = inter_event_distances
            events_within_100km['inter_event_distances'] = events_within_100km['inter_event_distances'].shift(1)

#             time_series = datetime_to_decimal_days(events_within_100km['DATETIME'])
            time_series = datetime_to_decimal_days(events_within_100km['DATETIME'])
            inter_event_times = np.diff(time_series)
            inter_event_times = np.insert(inter_event_times, 0, np.NAN)
            events_within_100km['inter_event_times'] = inter_event_times
            
            events_within_100km.to_csv(output_path + '/data/modelling_data/mainshock_slices/local_catalogues/' + csv_file, index=False)

        local_catalogue = events_within_100km.loc[(events_within_100km['KM_TO_MAINSHOCK'] < distance_threshold) & ((events_within_100km['DAYS_TO_MAINSHOCK']**2)**0.5 < time_threshold)].copy()
        print('    events within ' + str(time_threshold) + ' days: ' + str(len(local_catalogue)))

        nearby_large_earthquakes = local_catalogue.loc[(local_catalogue['MAGNITUDE']>=mag_thresh) & (mainshock.ID!=local_catalogue['ID'])].copy()

        pre_mainshock = local_catalogue.loc[local_catalogue['DAYS_TO_MAINSHOCK']<0].copy()
        post_mainshock = local_catalogue.loc[local_catalogue['DAYS_TO_MAINSHOCK']>=0].copy()

        sampling_results = []
        for day in window_points:
            windowed_data = local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] < day) & (local_catalogue['DAYS_TO_MAINSHOCK'] >= day - bin_width)]

            avg_x, avg_y = np.mean(windowed_data['EASTING']), np.mean(windowed_data['NORTHING'])
            SD_x, SD_y = np.std(windowed_data['EASTING']), np.std(windowed_data['NORTHING'])
            var_x, var_y = np.var(windowed_data['EASTING']), np.var(windowed_data['NORTHING'])
            
            results_dict = {'DAY':day,
                            'EVENT_RATE': len(windowed_data)/bin_width,
                            'MOMENT_RATE':sum(windowed_data['MOMENT'])/bin_width,
                            'AVG_KM_TO_MSHOCK':np.mean(windowed_data['KM_TO_MAINSHOCK']),
                            'inter_event_times':np.mean(windowed_data['inter_event_times']),
                            'inter_event_distances':np.mean(windowed_data['inter_event_distances']),
                            'avg_x':avg_x,
                            'avg_y':avg_y,
                            'SD_x':SD_x,
                            'SD_y':SD_y,
                            'var_x':var_x,
                            'var_y':var_y
                           }
            sampling_results.append(results_dict)
        sampling_results = pd.DataFrame.from_dict(sampling_results)
        
        sampling_results['NORMALISED_EVENT_RATE'] = preprocessing.minmax_scale(sampling_results['EVENT_RATE'], feature_range=(0, 1), axis=0, copy=True)
        sampling_results['NORMALISED_MOMENT_RATE'] = preprocessing.minmax_scale(sampling_results['MOMENT_RATE'], feature_range=(0, 1), axis=0, copy=True)
        sampling_results['LOG_MOMENT_RATE'] = np.log(sampling_results['MOMENT_RATE'].replace(0, np.nan))
        sampling_results['LOG_MOMENT_RATE'].replace(np.nan, 0, inplace=True)
        sampling_results['NORMALISED_LOG_MOMENT_RATE'] = preprocessing.minmax_scale(sampling_results['LOG_MOMENT_RATE'], feature_range=(0, 1), axis=0, copy=True)
        sampling_results['NORM_AVG_KM_TO_MSHOCK'] = preprocessing.minmax_scale(sampling_results['AVG_KM_TO_MSHOCK'], feature_range=(0, 1), axis=0, copy=True)
        sampling_results['NORM_IETs'] = preprocessing.minmax_scale(sampling_results['inter_event_times'], feature_range=(0, 1), axis=0, copy=True)
        sampling_results['NORM_IEDs'] = preprocessing.minmax_scale(sampling_results['inter_event_distances'], feature_range=(0, 1), axis=0, copy=True)

        points = np.array(sampling_results[['avg_x', 'avg_y']])
        d = np.diff(points, axis=0)
        segdists = np.sqrt((d ** 2).sum(axis=1))
        sampling_results['MIGRATION'] = pd.Series(segdists)
        
        points = np.array(sampling_results[['var_x', 'var_y']])
        d = np.diff(points, axis=0)
        segvars = np.sqrt((d ** 2).sum(axis=1))
        sampling_results['LOCALISATION'] = pd.Series(segvars)
        
        sampling_results['NORM_MIGRATION'] = preprocessing.minmax_scale(sampling_results['AVG_KM_TO_MSHOCK'], feature_range=(0, 1), axis=0, copy=True)
        sampling_results['NORM_LOCALISATION'] = preprocessing.minmax_scale(sampling_results['AVG_KM_TO_MSHOCK'], feature_range=(0, 1), axis=0, copy=True)
        
        sampling_results.to_csv(output_path + '/data/modelling_data/mainshock_slices/sampling_results/' + csv_file, index=False)

        list_of_sampling_results.append(sampling_results)

        fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(10,10))
        axs[0].scatter(pre_mainshock['DAYS_TO_MAINSHOCK'], pre_mainshock['MAGNITUDE'], alpha=0.5, color='black')
        axs[0].scatter(post_mainshock['DAYS_TO_MAINSHOCK'], post_mainshock['MAGNITUDE'], alpha=0.5, color='grey')
        axs[0].set_xlabel('Days to mainshock', fontsize=20)
        axs[0].set_ylabel('Mw', fontsize=20)
        axs[0].scatter(nearby_large_earthquakes['DAYS_TO_MAINSHOCK'], nearby_large_earthquakes['MAGNITUDE'], marker='*', s=200, color='red', zorder=2)
        axs[0].scatter(0, mainshock.MAGNITUDE, marker='*', s=200, color='red', zorder=2)
        axs[0].axvline(0, linestyle='--', color='red', zorder=3)
        axs[0].set_xlim(-time_threshold,time_threshold)

        axs[1].plot(sampling_results['DAY'], sampling_results['EVENT_RATE'], color='magenta', zorder=1)
        axs[1].set_xlabel('Days to mainshock', fontsize=20)
        axs[1].set_ylabel('Seismicity rate', fontsize=20)
        axs[1].axvline(0, linestyle='--', color='red', zorder=3)
        axs[1].axhline(sampling_results['EVENT_RATE'].iloc[-1], linestyle='--', color='red', alpha=0.5)

        axs[2].plot(sampling_results['DAY'], sampling_results['MOMENT_RATE'], zorder=1, color='dodgerblue')
        axs[2].set_xlabel('Days to mainshock', fontsize=20)
        axs[2].set_ylabel('Moment rate', fontsize=20)
        axs[2].axvline(0, linestyle='--', color='red', zorder=3)
        axs[2].axhline(sampling_results['MOMENT_RATE'].iloc[-1], linestyle='--', color='red', alpha=0.5)
        
        axs[3].plot(sampling_results['DAY'], sampling_results['AVG_KM_TO_MSHOCK'], zorder=1, color='orange')
        axs[3].set_xlabel('Days to mainshock', fontsize=20)
        axs[3].set_ylabel('km to Mshock', fontsize=20)
        axs[3].axvline(0, linestyle='--', color='red', zorder=3)
        axs[3].axhline(sampling_results['AVG_KM_TO_MSHOCK'].iloc[-1], linestyle='--', color='red', alpha=0.5)
        
        axs[4].plot(sampling_results['DAY'], sampling_results['NORM_IETs'], zorder=1)
        axs[4].set_xlabel('DAY', fontsize=20)
        axs[4].set_ylabel('IETs', fontsize=20)
        axs[4].axvline(0, linestyle='--', color='red', zorder=3)
        axs[4].axhline(sampling_results['NORM_IETs'].iloc[-2], linestyle='--', color='red', alpha=0.5)
        
        axs[5].plot(sampling_results['DAY'], sampling_results['NORM_IEDs'], zorder=1)
        axs[5].set_xlabel('DAY', fontsize=20)
        axs[5].set_ylabel('IEDs', fontsize=20)
        axs[5].axvline(0, linestyle='--', color='red', zorder=3)
        axs[5].axhline(sampling_results['NORM_IEDs'].iloc[-2], linestyle='--', color='red', alpha=0.5)

        fig.tight_layout()
        png_file = str(mainshock.ID) + '.png'
        fig.savefig(output_path + '/images/mainshock_slices/' + png_file)
        plt.show()
        plt.close()

        count += 1
        print("    Processed: " + str(count) + ' out of ' + n_mainshocks)
        print(" ")
    
    print("Processed all mainshocks")
    
    pd.DataFrame(solo_large_earthquakes, columns=['ID']).to_csv(output_path + '/data/mainshocks/' + n_mainshocks + '_Mw_'  + str(mag_thresh) + 's.csv', index=False)
    
    combined_sampling_results = pd.concat(list_of_sampling_results, ignore_index=True)
    
    stacked_results = combined_sampling_results.groupby(['DAY']).mean()
    stacked_results.reset_index(inplace=True)
    
    stacked_results.to_csv(output_path + '/data/stacked_mainshocks_results/stacked_Mw_' + str(mag_thresh) + '_results.csv', index=False)

    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(10,10))
    fig.suptitle(catalogue_name + '_Mc_' + str(Mc) + '_n_' + n_mainshocks + '_Mw_' + str(mag_thresh) + '+_' + str(distance_threshold) + 'km_' + str(bin_width) + 'daybin_' + str(time_threshold/365) + '_years',
                fontsize=20)
#     axs[0].plot(stacked_results['DAY'], stacked_results['EVENT_RATE'], alpha=0.5, label='event rate')
    axs[0].plot(stacked_results['DAY'], stacked_results['NORMALISED_EVENT_RATE'], alpha=0.5, color='magenta', label='event rate')
    axs[0].axhline(stacked_results['NORMALISED_EVENT_RATE'].iloc[-1], linestyle='--', color='red', alpha=0.5)
    axs[0].axvline(0, linestyle='--', color='red', alpha=0.5)
    axs[0].set_xlabel('Days to mainshock', fontsize=20)
    axs[0].set_ylabel('Avg seismicity rate', fontsize=20)
    axs[0].legend(fontsize=20)
#     ax_twin = axs[0].twinx()
#     ax_twin.plot(stacked_results['DAY'], stacked_results['NORMALISED_EVENT_RATE'], alpha=0.5, color='cyan', label='normalised')
#     ax_twin.set_ylabel('Normalised rate')
#     ax_twin.axhline(stacked_results['NORMALISED_EVENT_RATE'].iloc[-1], linestyle='--', color='red', alpha=0.5)
#     ax_twin.legend()

#     axs[1].plot(stacked_results['DAY'], stacked_results['LOG_MOMENT_RATE'], alpha=0.5, label='log Mo rate')
    axs[1].plot(stacked_results['DAY'], stacked_results['NORMALISED_LOG_MOMENT_RATE'], alpha=0.5, color='dodgerblue', label='moment rate')
    axs[1].axhline(stacked_results['NORMALISED_LOG_MOMENT_RATE'].iloc[-1], linestyle='--', color='red', alpha=0.5)
    axs[1].axvline(0, linestyle='--', color='red', alpha=0.5)
    axs[1].set_xlabel('Days to mainshock', fontsize=20)
    axs[1].set_ylabel('Avg moment rate', fontsize=20)
    axs[1].legend(fontsize=20)
#     ax_twin = axs[1].twinx()
#     ax_twin.plot(stacked_results['DAY'], stacked_results['NORMALISED_LOG_MOMENT_RATE'], alpha=0.5, color='cyan', label='normalised')
#     ax_twin.set_ylabel('Normalised rate')
#     ax_twin.axhline(stacked_results['NORMALISED_LOG_MOMENT_RATE'].iloc[-1], linestyle='--', color='red', alpha=0.5)
#     ax_twin.legend()
    
#     axs[2].plot(stacked_results['DAY'], stacked_results['AVG_KM_TO_MSHOCK'], alpha=0.5, label='km away')
    axs[2].plot(stacked_results['DAY'], stacked_results['NORM_AVG_KM_TO_MSHOCK'], alpha=0.5, color='orange', label='km to Mshock')
    axs[2].axhline(stacked_results['NORM_AVG_KM_TO_MSHOCK'].iloc[-1], linestyle='--', color='red', alpha=0.5)
    axs[2].axvline(0, linestyle='--', color='red', alpha=0.5)
    axs[2].set_xlabel('Days to mainshock', fontsize=20)
    axs[2].set_ylabel('Avg km to Mshock', fontsize=20)
    axs[2].legend(fontsize=20)
#     ax_twin = axs[2].twinx()
#     ax_twin.plot(stacked_results['DAY'], stacked_results['NORM_AVG_KM_TO_MSHOCK'], alpha=0.5, color='cyan', label='normalised')
#     ax_twin.set_ylabel('Normalised rate')
#     ax_twin.axhline(stacked_results['NORM_AVG_KM_TO_MSHOCK'].iloc[-1], linestyle='--', color='red', alpha=0.5)
#     ax_twin.legend()
        
    axs[3].plot(stacked_results['DAY'], stacked_results['NORM_IETs'], alpha=0.5, color='orange', label='IETs')
    axs[3].axhline(stacked_results['NORM_IETs'].iloc[-1], linestyle='--', color='red', alpha=0.5)
    axs[3].axvline(0, linestyle='--', color='red', alpha=0.5)
    axs[3].set_xlabel('Days to mainshock', fontsize=20)
    axs[3].set_ylabel('IETs', fontsize=20)
    axs[3].legend(fontsize=20)
    
    axs[4].plot(stacked_results['DAY'], stacked_results['NORM_IEDs'], alpha=0.5, color='orange', label='IEDs')
    axs[4].axhline(stacked_results['NORM_IEDs'].iloc[-1], linestyle='--', color='red', alpha=0.5)
    axs[4].axvline(0, linestyle='--', color='red', alpha=0.5)
    axs[4].set_xlabel('Days to mainshock', fontsize=20)
    axs[4].set_ylabel('IEDs', fontsize=20)
    axs[4].legend(fontsize=20)
    
    outfile_name  = n_mainshocks + '_Mw_' + str(mag_thresh) + 's_' + str(distance_threshold) + 'km_' + str(bin_width) + 'daybin_' + str(time_threshold) + 'days_results.png' 
    plt.savefig(output_path + '/images/big_picture_results/' + outfile_name)
    
    print("selected: " + str(len(solo_large_earthquakes)) + ' out of ' + n_mainshocks + ' earthquakes above Mw ' + str(mag_thresh))
    
    return stacked_results 

def apply_Mc_cut(earthquake_catalogue):
    Mc = seispy.Mc_by_maximum_curvature(earthquake_catalogue['MAGNITUDE'])
    # print(f"Mc: {Mc}")
    earthquake_catalogue = earthquake_catalogue.loc[earthquake_catalogue['MAGNITUDE']>= Mc].copy()
    return earthquake_catalogue

def load_local_catalogue(mainshock, catalogue_name='unspecified'):
    local_catalogue = pd.read_csv(f'../data/{catalogue_name}/local_catalogues/{mainshock.ID}.csv')
    seispy.string_to_datetime_df(local_catalogue)
    return local_catalogue

def create_local_catalogue(mainshock, earthquake_catalogue, catalogue_name, radius_km = 30, save=True):
    
    mainshock_LON = mainshock.LON
    mainshock_LAT = mainshock.LAT
    mainshock_DATETIME = mainshock.DATETIME

    box_halfwidth_km = radius_km
    min_box_lon, min_box_lat = seispy.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, -box_halfwidth_km, -box_halfwidth_km)
    max_box_lon, max_box_lat = seispy.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, box_halfwidth_km, box_halfwidth_km)

    local_catalogue = earthquake_catalogue.loc[
                                    (earthquake_catalogue['LON']>= min_box_lon) &\
                                    (earthquake_catalogue['LON']<= max_box_lon) &\
                                    (earthquake_catalogue['LAT']>= min_box_lat) &\
                                    (earthquake_catalogue['LAT']<= max_box_lat)
                                    ].copy()

    local_catalogue['DAYS_TO_MAINSHOCK'] = (mainshock_DATETIME - local_catalogue['DATETIME']).apply(lambda d: (d.total_seconds()/(24*3600)))

    local_catalogue['DISTANCE_TO_MAINSHOCK'] = seispy.calculate_distance_pyproj_vectorized(mainshock_LON, mainshock_LAT, local_catalogue['LON'],  local_catalogue['LAT'])
    # local_catalogue['DISTANCE_TO_MAINSHOCK'] = calculate_distances_haversine_vect(mainshock.LON, mainshock.LAT, local_catalogue['LON'],  local_catalogue['LAT'])

    # local_catalogue = local_catalogue[#(local_catalogue['DATETIME'] < mainshock_DATETIME) &\
    #                                     # (local_catalogue['DAYS_TO_MAINSHOCK'] < modelling_time_period+foreshock_window) &\
    #                                     # (local_catalogue['DAYS_TO_MAINSHOCK'] > 0)  &\
    #                                 #   (local_catalogue['ID'] != mainshock.ID) &\
    #                                     (local_catalogue['DISTANCE_TO_MAINSHOCK'] < radius_km) 
    #                                     ].copy()
    if save==True:
        Path(f'../data/{catalogue_name}/local_catalogues/').mkdir(parents=True, exist_ok=True)
        local_catalogue.to_csv(f'../data/{catalogue_name}/local_catalogues/{mainshock.ID}.csv', index=False)
    return local_catalogue

def create_spatial_plot(mainshock, local_cat, catalogue_name, Mc_cut, min_days=365, max_days=0, radius_km=10, save=True):
    
    mainshock_ID = mainshock.ID
    mainshock_M = mainshock.MAGNITUDE
    mainshock_LON = mainshock.LON
    mainshock_LAT = mainshock.LAT
    mainshock_DATETIME = mainshock.DATETIME

    box_halfwidth_km = 30
    min_box_lon, min_box_lat = seispy.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, -box_halfwidth_km, -box_halfwidth_km)
    max_box_lon, max_box_lat = seispy.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, box_halfwidth_km, box_halfwidth_km)

    aftershocks = local_cat.loc[(local_cat['DAYS_TO_MAINSHOCK'] < 0) &\
                                (local_cat['DAYS_TO_MAINSHOCK'] > -20)].copy()

    local_cat = local_cat.loc[(local_cat['DAYS_TO_MAINSHOCK'] < min_days) &\
                              (local_cat['DAYS_TO_MAINSHOCK'] > max_days)].copy()

    magnitude_fours = local_cat.loc[local_cat['MAGNITUDE']>=4].copy()

    fig = plt.figure()

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_title(f"ID: {mainshock_ID}", loc='right')
    
    # ax.set_extent(seispy.get_catalogue_extent(local_cat, buffer=0.025), crs=ccrs.PlateCarree())
    ax.set_extent([min_box_lon, max_box_lon, min_box_lat, max_box_lat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, edgecolor='none')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, zorder=0)
    gl.top_labels = False
    gl.right_labels = False

    ax.scatter(mainshock_LON, mainshock_LAT, color='red', s=np.exp(mainshock_M), marker='*', label=f'$M_w$ {mainshock_M} mainshock')
    new_LON, new_LAT = seispy.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, radius_km, 0)
    radius_degrees = new_LON - mainshock_LON
    circle = Circle((mainshock_LON, mainshock_LAT), radius_degrees, edgecolor='r', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(circle)
    z = np.exp(local_cat['MAGNITUDE'])
    local_cat = ax.scatter(local_cat['LON'], local_cat['LAT'], s=z, #c=seispy.datetime_to_decimal_year(local_cat['DATETIME']),
                c=local_cat['DAYS_TO_MAINSHOCK'], label=f'{len(local_cat)} earthquakes (1 year prior)', alpha=0.9)
    cbar = fig.colorbar(local_cat, ax=ax)
    cbar.set_label('Days to mainshock') 
    z = np.exp(aftershocks['MAGNITUDE'])
    ax.scatter(aftershocks['LON'], aftershocks['LAT'], s=z, #c=seispy.datetime_to_decimal_year(local_cat['DATETIME']),
                color='grey', label=f'{len(aftershocks)} aftershocks (20 days post)', alpha=0.3, zorder=0)
    # ax.scatter(magnitude_fours['LON'], magnitude_fours['LAT'], s=z, #c=seispy.datetime_to_decimal_year(local_cat['DATETIME']),
    #             c='black', label=f'{len(magnitude_fours)} $M_w$ $\ge$ 4 (1 year prior)', alpha=0.9)
    
    ax.legend(loc='lower right', bbox_to_anchor=(0.575,1))
    # ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    # ax.legend()
    ax.set_xlabel('LON')
    ax.set_ylabel('LAT')
    
    if save==True:
        if Mc_cut==False:
            Path(f"../outputs/{catalogue_name}/spatial_plots").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"../outputs/{catalogue_name}/spatial_plots/{mainshock_ID}_{radius_km}km_{min_days}_to_{max_days}.png")
        elif Mc_cut==True:
            Path(f"../outputs/{catalogue_name}/Mc_cut/spatial_plots").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"../outputs/{catalogue_name}/Mc_cut/spatial_plots/{mainshock_ID}_{radius_km}km_{min_days}_to_{max_days}.png")
    plt.show()

def identify_foreshocks_short(mainshock, earthquake_catalogue, local_catalogue, iterations=10000,
                              local_catalogue_radius = 10, foreshock_window = 20, modelling_time_period=365, Wetzler_cutoff=3):
    
    mainshock_ID = mainshock.ID
    mainshock_LON = mainshock.LON
    mainshock_LAT = mainshock.LAT
    mainshock_DATETIME = mainshock.DATETIME
    mainshock_Mc = mainshock.Mc
    mainshock_MAG = mainshock.MAGNITUDE
    
    method_dict = {"ESR":'ESR',
                    "VA_method":'G-IET',
                    "Max_window":'Max_rate',
                    "VA_half_method":'R-IET',
                    "TR_method":'BP'
                    }
    
    # try:
    #     Mc = round(seispy.Mc_by_maximum_curvature(local_catalogue['MAGNITUDE']),2) + 0.2
    # except:
    #     Mc = float('nan')

    local_catalogue = local_catalogue[(local_catalogue['DATETIME'] < mainshock_DATETIME) &\
                                        (local_catalogue['DAYS_TO_MAINSHOCK'] < modelling_time_period+foreshock_window) &\
                                        (local_catalogue['DAYS_TO_MAINSHOCK'] > 0)  &\
                                        (local_catalogue['DISTANCE_TO_MAINSHOCK'] < local_catalogue_radius) &\
                                        (local_catalogue['ID'] != mainshock_ID)
                                        ].copy()

    # local_catalogue_pre_Mc_cutoff = local_catalogue.copy()
    # local_catalogue_below_Mc = local_catalogue.loc[local_catalogue['MAGNITUDE']<mainshock_Mc].copy()
    # local_catalogue_below_Mc = local_catalogue_below_Mc.loc[(local_catalogue_below_Mc['DAYS_TO_MAINSHOCK']) < modelling_time_period].copy()
    # foreshocks_below_Mc = local_catalogue_below_Mc.loc[local_catalogue_below_Mc['DAYS_TO_MAINSHOCK']<foreshock_window]

    # if Mc_cut==True:
    #     local_catalogue = local_catalogue.loc[local_catalogue['MAGNITUDE']>=mainshock_Mc].copy()
    # else:
    #     local_catalogue = local_catalogue_pre_Mc_cutoff.copy()

    regular_seismicity_period = local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] >= foreshock_window)]
    foreshocks = local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] < foreshock_window)]

    # n_local_catalogue_pre_Mc_cutoff = len(local_catalogue_pre_Mc_cutoff)
    # n_local_catalogue_below_Mc = len(local_catalogue_below_Mc)
    n_local_catalogue = len(local_catalogue)
    n_regular_seismicity_events = len(regular_seismicity_period)
    n_events_in_foreshock_window = len(foreshocks)

    b_values = []
    for seismicity_period in [local_catalogue, foreshocks, regular_seismicity_period]:
        try:
            b_value = round(seispy.b_val_max_likelihood(seismicity_period['MAGNITUDE'], mc=Mc), 2)
        except:
            b_value = float('nan')
        b_values.append(b_value)
    overall_b_value, foreshock_b_value, regular_b_value = b_values

    ### WETZLER WINDOW METHOD ###
    Wetzler_foreshocks = foreshocks.loc[foreshocks['MAGNITUDE']>Wetzler_cutoff].copy()
    N_Wetzler_foreshocks = len(Wetzler_foreshocks)

    ### MAX RATE /ESR 2.0 METHOD ###
    catalogue_start_date = earthquake_catalogue['DATETIME'].iloc[0]
    time_since_catalogue_start = (mainshock_DATETIME - catalogue_start_date).total_seconds()/3600/24
    cut_off_day = math.floor(time_since_catalogue_start)
    if cut_off_day > 365:
        cut_off_day = 365
    range_scaler = 100    

    sliding_window_points = np.array(np.arange((-cut_off_day+foreshock_window)*range_scaler, -foreshock_window*range_scaler, 1))/range_scaler*-1
    sliding_window_counts = np.array([len(regular_seismicity_period[(regular_seismicity_period['DAYS_TO_MAINSHOCK'] > point) &\
                                                                    (regular_seismicity_period['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]) for point in sliding_window_points])

    try:
        max_window = max(sliding_window_counts)
    except:
        max_window = float('nan')

    if n_events_in_foreshock_window > max_window:
        max_window_method = 0.0
    elif n_events_in_foreshock_window <= max_window:
        max_window_method = 1.0
    else:
        max_window_method = float('nan')

    if (len(sliding_window_counts)==0) & (n_events_in_foreshock_window > 0):
        sliding_window_probability = 0.00
        sliding_window_99CI = float('nan')
    elif (len(sliding_window_counts)==0) & (n_events_in_foreshock_window == 0):    
        sliding_window_probability = 1.00
        sliding_window_99CI = float('nan')
    else:
        sliding_window_probability = len(sliding_window_counts[sliding_window_counts >= n_events_in_foreshock_window])/len(sliding_window_counts)
    # sliding_window_probability = len(list(filter(lambda c: c >= n_events_in_foreshock_window, sliding_window_counts)))/len(sliding_window_counts)
        sliding_window_99CI = np.percentile(sliding_window_counts,99)

    ### TR BACKGROUND POISSON MODEL ###
    if not regular_seismicity_period.empty:
        time_series = np.array(regular_seismicity_period['DATETIME'].apply(lambda d: (d-regular_seismicity_period['DATETIME'].iloc[0]).total_seconds()/3600/24))
    else:
        time_series = np.array([])
    if n_regular_seismicity_events >= 2:
        background_rate = seispy.gamma_law_MLE(time_series)
        TR_expected_events = background_rate*foreshock_window
        TR_probability = poisson.sf(n_events_in_foreshock_window, TR_expected_events)
        TR_99CI = poisson.ppf(0.99, TR_expected_events)
    elif n_regular_seismicity_events < 2:
        background_rate, TR_expected_events, TR_99CI = [float('nan')]*3
        if (n_events_in_foreshock_window==0):
            TR_probability = 1.00
        elif (n_events_in_foreshock_window > n_regular_seismicity_events):
            TR_probability = 0.00
        else:
            TR_probability = float('nan')
    else:
        background_rate, TR_expected_events, TR_probability, TR_99CI = [float('nan')]*4


    if n_regular_seismicity_events > 2:
        t_day = 3600 * 24.0
        t_win = foreshock_window * t_day
        IET = np.diff(time_series) ### V&As Gamma IET method
        IET = IET[IET>0]
        try:
            y_, loc_, mu_ = gamma.fit(IET, floc=0.0)
        except:
            y_, loc_, mu_ = gamma.fit(IET, loc=0.0)
        # print(f"y_ {y_}, loc_ {loc_}, mu_ {mu_}")

        if (np.isnan(y_)==False) & (np.isnan(mu_)==False):
            N_eq = np.zeros(iterations, dtype=int) # Buffer for the number of earthquakes observed in each random sample
            for i in range(0,iterations):
                prev_size = 200 # Generate a random IET sample with 200 events
                IET2 = gamma.rvs(a=y_, loc=0, scale=mu_, size=prev_size) * t_day # Sample from gamma distribution
                t0 = np.random.rand() * IET2[0] # Random shift of timing of first event
                t_sum = np.cumsum(IET2) - t0 # Cumulative sum of interevent times
                inds = (t_sum > t_win) # Find the events that lie outside t_win
                while (inds.sum() == 0):
                    prev_size *= 2 # If no events lie outside t_win, create a bigger sample and stack with previous sample
                    IET2 = np.hstack([IET2, gamma.rvs(a=y_, loc=0, scale=mu_, size=prev_size) * t_day])
                    t_sum = np.cumsum(IET2) # Cumulative sum of event times
                    inds = (t_sum > t_win) # Find the events that lie outside t_win
                N_inside_t_win = (~inds).sum()
                if N_inside_t_win == 0: 
                    N_eq[i] = 0 # No events inside t_win, seismicity rate = 0.
                else:
                    N_eq[i] =  N_inside_t_win - 1 # Store the number of events that lie within t_win (excluding shifted event)

            try:
                y_gam_IETs, loc_gam_IETs, mu_gam_IETs = gamma.fit(N_eq[N_eq > 0], floc=0.0)
            except:
                y_gam_IETs, loc_gam_IETs, mu_gam_IETs = gamma.fit(N_eq[N_eq > 0], loc=0.0)
        
        # print(f"y_gam_IETs {y_gam_IETs}, loc_gam_IETs {loc_gam_IETs}, mu_gam_IETs {mu_gam_IETs}")
        VA_gamma_probability = gamma.sf(n_events_in_foreshock_window, y_gam_IETs, loc_gam_IETs, mu_gam_IETs)
        VA_gamma_99CI = gamma.ppf(0.99, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
        VA_IETs_probability = len(N_eq[N_eq>=n_events_in_foreshock_window])/iterations
        VA_IETs_99CI = np.percentile(N_eq,99)

    elif n_regular_seismicity_events <= 2:
        y_gam_IETs, loc_gam_IETs, mu_gam_IETs = [float('nan')]*3
        N_eq = np.array([])
        VA_gamma_99CI,  VA_IETs_99CI = [float('nan')]*2
        if (n_events_in_foreshock_window == 0):
            VA_gamma_probability, VA_IETs_probability = [1.00]*2
        elif (n_events_in_foreshock_window > n_regular_seismicity_events):
            VA_gamma_probability, VA_IETs_probability = [0.00]*2
        else:
            VA_gamma_probability, VA_IETs_probability, VA_gamma_99CI,  VA_IETs_99CI = [float('nan')]*4
    else:
        N_eq = np.array([])
        y_gam_IETs, loc_gam_IETs, mu_gam_IETs = [float('nan')]*3
        VA_gamma_probability, VA_IETs_probability, VA_gamma_99CI,  VA_IETs_99CI = [float('nan')]*4

        ########################################################
                
    results_dict = {'ID':mainshock_ID,
                    'MAGNITUDE':mainshock_MAG,
                    'LON':mainshock_LON,
                    'LAT':mainshock_LAT,
                    'DATETIME':mainshock_DATETIME,
                    'DEPTH':mainshock.DEPTH,
                    'Mc':mainshock_Mc,
                    'time_since_catalogue_start':time_since_catalogue_start,
                    'n_regular_seismicity_events':n_regular_seismicity_events,
                    'n_events_in_foreshock_window':n_events_in_foreshock_window,
                    'n_Wetzler_foreshocks':N_Wetzler_foreshocks,
                    'max_20day_rate':max_window,
                    method_dict['Max_window']:max_window_method,
                    method_dict['ESR']:sliding_window_probability,
                    method_dict['VA_method']:VA_gamma_probability,
                    method_dict['VA_half_method']:VA_IETs_probability,
                    method_dict['TR_method']:TR_probability,
                    method_dict['ESR'] + '_99CI':sliding_window_99CI,
                    method_dict['VA_method'] + '_99CI':VA_gamma_99CI,
                    method_dict['VA_half_method'] + '_99CI':VA_IETs_99CI,
                    method_dict['TR_method'] + '_99CI':TR_99CI,
                    'overall_b_value':overall_b_value,
                    'regular_b_value':regular_b_value,
                    'foreshock_b_value':foreshock_b_value,
                    'y_gam_IETs':y_gam_IETs,
                    'loc_gam_IETs':loc_gam_IETs,
                    'mu_gam_IETs':mu_gam_IETs,
                    'background_rate':background_rate,
                    'cut_off_day':cut_off_day
                    }
    
    file_dict = {'local_catalogue':local_catalogue,
                #  'local_catalogue_pre_Mc_cutoff':local_catalogue_pre_Mc_cutoff,
                #  'local_catalogue_below_Mc':local_catalogue_below_Mc,
                 'foreshocks':foreshocks,
                #  'foreshocks_below_Mc':foreshocks_below_Mc,
                 'sliding_window_points':sliding_window_points,
                 'sliding_window_counts':sliding_window_counts,
                 'N_eq':N_eq
                 }
    
    return results_dict, file_dict

def plot_models(mainshock, results_dict, file_dict, catalogue_name, Mc_cut, foreshock_window = 20, save=True):
    
    colours = sns.color_palette("colorblind", 10)
    colour_names = ['dark blue', 
                'orange',
                'green',
                'red',
                'dark pink',
                'brown',
                'light pink',
                'grey',
                'yellow',
                'light blue']
    colour_dict = dict(zip(colour_names, colours))
    
    method_dict = {"ESR":'ESR',
                "VA_method":'G-IET',
                # "Max_window":'Max_rate',
                "VA_half_method":'R-IET',
                "TR_method":'BP'
                }
    
    mainshock_ID = results_dict['ID']
    mainshock_DATETIME = results_dict['DATETIME']
    cut_off_day = results_dict['cut_off_day']
    n_regular_seismicity_events = results_dict['n_regular_seismicity_events']
    n_events_in_foreshock_window = results_dict['n_events_in_foreshock_window']
    VA_IETs_probability = results_dict['R-IET']
    TR_expected_events = results_dict['background_rate']*foreshock_window
    TR_probability = results_dict['BP']
    y_gam_IETs = results_dict['y_gam_IETs']
    mu_gam_IETs = results_dict['mu_gam_IETs']
    loc_gam_IETs = results_dict['loc_gam_IETs']
    VA_gamma_probability = results_dict['G-IET']
    sliding_window_probability = results_dict['ESR']
    Mc = results_dict['Mc']

    local_catalogue = file_dict['local_catalogue']
    # local_catalogue_pre_Mc_cutoff= file_dict['local_catalogue_pre_Mc_cutoff']
    # local_catalogue_below_Mc= file_dict['local_catalogue_below_Mc']
    foreshocks= file_dict['foreshocks']
    # foreshocks_below_Mc= file_dict['foreshocks_below_Mc']
    sliding_window_counts = file_dict['sliding_window_counts']
    N_eq = file_dict['N_eq']
    
    align = 'left'
    foreshocks_colour = 'red'
    regular_earthquakes_colour = 'black'
    mainshock_colour = 'red'
    poisson_colour = colour_dict['orange']
    gamma_colour = colour_dict['green']
    ESR_colour = colour_dict['dark pink']
    Mc_colour = colour_dict['light blue']
    rate_colour = colour_dict['dark pink']
    
    range_scaler=100
    sliding_window_points_full = np.array(range((-cut_off_day+foreshock_window)*range_scaler, 0*range_scaler+1, 1))/range_scaler*-1
    sliding_window_counts_full = np.array([len(local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] > point) & (local_catalogue['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]) for point in sliding_window_points_full])

    time_series_plot, model_plot, CDF_plot, foreshock_window_plot, Mc_plot = 0, 1, 2, 3, 4
    panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)']

    histogram_alpha = 0.8
    fig, axs = plt.subplots(5,1, figsize=(10,15))

    axs[time_series_plot].set_title(f'Earthquake ID: {mainshock_ID}', loc='right', fontsize=20)

    radius_km=10
    modelling_time_period=365
    local_catalogue_pre_Mc_cutoff = load_local_catalogue(mainshock=mainshock, catalogue_name=catalogue_name)
    local_catalogue_pre_Mc_cutoff = local_catalogue_pre_Mc_cutoff[(local_catalogue_pre_Mc_cutoff['DATETIME'] < mainshock_DATETIME) &\
                                        (local_catalogue_pre_Mc_cutoff['DAYS_TO_MAINSHOCK'] < modelling_time_period+foreshock_window) &\
                                        (local_catalogue_pre_Mc_cutoff['DAYS_TO_MAINSHOCK'] > 0)  &\
                                        (local_catalogue_pre_Mc_cutoff['DISTANCE_TO_MAINSHOCK'] < radius_km) &\
                                        (local_catalogue_pre_Mc_cutoff['ID'] != mainshock_ID)
                                        ].copy()

    if len(local_catalogue_pre_Mc_cutoff)>0:
        bins = np.arange(math.floor(min(local_catalogue_pre_Mc_cutoff['MAGNITUDE'])), math.ceil(max(local_catalogue_pre_Mc_cutoff['MAGNITUDE'])), 0.1)
        values, base = np.histogram(local_catalogue_pre_Mc_cutoff['MAGNITUDE'], bins=bins)
        cumulative = np.cumsum(values)
        axs[Mc_plot].plot(base[:-1], len(local_catalogue_pre_Mc_cutoff)-cumulative, label='FMD', color='black')
        axs[Mc_plot].axvline(x=Mc, linestyle='--', color=Mc_colour, label=r'$M_{c}$: ' + str(round(Mc,1)))
    axs[Mc_plot].set_title(panel_labels[Mc_plot], fontsize=20, fontweight='bold', loc='left')
    axs[Mc_plot].set_xlabel('M')
    axs[Mc_plot].set_ylabel('N')
    axs[Mc_plot].legend()
    
    axs[time_series_plot].set_title(panel_labels[time_series_plot], fontsize=20, fontweight='bold', loc='left')
    axs[time_series_plot].scatter(0, mainshock.MAGNITUDE, marker='*', s=400, color=mainshock_colour,
                                    label=r'$M_{w}$ ' + str(mainshock.MAGNITUDE) + ' Mainshock',  
                                    zorder=3)
    axs[time_series_plot].axvline(x=foreshock_window, color=foreshocks_colour, linestyle='--', 
                                    label = f"{foreshock_window}-day foreshock window",
                                    zorder=4)
    axs[time_series_plot].set_xlabel('Days to mainshock', fontsize=20)
    axs[time_series_plot].set_ylabel('M', fontsize=20)
    axs[time_series_plot].set_xlim(-5,cut_off_day+foreshock_window)
    axs[time_series_plot].invert_xaxis()

    if len(local_catalogue) >0:
        axs[time_series_plot].set_yticks(np.arange(math.floor(min(local_catalogue['MAGNITUDE'])), math.ceil(mainshock.MAGNITUDE), 1))
        axs[time_series_plot].scatter(local_catalogue['DAYS_TO_MAINSHOCK'], local_catalogue['MAGNITUDE'],
                                        label= str(n_regular_seismicity_events) + ' Earthquakes for modelling',
                                        color=regular_earthquakes_colour, alpha=0.5,  zorder=1)
        # axs[time_series_plot].scatter(local_catalogue_below_Mc['DAYS_TO_MAINSHOCK'], local_catalogue_below_Mc['MAGNITUDE'], 
        #                             label= str(len(local_catalogue_below_Mc)) + ' Earthquakes below Mc', 
        #                             alpha=0.5, color=Mc_colour)
    if len(foreshocks) > 0:
        axs[time_series_plot].scatter(foreshocks['DAYS_TO_MAINSHOCK'], foreshocks['MAGNITUDE'],
                                        label= str(n_events_in_foreshock_window) + ' Earthquakes in foreshock window (' + r'$N_{obs}$)',
                                        color=foreshocks_colour, alpha=0.5, 
                                        zorder=2)
        axs[foreshock_window_plot].scatter(foreshocks['DAYS_TO_MAINSHOCK'], foreshocks['MAGNITUDE'], color=foreshocks_colour, alpha=0.5,
                                            label=r'$N_{obs}$: ' + str(n_events_in_foreshock_window))
        # axs[foreshock_window_plot].scatter(foreshocks_below_Mc['DAYS_TO_MAINSHOCK'], foreshocks_below_Mc['MAGNITUDE'], 
        #                                     label= str(len(foreshocks_below_Mc)) + ' Earthquakes below Mc', 
        #                                     alpha=0.2, color=Mc_colour)
        axs[foreshock_window_plot].set_yticks(np.arange(math.floor(min(foreshocks['MAGNITUDE'])), math.ceil(mainshock.MAGNITUDE), 1))
    if np.isnan(Mc)==False:
        axs[time_series_plot].axhline(y=Mc, color=Mc_colour, linestyle='--', 
                                        label = r'$M_{c}$: ' + str(round(Mc,1)),
                                        zorder=5)
        axs[foreshock_window_plot].axhline(y=Mc, color=Mc_colour, linestyle='--', label = r'$M_{c}$: ' + str(Mc), zorder=5)

    
    ax2 = axs[time_series_plot].twinx()
    ax2.plot(sliding_window_points_full, sliding_window_counts_full, color=rate_colour, label='Seismicity Rate (n/20 days)')
    ax2.axhline(y=n_events_in_foreshock_window, color=foreshocks_colour, alpha=0.5, 
                                        label = r'$N_{obs}$', zorder=0)
    ax2.set_ylabel('Rate (n/20 days)')
    lines, labels = axs[time_series_plot].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.set_yticks(seispy.estimate_axis_labels(sliding_window_counts_full))
    # axs[time_series_plot].legend(lines + lines2, labels + labels2, loc='upper left')

    axs[model_plot].set_title(panel_labels[model_plot], fontsize=20, fontweight='bold', loc='left')
    axs[model_plot].set_xlabel('Seismicity Rate (n/20 days)', fontsize=20)
    axs[model_plot].set_ylabel('PDF', fontsize=20)
    axs[model_plot].axvline(x=n_events_in_foreshock_window, color=foreshocks_colour, label=r'$N_{obs}$: ' + str(n_events_in_foreshock_window))      
    # axs[model_plot].set_xticks(range(0,20,2))

    if len(sliding_window_counts) > 0:
        event_counts_pdf = sliding_window_counts/sum(sliding_window_counts)
        event_counts_cdf = np.cumsum(event_counts_pdf)
        event_counts_sorted = np.sort(sliding_window_counts)

    if len(N_eq) > 0:
        axs[model_plot].hist(N_eq, bins=range(min(N_eq)-1, max(N_eq)+1), color=gamma_colour,
                            label=f"{method_dict['VA_half_method']}: {str(round(VA_IETs_probability,3))}",
                            density=True, rwidth=1.0, alpha=histogram_alpha/2, align=align)
        N_eq_pdf = N_eq/sum(N_eq)
        N_eq_cdf = np.cumsum(N_eq_pdf)
        N_eq_sorted = np.sort(N_eq)
        axs[CDF_plot].plot(N_eq_sorted, N_eq_cdf, label=method_dict["VA_half_method"], color=gamma_colour, alpha=histogram_alpha/2)

    if (np.isnan(TR_expected_events)==False) & (TR_expected_events!=0):
        x_TR_Poisson = np.arange(poisson.ppf(0.001, TR_expected_events), poisson.ppf(0.999, TR_expected_events))
        y_TR_Poisson = poisson.pmf(x_TR_Poisson, TR_expected_events)
        p_tail = [float('0.9999' + '9' * i) for i in range(3)]
        x_tail = [poisson.ppf(p_tail, TR_expected_events) for x in p_tail]
        y_tail = [poisson.pmf(x, TR_expected_events) for x in x_tail]
        x_TR_Poisson, y_TR_Poisson = np.append(x_TR_Poisson, x_tail), np.append(y_TR_Poisson, y_tail)
        axs[model_plot].plot(x_TR_Poisson, y_TR_Poisson, label=f"{method_dict['TR_method']}: {str(round(TR_probability,3))}",
                alpha=histogram_alpha, color=poisson_colour)
        # axs[model_plot].plot(x_TR_Poisson, y_TR_Poisson, 
        #         label=f"{method_dict['TR_method']}: {str(round(TR_probability,3))}",
        #         alpha=0.9, color=poisson_colour)
        TR_poisson_cdf = poisson.cdf(x_TR_Poisson, TR_expected_events)
        axs[CDF_plot].plot(x_TR_Poisson, TR_poisson_cdf, label=method_dict['TR_method'], alpha=histogram_alpha, color=poisson_colour)

    if (np.isnan(y_gam_IETs)==False) & (np.isnan(mu_gam_IETs)==False):
        x_gam_IETs = np.arange(gamma.ppf(0.001, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs),
                                gamma.ppf(0.999, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs))
        gamma_pdf = gamma.pdf(x_gam_IETs, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
        axs[model_plot].plot(x_gam_IETs, gamma_pdf, 
                                label= method_dict['VA_method'] + ': ' + str(round(VA_gamma_probability,3)),
                                alpha=histogram_alpha, color=gamma_colour)
        gamma_cdf = gamma.cdf(x_gam_IETs, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
        axs[CDF_plot].plot(x_gam_IETs, gamma_cdf, label= method_dict['VA_method'],
                    color=gamma_colour, alpha=histogram_alpha)
    if len(sliding_window_counts) > 0:
        axs[model_plot].hist(sliding_window_counts, bins=range(min(sliding_window_counts)-1, max(sliding_window_counts)+1), color=ESR_colour,
                density=True, rwidth=1.0, alpha=histogram_alpha, align=align, label=method_dict['ESR'] + ': ' + str(round(sliding_window_probability,3)))
        window_counts_pdf = sliding_window_counts/sum(sliding_window_counts)
        window_counts_cdf = np.cumsum(window_counts_pdf)
        window_counts_sorted = np.sort(sliding_window_counts)
        axs[CDF_plot].plot(window_counts_sorted, window_counts_cdf, label=method_dict['ESR'], color=ESR_colour, alpha=histogram_alpha)

    axs[model_plot].legend()

#         handles, labels = plt.gca().get_legend_handles_labels()       #specify order of items in legend
#         order = range(0,len(handles))
#         order = [0,1,5,2,4,3]
#         plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    axs[foreshock_window_plot].set_title(panel_labels[foreshock_window_plot], fontsize=20, fontweight='bold', loc='left')
    axs[foreshock_window_plot].scatter(0, mainshock.MAGNITUDE, marker='*', s=400, color=mainshock_colour, zorder=2)
    axs[foreshock_window_plot].axvline(x=foreshock_window, color='red', linestyle='--')

    axs[foreshock_window_plot].set_xlabel('Days to mainshock')
    axs[foreshock_window_plot].set_ylabel('M')
    axs[foreshock_window_plot].invert_xaxis()
    axs[foreshock_window_plot].set_xscale('log')
    axs[foreshock_window_plot].legend()

    axs[CDF_plot].set_title(panel_labels[CDF_plot], fontsize=20, fontweight='bold', loc='left')
    axs[CDF_plot].set_xlabel('Seismicity Rate (n/20 days)')
    axs[CDF_plot].set_ylabel('CDF')
    axs[CDF_plot].axvline(x=n_events_in_foreshock_window, color='red',
                            label=r'$N_{obs}$: ' + str(n_events_in_foreshock_window))
    axs[CDF_plot].legend()

    fig.tight_layout()

    if save==True:
        if Mc_cut==False:
            Path(f"../outputs/{catalogue_name}/model_plots").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"../outputs/{catalogue_name}/model_plots/{mainshock.ID}.png")
        elif Mc_cut==True:
            Path(f"../outputs/{catalogue_name}/Mc_cut/model_plots").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"../outputs/{catalogue_name}/Mc_cut/model_plots/{mainshock.ID}.png")
    plt.show()

def process_mainshocks(mainshocks_file, earthquake_catalogue, catalogue_name, Mc_cut, save):
    date = str(dt.datetime.now().date().strftime("%y%m%d"))
    results_list = []
    i = 1
    for mainshock in mainshocks_file.itertuples():
        print(f"{catalogue_name}")
        print(f"{i} of {len(mainshocks_file)} mainshocks")
        try:
            local_cat = load_local_catalogue(mainshock, catalogue_name=catalogue_name)
        except:
            local_cat = create_local_catalogue(mainshock, earthquake_catalogue, catalogue_name=catalogue_name, save=save)
        if Mc_cut==True:
            local_cat = apply_Mc_cut(local_cat)
        create_spatial_plot(mainshock=mainshock, local_cat=local_cat, Mc_cut=Mc_cut, catalogue_name=catalogue_name, save=save)
        results_dict, file_dict = identify_foreshocks_short(local_catalogue=local_cat, mainshock=mainshock, earthquake_catalogue=earthquake_catalogue)
        plot_models(mainshock=mainshock, results_dict=results_dict, file_dict=file_dict, Mc_cut=Mc_cut, catalogue_name=catalogue_name, save=save)
        results_list.append(results_dict)
        clear_output(wait=True)
        i+=1
    if len(results_list)<2:
        results_df = results_dict
    else:
        results_df = pd.DataFrame.from_dict(results_list)
        if save==True:
            if Mc_cut==False:
                Path(f'../data/{catalogue_name}/foreshocks/').mkdir(parents=True, exist_ok=True)
                results_df.to_csv(f'../data/{catalogue_name}/foreshocks/default_params_{date}.csv', index=False)
            if Mc_cut==True:
                Path(f'../data/{catalogue_name}/Mc_cut/foreshocks/').mkdir(parents=True, exist_ok=True)
                results_df.to_csv(f'../data/{catalogue_name}/Mc_cut/foreshocks/default_params_{date}.csv', index=False)
    return results_df

    def productivity_law(m, Q, alpha)
    """
    Shearer 2012. average number of direct (first generation) aftershocks,
     Nasl following an event of magnitude m follows a productivity law
     where m1 is the minimum magnitude earthquake that triggers other earthquakes,
      Q is an aftershock productivity parameter (denoted k by Sornette and Werner [2005b]),
       and alpha is a parameter that determines the rate of increase in the number of aftershocks
        observed for larger main shock magnitudes.
    """
        N_asl = Q*10**(alpha*(m-m1))
        return N_asl 



