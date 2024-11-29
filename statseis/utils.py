"""
Utility funcitons
"""

import datetime as dt
import math
from math import asin, cos, radians, sin, sqrt
import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer
# import statseis.mc as mc_methods
# import statseis.statseis as statseis

plot_colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
plot_color_dict = dict(zip(['teal', 'orange', 'purple', 'pink', 'green', 'yellow', 'brown', 'grey'], plot_colors))

def get_CDF(data):
    data_sorted = np.sort(data)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    return data_sorted, cdf

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

def string_to_datetime(list_of_datetimes, format='%Y-%m-%d %H:%M:%S'):
    """
    Turn datetimes from string into datetime objects
    """
    Datetime = pd.to_datetime(list_of_datetimes,
                              format = format)
    return Datetime


def string_to_datetime_df(dataframe, format='%Y-%m-%d %H:%M:%S'):
    """
    Find DATETIME column in df and change to datetime objects
    """
    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'],
                                           format = format)
    
def string_to_datetime_return(dataframe, format='%Y-%m-%d %H:%M:%S'):
    """
    Find DATETIME column in df and change to datetime objects
    Returns dataframe so function can be mapped
    """
    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'],
                                           format = format)
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


def find_nearest(array, value):
    """
    Returns the nearest value in an array to its argument.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

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
    lon1, lat1 = lon1.iloc[0], lat1.iloc[0]
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

def select_within_box(LON, LAT, df, r):
    min_box_lon, min_box_lat = add_distance_to_position_pyproj(LON, LAT, -r, -r)
    max_box_lon, max_box_lat = add_distance_to_position_pyproj(LON, LAT, r, r)

    selections = df.loc[(df['LON']>= min_box_lon) &\
                        (df['LON']<= max_box_lon) &\
                        (df['LAT']>= min_box_lat) &\
                        (df['LAT']<= max_box_lat)
                        ].copy()

    selections['DISTANCE_TO_MAINSHOCK'] = calculate_distance_pyproj_vectorized(LON, LAT, selections['LON'],  selections['LAT'])
    return selections
