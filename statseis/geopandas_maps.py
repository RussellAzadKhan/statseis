"""
This sub-module contains functions for aiding the plotting of earthquakes (source parameter data e.g. Longitude, Latitude).
"""

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.wkt import loads
# import contextily as cx
from shapely.geometry import Point
import numpy as np
from matplotlib.patches import Rectangle
from pyproj import Transformer
import utils

plot_colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
plot_color_dict = dict(zip(['teal', 'orange', 'purple', 'pink', 'green', 'yellow', 'brown', 'grey'], plot_colors))

def lat_lon_tick_labels(lon, lat, ax, fontsize=20):
    transformer_4326_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    x_ticks_3857 = [transformer_4326_to_3857.transform(lat[0], x)[0] for x in lon]
    y_ticks_3857 = [transformer_4326_to_3857.transform(y, lon[0])[1] for y in lat]

    ax.set_xticks(x_ticks_3857)
    ax.set_yticks(y_ticks_3857)
    ax.set_xticklabels(lon, fontsize=fontsize)
    ax.set_yticklabels(lat, fontsize=fontsize)

def lon_lat_to_geometry(lon, lat):
    geometry = [Point(xy) for xy in zip(lon, lat)]
    gdf = gpd.GeoDataFrame(geometry=geometry)
    gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=3857)
    return gdf

def convert_extent_to_epsg3857(extent):
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x_min, y_min = transformer.transform(extent[0], extent[2])
    x_max, y_max = transformer.transform(extent[1], extent[3])
    return [x_min, x_max, y_min, y_max]

def add_box_3857(rectangle_coords, ax, color='red', zorder=0, linewidth=1, alpha=0.9):
    rectangle_coords = convert_extent_to_epsg3857(rectangle_coords)

    rectangle = Rectangle(
        (rectangle_coords[0], rectangle_coords[2]),
        rectangle_coords[1] - rectangle_coords[0],
        rectangle_coords[3] - rectangle_coords[2],
        linewidth=linewidth, edgecolor=color, facecolor='none', zorder=zorder, alpha=alpha
    )
    ax.add_patch(rectangle)

def df_to_gdf(df, inplace=False):
    gdf = gpd.GeoDataFrame(df,
                           geometry=gpd.points_from_xy(df.LON, df.LAT),
                           crs="EPSG:4326")
    gdf.to_crs(epsg=3857, inplace=True)

    if inplace==True:
        df = gdf.copy()
    elif inplace==False:
        return gdf

def EPSG_transformer(points, current=4326, new=3857):
    transformer = Transformer.from_crs(f"EPSG:{current}", f"EPSG:{new}")
    
    if isinstance(points, tuple):
        points = [points]
    elif isinstance(points, np.ndarray):
        points = points.reshape(-1, 2)
    
    transformed_points = [transformer.transform(lat, lon) for lon, lat in points]
    return transformed_points