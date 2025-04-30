"""
This sub-module contains functions for aiding the plotting of earthquakes (source parameter data e.g. Longitude, Latitude).
"""

# To do: update last two functions as moved from statseis

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
# import geopandas as gpd
# from shapely.wkt import loads
# import contextily as cx
# from shapely.geometry import Point
import numpy as np
from matplotlib.patches import Rectangle
from pyproj import Transformer
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Circle
from pathlib import Path
import math
from matplotlib.lines import Line2D

# use if loading the package locally (comment out when uploading release)
import statseis
import utils
# uncomment when uploading release
# import statseis.statseis as statseis
# import statseis.utils as utils


plot_colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
plot_color_dict = dict(zip(['teal', 'orange', 'purple', 'pink', 'green', 'yellow', 'brown', 'grey'], plot_colors))


def basic_cartopy_map(ax):
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, edgecolor='none')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, zorder=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    gl.xlines = False
    gl.ylines = False

def lat_lon_tick_labels(lon, lat, ax, fontsize=20):
    transformer_4326_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857")
    x_ticks_3857 = [transformer_4326_to_3857.transform(lat[0], x)[0] for x in lon]
    y_ticks_3857 = [transformer_4326_to_3857.transform(y, lon[0])[1] for y in lat]

    ax.set_xticks(x_ticks_3857)
    ax.set_yticks(y_ticks_3857)
    ax.set_xticklabels(lon, fontsize=fontsize)
    ax.set_yticklabels(lat, fontsize=fontsize)

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

def EPSG_transformer(points, current=4326, new=3857):
    transformer = Transformer.from_crs(f"EPSG:{current}", f"EPSG:{new}")
    
    if isinstance(points, tuple):
        points = [points]
    elif isinstance(points, np.ndarray):
        points = points.reshape(-1, 2)
    
    transformed_points = [transformer.transform(lat, lon) for lon, lat in points]
    return transformed_points

def create_spatial_plot(mainshock, local_cat, catalogue_name, Mc_cut, min_days=365, max_days=0, radius_km=10, save=True):
    
    mainshock_ID = mainshock.ID
    mainshock_M = mainshock.MAGNITUDE
    mainshock_LON = mainshock.LON
    mainshock_LAT = mainshock.LAT
    mainshock_DATETIME = mainshock.DATETIME

    box_halfwidth_km = 30
    min_box_lon, min_box_lat = utils.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, -box_halfwidth_km, -box_halfwidth_km)
    max_box_lon, max_box_lat = utils.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, box_halfwidth_km, box_halfwidth_km)

    aftershocks = local_cat.loc[(local_cat['DAYS_TO_MAINSHOCK'] < 0) &\
                                (local_cat['DAYS_TO_MAINSHOCK'] > -20)].copy()

    local_cat = local_cat.loc[(local_cat['DAYS_TO_MAINSHOCK'] < min_days) &\
                              (local_cat['DAYS_TO_MAINSHOCK'] > max_days)].copy()

    magnitude_fours = local_cat.loc[local_cat['MAGNITUDE']>=4].copy()

    fig = plt.figure()

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_title(f"ID: {mainshock_ID}", loc='right')
    
    # ax.set_extent(utils.get_catalogue_extent(local_cat, buffer=0.025), crs=ccrs.PlateCarree())
    ax.set_extent([min_box_lon, max_box_lon, min_box_lat, max_box_lat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, edgecolor='none')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, zorder=0)
    gl.top_labels = False
    gl.right_labels = False

    ax.scatter(mainshock_LON, mainshock_LAT, color='red', s=np.exp(mainshock_M), marker='*', label=f'$M_w$ {mainshock_M} mainshock')
    new_LON, new_LAT = utils.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, radius_km, 0)
    radius_degrees = new_LON - mainshock_LON
    circle = Circle((mainshock_LON, mainshock_LAT), radius_degrees, edgecolor='r', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(circle)
    z = np.exp(local_cat['MAGNITUDE'])
    local_cat = ax.scatter(local_cat['LON'], local_cat['LAT'], s=z, #c=utils.datetime_to_decimal_year(local_cat['DATETIME']),
                c=local_cat['DAYS_TO_MAINSHOCK'], label=f'{len(local_cat)} earthquakes (1 year prior)', alpha=0.9)
    cbar = fig.colorbar(local_cat, ax=ax)
    cbar.set_label('Days to mainshock') 
    z = np.exp(aftershocks['MAGNITUDE'])
    ax.scatter(aftershocks['LON'], aftershocks['LAT'], s=z, #c=utils.datetime_to_decimal_year(local_cat['DATETIME']),
                color='grey', label=f'{len(aftershocks)} aftershocks (20 days post)', alpha=0.3, zorder=0)
    # ax.scatter(magnitude_fours['LON'], magnitude_fours['LAT'], s=z, #c=utils.datetime_to_decimal_year(local_cat['DATETIME']),
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

def plot_local_cat(mainshock, earthquake_catalogue, catalogue_name, Mc_cut, local_cat=None, min_days=365, max_days=0, radius_km=10, save=True, 
                   box_halfwidth_km=30, aftershock_days=-20, foreshock_days=20, stations=None, event_marker_size = (lambda x: 50+10**(x/1.25))):

    mainshock_ID = mainshock.ID
    mainshock_M = mainshock.MAGNITUDE
    mainshock_LON = mainshock.LON
    mainshock_LAT = mainshock.LAT
    mainshock_DATETIME = mainshock.DATETIME

    if local_cat is None:
        local_cat = statseis.create_local_catalogue(mainshock, catalogue_name=catalogue_name, earthquake_catalogue=earthquake_catalogue, radius_km=box_halfwidth_km, box=True)

    min_box_lon, min_box_lat = utils.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, -box_halfwidth_km, -box_halfwidth_km)
    max_box_lon, max_box_lat = utils.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, box_halfwidth_km, box_halfwidth_km)

    local_cat = local_cat.loc[(local_cat['DAYS_TO_MAINSHOCK'] < min_days) &\
                              (local_cat['DAYS_TO_MAINSHOCK']!=0) &\
                                     (local_cat['DAYS_TO_MAINSHOCK'] > aftershock_days)].copy()
    inside = local_cat.loc[(local_cat['DISTANCE_TO_MAINSHOCK']<radius_km)].copy()
    outside = local_cat.loc[(local_cat['DISTANCE_TO_MAINSHOCK']>=radius_km)].copy()

    aftershocks = inside.loc[(inside['DAYS_TO_MAINSHOCK'] < 0) &\
                             (inside['DAYS_TO_MAINSHOCK'] > aftershock_days)].copy()

    foreshocks = inside.loc[(inside['DAYS_TO_MAINSHOCK'] < foreshock_days) &\
                               (inside['DAYS_TO_MAINSHOCK'] > 0)].copy()
    
    # outside = local_cat.loc[(local_cat['DISTANCE_TO_MAINSHOCK']>=radius_km) &\
    #                            (local_cat['DAYS_TO_MAINSHOCK'] < min_days) &\
    #                           (local_cat['DAYS_TO_MAINSHOCK'] > 0)].copy()
    
    # outside_foreshocks = local_cat.loc[(local_cat['DISTANCE_TO_MAINSHOCK']>=radius_km) &\
    #                            (local_cat['DAYS_TO_MAINSHOCK'] < foreshock_days) &\
    #                           (local_cat['DAYS_TO_MAINSHOCK'] > 0)].copy()
    
    modelling_events = inside.loc[(inside['DAYS_TO_MAINSHOCK'] < min_days) &\
                                  (inside['DAYS_TO_MAINSHOCK'] > 0)].copy()
    
    # modelling_plus_outside = pd.concat([modelling_events, outside])
    modelling_plus_aftershocks = pd.concat([modelling_events, aftershocks])

    magnitude_fours = local_cat.loc[local_cat['MAGNITUDE']>=4].copy()

    mainshock_colour = 'black'

    # vmin, vmax = max_days, min_days
    vmin, vmax = aftershock_days, min_days

    fig = plt.figure(figsize=(8,8))
    gs = fig.add_gridspec(3,3)

    ax = fig.add_subplot(gs[0, :])
    # ax = fig.add_subplot(121)
    ax.set_title('a)', fontsize=20, loc='left')
    ax.set_title(f"ID: {mainshock_ID} - {mainshock.DATETIME.strftime('%b %d %Y')} - {catalogue_name}", loc='right')

    ax.scatter(0, mainshock.MAGNITUDE, marker='*', s = event_marker_size(mainshock.MAGNITUDE), #s=400, 
               color=mainshock_colour, label=r'$M_{w}$ ' + str(mainshock.MAGNITUDE) + ' Mainshock', zorder=3)
    ax.axvline(x=foreshock_days, color='red', linestyle='--', 
                                    # label = f"{foreshock_days}-day foreshock window",
                                    zorder=4)
    ax.set_xlabel('Days to mainshock', fontsize=20)
    ax.set_ylabel('Magnitude', fontsize=20)
    # ax.set_xlim(-25,365+20)
    try:
        max_mag = math.ceil(max([mainshock.MAGNITUDE, max(local_cat['MAGNITUDE'])]))
        min_mag = round(min(local_cat['MAGNITUDE']))
        mid_mag = max_mag - (max_mag - min_mag)/2
        mag_y_ticks = [min_mag, mid_mag, max_mag]
        ax.set_yticks(mag_y_ticks)
        ax.set_yticklabels(mag_y_ticks)
    except:
        print('Auto Mag y-ticks, not manual')

    ax.invert_xaxis()

    # if len(modelling_events) >0:
    #     # ax.set_yticks(np.arange(math.floor(min(local_cat['MAGNITUDE'])), math.ceil(mainshock.MAGNITUDE), 1))
    #     ax.scatter(modelling_events['DAYS_TO_MAINSHOCK'], modelling_events['MAGNITUDE'], #s=6*np.exp(modelling_events['MAGNITUDE']),
    #                s=event_marker_size(modelling_events['MAGNITUDE']), vmin=vmin, vmax=vmax, ec='white', linewidth=0.25,
    #                label= f'{len(modelling_events)- len(foreshocks)} modelling events',
    #                                     c=modelling_events['DAYS_TO_MAINSHOCK'], alpha=0.5,  zorder=1)
    # outside_plus_foreshocks = pd.concat([outside, outside_foreshocks])
    if len(inside) > 0:
        ax.scatter(inside['DAYS_TO_MAINSHOCK'], inside['MAGNITUDE'],
            s= event_marker_size(inside['MAGNITUDE']), ec='white', linewidth=0.25,
               c=inside['DAYS_TO_MAINSHOCK'], 
               alpha=0.75, zorder=1)
    if len(foreshocks) > 0:
        ax.scatter(foreshocks['DAYS_TO_MAINSHOCK'], foreshocks['MAGNITUDE'],
                    # s=6*np.exp(foreshocks['MAGNITUDE']),
                    s = event_marker_size(foreshocks['MAGNITUDE']), ec='white', linewidth=0.25,
                   label= fr"$N_obs$: {len(foreshocks)}", color='red', alpha=0.75, zorder=2)
    if len(outside) > 0:
        ax.scatter(outside['DAYS_TO_MAINSHOCK'], outside['MAGNITUDE'], 
            #    s=np.exp(aftershocks['MAGNITUDE']),
            s = event_marker_size(outside['MAGNITUDE']), ec='white', linewidth=0.25,
               label= f"Aftershocks: {len(outside)}", color='grey', alpha=0.25, zorder=0)

    
    ax2 = ax.twinx()
    cut_off_day=min_days
    foreshock_window=foreshock_days
    range_scaler=100
    modelling_plus_foreshocks = pd.concat([modelling_events, foreshocks])
    # sliding_window_points_full = np.array(range((-cut_off_day+foreshock_window)*range_scaler, 0*range_scaler+1, 1))/range_scaler*-1
    sliding_window_points_full = np.array(range((-cut_off_day+foreshock_window), 0+1, 1))/-1
    sliding_window_counts_full = np.array([len(modelling_plus_foreshocks[(modelling_plus_foreshocks['DAYS_TO_MAINSHOCK'] > point) &\
                                                                          (modelling_plus_foreshocks['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]) for point in sliding_window_points_full])
    average = np.mean
    # sliding_window_distances = np.array([average(modelling_plus_foreshocks.loc[(modelling_plus_foreshocks['DAYS_TO_MAINSHOCK'] > point) &\
    #                                                                 (modelling_plus_foreshocks['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window)), 'DISTANCE_TO_MAINSHOCK']) for point in sliding_window_points_full])

    # sliding_window_df_full = pd.DataFrame({'points':sliding_window_points_full,
    #                                        'counts':sliding_window_counts_full,
    #                                        'distances':sliding_window_distances})
    
    ax2.step(sliding_window_points_full, sliding_window_counts_full, zorder=6,
            #  c=sliding_window_points_full.astype(int),
             color='black', alpha=0.7,
             label='20-day count')
    # ax2.axhline(y=len(foreshocks), color='red', alpha=0.5, label = r'$N_{obs}$', zorder=0)
    ax2.set_ylabel('20-day count')
    # ax.set_zorder(ax2.get_zorder()+1)
    ax.patch.set_visible(False)
    try:
        y_min, y_max = round(sliding_window_counts_full.min()), round(sliding_window_counts_full.max())
        y_mid = round(y_min + (y_max - y_min)/2)
        y_ticks = [y_min, y_mid, y_max]
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(y_ticks)
    except:
        print('Could not update yticks')

    ax = fig.add_subplot(gs[1:3, :], projection=ccrs.PlateCarree())
    ax.set_title('b)', fontsize=20, loc='left')

    # ax = fig.add_subplot(212, projection=ccrs.PlateCarree())
    
    # ax.set_extent(utils.get_catalogue_extent(local_cat, buffer=0.025), crs=ccrs.PlateCarree())
    ax.set_extent([min_box_lon, max_box_lon, min_box_lat, max_box_lat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, edgecolor='none')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, zorder=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    gl.xlines = False
    gl.ylines = False
    # gl.ylocator = mticker.FixedLocator([min_box_lat, max_box_lat])
    # gl.xlocator = mticker.FixedLocator([min_box_lon, max_box_lon])

    plot_scalar=6
    ax.scatter(mainshock_LON, mainshock_LAT, color=mainshock_colour, s=event_marker_size(mainshock_M), marker='*', label=f'$M_w$ {mainshock_M} mainshock')
    new_LON, new_LAT = utils.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, radius_km, 0)
    radius_degrees = new_LON - mainshock_LON
    circle = Circle((mainshock_LON, mainshock_LAT), radius_degrees, edgecolor='r', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(circle)
    if len(outside)>0:
        ax.scatter(outside['LON'], outside['LAT'], s=event_marker_size(outside['MAGNITUDE']), ec='white', linewidth=0.25,
                   color='grey', alpha=0.25, zorder=0)
    if len(inside) > 0:
        local_cat_plot = ax.scatter(inside['LON'], inside['LAT'], s=event_marker_size(inside['MAGNITUDE']), zorder=1, ec='white', linewidth=0.25,
                                    c=inside['DAYS_TO_MAINSHOCK'], label=f'inside: {len(inside)}', alpha=0.75, vmin=vmin, vmax=vmax) 
        cbar = fig.colorbar(local_cat_plot, ax=ax) #, shrink=0.5
        cbar.set_label('Days to mainshock')
    if len(foreshocks) > 0:
        ax.scatter(foreshocks['LON'], foreshocks['LAT'], s=event_marker_size(foreshocks['MAGNITUDE']),
                   color='red', alpha=0.75, ec='white', linewidth=0.25, zorder=2)
    
    if stations is not None:
        stations = utils.select_within_box(mainshock.LON, mainshock.LAT, df=stations, r=box_halfwidth_km)
        ax.scatter(stations['LON'], stations['LAT'], ec='white', linewidth=0.25, marker='^', color=plot_color_dict['orange'], zorder=100000)
    ax.set_xlabel('LON')
    ax.set_ylabel('LAT')

    cmap = plt.get_cmap('Spectral')
    rgba = cmap(0.5)
    
    custom_legend_items = [Line2D([], [], color='red', marker='*', markersize=10,
                                label=f'$M_w$ {mainshock_M} mainshock', linestyle='None'),
                        Line2D([], [], color=rgba, marker='o', markersize=10, 
                                label=f'{len(modelling_events)} events 1 year prior)', linestyle='None'),
                        Line2D([], [], color='grey', marker='o', markersize=10, 
                                label=f'{len(aftershocks)} aftershocks (20 days post)', linestyle='None')
                        #    Line2D([0], [0], color='black', lw=0, marker='', label=f"Spearmanr: {round(stats_dict['QTM_12_spearmanr'],2)}"),
                        #    Line2D([0], [0], color='black', lw=0, marker='', label=f"Spearmanr: {round(stats_dict['SCSN_spearmanr'],2)}")
                        ]
    # ax.legend(handles=custom_legend_items, loc='lower right', bbox_to_anchor=(0.575,1))
    # ax.legend(loc='lower right', bbox_to_anchor=(0.575,1))

    if save==True:
        Path(f"../outputs/{catalogue_name}/data_plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"../outputs/{catalogue_name}/data_plots/{mainshock_ID}.png")
    plt.show()