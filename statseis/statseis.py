"""
This module contains functions to statistically analyse seismicity (source parameters e.g. time, location, and magnitude).
Many functions require the renaming of earthquake catalog dataframe columns to: ID, MAGNITUDE, DATETIME, LON, LAT, DEPTH.
"""

### Notes ###
# Moved create spatial plot and plot local cat to cartopy maps, may have messed things up


import datetime as dt
import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as special
import scipy.stats as stats
from scipy.stats import gamma, poisson
import pyproj
from IPython.display import clear_output
import string
import matplotlib.collections
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Circle
from collections import namedtuple
import shutil
from tqdm import tqdm
import statseis.utils
import statseis.mc
import statseis.cartopy_maps as cartmaps

date = str(dt.datetime.now().date().strftime("%y%m%d"))

# To do
# Some functions may not work since I made the statseis.mc submodule, update

plot_colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
plot_color_dict = dict(zip(['teal', 'orange', 'purple', 'pink', 'green', 'yellow', 'brown', 'grey'], plot_colors))

alphabet = string.ascii_lowercase
panel_labels = [letter + ')' for letter in alphabet]

scale_eq_marker = (lambda x: 10 + np.exp(1.1*x))

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

def move_plots(mshock_file, catalog, plot_type, out_folder_name):
    for mainshock in mshock_file.itertuples():
        filename = f'{mainshock.ID}.png'

        source_folder = f'../outputs/{catalog}/{plot_type}/'

        destination_folder = f'../outputs/{out_folder_name}/{plot_type}'
        Path(destination_folder).mkdir(parents=True, exist_ok=True)

        source_file = os.path.join(source_folder, filename)
        
        if os.path.exists(source_file):
            destination_file = os.path.join(destination_folder, filename)
            
            shutil.copyfile(source_file, destination_file)
            print(f"File '{filename}' copied to '{destination_folder}'")
        else:
            print(f"File '{filename}' not found in '{source_folder}'")

def mainshock_selections_counts(mainshock_file):
    print('Total', len(mainshock_file))
    for s in mainshock_file['Selection'].unique():
        print(s, len(mainshock_file.loc[mainshock_file['Selection']==s]))

def iterable_mainshock(ID, mainshock_file):
    mainshock = mainshock_file.loc[mainshock_file['ID']==ID].copy()
    RowTuple = namedtuple('RowTuple', mainshock.columns)
    mainshock = [RowTuple(*row) for row in mainshock.values][0]
    return mainshock

def load_local_catalogue(mainshock, catalogue_name='unspecified'):
    local_catalogue = pd.read_csv(f'../data/{catalogue_name}/local_catalogues/{mainshock.ID}.csv')
    utils.string_to_datetime_df(local_catalogue)
    return local_catalogue

def create_local_catalogue(mainshock, earthquake_catalogue, catalogue_name, radius_km = 30, box=False, save=True):

    mainshock_LON = mainshock.LON
    mainshock_LAT = mainshock.LAT
    mainshock_DATETIME = mainshock.DATETIME

    box_halfwidth_km = radius_km

    local_catalogue = utils.select_within_box(mainshock.LON, mainshock.LAT, earthquake_catalogue, r=box_halfwidth_km)
    # min_box_lon, min_box_lat = utils.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, -box_halfwidth_km, -box_halfwidth_km)
    # max_box_lon, max_box_lat = utils.add_distance_to_position_pyproj(mainshock_LON, mainshock_LAT, box_halfwidth_km, box_halfwidth_km)

    # local_catalogue = earthquake_catalogue.loc[
    #                                 (earthquake_catalogue['LON']>= min_box_lon) &\
    #                                 (earthquake_catalogue['LON']<= max_box_lon) &\
    #                                 (earthquake_catalogue['LAT']>= min_box_lat) &\
    #                                 (earthquake_catalogue['LAT']<= max_box_lat)
    #                                 ].copy()
    local_catalogue['DISTANCE_TO_MAINSHOCK'] = utils.calculate_distance_pyproj_vectorized(mainshock_LON, mainshock_LAT, local_catalogue['LON'],  local_catalogue['LAT'])

    if box==False:
        local_catalogue = local_catalogue.loc[local_catalogue['DISTANCE_TO_MAINSHOCK']<radius_km].copy()

    local_catalogue['DAYS_TO_MAINSHOCK'] = (mainshock_DATETIME - local_catalogue['DATETIME']).apply(lambda d: (d.total_seconds()/(24*3600)))

    if save==True:
        Path(f'../data/{catalogue_name}/local_catalogues/').mkdir(parents=True, exist_ok=True)
        local_catalogue.to_csv(f'../data/{catalogue_name}/local_catalogues/{mainshock.ID}.csv', index=False)
    return local_catalogue

def select_mainshocks(earthquake_catalogue,
                      search_style='radius',
                      search_distance_km=10,
                      mainshock_magnitude_threshold = 4,
                      minimum_exclusion_distance = 20,
                      scaling_exclusion_distance = 5,
                      minimum_exclusion_time = 50,
                      scaling_exclusion_time = 25,
                      station_file=None,
                      restrict_by_year=True
                      ):
    """
    Select mainshocks from an earthquake catalogue using the following methods:
    MDET - Magnitude-Dependent Exclusion Thresholds (Trugman & Ross, 2019);
    FET - Fixed Exclusion Thresholds (Moutote et al., 2021).
    DDET - Distance-Dependent Exclusion Thresholds.
    """
    
    earthquakes_above_magnitude_threshold = earthquake_catalogue.loc[earthquake_catalogue['MAGNITUDE'] >= mainshock_magnitude_threshold].copy()
    
    exclusion_criteria_results = []
    TR_mainshocks_to_exclude = []
    for mainshock in tqdm(earthquakes_above_magnitude_threshold.itertuples(), total=len(earthquakes_above_magnitude_threshold)):

        local_catalogue = utils.select_within_box(mainshock.LON, mainshock.LAT, df=earthquake_catalogue, r=search_distance_km)        
        # min_box_lon, min_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, -search_distance_km, -search_distance_km)
        # max_box_lon, max_box_lat = add_distance_to_position_pyproj(mainshock.LON, mainshock.LAT, search_distance_km, search_distance_km)

        # local_catalogue = earthquake_catalogue.loc[
        #                                 (earthquake_catalogue['LON']>= min_box_lon) &\
        #                                 (earthquake_catalogue['LON']<= max_box_lon) &\
        #                                 (earthquake_catalogue['LAT']>= min_box_lat) &\
        #                                 (earthquake_catalogue['LAT']<= max_box_lat)
        #                                 ].copy()
        # local_catalogue['DISTANCE_TO_MAINSHOCK'] = calculate_distance_pyproj_vectorized(mainshock.LON, mainshock.LAT, local_catalogue['LON'],  local_catalogue['LAT'])

        if search_style=='radius':    
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

        if n_local_catalogue > 0:
            try:
                Mbass, this_fmd, b, b_avg, mc_shibolt_unc = mc.get_mbs(np.array(local_catalogue['MAGNITUDE']), mbin=0.1)
                a, b_Mbass, aki_unc, shibolt_unc = mc.b_est(np.array(local_catalogue['MAGNITUDE']), mbin=0.1, mc=Mbass)
            except:
                Mbass, b_Mbass = [np.nan]*2
            maxc = mc.get_maxc(np.array(local_catalogue['MAGNITUDE']), mbin=0.1)
            a, b_maxc, aki_unc, shibolt_unc = mc.b_est(np.array(local_catalogue['MAGNITUDE']), mbin=0.1, mc=maxc)
        else:
            Mbass, b_Mbass, maxc, b_maxc = [np.nan]*4
                    
        subsurface_rupture_length = 10**((mainshock.MAGNITUDE - 4.38)/1.49)
        distance_exclusion_threshold = minimum_exclusion_distance + scaling_exclusion_distance * subsurface_rupture_length
        time_exclusion_threshold = minimum_exclusion_time + scaling_exclusion_time * (mainshock.MAGNITUDE - mainshock_magnitude_threshold)

        distances_between_earthquakes = utils.calculate_distance_pyproj_vectorized(mainshock.LON, mainshock.LAT, earthquakes_above_magnitude_threshold['LON'],  earthquakes_above_magnitude_threshold['LAT'])

        earthquakes_within_exclusion_criteria = earthquakes_above_magnitude_threshold.loc[
            (mainshock.ID != earthquakes_above_magnitude_threshold.ID) &\
            (distances_between_earthquakes <= distance_exclusion_threshold) &\
            (earthquakes_above_magnitude_threshold.MAGNITUDE < mainshock.MAGNITUDE) &\
            (((earthquakes_above_magnitude_threshold['DATETIME'] - mainshock.DATETIME).apply(lambda d: d.total_seconds()/(3600*24))) < time_exclusion_threshold) &\
            (((earthquakes_above_magnitude_threshold['DATETIME'] - mainshock.DATETIME).apply(lambda d: d.total_seconds()/(3600*24)) > 0))
            ]
        
        TR_mainshocks_to_exclude.extend(list(earthquakes_within_exclusion_criteria['ID']))

        if station_file is not None and not station_file.empty:
            distance_to_stations = np.array(utils.calculate_distance_pyproj_vectorized(mainshock.LON, mainshock.LAT, station_file['LON'],  station_file['LAT']))
            distance_to_stations = np.sort(distance_to_stations)
        else:
            distance_to_stations = [np.nan]*4
        
        results_dict = {'ID':mainshock.ID,
                        'DATETIME':mainshock.DATETIME,
                        'MAGNITUDE':mainshock.MAGNITUDE,
                        'LON':mainshock.LON,
                        'LAT':mainshock.LAT,
                        'DEPTH':mainshock.DEPTH,
                        'Maxc':maxc,
                        'Mbass':Mbass,
                        'b_Mbass':b_Mbass,
                        'b_maxc':b_maxc,
                        'n_local_cat':n_local_catalogue,
                        'n_local_cat_1yr':n_local_catalogue_1yr,
                        'Largest_preceding_1yr':max_magnitude,
                        'Moutote_method':Moutote_method,
                        'Moutote_excluded_by':Moutote_excluded_by,
                        'subsurface_rupture_length':subsurface_rupture_length,
                        'distance_exclusion_threshold':distance_exclusion_threshold,
                        'time_exclusion_threshold':time_exclusion_threshold,
                        'TR_excludes':list(earthquakes_within_exclusion_criteria['ID']),
                        'km_to_STA':distance_to_stations[0:4],
                        'STA_4_km':distance_to_stations[3]}
        
        exclusion_criteria_results.append(results_dict)
        clear_output(wait=True)
    
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

    if restrict_by_year==True:
        exclusion_criteria_results = exclusion_criteria_results.loc[exclusion_criteria_results['DATETIME'] >= earthquake_catalogue.iloc[0]['DATETIME'] + dt.timedelta(days=365)].copy()  

    return exclusion_criteria_results

# Defunct? Same as plot local cat?
def plot_single_mainshock(ID, mainshock_file, catalogue_name, earthquake_catalogue, Mc_cut = False):
    mainshock = iterable_mainshock(ID, mainshock_file)
    local_cat = create_local_catalogue(mainshock=mainshock, earthquake_catalogue=earthquake_catalogue, catalogue_name=catalogue_name)
    if Mc_cut==True:
        local_cat = mc.apply_Mc_cut(local_cat)
    cartmaps.plot_local_cat(mainshock=mainshock, local_cat=local_cat, Mc_cut=Mc_cut, catalogue_name=catalogue_name, earthquake_catalogue=earthquake_catalogue)

def identify_foreshocks_short(mainshock, earthquake_catalogue, local_catalogue, iterations=10000,
                              local_catalogue_radius = 10, foreshock_window = 20, modelling_time_period=345, Wetzler_cutoff=3):
    """
    Identify foreshocks before mainshocks using the following methods:
        BP - Background Poisson (Trugman and Ross, 2019);
        G-IET - Gamma inter-event time (van den Ende & Ampuero, 2020);
        ESR - Empirical Seismicity Rate (van den Ende & Ampuero, 2020).
        We create code for the BP and ESR methods. We integrate the publically available code for the G-IET method (van den Ende & Ampuero, 2020).
    """
    
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
    #     Mc = round(Mc_by_maximum_curvature(local_catalogue['MAGNITUDE']),2) + 0.2
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
            b_value = round(mc.b_val_max_likelihood(seismicity_period['MAGNITUDE'], mc=mainshock_Mc), 2)
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

    variance = np.var(sliding_window_counts)
    q25 = np.percentile(sliding_window_counts, 25)
    q75 = np.percentile(sliding_window_counts, 75)
    ESR_median = np.median(sliding_window_counts)

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


    if n_regular_seismicity_events > 2:
        t_day = 3600 * 24.0
        t_win = foreshock_window * t_day
        IET = np.diff(time_series) ### V&As Gamma IET method
        IET = IET[IET>0]
        try:
            y_, loc_, mu_ = stats.gamma.fit(IET, floc=0.0)
        except:
            y_, loc_, mu_ = stats.gamma.fit(IET, loc=0.0)
        # print(f"y_ {y_}, loc_ {loc_}, mu_ {mu_}")

        if (np.isnan(y_)==False) & (np.isnan(mu_)==False):
            N_eq = np.zeros(iterations, dtype=int) # Buffer for the number of earthquakes observed in each random sample
            for i in range(0,iterations):
                prev_size = 200 # Generate a random IET sample with 200 events
                IET2 = stats.gamma.rvs(a=y_, loc=0, scale=mu_, size=prev_size) * t_day # Sample from gamma distribution
                t0 = np.random.rand() * IET2[0] # Random shift of timing of first event
                t_sum = np.cumsum(IET2) - t0 # Cumulative sum of interevent times
                inds = (t_sum > t_win) # Find the events that lie outside t_win
                while (inds.sum() == 0):
                    prev_size *= 2 # If no events lie outside t_win, create a bigger sample and stack with previous sample
                    IET2 = np.hstack([IET2, stats.gamma.rvs(a=y_, loc=0, scale=mu_, size=prev_size) * t_day])
                    t_sum = np.cumsum(IET2) # Cumulative sum of event times
                    inds = (t_sum > t_win) # Find the events that lie outside t_win
                N_inside_t_win = (~inds).sum()
                if N_inside_t_win == 0: 
                    N_eq[i] = 0 # No events inside t_win, seismicity rate = 0.
                else:
                    N_eq[i] =  N_inside_t_win - 1 # Store the number of events that lie within t_win (excluding shifted event)

            try:
                y_gam_IETs, loc_gam_IETs, mu_gam_IETs = stats.gamma.fit(N_eq[N_eq > 0], floc=0.0)
            except:
                y_gam_IETs, loc_gam_IETs, mu_gam_IETs = stats.gamma.fit(N_eq[N_eq > 0], loc=0.0)
        
        # print(f"y_gam_IETs {y_gam_IETs}, loc_gam_IETs {loc_gam_IETs}, mu_gam_IETs {mu_gam_IETs}")
        VA_gamma_probability = stats.gamma.sf(n_events_in_foreshock_window, y_gam_IETs, loc_gam_IETs, mu_gam_IETs)
        VA_gamma_99CI = stats.gamma.ppf(0.99, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
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
                    'cut_off_day':cut_off_day,
                    'M3_IDs':Wetzler_foreshocks['ID'],
                    'ESR_median':ESR_median,
                    'var':variance,
                    'q25':q25,
                    'q75':q75
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
    min_days=365
    max_days=0
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
    
    # event_marker_size = (lambda x: 12*np.exp(x))
    # event_marker_size = (lambda x: 7.5**(x))
    event_marker_size = (lambda x: 50+10**(x/1.25))
    linewidth = 3
    
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
    local_cat = file_dict['local_catalogue']
    # local_catalogue_pre_Mc_cutoff= file_dict['local_catalogue_pre_Mc_cutoff']
    # local_catalogue_below_Mc= file_dict['local_catalogue_below_Mc']
    foreshocks= file_dict['foreshocks']
    # foreshocks_below_Mc= file_dict['foreshocks_below_Mc']
    sliding_window_counts = file_dict['sliding_window_counts']
    N_eq = file_dict['N_eq']
    N_eq = N_eq[N_eq>0]
    
    align = 'mid' #left, or right
    # foreshocks_colour = 'red'
    # regular_earthquakes_colour = 'black'
    # mainshock_colour = 'red'
    # poisson_colour = colour_dict['orange']
    # gamma_colour = colour_dict['green']
    # ESR_colour = colour_dict['dark pink']
    # Mc_colour = colour_dict['light blue']
    # rate_colour = colour_dict['dark pink']

    foreshocks_colour = 'red'
    regular_earthquakes_colour = 'black'
    mainshock_colour = 'black'
    poisson_colour = plot_color_dict['orange']
    gamma_colour = plot_color_dict['teal']
    ESR_colour = plot_color_dict['purple']
    Mc_colour = plot_color_dict['grey']
    rate_colour = 'black'
    rate_alpha = 0.5
    vmin, vmax = 0, 365

    modelling_events = local_cat.loc[(local_cat['DAYS_TO_MAINSHOCK'] < min_days) &\
                                    (local_cat['DAYS_TO_MAINSHOCK'] > max_days+20) &\
                                    (local_cat['DISTANCE_TO_MAINSHOCK']<10)].copy()
    aftershocks = local_cat.loc[(local_cat['DAYS_TO_MAINSHOCK'] < 0) &\
                                (local_cat['DAYS_TO_MAINSHOCK'] > -20)].copy()
    
    range_scaler=100
    sliding_window_points_full = np.array(range((-cut_off_day+foreshock_window)*range_scaler, 0*range_scaler+1, 1))/range_scaler*-1
    sliding_window_counts_full = np.array([len(local_catalogue[(local_catalogue['DAYS_TO_MAINSHOCK'] > point) & (local_catalogue['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]) for point in sliding_window_points_full])

    sliding_window_df_full = pd.DataFrame({'points':sliding_window_points_full,
                                           'counts':sliding_window_counts_full})
    
    # time_series_plot, model_plot, CDF_plot, foreshock_window_plot, Mc_plot = 0, 1, 2, 3, 4
    time_series_plot, model_plot, CDF_plot, foreshock_window_plot= 0, 1, 2, 3

    panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)']

    histogram_alpha = 0.95
    fig, axs = plt.subplots(5,1, figsize=(10,15))
    # fig, axs = plt.subplots(4,1, figsize=(10,15))

    axs[time_series_plot].set_title('a)', fontsize=20, loc='left')
    axs[time_series_plot].set_title(f"ID: {mainshock_ID} - {mainshock.DATETIME.strftime('%b %d %Y')} - {catalogue_name}", loc='right')

    axs[time_series_plot].scatter(0, mainshock.MAGNITUDE, s= event_marker_size(mainshock.MAGNITUDE), #s=400, 
                                  ec=mainshock_colour, fc='grey', alpha=0.5,
                                    label=r'$M_{w}$ ' + str(mainshock.MAGNITUDE) + ' Mainshock',  
                                    zorder=1)
    axs[time_series_plot].axvline(x=20, color='red', linestyle='--', linewidth=linewidth,
                                    label = f"20-day foreshock window",
                                    zorder=4)
    axs[time_series_plot].set_xlabel('Days to mainshock', fontsize=20)
    axs[time_series_plot].set_ylabel('Magnitude', fontsize=20)
    axs[time_series_plot].set_xlim(-25,365+20)
    # axs[time_series_plot].set_ylim(axs[time_series_plot].get_extent(),365+20)
    current_ylim = axs[time_series_plot].get_ylim()
    # axs[time_series_plot].set_ylim(current_ylim[0], current_ylim[1]+0.5)
    axs[time_series_plot].invert_xaxis()

    if len(modelling_events) >0:
        # ax.set_yticks(np.arange(math.floor(min(local_cat['MAGNITUDE'])), math.ceil(mainshock.MAGNITUDE), 1))
        axs[time_series_plot].scatter(modelling_events['DAYS_TO_MAINSHOCK'], modelling_events['MAGNITUDE'],
                   s=event_marker_size(modelling_events['MAGNITUDE']), vmin=vmin, vmax=vmax, 
                                        label= f'{len(modelling_events)- len(foreshocks)} modelling events',
                                        c=modelling_events['DAYS_TO_MAINSHOCK'], alpha=0.5,  zorder=1)
        # axs[time_series_plot].scatter(local_catalogue_below_Mc['DAYS_TO_MAINSHOCK'], local_catalogue_below_Mc['MAGNITUDE'], 
        #                             label= str(len(local_catalogue_below_Mc)) + ' Earthquakes below Mc', 
        #               
        #               alpha=0.5, color=Mc_colour)
    if len(foreshocks) > 0:
        axs[time_series_plot].scatter(foreshocks['DAYS_TO_MAINSHOCK'], foreshocks['MAGNITUDE'],
                    s=event_marker_size(foreshocks['MAGNITUDE']),
                   label= fr"$N_obs$: {len(foreshocks)}", color='red', alpha=0.5, zorder=5)
        
    if len(aftershocks) > 0:
        axs[time_series_plot].scatter(aftershocks['DAYS_TO_MAINSHOCK'], aftershocks['MAGNITUDE'], 
               s=event_marker_size(aftershocks['MAGNITUDE'])/4,
               label= f"Aftershocks: {len(aftershocks)}", color='grey', alpha=0.5, zorder=0)

    
    ax2 = axs[time_series_plot].twinx()
    cut_off_day=365
    foreshock_window=20
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
    
    ax2.step(sliding_window_points_full, sliding_window_counts_full, zorder=6, linewidth=linewidth, where='post',
            #  c=sliding_window_points_full.astype(int),
             color='black', alpha=rate_alpha,
             label='Count')
    ax2.axhline(y=len(foreshocks), color='red', alpha=0.5, label = r'$N_{obs}$', zorder=100, linewidth=linewidth,)
    ax2.set_ylabel('20-day Count')
    # ax.set_zorder(ax2.get_zorder()+1)
    axs[time_series_plot].patch.set_visible(False)
    try:
        y_min, y_max = round(sliding_window_counts_full.min()), round(sliding_window_counts_full.max())
        y_mid = round(y_min + (y_max - y_min)/2)
        y_ticks = [y_min, y_mid, y_max]
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(y_ticks)
    except:
        print('Could not update yticks')

    radius_km=10
    modelling_time_period=365
    local_catalogue_pre_Mc_cutoff = load_local_catalogue(mainshock=mainshock, catalogue_name=catalogue_name)
    # local_catalogue_pre_Mc_cutoff = create_local_catalogue(mainshock=mainshock, catalogue_name=catalogue_name, earthquake_catalogue=catalogue_dict[catalogue_name])
    local_catalogue_pre_Mc_cutoff = local_catalogue_pre_Mc_cutoff[(local_catalogue_pre_Mc_cutoff['DATETIME'] < mainshock_DATETIME) &\
                                        (local_catalogue_pre_Mc_cutoff['DAYS_TO_MAINSHOCK'] < modelling_time_period+foreshock_window) &\
                                        (local_catalogue_pre_Mc_cutoff['DAYS_TO_MAINSHOCK'] > 0)  &\
                                        (local_catalogue_pre_Mc_cutoff['DISTANCE_TO_MAINSHOCK'] < radius_km) &\
                                        (local_catalogue_pre_Mc_cutoff['ID'] != mainshock_ID)
                                        ].copy()
    
    # if len(local_catalogue_pre_Mc_cutoff)>0:
    #     print(local_catalogue_pre_Mc_cutoff['MAGNITUDE'].head())
    #     bins = np.arange(math.floor(local_catalogue_pre_Mc_cutoff['MAGNITUDE'].min()), math.ceil(local_catalogue_pre_Mc_cutoff['MAGNITUDE'].max()), 0.1)
    #     values, base = np.histogram(local_catalogue_pre_Mc_cutoff['MAGNITUDE'], bins=bins)
    #     cumulative = np.cumsum(values)
    #     axs[Mc_plot].step(base[:-1], len(local_catalogue_pre_Mc_cutoff)-cumulative, label='FMD', color='black', linewidth=linewidth)
    #     axs[Mc_plot].axvline(x=Mc, linestyle='--', color=Mc_colour, label=r'$M_{c}$: ' + str(round(Mc,1)), linewidth=linewidth)
    # axs[Mc_plot].set_title(panel_labels[Mc_plot], fontsize=20, loc='left')
    # axs[Mc_plot].set_xlabel('Magnitude')
    # axs[Mc_plot].set_ylabel('N')
    # # axs[Mc_plot].legend()
    # axs[Mc_plot].set_yscale('log')
    
    # axs[time_series_plot].set_title(panel_labels[time_series_plot], fontsize=20, loc='left')
    # axs[time_series_plot].scatter(0, mainshock.MAGNITUDE, marker='*', s=400, color=mainshock_colour,
    #                                 label=r'$M_{w}$ ' + str(mainshock.MAGNITUDE) + ' Mainshock',  
    #                                 zorder=3)
    # axs[time_series_plot].axvline(x=foreshock_window, color=foreshocks_colour, linestyle='--', 
    #                                 label = f"{foreshock_window}-day foreshock window",
    #                                 zorder=4)
    # axs[time_series_plot].set_xlabel('Days to mainshock', fontsize=20)
    # axs[time_series_plot].set_ylabel('M', fontsize=20)
    # axs[time_series_plot].set_xlim(-5,cut_off_day+foreshock_window)
    # axs[time_series_plot].invert_xaxis()

    # if len(local_catalogue) >0:
    #     # axs[time_series_plot].set_yticks(np.arange(math.floor(min(local_catalogue['MAGNITUDE'])), math.ceil(mainshock.MAGNITUDE), 1))
    #     axs[time_series_plot].scatter(local_catalogue['DAYS_TO_MAINSHOCK'], local_catalogue['MAGNITUDE'],
    #                                     label= str(n_regular_seismicity_events) + ' Earthquakes for modelling',
    #                                     color=regular_earthquakes_colour, alpha=0.5,  zorder=1)
    #     # axs[time_series_plot].scatter(local_catalogue_below_Mc['DAYS_TO_MAINSHOCK'], local_catalogue_below_Mc['MAGNITUDE'], 
    #     #                             label= str(len(local_catalogue_below_Mc)) + ' Earthquakes below Mc', 
    #     #                             alpha=0.5, color=Mc_colour)
    if len(foreshocks) > 0:
        # axs[time_series_plot].scatter(foreshocks['DAYS_TO_MAINSHOCK'], foreshocks['MAGNITUDE'],
        #                                 label= str(n_events_in_foreshock_window) + ' Earthquakes in foreshock window (' + r'$N_{obs}$)',
        #                                 color=foreshocks_colour, alpha=0.5, 
        #                                 zorder=2)
        axs[foreshock_window_plot].scatter(foreshocks['DAYS_TO_MAINSHOCK'], foreshocks['MAGNITUDE'],
                                           s=event_marker_size(foreshocks['MAGNITUDE']),
                                           label=r'$N_{obs}$: ' + str(len(foreshocks)), color='red', alpha=0.5, zorder=5)
        # axs[foreshock_window_plot].scatter(foreshocks['DAYS_TO_MAINSHOCK'], foreshocks['MAGNITUDE'], color=foreshocks_colour, alpha=0.5,
        #                                     label=r'$N_{obs}$: ' + str(n_events_in_foreshock_window))
        # axs[foreshock_window_plot].scatter(foreshocks_below_Mc['DAYS_TO_MAINSHOCK'], foreshocks_below_Mc['MAGNITUDE'], 
        #                                     label= str(len(foreshocks_below_Mc)) + ' Earthquakes below Mc', 
        #                                     alpha=0.2, color=Mc_colour)
        foreshock_sliding_window = sliding_window_df_full.loc[sliding_window_df_full['points']<=20].copy()
        ax_foreshock_window_twin = axs[foreshock_window_plot].twinx()
        ax_foreshock_window_twin.step(foreshock_sliding_window['points'], foreshock_sliding_window['counts'], color=rate_colour, label='20-day count)', where='post',
                                      alpha=rate_alpha, linewidth=linewidth)
        ax_foreshock_window_twin.set_ylabel('20-day Count')
        axs[foreshock_window_plot].set_yticks(np.arange(math.floor(foreshocks['MAGNITUDE'].min()), math.ceil(mainshock.MAGNITUDE), 1))
    if np.isnan(Mc)==False:
        axs[time_series_plot].axhline(y=Mc, color=Mc_colour, linestyle='--', linewidth=linewidth,
                                        label = r'$M_{c}$: ' + str(round(Mc,1)),
                                        zorder=5)
        axs[foreshock_window_plot].axhline(y=Mc, color=Mc_colour, linestyle='--', label = r'$M_{c}$: ' + str(round(Mc,1)), zorder=5, linewidth=linewidth)

    axs[foreshock_window_plot].set_title(panel_labels[foreshock_window_plot], fontsize=20, loc='left')
    # axs[foreshock_window_plot].scatter(1E-10, mainshock.MAGNITUDE, marker='*', s=400, color=mainshock_colour, zorder=2)
    axs[foreshock_window_plot].axvline(x=foreshock_window, color='red', linestyle='--', linewidth=linewidth,)

    axs[foreshock_window_plot].set_xlabel('Days to mainshock')
    axs[foreshock_window_plot].set_ylabel('Magnitude')
    
    axs[foreshock_window_plot].invert_xaxis()
    axs[foreshock_window_plot].set_xscale('log')
    axs[foreshock_window_plot].set_xticks([10, 1, 0.1, 0.01])
    axs[foreshock_window_plot].set_xticklabels([10, 1, 0.1, 0.01])

    # ax2 = axs[time_series_plot].twinx()
    # ax2.plot(sliding_window_points_full, sliding_window_counts_full, color=rate_colour, label='Count')
    # ax2.axhline(y=n_events_in_foreshock_window, color=foreshocks_colour, alpha=0.5, 
    #                                     label = r'$N_{obs}$', zorder=0)
    # ax2.set_ylabel('Count')
    # lines, labels = axs[time_series_plot].get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.set_yticks(utils.estimate_axis_labels(sliding_window_counts_full))
    # axs[time_series_plot].legend(lines + lines2, labels + labels2, loc='upper left')

    axs[model_plot].set_title(panel_labels[model_plot], fontsize=20, loc='left')
    axs[model_plot].set_xlabel('20-day Count', fontsize=20)
    axs[model_plot].set_ylabel('PDF', fontsize=20)
    axs[model_plot].axvline(x=n_events_in_foreshock_window, color=foreshocks_colour, label=r'$N_{obs}$: ' + str(n_events_in_foreshock_window), linewidth=linewidth)      
    axs[CDF_plot].axvline(x=n_events_in_foreshock_window, color='red', linewidth=linewidth,
                            label=r'$N_{obs}$: ' + str(n_events_in_foreshock_window))
    # axs[model_plot].set_xticks(range(0,20,2))

    if len(sliding_window_counts) > 0:
        event_counts_pdf = sliding_window_counts/sum(sliding_window_counts)
        event_counts_cdf = np.cumsum(event_counts_pdf)
        event_counts_sorted = np.sort(sliding_window_counts)

    if len(N_eq) > 0:
        N_eq_counts, N_eq_bins = np.histogram(N_eq, bins=range(math.floor(min(N_eq))-1, math.ceil(max(N_eq))+2))
        # axs[model_plot].step(N_eq_bins[:-1], N_eq_counts/N_eq_counts.sum(), color=gamma_colour, where='post',
        #                     label=f"{method_dict['VA_half_method']}: {round(VA_IETs_probability,3)}",
        #                     alpha=histogram_alpha)
        # axs[model_plot].hist(N_eq, bins=range(min(N_eq)-1, max(N_eq)+1), color=gamma_colour,
        #                     label=f"{method_dict['VA_half_method']}: {str(round(VA_IETs_probability,3))}",
        #                     density=True, rwidth=1.0, alpha=histogram_alpha/2, align=align)
        N_eq_pdf = N_eq/sum(N_eq)
        N_eq_cdf = np.cumsum(N_eq_pdf)
        N_eq_sorted = np.sort(N_eq)
        # axs[CDF_plot].plot(N_eq_sorted, N_eq_cdf, label=method_dict["VA_half_method"], color=gamma_colour, alpha=histogram_alpha/2)
    if (np.isnan(TR_expected_events)==False) & (TR_expected_events!=0):
        min_x, max_x = round(poisson.ppf(0.001, TR_expected_events)), round(poisson.ppf(0.999, TR_expected_events))
        x_TR_Poisson = range(min_x, max_x+1)
        y_TR_Poisson = poisson.pmf(x_TR_Poisson, TR_expected_events)
        axs[model_plot].step(x_TR_Poisson, y_TR_Poisson, label=f"{method_dict['TR_method']}: {str(round(TR_probability,3))}", where='post',
                alpha=histogram_alpha, color=poisson_colour, linewidth=linewidth)
        TR_poisson_cdf = poisson.cdf(x_TR_Poisson, TR_expected_events)
        axs[CDF_plot].step(x_TR_Poisson, TR_poisson_cdf, label=method_dict['TR_method'], alpha=histogram_alpha, color=poisson_colour, linewidth=linewidth, where='post',)

    if (np.isnan(y_gam_IETs)==False) & (np.isnan(mu_gam_IETs)==False):
        # x_gam_IETs = np.arange(gamma.ppf(0.001, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs),
                                # gamma.ppf(0.999, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs))
        # x_gam_IETs = range(min(N_eq), max(N_eq))
        min_x, max_x = round(gamma.ppf(0.001, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)), round(gamma.ppf(0.999, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs))
        if min_x > min(sliding_window_counts):
            min_x = min(sliding_window_counts)
        x_gam_IETs = range(min_x, max_x+1)
        gamma_pdf = gamma.pdf(x_gam_IETs, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
        axs[model_plot].step(x_gam_IETs, gamma_pdf, linewidth=linewidth, where='post',
                                label= method_dict['VA_method'] + ': ' + str(round(VA_gamma_probability,3)),
                                alpha=histogram_alpha, color=gamma_colour)
        gamma_cdf = gamma.cdf(x_gam_IETs, a=y_gam_IETs, loc=loc_gam_IETs, scale=mu_gam_IETs)
        axs[CDF_plot].step(x_gam_IETs, gamma_cdf, label= method_dict['VA_method'],linewidth=linewidth, where='post',
                    color=gamma_colour, alpha=histogram_alpha)
    if len(sliding_window_counts) > 0:
        # axs[model_plot].hist(sliding_window_counts, bins=range(math.floor(min(sliding_window_counts))-1, math.ceil(max(sliding_window_counts))+2), color=ESR_colour,
        #         density=True, rwidth=1.0, alpha=histogram_alpha, align=align, label=method_dict['ESR'] + ': ' + str(round(sliding_window_probability,3)))
        min_x, max_x = min(sliding_window_counts), max(sliding_window_counts)
        ESR_counts, ESR_bins = np.histogram(sliding_window_counts, bins=range(min_x, max_x+3))
        axs[model_plot].step(ESR_bins[:-1], ESR_counts/ESR_counts.sum(), color=ESR_colour, linewidth=linewidth, alpha=histogram_alpha, where='post',
                             label=f"{method_dict['ESR']} :{round(sliding_window_probability,3)}")
        # window_counts_pdf = sliding_window_counts/sum(sliding_window_counts)
        window_counts_pdf = ESR_counts/sum(ESR_counts)
        window_counts_cdf = np.cumsum(window_counts_pdf)
        # window_counts_sorted = np.sort(sliding_window_counts)
        window_counts_sorted = np.sort(ESR_counts)

        axs[CDF_plot].step(ESR_bins[:-1], window_counts_cdf, where='post', label=method_dict['ESR'], color=ESR_colour, alpha=histogram_alpha, linewidth=linewidth)
        # axs[CDF_plot].step(window_counts_sorted, window_counts_cdf, where='post')

    for ax in [model_plot, CDF_plot]:
        axs[ax].axvline(x=results_dict['BP_99CI'], color=poisson_colour, alpha=0.5, linewidth=linewidth+1, linestyle='--')
        axs[ax].axvline(x=results_dict['G-IET_99CI'], color=gamma_colour, alpha=0.5, linewidth=linewidth+1, linestyle='--')
        axs[ax].axvline(x=results_dict['ESR_99CI'], color=ESR_colour, alpha=0.5, linewidth=linewidth+1, linestyle='--')

    # axs[model_plot].legend(fontsize=20, loc='upper center', bbox_to_anchor=(1.1, 1), ncols=1)
    axs[model_plot].set_ylim(axs[model_plot].get_ylim()[0], axs[model_plot].get_ylim()[1]+0.025)

#         handles, labels = plt.gca().get_legend_handles_labels()       #specify order of items in legend
#         order = range(0,len(handles))
#         order = [0,1,5,2,4,3]
#         plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    axs[CDF_plot].set_title(panel_labels[CDF_plot], fontsize=20, loc='left')
    axs[CDF_plot].set_xlabel('20-day Count')
    axs[CDF_plot].set_ylabel('CDF')

    fig.tight_layout()

    if save==True:
        if Mc_cut==False:
            Path(f"../outputs/{catalogue_name}/model_plots").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"../outputs/{catalogue_name}/model_plots/{mainshock.ID}.png")
        elif Mc_cut==True:
            Path(f"../outputs/{catalogue_name}/Mc_cut/model_plots").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"../outputs/{catalogue_name}/Mc_cut/model_plots/{mainshock.ID}.png")
    plt.show()

def process_mainshocks(mainshocks_file, earthquake_catalogue, catalogue_name, Mc_cut, save, save_name='default_params'):
    date = str(dt.datetime.now().date().strftime("%y%m%d"))

    results_list = []
    i = 1
    for mainshock in mainshocks_file.itertuples():
        print(f"{catalogue_name}")
        print(f"{i} of {len(mainshocks_file)} mainshocks")
        print(f"ID: {mainshock.ID}")
        local_cat = create_local_catalogue(mainshock, earthquake_catalogue, catalogue_name=catalogue_name, save=save)
        # try:
        #     local_cat = load_local_catalogue(mainshock, catalogue_name=catalogue_name)
        # except:
        #     local_cat = create_local_catalogue(mainshock, earthquake_catalogue, catalogue_name=catalogue_name, save=save)
        if Mc_cut==True:
            # local_cat = mc.apply_Mc_cut(local_cat)
            local_cat = local_cat.loc[local_cat['MAGNITUDE']>=mainshock.Mc].copy()
        # create_spatial_plot(mainshock=mainshock, local_cat=local_cat, Mc_cut=Mc_cut, catalogue_name=catalogue_name, save=save)
        cartmaps.plot_local_cat(mainshock=mainshock, local_cat=local_cat, earthquake_catalogue=earthquake_catalogue, catalogue_name=catalogue_name, Mc_cut=Mc_cut)
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
                results_df.to_csv(f'../data/{catalogue_name}/foreshocks/{save_name}_{date}.csv', index=False)
            if Mc_cut==True:
                Path(f'../data/{catalogue_name}/Mc_cut/foreshocks/').mkdir(parents=True, exist_ok=True)
                results_df.to_csv(f'../data/{catalogue_name}/Mc_cut/foreshocks/{save_name}_{date}.csv', index=False)
    return results_df

def ESR_model(mainshock, earthquake_catalogue, local_catalogue,
              local_catalogue_radius = 10, foreshock_window = 20, modelling_time_period=365):
    
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

    sliding_window_points = np.array(np.arange((-cut_off_day+foreshock_window)*range_scaler, -foreshock_window*range_scaler+1, 1))/range_scaler*-1
    sliding_window_counts = np.array([len(regular_seismicity_period.loc[(regular_seismicity_period['DAYS_TO_MAINSHOCK'] > point) &\
                                                                    (regular_seismicity_period['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]) for point in sliding_window_points])
    average = np.mean
    sliding_window_distances = np.array([average(regular_seismicity_period.loc[(regular_seismicity_period['DAYS_TO_MAINSHOCK'] > point) &\
                                                                    (regular_seismicity_period['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window)), 'DISTANCE_TO_MAINSHOCK']) for point in sliding_window_points])

    ESR_median = np.median(sliding_window_counts)
    variance = np.var(sliding_window_counts)
    q25 = np.percentile(sliding_window_counts, 25)
    q75 = np.percentile(sliding_window_counts, 75)
    
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
                    'ESR_median':ESR_median,
                    'ESD':distance_probability,
                    'cut_off_day':cut_off_day,
                    'var':variance,
                    'q25':q25,
                    'q75':q75
                    }
    
    sliding_window_points_full = np.array(np.arange((-cut_off_day+foreshock_window)*range_scaler, 1, 1))/range_scaler*-1
    sliding_window_counts_full = np.array([len(local_catalogue.loc[(local_catalogue['DAYS_TO_MAINSHOCK'] > point) &\
                                                                    (local_catalogue['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window))]) for point in sliding_window_points_full])
    average = np.mean
    sliding_window_distances_full = np.array([average(local_catalogue.loc[(local_catalogue['DAYS_TO_MAINSHOCK'] > point) &\
                                                                    (local_catalogue['DAYS_TO_MAINSHOCK'] <= (point + foreshock_window)), 'DISTANCE_TO_MAINSHOCK']) for point in sliding_window_points_full])


    sliding_window_counts_df = pd.DataFrame({'points':sliding_window_points_full,
                                             'counts':sliding_window_counts_full,
                                             'distances':sliding_window_distances_full})
    
    file_dict = {'local_catalogue':local_catalogue,
                #  'local_catalogue_pre_Mc_cutoff':local_catalogue_pre_Mc_cutoff,
                #  'local_catalogue_below_Mc':local_catalogue_below_Mc,
                 'foreshocks':foreshocks,
                #  'foreshocks_below_Mc':foreshocks_below_Mc,
                 'sliding_window_points':sliding_window_points_full,
                 'sliding_window_counts':sliding_window_counts_full,
                 'sliding_window_distances':sliding_window_distances_full
                 }
    
    return results_dict, file_dict, sliding_window_counts_df

def run_ESR_for_mainshock_file(mainshock_file, earthquake_catalog, input_name, mcut=True):
    data_dict = {}
    count=0
    results_dict_list = []
    for mainshock in tqdm(mainshock_file.itertuples(), total=len(mainshock_file)):
        count+=1
        print(f"{mainshock.ID} - {count} of {len(mainshock_file)}")
        local_cat = create_local_catalogue(mainshock=mainshock, earthquake_catalogue=earthquake_catalog, catalogue_name=input_name, save=False)
        if mcut==True:
            local_cat = local_cat.loc[local_cat['MAGNITUDE']>=mainshock.Mc].copy()
        results_dict, file_dict, window_df = ESR_model(mainshock=mainshock, earthquake_catalogue=earthquake_catalog, local_catalogue=local_cat)
        data_dict.update({mainshock.ID:{'results_dict':results_dict,
                                        'file_dict':file_dict,
                                        'window_df':window_df}})
        results_dict_list.append(results_dict)
        if mcut==True:
            path = f'../data/{input_name}/Mc_cut/ESR'
        else:
            path = f'../data/{input_name}/no_Mc_cut/ESR'
        Path(path).mkdir(exist_ok=True, parents=True)
        window_df.to_csv(path + f'/{mainshock.ID}.csv', index=False)
        clear_output(wait=True)

    ESR_results = pd.DataFrame.from_dict(results_dict_list)
    if mcut==True:
        save_name = f'{input_name}_mcut_{date}.csv'
    else:
        save_name = f'{input_name}_no_mcut_{date}.csv'
    ESR_results.to_csv(f'../p2_outputs/ESR_results/{save_name}', index=False)
    return ESR_results