"""
This sub-module contains functions to estimate the mc and b-values of an earthquake catalog, and plot the fmd.
Many functions require the renaming of earthquake catalog dataframe columns to: ID, MAGNITUDE, DATETIME, LON, LAT, DEPTH.
"""

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

# use if loading the package locally (comment out when uploading release)
import utils
import statseis
import mc_lilliefors

# uncomment when uploading release
# import statseis.utils as utils
# import statseis.statseis as statseis
# import statseis.mc_lilliefors as mc_lilliefors

plot_colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
plot_color_dict = dict(zip(['teal', 'orange', 'purple', 'pink', 'green', 'yellow', 'brown', 'grey'], plot_colors))

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
    Written by Sacah Lapins. This code calculates b values by maximum likelihood estimate. It takes in an array of magnitude (mag), a
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
    return a, b, aki_unc, shibolt_unc # Return b-value and estimates of uncertainty

def Mc_by_maximum_curvature(mag, mbin=0.1, correction=0.2):
    """
    Written by Sacha Lapins. This code returns the magnitude of completeness estimates using the maximum curvature method. It takes a magnitude
    array (mag) and binning (mbin). [Wiemer & Wyss (2000), Lapins, CORSSA] - modified
    """
    mag = np.array(mag)
    this_fmd = freq_mag_dist(mag, mbin) # uses the fmd distribution (a previous function)
    maxc = this_fmd[0][np.argmax(this_fmd[1])] # Mag bin with highest no. of events
    print(f"with a correction of: {correction}")
    return maxc + correction 
 
def Mc_by_goodness_of_fit(mag, mbin=0.1):
    """
    Written by Sacha Lapins. This code returns the magnitude of completeness estimates using a goodness of fit method. It takes a magnitude
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
            mc = np.nan
            print("No fits within confidence levels")
            break
        else:
            if len(GFT_test[i][0]) > 0:
                mc = round(this_fmd[0][GFT_test[i][0][0]], 1) # Use first cut-off magnitude within confidence level
                break
#     return mc, this_fmd[0], R
    return mc
 
def Mc_by_b_value_stability(mag, mbin=0.1, dM = 0.4, min_mc = -3, return_b=False):
    """
    Written by Sacha Lapins. This code returns the magnitude of completeness estimates using a b value stability method. It takes a magnitude
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
        mc = np.nan 
#     return mc, this_fmd[0], b, b_avg, shibolt_unc
    return mc, b#, this_fmd[0], b, b_avg, shibolt_unc

def fmd(mag, mbin):
    """
    Written by Sacha Lapins.
    """
    minmag = math.floor(min(mag/mbin)) * mbin # Lowest bin
    maxmag = math.ceil(max(mag/mbin)) * mbin # Highest bin bin
    mi = np.arange(minmag, maxmag + mbin, mbin) # Make array of bins
    nbm = len(mi)
    cumnbmag = np.zeros(nbm) # Array for cumulative no. of events
    for i in range(nbm): # cumulative no. of events
        cumnbmag[i] = np.where(mag > mi[i] - mbin/2)[0].shape[0]
    nbmag = abs(np.diff(np.append(cumnbmag, 0))) # no. of events
    return mi, nbmag, cumnbmag
print('FMD Function Loaded')

def b_est(mag, mbin, mc):
    """
    Written by Sacha Lapins.
    """
    mag_above_mc = mag[np.where(mag > round(mc,1)-mbin/2)[0]]# Magnitudes for events larger than cut-off magnitude mc
    n = mag_above_mc.shape[0] # No of. events larger than cut-off magnitude mc
    mbar = np.mean(mag_above_mc) # Mean magnitude for events larger than cut-off magnitude mc
    b = math.log10(math.exp(1)) / (mbar - (mc - mbin/2)) # b-value from Eq 2
    a = math.log10(n) + b * mc # 'a-value' for Eq 1
    aki_unc = b / math.sqrt(n) # Uncertainty estimate from Eq 3
    shibolt_unc = 2.3 * b**2 * math.sqrt(sum((mag_above_mc - mbar)**2) / (n * (n-1))) # Uncertainty estimate from Eq 4
    return a, b, aki_unc, shibolt_unc # Return b-value and estimates of uncertainty
print('MLM B Function Loaded')

def get_maxc(mag, mbin):
    """
    Written by Sacha Lapins.
    """
    this_fmd = fmd(mag, mbin) # uses the fmd distribution (a previous function)
    maxc = this_fmd[0][np.argmax(this_fmd[1])] # Mag bin with highest no. of events
    return round(maxc, 1)
print('MAXC Function Loaded')

def get_mbs(mag, mbin, dM = 0.4, min_mc = -3):
    """
    Written by Sacha Lapins.
    """
    this_fmd = fmd(mag, mbin) # FMD
    this_maxc = get_maxc(mag, mbin) # Needed further down
    # Zeros to accommodate synthetic GR distributions for each magnitude bin
    a = np.zeros(this_fmd[0].shape[0]) # Pre-allocate array to accommodate a values from Eq 1
    b = np.zeros(this_fmd[0].shape[0]) # Pre-allocate array to accommodate b values from Eq 1 & 2
    b_avg = np.zeros(this_fmd[0].shape[0]) # Pre-allocate array to accommodate b values from Eq 1 & 2
    shibolt_unc = np.zeros(this_fmd[0].shape[0]) # Pre-allocate array to accommodate uncertainty values from Eq 4
    for i in range(this_fmd[0].shape[0]): # Loop through each magnitude bin, using it as cut-off magnitude
        mi = round(this_fmd[0][i], 1) # Cut-off magnitude
        if this_fmd[2][i] > 1:
            a[i], b[i], tmp1, shibolt_unc[i] = b_est(mag, mbin, mi) # a and b-values for this cut-off magnitude
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
        # mc = this_maxc # If no stability point, use MAXC
        mc = np.nan # my addition
    return mc, this_fmd[0], b, b_avg, shibolt_unc
print('MBS Funtion Loaded')

def get_Mcs_400(mainshocks_file, earthquake_catalogue, catalogue_name, start_radius=10, inc=5, max_r=50, min_n=400):
    """
    Calculate Mc using b-value stability (Mbass) and maximumum curvature (Maxc) around mainshock epicenters.
    """
    date = str(dt.datetime.now().date().strftime("%y%m%d"))
    Mbass = []
    Maxc = []
    n_local_cat = []
    radii = []
    Mbass_b = []
    Maxc_b = []
    Gft_Mc = []
    i = 1
    for mainshock in tqdm(mainshocks_file.itertuples(), total=len(mainshocks_file)):
        # print(f"{catalogue_name}")
        # print(f"{i} of {len(mainshocks_file)} mainshocks")
        radius = start_radius
        local_cat = statseis.create_local_catalogue(mainshock, earthquake_catalogue, catalogue_name=catalogue_name, save=False, radius_km=radius)
        while len(local_cat)<min_n:
            if radius <max_r:
                radius+=inc
                local_cat = statseis.create_local_catalogue(mainshock, earthquake_catalogue, catalogue_name=catalogue_name, save=False, radius_km=radius)
            elif radius>=max_r:
                break
        try:
            Mbass_mc = get_mbs(np.array(local_cat['MAGNITUDE']), mbin=0.1)[0]
            Maxc_mc = get_maxc(local_cat['MAGNITUDE'], mbin=0.1)+0.2
            Mbass_b_val = b_est(np.array(local_cat['MAGNITUDE']), mbin=0.1, mc=Mbass_mc)[1]
            Maxc_b_val = b_est(np.array(local_cat['MAGNITUDE']), mbin=0.1, mc=Maxc_mc)[1]

        except:
            Mbass_mc, Maxc_mc, Mbass_b_val, Maxc_b_val = [np.nan]*4
        
        Mbass.append(Mbass_mc)
        Maxc.append(Maxc_mc)
        n_local_cat.append(len(local_cat))
        radii.append(radius)
        Mbass_b.append(Mbass_b_val)
        Maxc_b.append(Maxc_b_val)

        i+=1
        clear_output(wait=True)
    mainshocks_file[f'Mbass_50'] = Mbass
    mainshocks_file[f'b_Mbass_50'] = Mbass_b
    mainshocks_file[f'Mc'] = Mbass
    mainshocks_file[f'Maxc_50'] = Maxc
    mainshocks_file[f'b_Maxc_50'] = Maxc_b
    mainshocks_file[f'n_for_Mc_50'] = n_local_cat
    mainshocks_file[f'radii_50'] = radii
    # mainshocks_file[f'Mbass_{max_r}'] = Mbass
    # mainshocks_file[f'b_Mbass_{max_r}'] = Mbass_b
    # mainshocks_file[f'Mc'] = Mbass
    # mainshocks_file[f'Maxc_{max_r}'] = Maxc
    # mainshocks_file[f'b_Maxc_{max_r}'] = Maxc_b
    # mainshocks_file[f'n_for_Mc_{max_r}'] = n_local_cat
    # mainshocks_file[f'radii_{max_r}'] = radii
    return mainshocks_file

def Mc_by_Lilliefors(mag, n_repeats=50, Mstart=0.0):
    lill = mc_lilliefors.McLilliefors(mag)
    lill.calc_testdistr_mcutoff(
        n_repeats=n_repeats,  # number of iterations for the random noise
        Mstart=Mstart,  # lowest magnitude for which to perform the test
        # log=False,  # whether to show anythe progress bar
    )
    Mc = lill.estimate_Mc_expon_test()
    return Mc

def get_Mcs_ensemble(mainshocks_file, earthquake_catalogue, catalogue_name,
                      start_radius=10, inc=5, max_r=50, min_n=400, Lilliefors=False):
    """
    Calculate Mc using b-value stability (Mbass) and maximumum curvature (Maxc) around mainshock epicenters.
    """
    date = str(dt.datetime.now().date().strftime("%y%m%d"))
    Mbass = []
    Maxc = []
    n_local_cat = []
    radii = []
    Mbass_b = []
    Maxc_b = []
    Gft_Mc = []
    Lilliefors_Mc_list = []
    Lilliefors_Mc_b = []
    i = 1
    for mainshock in tqdm(mainshocks_file.itertuples(), total=len(mainshocks_file)):
        # print(f"{catalogue_name}")
        # print(f"{i} of {len(mainshocks_file)} mainshocks")
        radius = start_radius
        local_cat = statseis.create_local_catalogue(mainshock, earthquake_catalogue, catalogue_name=catalogue_name, save=False, radius_km=radius)
        while len(local_cat)<min_n:
            if radius <max_r:
                radius+=inc
                local_cat = statseis.create_local_catalogue(mainshock, earthquake_catalogue, catalogue_name=catalogue_name, save=False, radius_km=radius)
            elif radius>=max_r:
                break
        if Lilliefors==True:
            McLil = Mc_by_Lilliefors(local_cat['MAGNITUDE'])
            Lilliefors_Mc_list.append(McLil)
            Lilliefors_Mc_b.append(b_est(np.array(local_cat['MAGNITUDE']), mbin=0.1, mc=McLil)[1])
        Mbass_mc = get_mbs(np.array(local_cat['MAGNITUDE']), mbin=0.1)[0]
        Mbass.append(Mbass_mc)
        Maxc_mc = get_maxc(local_cat['MAGNITUDE'], mbin=0.1)+0.2
        Maxc.append(Maxc_mc)
        n_local_cat.append(len(local_cat))
        radii.append(radius)
        Mbass_b.append(b_est(np.array(local_cat['MAGNITUDE']), mbin=0.1, mc=Mbass_mc)[1])
        Maxc_b.append(b_est(np.array(local_cat['MAGNITUDE']), mbin=0.1, mc=Maxc_mc)[1])
        i+=1
        clear_output(wait=True)
    if Lilliefors==True:
        mainshocks_file[f'Mc_lil'] = Lilliefors_Mc_list
        mainshocks_file[f'b_lil'] = Lilliefors_Mc_b
    mainshocks_file[f'Mc_mbs'] = Mbass
    mainshocks_file[f'b_mbs'] = Mbass_b
    # mainshocks_file[f'Mc'] = Mbass
    mainshocks_file[f'Mc_maxc'] = Maxc
    mainshocks_file[f'b_maxc'] = Maxc_b
    mainshocks_file[f'n_for_Mc'] = n_local_cat
    mainshocks_file[f'radii'] = radii
    # mainshocks_file[f'Mbass_{max_r}'] = Mbass
    # mainshocks_file[f'b_Mbass_{max_r}'] = Mbass_b
    # mainshocks_file[f'Mc'] = Mbass
    # mainshocks_file[f'Maxc_{max_r}'] = Maxc
    # mainshocks_file[f'b_Maxc_{max_r}'] = Maxc_b
    # mainshocks_file[f'n_for_Mc_{max_r}'] = n_local_cat
    # mainshocks_file[f'radii_{max_r}'] = radii
    return mainshocks_file

def get_Mc_expanding_r(mainshocks_file, earthquake_catalogue, catalogue_name, start_radius=10, inc=5, max_r=100, min_n=1000):
    """
    Calculate Mc using b-value stability (Mbass) and maximumum curvature (Maxc) around mainshock epicenters
    at each point for an expanding radius.
    """
    
    mainshock_results = []
    for mainshock in tqdm(mainshocks_file.itertuples(), total=len(mainshocks_file)):
        results_list = []
        for radius in np.arange(start_radius,max_r,inc):
            if radius==0:
                radius+=1

            local_cat = statseis.create_local_catalogue(mainshock, earthquake_catalogue, catalogue_name=catalogue_name, save=False, radius_km=radius)
            # print(radius, len(local_cat))
            try:
                Mbass_mc = get_mbs(np.array(local_cat['MAGNITUDE']), mbin=0.1)[0]
                Maxc_mc = get_maxc(np.array(local_cat['MAGNITUDE']), mbin=0.1)+0.2

                results_dict = {'radii':radius,
                                'n_local_cat':len(local_cat),
                                'Mbass':Mbass_mc,
                                'Maxc':Maxc_mc,
                                'b_Mbass':b_est(np.array(local_cat['MAGNITUDE']), mbin=0.1, mc=Mbass_mc)[1],
                                'b_Maxc':b_est(np.array(local_cat['MAGNITUDE']), mbin=0.1, mc=Maxc_mc)[1]}
                
            except:
                results_dict = {'radii':radius,
                                'n_local_cat':len(local_cat),
                                'Mbass':np.nan,
                                'Maxc':np.nan,
                                'b_Mbass':np.nan,
                                'b_Maxc':np.nan}

            results_list.append(results_dict)

        mainshock_results.append({'ID':mainshock.ID, 'df':pd.DataFrame.from_dict(results_list)})
        clear_output(wait=True)

    return mainshock_results

def apply_Mc_cut(earthquake_catalogue):
    mag = np.array(earthquake_catalogue['MAGNITUDE'])
    Mc = get_mbs(mag, mbin=0.1)[0]
    earthquake_catalogue = earthquake_catalogue.loc[earthquake_catalogue['MAGNITUDE']>= Mc].copy()
    return earthquake_catalogue

def plot_fmd(local_cat, save_path=None, ID=np.nan, radius=50):

    local_cat = local_cat.loc[local_cat['DISTANCE_TO_MAINSHOCK']<radius].copy()
    print(len(local_cat))
    magnitudes = np.array(local_cat['MAGNITUDE'])
    bins = np.arange(math.floor(magnitudes.min()), math.ceil(magnitudes.max()), 0.1)
    values, base = np.histogram(magnitudes, bins=bins)
    cumulative = np.cumsum(values)
    Mc, this_fmd, b, b_avg, shibolt_unc = get_mbs(mag=magnitudes, mbin=0.1)
    a, b_value, _a, _b = b_est(mag=magnitudes, mbin=0.1, mc=Mc)
    N = [10**(a-b_value*M) for M in this_fmd]
    ratio_above_Mc = round(100*len(magnitudes[magnitudes>Mc])/len(magnitudes))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot([], [], label=f"ID: {ID}", marker=None, linestyle='')
    ax.plot([], [], label=f'{ratio_above_Mc}% above $M_c$', marker=None, linestyle='')
    ax.plot([], [], label=f'N: {len(magnitudes)}', marker=None, linestyle='')
    ax.plot([], [], label=f'Radius: {radius}', marker=None, linestyle='')
    ax.step(base[:-1], len(magnitudes)-cumulative, color='black')
    ax.axvline(x=Mc, linestyle='--', label=r'$M_{c}$: ' + str(round(Mc,1)), color=plot_colors[0])
    ax.plot(this_fmd, N, label=f'b: {round(b_value,2)}',  color=plot_colors[1])

    ax.set_xlabel('Magnitude')
    ax.set_ylabel('N')
    ax.set_yscale('log')
    ax.legend()
    if save_path!=None:
        plt.savefig(save_path)

def plot_FMD_mainshock_subset(mshock_file, name, outfile_name, catalog, stations=None):
    Path(f'../outputs/{outfile_name}/FMD').mkdir(parents=True, exist_ok=True)
    for mainshock in tqdm(mshock_file.itertuples(), total=len(mshock_file)):
        local_cat = statseis.create_local_catalogue(mainshock, earthquake_catalogue=catalog, catalogue_name=name, radius_km=100)
        statseis.plot_local_cat(mainshock=mainshock, local_cat=local_cat, catalogue_name=name, Mc_cut=False, stations=stations, earthquake_catalogue=catalog,
                    min_days=math.ceil(local_cat['DAYS_TO_MAINSHOCK'].max()), max_days=0,
                    radius_km=mainshock.radii_50, box_halfwidth_km=100, aftershock_days=math.floor(local_cat['DAYS_TO_MAINSHOCK'].min()))
        print(mainshock.n_for_Mc_50)
        plot_fmd(local_cat, save_path=f'../outputs/{outfile_name}/FMD/{mainshock.ID}.png', ID=mainshock.ID, radius=mainshock.radii_50)
        plt.close()
        clear_output(wait=True)