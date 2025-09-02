#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 18:59:38 2025

@author: fiendedoncker
"""

# 1. Imports, pathnames and constants
# 1.1. Imports
import numpy as np

import rasterio
from rasterio.plot import show

import scipy.io as sio
from scipy.special import logsumexp
from scipy.stats import kruskal
from scipy.spatial.distance import jensenshannon

from sklearn.decomposition import PCA

import math

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar

import seaborn as sns

import geopandas as gpd
import pandas as pd

from datetime import timedelta
from datetime import datetime

import re

from collections import Counter
from collections import defaultdict

from tqdm import tqdm
from itertools import combinations

import os

import pickle

import textwrap


# 1.2. Pathnames
INPUT_FOLDER = '/Users/fiendedoncker/Documents/GitHub/Non-Linear-Inversion/Testing Database/INPUT'  # Path to the input data folder
OUTPUT_FOLDER = '/Users/fiendedoncker/Documents/GitHub/Non-Linear-Inversion/Testing Database/OUTPUT' 

SHAPEFILE_PATH = f'{INPUT_FOLDER}/Gornergletscher_2018.shp'

hillshade_path = f'{INPUT_FOLDER}/hillshade_clipped.tif'
glacier_shapefile = f'{INPUT_FOLDER}/Gornergletscher_2018.shp'
geology_shapefile = f'{INPUT_FOLDER}/Geology.gpkg'
watershed_shapefile = f'{INPUT_FOLDER}/Watershed.gpkg'

DETRITAL_PATH = f'{INPUT_FOLDER}/XRD_data/DetritalData/detrital_cleaned.pkl'
LOAD_PATH = f'{INPUT_FOLDER}/XRD_data/DetritalData/SampleName_Load_AreaXRD.csv'
SOURCE_PATH = f'{INPUT_FOLDER}/XRD_data/SourceData/source_cleaned.pkl'


# 1.3. Flags and Settings
savename = 'XRD '
savefolder = 'Zircon tests'

dates_of_interest = ['20190605','20190628'] # dates for 

age_labels = ['40-140', '140-200', '200-210', '210-250', '250-263', '263-270', 
              '270-340', '340-430', '430-475', '475-555', '555-585', '585-750', 
              '750-800 Ma']

lithology_colors = {
    "ZSF ophiolites (serpentinites)": "#e3dd19",
    "ZSF sediments": "#809847",
    "Stockhorn, Tuftgrat, Gornergrat": "#ff5001",
    "Monte Rosa (granite)": "#fa9b9a",
    "ZSF ophiolites (metabasites, eclogites)": "#32a02d",
    "Monte Rosa (gneiss, micaschist)": "#ff7f41",
    "Furgg series": "#badd68"
}

geo_names = {1: 'ZSF ophiolites (serpentinites)',
 2: 'ZSF sediments',
 3: 'Stockhorn, Tuftgrat, Gornergrat',
 4: 'Monte Rosa (granite)',
 5: 'ZSF ophiolites (metabasites, eclogites)',
 6: 'Monte Rosa (gneiss, micaschist)',
 7: 'Furgg series'}

n_minerals_to_keep = None
flag_uniformprior = 1 # if == 1: etot uniformally distributed, else: eprior ~ Kg*us^l
flag_firstprior = 1 # if == 1: start with eprior, else: start from linear closed form solution
flag_sigma_d_meas = 0 # if == 1: use observed sigma, else: use parameter d sigma
flag_quasi_newton = 1 # if == 1: quasi newton, else: steepest descent
flag_test_optimal_source = 1  # if == 1: test with very distinct synthetic sediment sources
flag_test_parameters = 1 # if == 1: test values for different par.s
flag_test_scenarios = 1 # if == 1: test with synthetic source data
flag_test_SD = 0 # if == 1: test with steepest descent
flag_use_zircon_data = 1
flag_combine_data = 1

if flag_use_zircon_data == 1:
    savename = 'Zr '

if flag_combine_data == 1:
    savename = 'Combined XRD-Zr '
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2. Helper functions
def v_to_xy(map_masked_v, mask_vector, ny, nx, zero_val):
    # Ensure the output array has float dtype and is initialized to zero_val
    map_v = np.full(mask_vector.shape, zero_val, dtype=float)

    # Fill in values at masked (==1) positions
    map_v[mask_vector == 1] = map_masked_v
    map_v[mask_vector == 0] = np.nan
    
    return map_v.reshape((ny, nx))

def create_output_path(base_output_folder, savename):
    full_output_path = os.path.join(base_output_folder, savename)
    os.makedirs(full_output_path, exist_ok=True)
    return full_output_path

def get_js_dist(source,n_lithologies):
    dists = []
    for i, j in combinations(range(n_lithologies), 2):
        p = source[:, i]
        q = source[:, j]
        # Normalize in case columns don’t sum to 1
        p = p / p.sum()
        q = q / q.sum()
        d = jensenshannon(p, q)
        dists.append(d)
    
    avg_js_dist = np.round(np.mean(dists),3)
    return avg_js_dist
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 3. Loading data functions
def load_geology(filename):
    with rasterio.open(f'{INPUT_FOLDER}/{filename}.tif') as src:
        B = src.read(1)
        R = src.transform
        crs_raster = src.crs
    return B, R, crs_raster

def load_mask_GG():
    with rasterio.open(f'{INPUT_FOLDER}/mask_GGsamples.tif') as src:
        mask_GG = src.read(1)
    return mask_GG

def load_dem_and_velocity():
    with rasterio.open(f'{INPUT_FOLDER}/DEM_lowres.tif') as src:
        DEM = src.read(1)
    with rasterio.open(f'{INPUT_FOLDER}/USurf_lowres.tif') as src:
        vel_obs = src.read(1)
    return DEM, vel_obs

def load_source_data(n_minerals_to_keep=None):
    # source_data = sio.loadmat(f'{INPUT_FOLDER}/XRD data/source.mat')
    # source = source_data['source']
    
    with open(SOURCE_PATH, 'rb') as f:
        source_data = pickle.load(f)

    #source = source_data['source_rel_sample']
    source = source_data['source']
    
    source[source == 0] = 1e-5
    source[source == 1] = 0.9999
    

    # Perform PCA-based reduction if requested
    if n_minerals_to_keep is not None:
        pca = PCA()
        pca.fit(source.T)
        importance = np.abs(pca.components_[0])
        top_k_idx = np.argsort(importance)[::-1][:n_minerals_to_keep]
        source = source[top_k_idx, :]  # reduce minerals
        return source, top_k_idx
    else:
        return source, np.arange(source.shape[0])
    
def estimate_sigma_from_replicates(data_percentage, names, datenames):
    tracer_values_by_date = defaultdict(list)

    for i, date in enumerate(datenames):
        tracer_values_by_date[date].append(data_percentage[:, i])  # shape (n_tracers,)

    all_std = []
    for tracer_idx in range(data_percentage.shape[0]):  # for each mineral
        tracer_values = []
        for date, replicate_columns in tracer_values_by_date.items():
            replicates = np.array(replicate_columns)  # shape (n_replicates, n_tracers)
            if replicates.shape[0] > 1:  # only if more than 1 replicate
                tracer_values.append(np.std(replicates[:, tracer_idx], ddof=1))
        if len(tracer_values) > 0:
            all_std.append(np.mean(tracer_values))  # average std across dates
        else:
            all_std.append(1e-2)  # fallback value if no replicates found
    return np.array(all_std)  # shape (n_tracers,)

def load_detrital_data_from_pickle(DETRITAL_PATH, LOAD_PATH, selected_mineral_indices=None):
    with open(DETRITAL_PATH, 'rb') as f:
        data = pickle.load(f)

    datenames = data['dates'].to_list()
    names = [re.sub(r'_pf_.*$', '', name, flags=re.IGNORECASE) for name in data['names']]
    data_percentage = data['data_percentage']
    #data_percentage = data['data_percentage_per_sample']
    
    # Optional: estimate data uncertainty
    data_sigma = estimate_sigma_from_replicates(data_percentage, names, datenames)

    # I now merge the data from different filters for the same dates
    # for this, I use a weighed average of the signal, with weight ~ sample load
    df_weights = pd.read_csv(LOAD_PATH)
    weights_dict = dict(zip(df_weights['Name'], df_weights['Load']))
    
    date_to_indices = defaultdict(list)
    for i, name in enumerate(names):
        date = datenames[i]
        weight = weights_dict.get(name, None)
        if weight is not None:
            date_to_indices[date].append((i, weight))
        else:
            print(f"Warning: No weight found for sample '{name}' – skipping.")
    
    # Merge data_percentage by date with weights
    merged_data = []
    merged_dates = []

    for date, idx_weights in date_to_indices.items():
        indices, weights = zip(*idx_weights)
        weights = np.array(weights)
        weights = weights / weights.sum() # normalize weights

        weighted_sum = np.sum(data_percentage[:, indices] * weights, axis=1)
        merged_data.append(weighted_sum)
        merged_dates.append(date)

    merged_data_percentage = np.array(merged_data).T # shape: (n_minerals, n_dates)
    
    # Reduce minerals if needed
    if selected_mineral_indices is not None:
        merged_data_percentage = merged_data_percentage[selected_mineral_indices, :]
        data_sigma = data_sigma[selected_mineral_indices]

    return merged_dates, merged_data_percentage, data_sigma

def load_detrital_data_Bruno(loadname):
    loadpath = f'{INPUT_FOLDER}/Zircon_Data_to_Generate_Fingerprints/New_Fingerprints'
    data = sio.loadmat(f'{loadpath}/DataSamples_{loadname}.mat')
    data_percentage = data['data_percentage']
    data_sigma = data['data_sigma']
    data_labels = ['GS01R1', 'GS09R1', 'GS20', 
                   'GW21', 'GW22', 'GW23', 'GW25', 
                   'GW27','GW29']
    datenames = ['30-Sep-2018 23:59', '30-Oct-2018 23:59', '30-Sep-2019 23:59',
                 '28-Jun-2019 12:30', '28-Jun-2019 10:30', '28-Jun-2019 13:30', 
                 '28-Jun-2019 14:30', '05-Jun-2019 13:30', '05-Jun-2019 15:00']
    
    gg_df = pd.DataFrame(['30-Oct-2018 23:59'],columns=['date'])
    
    return datenames, data_percentage, data_sigma, gg_df

def load_source_data_Bruno(n_minerals_to_keep=None):
    loadpath = f'{INPUT_FOLDER}/Zircon_Data_to_Generate_Fingerprints/New_Fingerprints'
    source_data = sio.loadmat(f'{loadpath}/source_B.mat')
    source = source_data['source_percentage']
    source[source == 0] = 1e-5
    source[source == 1] = 0.9999

    # Perform PCA-based reduction if requested
    if n_minerals_to_keep is not None:
        pca = PCA()
        pca.fit(source)
        importance = np.abs(pca.components_[0])
        top_k_idx = np.argsort(importance)[::-1][:n_minerals_to_keep]
        source = source[top_k_idx, :]  # reduce minerals
        return source, top_k_idx
    else:
        return source, np.arange(source.shape[0])
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 4. Set up parameters, dimensions etc
def setup_model_params():
    sigma_cov = 3
    
    L = 1200 #m
    
    d_sigma_val = 0.1
    
    n_m = 15
    mu = 0.01
    #mu = 0.001 # use for Zr data otherwise it explosdes
    
    misfit = np.zeros(n_m)
    d_misfit = np.zeros(n_m)
    m_misfit = np.zeros(n_m)
    
    return sigma_cov, L, d_sigma_val, n_m, mu, misfit, d_misfit, m_misfit

def setup_grid_dimensions(B, R):
    ny, nx = B.shape
    nn = nx * ny
    dx = R[0]
    dy = -R[4]
    Ly = dy * (ny - 1)
    Lx = dx * (nx - 1)
    dA = dx * dy
    x2 = np.arange(0, Lx + dx, dx)
    y2 = np.arange(Ly, -dy, -dy)
    X2, Y2 = np.meshgrid(x2, y2)
    return ny, nx, nn, dx, dy, Lx, Ly, dA, X2, Y2

def setup_observed_data(B):
    mask = (B > 0).astype(int)
    return mask

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 5. Functions to prepare inversion input
def generate_A(B, mask, source, n_tracers):
    active_idx = np.where(mask.flatten() > 0)[0]
    geol_v = B.flatten()[active_idx]
    S = np.unique(geol_v)
    mask_size = len(geol_v)
    
    A_0 = np.zeros((n_tracers, mask_size))
    
    for tr in range(n_tracers):
        A_v = np.zeros(mask_size)
        for geol_i, code in enumerate(S):
            A_v[geol_v == code] = source[tr, geol_i]
        A_0[tr, :] = A_v
    
    return A_0

def generate_cov(x, y, sigma_cov, L, nugget=1e-2):
    mask_size = len(x)
    cov_0 = np.zeros((mask_size, mask_size))
    
    for i in range(mask_size):
        for j in range(i, mask_size):
            dist_ij = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            cov_ij = sigma_cov**2 * np.exp(-(dist_ij / L)**2)
            cov_0[i, j] = cov_ij
            cov_0[j, i] = cov_ij  # symmetry
    
    # Add regularization to the diagonal
    cov_0 += np.eye(mask_size) * (nugget * sigma_cov**2)
    
    return cov_0

def generate_cov_d(data_sigma, d_sigma_val, flag_sigma_d_meas, n_tracers):
    if flag_sigma_d_meas and data_sigma is not None:
        d_sigma = np.copy(data_sigma)
        d_sigma[d_sigma == 0] = 1e-5
    else:
        d_sigma = np.ones(n_tracers) * d_sigma_val
    
    cov_d = np.diag(d_sigma**2)
    
    return cov_d

def compute_etot(dx, dy, Q_s_kg_per_year, rho=2650, dA=1.0):
    dA = dx * dy
    V_s = Q_s_kg_per_year / rho  # m³/year
    etot = (V_s / dA) * 1e3      # mm/year
    return etot

def compute_eps_prior_uniform(etot, mask):
    mask_size = np.sum(mask > 0)
    edot_prior = np.ones(mask_size) * etot / mask_size  # mm/year per cell
    eps_prior = np.log(edot_prior)
    return eps_prior

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 6. Functions to loop over data for different dates
def prepare_data_for_iteration(data_percentage, data_i, etot):
    d = data_percentage[:, data_i] * etot # IMPORTANT: here, i multiply d times etot so i dont have to take care of etot in the computation of G
    d[np.isnan(d)] = 0
    d[d == 0] = 0.00001 # deal with zero bc log(d) 
    d[d == 1] = 0.99

    return d

def nonlinear_iteration(A, cov, cov_d, eps_prior, mu, n_m, etot, d, mask_size, flag_firstprior, flag_quasi_newton):
    misfit = np.zeros(n_m)
    d_misfit = np.zeros(n_m)
    m_misfit = np.zeros(n_m)
    
    eps_m = eps_prior
    
    # To store edot_post at each iteration
    edot_post_list = []
    Cm_post_list = []
    
    for m in tqdm(range(n_m)):
        if m == 0:
            if flag_firstprior == 1:
                edot_post = np.exp(eps_prior)
            else:
                H = cov @ A.T @ np.linalg.pinv((A @ cov @ A.T) + cov_d)
                edot_post = np.exp(eps_prior) + H @ (d - A @ np.exp(eps_prior))
                edot_post[edot_post <= 0] = 0.0001
                eps_prior = np.log(edot_post)
            
            # Compute epsilon initial
            eps_m = np.log(edot_post)
            
            # Compute d initial
            d_model = A @ np.exp(eps_m)
        
        
        # Compute G (jacobian) (how model-predicted output changes with respect to the parameters)
        #G = A * np.exp(eps_m)
        G = A @ np.diag(np.exp(eps_m))
        
        
        if flag_quasi_newton == 1:
            # Compute gradient
            residual = d_model - d
            grad_data = G.T @ np.linalg.solve(cov_d, residual)
            grad_prior = np.linalg.solve(cov, eps_m - eps_prior)
            total_grad = grad_data + grad_prior
            
            # Compute Hessian
            H = G.T @ np.linalg.solve(cov_d, G) + np.linalg.inv(cov)
            
            # Quasi Newton update
            delta_eps = -np.linalg.solve(H, total_grad)
            eps_m1 = eps_m + mu * delta_eps
            
        else: # Steepest descent
            residual = d_model - d
            grad_data = cov @ G.T @ np.linalg.solve(cov_d, residual)
            grad_prior = (eps_m - eps_prior)
            
            delta_eps = - (grad_data + grad_prior )
            delta_eps /= np.linalg.norm(delta_eps) # Normalize gradient
            eps_m1 = eps_m + mu * delta_eps

        # Transform to erosion rates
        edot_post = np.exp(eps_m1)
        edot_post_list.append(edot_post.copy())
        
        eps_m = eps_m1

        
        # Misfit calculation
        d_model = A @ np.exp(eps_m)
        diff_d = d_model - d
        diff_eps = eps_m - eps_prior

        misfit[m] = (
            diff_d.T @ np.linalg.pinv(cov_d) @ diff_d
            + diff_eps.T @ np.linalg.pinv(cov) @ diff_eps
        )
        m_misfit[m] = diff_eps.T @ np.linalg.pinv(cov) @ diff_eps
        d_misfit[m] = diff_d.T @ np.linalg.pinv(cov_d) @ diff_d
        

        if m > 0: 
            delta_misfit = np.abs(d_misfit[m] - d_misfit[m - 1])
            if delta_misfit < 1e-3:
                print(f"Stopping early at iteration {m} — data misfit change too small ({delta_misfit:.2e})")
                misfit = misfit[:m+1]
                edot_post_list = edot_post_list[:m+1]
                break
        
        # Posterior covariance
        G_Cm = G @ cov
        try:
            middle_term = np.linalg.inv(G_Cm @ G.T + cov_d)
        except np.linalg.LinAlgError:
            print("Computation of middle matrix for Cm_post was singular — using pseudo-inverse instead.")
            middle_term = np.linalg.pinv(G_Cm @ G.T + cov_d, rcond=1e-6)
            
        Cm_post = cov - cov @ G.T @ middle_term @ G @ cov
        
        Cm_post_list.append(Cm_post)
    
    # Resolution and spread
    resolution = Cm_post @ np.linalg.inv(cov)
        
    return d_misfit, m_misfit, edot_post, edot_post_list, Cm_post, resolution, Cm_post_list

def plot_true_and_posterior_erosion_maps(R, true_erosion_map, edot_posts, resolution, ratio, mask, ny, nx, shape_gdf, OUTPUT_PATH, savename):
    n_steps = len(edot_posts)
    n_plots = n_steps + 2  # One for true, one for difference, rest for steps

    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axs = axs.flatten() if n_plots > 1 else [axs]

    # Shared color scale
    min_e_true = np.nanmin(true_erosion_map)
    max_e_true = np.nanmax(true_erosion_map)
    all_maps = [true_erosion_map] + [v_to_xy(post, mask.flatten(), ny, nx, zero_val=np.nan) for post in edot_posts]
    vmin = np.nanpercentile(np.concatenate([m[~np.isnan(m)] for m in all_maps]), 1)
    vmax = np.nanpercentile(np.concatenate([m[~np.isnan(m)] for m in all_maps]), 99)
    norm = Normalize(vmin=np.min([vmin,min_e_true]), vmax=np.max([vmax,max_e_true]))
    
    # Set extent
    extent = [
        R.c,  # left (x min)
        R.c + R.a * B.shape[1],  # right (x max)
        R.f + R.e * B.shape[0],  # bottom (y min)
        R.f  # top (y max)
    ]
    
    # Prepare difference erosion map
    # Calculate difference map (last posterior - true)
    diff_map = v_to_xy(edot_posts[-1], mask.flatten(), ny, nx, zero_val=np.nan) - true_erosion_map

    # Diverging color scale for difference map, centered at zero
    diff_abs_max = np.nanmax(np.abs(diff_map))
    diff_norm = Normalize(vmin=-diff_abs_max, vmax=diff_abs_max)

    # Plot true erosion map
    im = axs[0].imshow(true_erosion_map, cmap='viridis', norm=norm, extent=extent, origin='upper')
    shape_gdf.boundary.plot(ax=axs[0], edgecolor='black', linewidth=1)
    axs[0].set_title("True Erosion Map")
    axs[0].axis('off')
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    # Plot posterior maps
    for i, post in enumerate(edot_posts):
        edot_map = v_to_xy(post, mask.flatten(), ny, nx, zero_val=np.nan)
        ax = axs[i + 1]
        im = ax.imshow(edot_map, cmap='viridis', norm=norm, extent=extent, origin='upper')
        shape_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
        
        # Add hatching only on the last posterior map
        if i == n_steps - 1 and resolution is not None and ratio is not None:
            resolution_diag = np.diag(resolution)
            resolution_map = v_to_xy(resolution_diag, mask.flatten(), ny, nx, zero_val=np.nan)
            ratio_map = v_to_xy(ratio, mask.flatten(), ny, nx, zero_val=np.nan)

            hatch_mask = (resolution_map < 0.95) | (resolution_map > 1) | (ratio_map > 1)

            ax.contourf(hatch_mask, levels=[0.5, 1], colors='none', hatches=['////'], 
                        extent=extent, origin='upper', alpha=0.25)
            
            # Add legend for hatched region to this specific subplot
            hatch_patch = mpatches.Patch(facecolor='lightgrey', hatch='////',
                                         label='Low-confidence area', 
                                         edgecolor='k', alpha = 0.25)
            ax.legend(handles=[hatch_patch], 
                      loc='lower left', 
                      bbox_to_anchor=(0, -0.2), 
                      bbox_transform=ax.transAxes, 
                      frameon=False, 
                      fontsize=10)
        
        # Compute and annotate L1 error
        abs_diff = np.nansum(np.abs(edot_map - true_erosion_map))
        ax.set_title(f"NL Step {i+1}", fontsize=10)
        ax.text(0.5, -0.2, r"$\sum_i |y_i - \hat{y}_i|$" + f"= {abs_diff:.2f}",
        transform=ax.transAxes, ha='center', va='top', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot difference map (last posterior - true)
    im_diff = axs[i+2].imshow(diff_map, cmap='RdBu', norm=diff_norm, extent=extent, origin='upper')
    shape_gdf.boundary.plot(ax=axs[i+2], edgecolor='black', linewidth=1)
    axs[i+2].set_title("Difference (Last Posterior - True)")
    axs[i+2].axis('off')
    plt.colorbar(im_diff, ax=axs[i+2], fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for j in range(n_steps + 2, len(axs)):
        axs[j].set_visible(False)

    fig.tight_layout()
    fig.suptitle("True and Posterior Erosion Maps", fontsize=16, y=1.02)
    fig.savefig(f"{OUTPUT_PATH}/{savename}_true_and_posteriors.png", bbox_inches='tight')
    plt.close(fig)

def save_settings_and_params(n_minerals_to_keep,flag_uniformprior,flag_firstprior,flag_quasi_newton,sigma_cov,d_sigma_val,n_m,mu,OUTPUT_PATH,savename):
    
    if flag_uniformprior == 1:
        flag_u_p_txt = 'Total erosion uniformally distributed'
    else:
        flag_u_p_txt = 'e ~ Kg*us^l'
    
    if flag_firstprior == 1:
        flag_f_p_txt = 'Specified in the "Uniform Prior?" section'
    else:
        flag_f_p_txt = 'Linear closed from solution'
        
    if flag_quasi_newton == 1:
        flag_q_n_txt = 'Quasi Newton solution - preconditioning with Hessian'
    else:
        flag_q_n_txt = 'Steepest Descent solution - preconditioning with model covariance'
    
    flag_settings = {
    'Number of most relevant minerals to keep (if None: keep all)': n_minerals_to_keep,
    'Prior erosion rates': flag_u_p_txt,
    'Starting point of non linear iteration': flag_f_p_txt,
    'Nonlinear inversion': flag_q_n_txt}
    
    model_params = {
    'sigma_cov (for building model covariance)': sigma_cov,
    'L (for building model covariance)': L,
    'sigma_d (for building data covariance)': d_sigma_val,
    'n_m (max number of non linear steps)': n_m,
    'mu (step size of non-lin. iterations)': mu}
    
    filepath = os.path.join(OUTPUT_PATH, f"{savename}_params.txt")
    
    with open(filepath, 'w') as f:
        f.write(f"Run Parameters for {savename}\n")
        f.write("=" * 40 + "\n")
        
        f.write("\n# Flags and Settings\n")
        for key, val in flag_settings.items():
            f.write(f"{key}     :    {val}\n")
        
        f.write("\n# Model Parameters\n")
        for key, val in model_params.items():
            f.write(f"{key} = {val}\n")

def summary_figure_sample(extent, mask, ny, nx, e_post, hillshade_path, glacier_shapefile, 
                          geology_shapefile, source_data, lithology_colors,  
                          resolution, ratio, cov, fig_name):
    
    # Setup
    n_steps = len(edot_posts)
    iterations = list(range(1, n_steps + 1))

    # Load hillshade and shapefiles
    with rasterio.open(hillshade_path) as src:
        hillshade = src.read(1)
        hillshade[hillshade < 0] = np.nan
        hs_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    glacier = gpd.read_file(glacier_shapefile)
    geology = gpd.read_file(geology_shapefile)

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    cmap_e = plt.cm.viridis
    cmap_unc = plt.cm.magma_r

    # Normalize
    vmin = np.nanpercentile(e_post[~np.isnan(e_post)], 1)
    vmax = np.nanpercentile(e_post[~np.isnan(e_post)], 99)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Posterior uncertainty
    resolution_diag = np.diag(resolution)
    resolution_map = v_to_xy(resolution_diag, mask.flatten(), ny, nx, zero_val=np.nan)
    
    ratio_map = v_to_xy(ratio, mask.flatten(), ny, nx, zero_val=np.nan)
    
    hatch_mask = (resolution_map < 0.95) | (resolution_map > 1) | (ratio_map > 1)
    
    diag = np.diag(Cm_posts[-1])
    diag_map = v_to_xy(diag, mask.flatten(), ny, nx, zero_val=np.nan)

    # Row 1: Posterior erosion rate map, Resolution, Ratio
    for idx, data, title in zip(range(3), [e_post, resolution_map, ratio_map],
                                 ["Posterior Erosion Rate (mm/y): " + fig_name, "Resolution Diagonal", 
                                  "Normalized Posterior Uncertainty"]):
        ax = fig.add_subplot(gs[0, idx])
        
        norm_used = norm if idx < 1 else None
        cmap_used = cmap_e if idx < 1 else cmap_unc
        
        im = ax.imshow(data, extent=extent, cmap=cmap_used, norm=norm_used)
        
        ax.imshow(hillshade, extent=hs_extent, cmap='gray', alpha=0.5)
        glacier.boundary.plot(ax=ax, edgecolor='black', linewidth=0.8)
        
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        if 'Posterior Erosion Rate' in title:
            ax.contourf(hatch_mask, levels=[0.5, 1], colors=['lightgrey'], hatches=['////'],
                        extent=extent, origin='upper', alpha=0.5)

            hatch_patch = mpatches.Patch(facecolor='lightgrey', hatch='////',
                                         label='Low-confidence area',
                                         edgecolor='k', linewidth=0.5)
            ax.legend(handles=[hatch_patch],
                      loc='lower left',
                      bbox_to_anchor=(0, -0.2),
                      bbox_transform=ax.transAxes,
                      frameon=False,
                      fontsize=10)
    
        if title == "Normalized Posterior Uncertainty":
            diag_tot = np.trace(Cm_post)/np.trace(cov)
            ax.text(0.5, -0.1, "Normalized Posterior Uncertainty"+ f"= {diag_tot:.2f}", 
                    transform=ax.transAxes, ha='center', va='top', fontsize=10)
    
    # Row 2 col 1: Geological Map
    ax_geol = fig.add_subplot(gs[1, 0])
    for geo_name, color in lithology_colors.items():
        geology[geology['litho'] == geo_name].plot(ax=ax_geol, facecolor=color, edgecolor='none', alpha=0.5)
    ax_geol.imshow(hillshade, extent=hs_extent, cmap='gray')
    glacier.boundary.plot(ax=ax_geol, edgecolor='black', linewidth=0.8)
    ax_geol.set_title("Geological Map")
    ax_geol.set_xticks([])
    ax_geol.set_yticks([])
    ax_geol.add_artist(ScaleBar(1))
    for spine in ax_geol.spines.values():
        spine.set_visible(False)
    
    legend_elements = [Patch(facecolor=color, edgecolor='k', label=name) for name, color in lithology_colors.items()]
    ax_geol.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.5),  # move it slightly lower
        ncol=1,                        # one column
        fontsize=10,
        frameon=False)

    # Row 4 col 2+3: Source concentrations
    ax_bar = fig.add_subplot(gs[1, 1:])
    mineral_names = source_data['Mineral names (rows)']
    geo_names = list(source_data['Geology names (columns)'].values())
    source_matrix = source_data['source']

    matrix_norm = source_matrix / (source_matrix.sum(axis=1, keepdims=True) + 1e-12)
    bottom = np.zeros(matrix_norm.shape[0])
    for j in range(matrix_norm.shape[1]):
        ax_bar.bar(np.arange(matrix_norm.shape[0]), matrix_norm[:, j],
                   bottom=bottom,
                   color=list(lithology_colors.values())[j % len(lithology_colors)],
                   alpha=0.7)
        bottom += matrix_norm[:, j]

    ax_bar.set_ylabel("Normalized abundance per mineral",fontsize=12)
    ax_bar.set_xticks(np.arange(len(mineral_names)))
    ax_bar.set_xticklabels(mineral_names, rotation=90)
    ax_bar.set_yticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    plt.savefig(f"{OUTPUT_PATH}/summary_nonlinear_inversion_{fig_name}.pdf", bbox_inches="tight")
    plt.show()
  
def plot_iterations(extent, edot_posts, resolution, ratio, mask, ny, nx, shape_gdf, OUTPUT_PATH, savename):
    n_steps = len(edot_posts)
    n_plots = n_steps

    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axs = axs.flatten() if n_plots > 1 else [axs]

    # Shared color scale
    all_maps = [v_to_xy(post, mask.flatten(), ny, nx, zero_val=np.nan) for post in edot_posts]
    vmin = np.nanpercentile(np.concatenate([m[~np.isnan(m)] for m in all_maps]), 1)
    vmax = np.nanpercentile(np.concatenate([m[~np.isnan(m)] for m in all_maps]), 99)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot posterior maps
    for i, post in enumerate(edot_posts):
        edot_map = v_to_xy(post, mask.flatten(), ny, nx, zero_val=np.nan)
        ax = axs[i]
        im = ax.imshow(edot_map, cmap='viridis', norm=norm, extent=extent, origin='upper')
        shape_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
        
        # Add hatching only on the last posterior map
        if i == n_steps - 1 and resolution is not None and ratio is not None:
            resolution_diag = np.diag(resolution)
            resolution_map = v_to_xy(resolution_diag, mask.flatten(), ny, nx, zero_val=np.nan)
            ratio_map = v_to_xy(ratio, mask.flatten(), ny, nx, zero_val=np.nan)

            hatch_mask = (resolution_map < 0.95) | (resolution_map > 1) | (ratio_map > 1)

            ax.contourf(hatch_mask, levels=[0.5, 1], colors='none', hatches=['////'], 
                        extent=extent, origin='upper', alpha=0.25)
            
            # Add legend for hatched region to this specific subplot
            hatch_patch = mpatches.Patch(facecolor='lightgrey', hatch='////',
                                         label='Low-confidence area', 
                                         edgecolor='k', alpha = 0.25)
            ax.legend(handles=[hatch_patch], 
                      loc='lower left', 
                      bbox_to_anchor=(0, -0.2), 
                      bbox_transform=ax.transAxes, 
                      frameon=False, 
                      fontsize=10)
        
        # Compute and annotate normalized uncertainty 
        diag_tot = np.trace(Cm_post)/np.trace(cov)
        
        ax.set_title(f"NL Step {i+1}", fontsize=10)
        ax.text(0.5, -0.1, "Normalized uncertainty"+ f"= {diag_tot:.2f}", 
                transform=ax.transAxes, ha='center', va='top', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for j in range(n_steps, len(axs)):
        axs[j].set_visible(False)

    fig.tight_layout()
    fig.suptitle("Posterior Erosion Maps", fontsize=16, y=1.02)
    #fig.savefig(f"{OUTPUT_PATH}/{savename}_true_and_posteriors.png", bbox_inches='tight')
    fig.show()
     
# -----------------------------------------------------------------------------
# _____________________________________________________________________________
# MAIN BLOCK
# 1. Load maps and data
OUTPUT_PATH = create_output_path(OUTPUT_FOLDER, savefolder)
B, R, crs_raster = load_geology('geology')
DEM, vel_obs = load_dem_and_velocity() if flag_uniformprior != 1 else (None, None)
shape_gdf = gpd.read_file(SHAPEFILE_PATH).to_crs(crs_raster)
mask_GG = load_mask_GG()

source, top_k_idx = load_source_data(n_minerals_to_keep)
with open(f'{INPUT_FOLDER}/XRD_data/SourceData/source_cleaned.pkl', 'rb') as f:
     source_data = pickle.load(f)

n_tracers = source.shape[0]

datenames, data_percentage, data_sigma = load_detrital_data_from_pickle(DETRITAL_PATH, LOAD_PATH, top_k_idx)
n_dates = data_percentage.shape[1]

if flag_use_zircon_data:
    datenames, data_percentage, data_sigma, gg_df = load_detrital_data_Bruno('B')
    n_dates = data_percentage.shape[1]
    gg_dates = set(pd.to_datetime(gg_df['date'], format='%d-%b-%Y %H:%M'))
    sample_dates = [datetime.strptime(date, "%d-%b-%Y %H:%M") for date in datenames]
    dates_of_interest = [date.strftime('%Y%m%d') for date in sample_dates]
    
    source, top_k_idx = load_source_data_Bruno(n_minerals_to_keep)
    source_data['source'] = source
    source_data['Geology names (columns)'] = geo_names
    source_data['Mineral names (rows)'] = age_labels
    
    n_tracers = source.shape[0]
    
else:
    gg_dates = [None]
    sample_dates = [datetime.strptime(date, '%d.%m.%Y %H:%M') for date in datenames]
    

if flag_combine_data == 1:
    gg_dates = [None]
    source_XRD, top_k_idx = load_source_data(None)
    with open(f'{INPUT_FOLDER}/XRD_data/SourceData/source_cleaned.pkl', 'rb') as f:
         source_data_XRD = pickle.load(f)

    n_tracers_XRD = source.shape[0]

    datenames_XRD, data_percentage_XRD, data_sigma_XRD = load_detrital_data_from_pickle(DETRITAL_PATH, LOAD_PATH, top_k_idx)
    n_dates_XRD = data_percentage.shape[1]
    
    source_z, top_k_idx = load_source_data_Bruno(None)
    source_data = {}
    source = np.vstack([source_XRD,source_z])
    source_data['source'] = np.vstack([source_XRD,source_z])
    source_data['Geology names (columns)'] = geo_names
    source_data['Mineral names (rows)'] = source_data_XRD['Mineral names (rows)']+ age_labels
    
    n_tracers_XRD = source_XRD.shape[0]; n_tracers_z = source_z.shape[0]
    n_tracers = source.shape[0]
    
    datenames_z, data_percentage_z, data_sigma, gg_df = load_detrital_data_Bruno('B')
    
    # Extract datetimes with hour-level precision
    xrd_times = [datetime.strptime(date, '%d.%m.%Y %H:%M') for date in datenames_XRD]
    zircon_times = [datetime.strptime(date, "%d-%b-%Y %H:%M") for date in datenames_z]

    # Create tuples (day, hour) for comparison
    xrd_day_hour = [(dt.date(), dt.hour) for dt in xrd_times]
    zircon_day_hour = [(dt.date(), dt.hour) for dt in zircon_times]

    # Find matching (day, hour) timestamps
    matched_indices = [(i, j) for i, xrd_dh in enumerate(xrd_day_hour)
                       for j, zir_dh in enumerate(zircon_day_hour)
                       if xrd_dh == zir_dh]
    
    # Get J-S distance
    n_lithologies = len(lithology_colors)
    print("XRD  avg JS =" + str(get_js_dist(source_XRD,n_lithologies)))
    print("Zr  avg JS =" + str(get_js_dist(source_z,n_lithologies)))
    print("XRD & Zr  avg JS =" + str(get_js_dist(source,n_lithologies)))
    
    # Combine matching in dataframe
    data_percentage = np.empty(shape=(source.shape[0],len(matched_indices)))
    col = -1
    sample_dates = []
    datenames = []
    for idx_xrd, idx_z in matched_indices:
        col = col + 1
        data_percentage[0:n_tracers_XRD,col] = data_percentage_XRD[:,idx_xrd]
        data_percentage[n_tracers_XRD:,col] = data_percentage_z[:,idx_z]
        sample_dates.append(xrd_times[idx_xrd])
        datenames.append(datenames_XRD[idx_xrd])
    
    dates_of_interest = [date.strftime('%Y%m%d') for date in sample_dates]
    

# 2. Set up model parameters
sigma_cov, L, d_sigma_val, n_m, mu, misfit, d_misfit, m_misfit = setup_model_params()
Q_s = 100e6; # sediment load transported in river in one year (kg/y) (here: year 2018)

# 3. Set up grid and observed data
ny, nx, nn, dx, dy, Lx, Ly, dA, X2, Y2 = setup_grid_dimensions(B, R)
mask = setup_observed_data(B)

# 4. Set up the forward-inverse matrices for the inversion
# Masked coordinates of active cells
active_idx = np.where(mask.flatten() > 0)[0]
x = X2.flatten()[active_idx]
y = Y2.flatten()[active_idx]
mask_size = len(x)

# 5. Generate eprior and etot
etot = compute_etot(dx, dy, Q_s, rho=2650, dA=1.0)

if flag_uniformprior == 1:
    eps_prior = compute_eps_prior_uniform(etot, mask)
else:
    e_prior = 2.7e-4 * np.power(vel_obs,2.02)
    e_prior[e_prior == 0] = 0.001
    eps_prior_xy = np.log(e_prior)
    eps_prior = eps_prior_xy.flatten()[active_idx]

data_sigma = None # measured data_sigma

# 6. Generate all required matrices (also for mask_GG)
A = generate_A(B, mask, source, n_tracers)
cov = generate_cov(x, y, sigma_cov=sigma_cov, L=L, nugget=1e-2)
cov_d = generate_cov_d(data_sigma, d_sigma_val, flag_sigma_d_meas, n_tracers)

mask_GG = load_mask_GG()
mask_GG_combined = mask_GG.astype(bool)
kept_idx = np.where(mask_GG_combined.flatten())[0]
A_GG = A[:, kept_idx]
cov_GG = cov[np.ix_(kept_idx, kept_idx)]
eps_prior_GG = eps_prior[kept_idx]
mask_size_GG = len(kept_idx)
etot_GG = sum(eps_prior_GG)
    
# 7. Only get data for dates of interest
idx_data = [i for i in range(len(datenames)) if sample_dates[i].strftime('%Y%m%d') in dates_of_interest]

# 8. Set-up figure environment
cmap = cm.viridis
extent = [
    R.c,  # left (x min)
    R.c + R.a * B.shape[1],  # right (x max)
    R.f + R.e * B.shape[0],  # bottom (y min)
    R.f  # top (y max)
]


# 9. Inversion
for date_idx in idx_data:
    date_name_i = sample_dates[date_idx].strftime('%d.%m.%Y %H:%M')
    
    date = sample_dates[date_idx]
    
    if date not in gg_dates:
        d = prepare_data_for_iteration(data_percentage, date_idx, etot)
        d_misfit, m_misfit, edot_post, edot_posts, Cm_post, resolution, Cm_posts = nonlinear_iteration(A, 
                                                                                 cov, 
                                                                                 cov_d, 
                                                                                 eps_prior, 
                                                                                 mu, 
                                                                                 n_m, 
                                                                                 etot,
                                                                                 d, 
                                                                                 mask_size, 
                                                                                 flag_firstprior, 
                                                                                 flag_quasi_newton)
        ratio = np.sqrt(np.diag(Cm_post))/np.sqrt(np.diag(cov))
        mask_to_use = mask
        cov_to_use = cov
        
    else:
        d = prepare_data_for_iteration(data_percentage, date_idx, etot_GG)
        d_misfit, m_misfit, edot_post, edot_posts, Cm_post, resolution, Cm_posts = nonlinear_iteration(A_GG, 
                                                                                 cov_GG, 
                                                                                 cov_d, 
                                                                                 eps_prior_GG, 
                                                                                 mu, 
                                                                                 n_m, 
                                                                                 etot_GG,
                                                                                 d, 
                                                                                 mask_size_GG, 
                                                                                 flag_firstprior, 
                                                                                 flag_quasi_newton)
        ratio = np.sqrt(np.diag(Cm_post))/np.sqrt(np.diag(cov_GG))
        mask_to_use = mask_GG
        cov_to_use = cov_GG
    
    # 10. Plotting and saving
    e_post = v_to_xy(edot_posts[-1], mask_to_use.flatten(), ny, nx, zero_val=np.nan)
    summary_figure_sample(extent, mask_to_use, ny, nx, e_post, hillshade_path, 
                          glacier_shapefile, geology_shapefile, source_data, 
                          lithology_colors, resolution, ratio, cov_to_use, 
                          savename + date_name_i)
    
    # plot_iterations(extent, edot_posts, resolution, ratio, mask_to_use, ny, nx, 
    #                 shape_gdf, OUTPUT_PATH, savename + date_name_i)

save_settings_and_params(n_minerals_to_keep,
                         flag_uniformprior,
                         flag_firstprior,
                         flag_quasi_newton,
                         sigma_cov,
                         d_sigma_val,
                         n_m,mu,
                         OUTPUT_PATH,savename)

