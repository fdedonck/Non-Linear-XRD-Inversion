#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:26:21 2025

@author: fiendedoncker
"""

# Modified from last version: 22 Jul: added normalization with avg erosion rate
# Instead of e_mean, we just use normalization by e_0 = 1, like that nothing else changes
# Normalisation was needed to prevent taking log of variable with units (mm/y)

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


# 1.3. Flags and Settings
savename = 'Forw_Inv_Synth_Source'
#savefolder = 'Quasi Newton Forward-Inverse tests XRD area data - e true hotspots - cross sample norm'
savefolder = 'QN Forw-Inv tests SYNTHETIC SOURCE'

lithology_colors = {
    "ZSF ophiolites (serpentinites)": "#e3dd19",
    "ZSF sediments": "#809847",
    "Stockhorn, Tuftgrat, Gornergrat": "#ff5001",
    "Monte Rosa (granite)": "#fa9b9a",
    "ZSF ophiolites (metabasites, eclogites)": "#32a02d",
    "Monte Rosa (gneiss, micaschist)": "#ff7f41",
    "Furgg series": "#badd68"
}

ltr =" ABCDEFGHIJKLMNOPQRSTUVWXYZ"

n_minerals_to_keep = None
flag_uniformprior = 1 # if == 1: etot uniformally distributed, else: eprior ~ Kg*us^l
flag_firstprior = 1 # if == 1: start with eprior, else: start from linear closed form solution
flag_sigma_d_meas = 0 # if == 1: use observed sigma, else: use parameter d sigma
flag_quasi_newton = 1 # if == 1: quasi newton, else: steepest descent
flag_test_optimal_source = 1  # if == 1: test with very distinct synthetic sediment sources
flag_test_parameters = 1 # if == 1: test values for different par.s
flag_test_scenarios = 1 # if == 1: test with synthetic source data
flag_test_SD = 1 # if == 1: test with steepest descent
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
    
    with open(f'{INPUT_FOLDER}/XRD_data/SourceData/source_cleaned.pkl', 'rb') as f:
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
    
def generate_distinct_source(n_minerals=21, n_lithologies=7, seed=42, 
                             n_minerals_to_keep=None):
    np.random.seed(seed)
    
    source = np.zeros((n_minerals, n_lithologies))
    available_minerals = list(range(n_minerals))
    np.random.shuffle(available_minerals)
    
    used_minerals = set()

    for lith in range(n_lithologies):
        # Select dominant minerals, avoiding reuse if possible
        minerals_per_lith = np.random.randint(2, 5)  # number of dominant minerals for this lithology
        remaining = [m for m in available_minerals if m not in used_minerals]
        if len(remaining) < minerals_per_lith:
            # allow overlap if we run out
            dominant = np.random.choice(n_minerals, size=minerals_per_lith, replace=False)
        else:
            dominant = np.random.choice(remaining, size=minerals_per_lith, replace=False)
            used_minerals.update(dominant)

        # Assign high values to dominant minerals
        for m in range(n_minerals):
            if m in dominant:
                source[m, lith] = np.random.uniform(0.8, 1.0)
            else:
                source[m, lith] = np.random.uniform(0.001, 0.1)
        
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
    
def degrade_source_data(source, degradation_level):
    # degradation level between 0 and 1
    # slowly blend to average
    mean_profile = source.mean(axis=1, keepdims=True)  # average over lithologies
    degraded_source = (1 - degradation_level) * source + degradation_level * mean_profile

    n_minerals, n_lithologies = degraded_source.shape
    H_stats = []

    dists = []
    for i, j in combinations(range(n_lithologies), 2):
        p = degraded_source[:, i]
        q = degraded_source[:, j]
        # Normalize in case columns don’t sum to 1
        p = p / p.sum()
        q = q / q.sum()
        d = jensenshannon(p, q)
        dists.append(d)
    
    avg_js_dist = np.round(np.mean(dists),3);
    
    return degraded_source, avg_js_dist

def get_n_tracers(source, n_minerals_to_keep):
    pca = PCA()
    pca.fit(source.T)
    importance = np.abs(pca.components_[0])
    top_k_idx = np.argsort(importance)[::-1][:n_minerals_to_keep]
    source = source[top_k_idx, :]  # reduce minerals
    return source, top_k_idx

def create_synthetic_erosion_map_hotspots(B, R):
    sigma = 5 # bump width in pixels
    e_min = 0.1
    e_max = 3
    
    nrows, ncols = B.shape
    erosion_map = np.ones((nrows, ncols)) * e_min  # Base erosion rate

    # Define a single Gaussian bump in the center
    center_r, center_c = (nrows-4) // 2, (ncols-2) // 2
    sigma = 4  # controls the width of the bump

    for i in range(-3*sigma, 3*sigma + 1):
        for j in range(-3*sigma, 3*sigma + 1):
            rr, cc = center_r + i, center_c + j
            if 0 <= rr < nrows and 0 <= cc < ncols:
                erosion_map[rr, cc] += e_max * np.exp(-(i**2 + j**2) / (2 * sigma**2))
   
    # Mask invalid geology
    erosion_map[B == 0] = np.nan

    return erosion_map

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 4. Set up parameters, dimensions etc
def setup_model_params():
    sigma_cov = 3
    #sigma_cov = 0.055
    
    L = 1200 #m
    #L = 1800
    
    d_sigma_val = 0.01
    #d_sigma_val = 0.007
    
    n_m = 20
    mu = 0.1
    
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
        middle_term = np.linalg.inv(G_Cm @ G.T + cov_d)
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

def generate_summary_figure(
    R, e_true, e_true_v, e_post, hillshade_path, glacier_shapefile, geology_shapefile,
    source_data, lithology_colors,
    data_misfits, model_misfits, edot_posts, Cm_posts, resolution, ratio, fig_name):
    # Setup
    extent = [R.c, R.c + R.a * e_true.shape[1], R.f + R.e * e_true.shape[0], R.f]
    n_steps = len(edot_posts)
    iterations = list(range(1, n_steps + 1))

    # Load hillshade and shapefiles
    with rasterio.open(hillshade_path) as src:
        hillshade = src.read(1)
        hillshade[hillshade < 0] = np.nan
        hs_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    glacier = gpd.read_file(glacier_shapefile)
    geology = gpd.read_file(geology_shapefile)

    fig = plt.figure(figsize=(20, 19))
    gs = gridspec.GridSpec(4, 3, figure=fig, 
                           height_ratios=[1, 1, 1.5, 1.2],  # adjust second and third row height 
                           hspace=0.2  # reduce space between rows
                           )

    cmap = plt.cm.viridis
    diff_cmap = plt.cm.RdBu_r

    # Normalize
    vmin = np.nanpercentile(np.concatenate([e_true[~np.isnan(e_true)], e_post[~np.isnan(e_post)]]), 1)
    vmax = np.nanpercentile(np.concatenate([e_true[~np.isnan(e_true)], e_post[~np.isnan(e_post)]]), 99)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Posterior uncertainty
    resolution_diag = np.diag(resolution)
    resolution_map = v_to_xy(resolution_diag, mask.flatten(), ny, nx, zero_val=np.nan)
    ratio_map = v_to_xy(ratio, mask.flatten(), ny, nx, zero_val=np.nan)
    hatch_mask = (resolution_map < 0.95) | (resolution_map > 1) | (ratio_map > 1)
    diag = np.diag(Cm_posts[-1])
    diag_map = v_to_xy(diag, mask.flatten(), ny, nx, zero_val=np.nan)

    # Row 1: Erosion maps
    for idx, data, title in zip(range(3), [e_true, e_post, e_true - e_post],
                                 ["True Erosion Rate", "Posterior Erosion Rate", "Difference (True - Posterior)"]):
        ax = fig.add_subplot(gs[0, idx])
        norm_used = norm if idx < 2 else Normalize(vmin=-np.nanmax(np.abs(data)), vmax=np.nanmax(np.abs(data)))
        cmap_used = cmap if idx < 2 else diff_cmap
        im = ax.imshow(data, extent=extent, cmap=cmap_used, norm=norm_used)
        ax.imshow(hillshade, extent=hs_extent, cmap='gray', alpha=0.5)
        glacier.boundary.plot(ax=ax, edgecolor='black', linewidth=0.8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        if title == 'Posterior Erosion Rate':
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

    # Row 2: Posterior uncertainty (Resolution, Covariance diag, Ratio)
    for idx, data, title in zip(range(3), [resolution_map, diag_map, ratio_map],
                                 ["Resolution Diagonal", "Diag(Cm_post)", "Ratio diag(Cm_post)/diag(Cm)"]):
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(data, extent=extent, cmap='magma_r')
        ax.imshow(hillshade, extent=hs_extent, cmap='gray', alpha=0.5)
        glacier.boundary.plot(ax=ax, edgecolor='black', linewidth=0.8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Row 3: Error and trade-off plots
    abs_errors = [np.sum(np.abs(e_true_v - edot)) for edot in edot_posts]
    traces = [np.trace(C)/np.trace(cov) for C in Cm_posts]
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])

    sc7 = ax7.scatter(iterations, abs_errors, c=iterations, cmap='Purples', edgecolor='k', linewidths=0.5)
    sc8 = ax8.scatter(iterations[:len(traces)], traces, c=iterations[:len(traces)], cmap='Purples', edgecolor='k', linewidths=0.5)
    sc9 = ax9.scatter(data_misfits[:len(traces)], model_misfits[:len(traces)], c=iterations[:len(traces)], cmap='Purples', edgecolor='k', linewidths=0.5)
    
    for ax in [ax7, ax8, ax9]:
        ax.set_box_aspect(1)
    
    ax7.set_title("∑|e_true - e_post|")
    ax7.set_xlabel("Iteration",fontsize=12)
    ax8.set_title("Trace(Cm_post)/trace(Cm)")
    ax8.set_xlabel("Iteration",fontsize=12)
    ax9.set_title("Trade-off")
    ax9.set_xlabel("Data Misfit",fontsize=12)
    ax9.set_ylabel("Model Misfit",fontsize=12)
    cbar = fig.colorbar(sc9, ax=ax9, fraction=0.025, pad=0.02)
    cbar.set_label("Iteration", fontsize=12)
    
    # Row 4 col 1: Geological Map
    ax10 = fig.add_subplot(gs[3, 0])
    for geo_name, color in lithology_colors.items():
        geology[geology['litho'] == geo_name].plot(ax=ax10, facecolor=color, edgecolor='none', alpha=0.5)
    ax10.imshow(hillshade, extent=hs_extent, cmap='gray')
    glacier.boundary.plot(ax=ax10, edgecolor='black', linewidth=0.8)
    ax10.set_title("Geological Map")
    ax10.set_xticks([])
    ax10.set_yticks([])
    for spine in ax10.spines.values():
        spine.set_visible(False)
    
    legend_elements = [Patch(facecolor=color, edgecolor='k', label=name) for name, color in lithology_colors.items()]
    ax10.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.5),  # move it slightly lower
        ncol=1,                        # one column
        fontsize=10,
        frameon=False)

    # Row 4 col 2+3: Source concentrations
    ax_bar = fig.add_subplot(gs[3, 1:])
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
        
# -----------------------------------------------------------------------------
# Test param values
def parameter_sweep(param_name, param_values, d,
                    erosion_true, etot, source, B, A, x, y, 
                    cov, cov_d, eps_prior, mu, n_m, mask_size, 
                    flag_firstprior, flag_quasi_newton):
    
    # Load data
    d_original = d.copy()
    A_original = A.copy()
    B_original = B.copy()
    source_original = source.copy()
    n_tracers = A.shape[0]

    results = []
    source_out = {}

    for val in tqdm(param_values, desc=f"Sweeping {param_name}"):
        # Use default values for everything else
        sigma_cov, L, d_sigma_val, n_m, mu, *_ = setup_model_params()

        if param_name == "L":
            L = val
            cov = generate_cov(x, y, sigma_cov=sigma_cov, L=L)
            
        elif param_name == "sigma_cov":
            sigma_cov = val
            cov = generate_cov(x, y, sigma_cov=sigma_cov, L=L)
            
        elif param_name == "d_sigma_val":
            d_sigma_val = val
            cov_d = generate_cov_d(None, d_sigma_val=d_sigma_val, flag_sigma_d_meas=0, n_tracers=A.shape[0])
        
        elif param_name == "mu":
            mu = val
        
        elif param_name == "n_m":
            n_m = val
        
        elif param_name == "geo_diff":
            geol_loadname = val
            
            B_test, R, crs_raster = load_geology(val)
            B_test[B_test<=0] = 0
            A = generate_A(B_test, mask, source_original, n_tracers)
            
            diff_mask = B_original != B_test
            diff_cells = np.sum(diff_mask)
            val = diff_cells * x.min() * x.min() # difference area in m2
        
        elif param_name == "avg_js_dist":
            source, val = degrade_source_data(source_original, val)
            source_out[val] = source
            A = generate_A(B, mask, source, n_tracers)
            
        elif param_name == "n_tracers":
            n_tracers = val
            
            # select n most important tracers
            #source, top_k_idx = load_source_data(n_tracers)
            source, top_k_idx = get_n_tracers(source_original, n_tracers)
            
            source_to_store = source_original * 0
            source_to_store[top_k_idx,:] = source_original[top_k_idx,:]
            source_out[val] = source_to_store
            
            # create A from cropped source data
            A = A_original[top_k_idx,:]
            
            # crop d 
            d = d_original[top_k_idx]
            
            # crop cov_d
            cov_d = generate_cov_d(None, d_sigma_val=d_sigma_val, flag_sigma_d_meas=0, n_tracers=A.shape[0])
            

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

        e_true_v = erosion_true[mask == 1]
        abs_error = np.sum(np.abs(e_true_v - edot_post))
        trace_cov = np.trace(Cm_post) / np.trace(cov)
        
        results.append({
            "param_name": param_name,
            "param_value": val,
            "abs_error": abs_error,
            "data_misfit": d_misfit[-1],
            "model_misfit": m_misfit[-1],
            "trace_cov": trace_cov
        })
    
    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_PATH}/sweep_results_{param_name}.csv", index=False)
    
    if (param_name == "avg_js_dist") or (param_name == "n_tracers"):
        return source_out, df
    else:
        return df

def plot_all_sweeps(all_results_df):
    param_order = ["L", "sigma_cov", "d_sigma_val", "mu", "n_m"]
    fig, axs = plt.subplots(len(param_order), 3, figsize=(16, 22))
    palettes = {
        "L": sns.color_palette("Blues", as_cmap=True),
        "sigma_cov": sns.color_palette("Oranges", as_cmap=True),
        "d_sigma_val": sns.color_palette("Greens", as_cmap=True),
        "mu": sns.color_palette("Purples", as_cmap=True),
        "n_m": sns.color_palette("Reds", as_cmap=True)
    }

    for i, param in enumerate(param_order):
        df = all_results_df[all_results_df['param_name'] == param]
        cmap = palettes[param]

        # Use log color scale for logspaced params
        if param in ["sigma_cov", "d_sigma_val", "mu"]:
            norm = LogNorm(vmin=df['param_value'].min(), vmax=df['param_value'].max())
        else:
            norm = Normalize(vmin=df['param_value'].min(), vmax=df['param_value'].max())

        colors = [cmap(norm(v)) for v in df['param_value']]

        # 1st column: Abs error
        axs[i, 0].scatter(df['param_value'], df['abs_error'], color=colors,
                          edgecolor='k', s=60, marker='o')
        axs[i, 0].set_ylabel("sum |e_true - e_post|")
        axs[i, 0].set_xlabel(f"{param}")
        axs[i, 0].set_title("Total absolute difference e_post - e_true")

        # 2nd column: Trace of posterior covariance
        axs[i, 1].scatter(df['param_value'], df['trace_cov'], color=colors,
                          edgecolor='k', s=60, marker='o')
        axs[i, 1].set_title("Normalized posterior uncertainty")
        axs[i, 1].set_ylabel("Trace(Cm_post)")
        axs[i, 1].set_xlabel(f"{param}")

        # 3rd column: Trade-off
        axs[i, 2].scatter(df['data_misfit'], df['model_misfit'], c=colors,
                          edgecolor='k', s=60, marker='o')
        axs[i, 2].set_title("Trade-off")
        axs[i, 2].set_xlabel("Data Misfit")
        axs[i, 2].set_ylabel("Model Misfit")
        
        # Best point (lowest total abs error)
        best_idx = df['abs_error'].idxmin()
        best_row = df.loc[best_idx]
        x, y = best_row['data_misfit'], best_row['model_misfit']
        param_val = best_row['param_value']
        
        # Plot emphasized circle
        axs[i, 2].scatter(x, y, s=100, facecolor='none', edgecolor='black', linewidth=1.5, zorder=10)
        
        # Get axis limits
        xlim = axs[i, 2].get_xlim()
        ylim = axs[i, 2].get_ylim()
        
        # Normalized (0–1) coordinates for annotation (safe position)
        x_text_norm = 0.7  # Adjust if needed
        y_text_norm = 0.85
        
        # Convert to data coords
        x_text = xlim[0] + x_text_norm * (xlim[1] - xlim[0])
        y_text = ylim[0] + y_text_norm * (ylim[1] - ylim[0])
        
        # Annotate inside the box
        axs[i, 2].annotate(
            f"{param} = {param_val:.2g}",
            xy=(x, y),
            xytext=(x_text, y_text),
            textcoords='data',
            ha='right',
            va='bottom',
            fontsize=9,
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
            arrowprops=dict(arrowstyle="->", lw=1, color='black'),
            clip_on=True
        )
        
        # Add colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # required for ScalarMappable without data
        plt.colorbar(sm, ax=axs[i, 2], label=param)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/summary_parameter_sweeps.pdf")
    plt.show()
    
def plot_scenarios(all_results_df, source_matrices_all, geo_maps, geo_labels, lithology_colors):
    # Helper function to get keys corresponding to two extreme values and midpoint value
    def get_three_keys(source_dict):
        keys = sorted(source_dict.keys())
        return [keys[0], keys[len(keys)//2], keys[-1]]
    
    # Customization
    scenarios = ['geo_diff', 'avg_js_dist', 'n_tracers']
    scenario_titles = {
        'geo_diff': 'Misaligned geology',
        'avg_js_dist': 'Blended fingerprints',
        'n_tracers': 'Reduced tracer count'}
    
    scenario_xlabels = {
        'geo_diff': 'Area difference',
        'avg_js_dist': 'Average JS distance between lithologies',
        'n_tracers': 'Number of tracers'}
    
    metric_ylabels = {
        'abs_error': 'Absolute Error',
        'trace_cov': 'Posterior uncertainty',
        'data_misfit': 'Data Misfit',
        'model_misfit': 'Model Misfit'}
    
    colors_dict = {
        'geo_diff': sns.color_palette("OrRd", as_cmap=True),
        'avg_js_dist': sns.color_palette("PuBu", as_cmap=True),
        'n_tracers': sns.color_palette("RdPu", as_cmap=True)
    }

    # Metrics to plot
    metrics = ['abs_error', 'trace_cov']
    
    fig = plt.figure(figsize=(20, 30))
    gs = gridspec.GridSpec(6+3, 3, width_ratios=[1, 1, 1], 
                           height_ratios=[1, 2.5, 0.2, 1, 2.5,0.2,1,2.5,0.2], 
                           wspace=0.1, hspace=0.2)

    row = 0
    for i, scenario in enumerate(scenarios):
        df = all_results_df[all_results_df['param_name'] == scenario]
        c_map = colors_dict[scenario]
        param_values = df['param_value']
        
        # Row 1, column 0,1,2: mineral fingerprints (or geological maps)
        if scenario == 'geo_diff':
            # Load hillshade and shapefiles
            with rasterio.open(hillshade_path) as src:
                hillshade = src.read(1)
                hillshade[hillshade < 0] = np.nan
                hs_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
                
            for ii, (geo_map, label) in enumerate(zip(geo_maps, geo_labels)):
                ax = fig.add_subplot(gs[row, ii])
                for lith, color in lithology_colors.items():
                    geo_map[geo_map['litho'] == lith].plot(ax=ax, facecolor=color, edgecolor='none', alpha=0.6)
                
                ax.imshow(hillshade, extent=hs_extent, cmap='gray')
                ax.set_title(label)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
        else:   
            keys3 = get_three_keys(source_matrices_all[scenario])
            for ii, key in enumerate(keys3):
                ax = fig.add_subplot(gs[row, ii])
                matrix = source_matrices_all[scenario][key]
                matrix_norm = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-12)  # normalize each row
                bottom = np.zeros(matrix.shape[0])  # start at zero for each mineral
                for j in range(matrix.shape[1]):
                    ax.bar(np.arange(matrix.shape[0]), matrix_norm[:, j],
                           bottom=bottom,
                           color=list(lithology_colors.values())[j % len(lithology_colors)],
                           alpha = 0.7)
                    bottom += matrix_norm[:, j]
                ax.set_title(f"{scenario_xlabels[scenario]}= {key}")
                ax.set_ylabel("Normalized abundance")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                
                
        row = row + 1 # Add 1, because we've filled 1 row with fingerprints
        
        # Row 2, column 0: Scatter: absolute error
        ax1 = fig.add_subplot(gs[row, 0])
        norm = Normalize(vmin=df['param_value'].min(), vmax=df['param_value'].max())
        colors = [c_map(norm(v)) for v in param_values]
        ax1.scatter(param_values, df['abs_error'], c=colors, edgecolor='k', 
                    linewidth=0.3, s=60)
        ax1.set_title(metric_ylabels['abs_error'])
        ax1.set_xlabel(scenario_xlabels[scenario])
        ax1.set_ylabel("∑|e_true - e_post|")
        
        # Row 2, column 1: Scatter: Trace(C_m)
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.scatter(param_values, df['trace_cov'], c=colors, edgecolor='k', 
                    linewidth=0.3, s=60)
        ax2.set_title(metric_ylabels['trace_cov'])
        ax2.set_xlabel(scenario_xlabels[scenario])
        ax2.set_ylabel("Trace(Cm_post)")
        
        # Row 2, column 2: Scatter: trade-off model misfit - data misfit
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.scatter(df['data_misfit'], df['model_misfit'], c=colors, 
                    edgecolor='black', linewidth=0.3, s = 60)
        ax3.set_title("Trade-off")
        ax3.set_xlabel("Data Misfit")
        ax3.set_ylabel("Model Misfit")
        
        for ax in [ax1, ax2, ax3]:
            ax.set_box_aspect(1)
            
        # Highlight and annotate the 3 selected param_values
        if scenario == 'geo_diff':
            highlight_vals = param_values.to_list()
            highlight_labels = geo_labels
        else:
            highlight_vals = keys3
            highlight_labels = highlight_vals
        highlight_color = 'none'
        highlight_edge = 'black'
        highlight_size = 120
        highlight_linewidth = 1.2
        
        # Scatter on each of the 3 subplots
        for val_i, val in enumerate(highlight_vals):
            # Find corresponding row in df
            subset = df[df['param_value'] == val]
            if subset.empty:
                continue
            row_data = subset.iloc[0]
            
            # Big hollow circle on abs_error
            ax1.scatter(val, row_data['abs_error'], facecolor=highlight_color,
                        edgecolor=highlight_edge, s=highlight_size, linewidth=highlight_linewidth)
            wrapped_label = "\n".join(textwrap.wrap(str(highlight_labels[val_i]), width=20))
            ax1.annotate(wrapped_label, (val, row_data['abs_error']), 
                         textcoords="offset points", 
                         xytext=(0, 8), ha='center', fontsize=10, 
                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                   edgecolor='none', alpha=0.7))
        
            # Big hollow circle on trace_cov
            ax2.scatter(val, row_data['trace_cov'], facecolor=highlight_color,
                        edgecolor=highlight_edge, s=highlight_size, linewidth=highlight_linewidth)
            # ax2.annotate(f"{highlight_labels[val_i]}", (val, row_data['trace_cov']), textcoords="offset points",
            #              xytext=(0, 8), ha='center', fontsize=10)
        
            # Big hollow circle on trade-off plot
            ax3.scatter(row_data['data_misfit'], row_data['model_misfit'], 
                        facecolor=highlight_color, edgecolor=highlight_edge, 
                        s=highlight_size, linewidth=highlight_linewidth)
            # ax3.annotate(f"{highlight_labels[val_i]}", (row_data['data_misfit'], row_data['model_misfit']), 
            #              textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
        
        # Row 2, column 3: Colorbar
        # -- Add shared colorbar --
        sm = ScalarMappable(norm=norm, cmap=c_map)
        sm.set_array([])
        fig.colorbar(sm, ax=ax3, label=scenario_titles[scenario],fraction=0.046, pad=0.02)
        
        row = row + 1
        
        row = row + 1
    
    fig.savefig(f"{OUTPUT_PATH}/summary_scenario_sweeps.pdf")
    fig.show()
# -----------------------------------------------------------------------------
# _____________________________________________________________________________
# MAIN BLOCK
# 1. Load maps and data
OUTPUT_PATH = create_output_path(OUTPUT_FOLDER, savefolder)
B, R, crs_raster = load_geology('geology')
DEM, vel_obs = load_dem_and_velocity() if flag_uniformprior != 1 else (None, None)
shape_gdf = gpd.read_file(SHAPEFILE_PATH).to_crs(crs_raster)

source, top_k_idx = load_source_data(n_minerals_to_keep)
with open(f'{INPUT_FOLDER}/XRD_data/SourceData/source_cleaned.pkl', 'rb') as f:
     source_data = pickle.load(f)

n_tracers = source.shape[0]

if flag_test_optimal_source == 1:
    source, top_k_idx = generate_distinct_source(n_minerals=n_tracers, n_lithologies=source.shape[1])
    source_data['source'] = source
    source_data['Mineral names (rows)'] = [ltr[i+1] for i in range(n_tracers)]


# 2. Set up model parameters
sigma_cov, L, d_sigma_val, n_m, mu, misfit, d_misfit, m_misfit = setup_model_params()


# 3. Set up grid and observed data
ny, nx, nn, dx, dy, Lx, Ly, dA, X2, Y2 = setup_grid_dimensions(B, R)
mask = setup_observed_data(B)

# 4. Set up the forward-inverse matrices for the inversion
# Masked coordinates of active cells
active_idx = np.where(mask.flatten() > 0)[0]
x = X2.flatten()[active_idx]
y = Y2.flatten()[active_idx]
mask_size = len(x)

# 5. Generate synthetic erosion rate map
e_true_xy = create_synthetic_erosion_map_hotspots(B, R)
e_true_v = e_true_xy.flatten()[active_idx]

etot = sum(e_true_v)

data_sigma = None # measured data_sigma

# 6. Generate all required matrices
A = generate_A(B, mask, source, n_tracers)
cov = generate_cov(x, y, sigma_cov=sigma_cov, L=L, nugget=1e-2)
cov_d = generate_cov_d(data_sigma, d_sigma_val, flag_sigma_d_meas, n_tracers)

if flag_uniformprior == 1:
    eps_prior = compute_eps_prior_uniform(etot, mask)
else:
    e_prior = 2.7e-4 * np.power(vel_obs,2.02)
    e_prior[e_prior == 0] = 0.001
    eps_prior_xy = np.log(e_prior)
    eps_prior = eps_prior_xy.flatten()[active_idx]
    
# 7. Generate synthetic data
eps_true = np.log(e_true_v)
d = A @ np.exp(eps_true)

# 8. Set-up figure environment
cmap = cm.viridis
extent = [
    R.c,  # left (x min)
    R.c + R.a * B.shape[1],  # right (x max)
    R.f + R.e * B.shape[0],  # bottom (y min)
    R.f  # top (y max)
]


# 9. Inversion
d_misfit, m_misfit, edot_post, edot_posts, Cm_post, resolution, Cm_posts = nonlinear_iteration(A, 
                                                                         cov, 
                                                                         cov_d, 
                                                                         eps_prior, 
                                                                         mu, 
                                                                         100, 
                                                                         etot,
                                                                         d, 
                                                                         mask_size, 
                                                                         flag_firstprior, 
                                                                         flag_quasi_newton)

# 10. Plotting and saving
ratio = np.sqrt(np.diag(Cm_post))/np.sqrt(np.diag(cov))
plot_true_and_posterior_erosion_maps(R, 
                                     e_true_xy, 
                                     edot_posts, 
                                     resolution, 
                                     ratio, 
                                     mask, ny, nx, 
                                     shape_gdf, 
                                     OUTPUT_PATH, savename)

save_settings_and_params(n_minerals_to_keep,
                         flag_uniformprior,
                         flag_firstprior,
                         flag_quasi_newton,
                         sigma_cov,
                         d_sigma_val,
                         n_m,mu,
                         OUTPUT_PATH,savename)

e_post = v_to_xy(edot_posts[-1], mask.flatten(), ny, nx, zero_val=np.nan)

generate_summary_figure(
    R, e_true_xy, e_true_v, e_post, hillshade_path, glacier_shapefile, geology_shapefile,
    source_data, lithology_colors,
    d_misfit, m_misfit, edot_posts, Cm_posts, resolution, ratio,
    savename)


# 10. Test scenarios
if flag_test_scenarios == 1:
    all_results = []
    source_matrices = {}
    for param_to_test in ['geo_diff','avg_js_dist','n_tracers']:
        if param_to_test == "geo_diff":
            values = ['geology','geology_test','geology_steck']
            labels_geo = {'geology':'Original geology (same as in forward)',
                          'geology_test': 'Geology with minimal changes',
                          'geology_steck': 'Geology by Steck et al. 2015'}
            geo_maps = []
            geo_labels = []
            for loadname in values:
                geology = gpd.read_file(f'{INPUT_FOLDER}/{loadname}.gpkg')
                geo_maps.append(geology)
                geo_labels.append(labels_geo[loadname])
                
        elif param_to_test == "avg_js_dist":
            values = np.linspace(0, 0.7,20)
            
        elif param_to_test == "n_tracers":
            values = np.arange(1,n_tracers+1,1, dtype=int)
        
        if param_to_test == "geo_diff":
            df_result = parameter_sweep(param_to_test, values, d,
                                e_true_xy, etot, source, B, A, x, y, 
                                cov, cov_d, eps_prior, mu, n_m, mask_size, 
                                flag_firstprior, flag_quasi_newton)
        else:
            source_matrices[param_to_test], df_result = parameter_sweep(param_to_test, values, d,
                                e_true_xy, etot, source, B, A, x, y, 
                                cov, cov_d, eps_prior, mu, n_m, mask_size, 
                                flag_firstprior, flag_quasi_newton)
            
        all_results.append(df_result)
        
    all_results_df = pd.concat(all_results, ignore_index = True)
    
    watershed = gpd.read_file(watershed_shapefile)
    geo_maps_clipped = [gpd.clip(gdf, watershed) for gdf in geo_maps]
    
    plot_scenarios(all_results_df, source_matrices, geo_maps_clipped, geo_labels, 
                   lithology_colors)
    all_results_df.to_csv(f"{OUTPUT_PATH}/summary_scenario_sweeps.csv")

if flag_test_parameters == 1:
    all_results = []
    for param_to_test in ['L','sigma_cov','d_sigma_val','mu', 'n_m']:
        if param_to_test == "L":
            values = np.linspace(200, 3000, 20)
        elif param_to_test == "sigma_cov":
            values = np.logspace(-3, 1, 20)
        elif param_to_test == "d_sigma_val":
            values = np.logspace(-3, -0.05, 20)
        elif param_to_test == "mu":
            values = np.round(np.logspace(-2,-0.5,10),3)
        elif param_to_test == "n_m":
            values = np.arange(5,200,10)
    
        df_result = parameter_sweep(param_to_test, values, d,
                            e_true_xy, etot, source, B, A, x, y, 
                            cov, cov_d, eps_prior, mu, n_m, mask_size, 
                            flag_firstprior, flag_quasi_newton)
        all_results.append(df_result)
        
    all_results_df = pd.concat(all_results, ignore_index = True)
    plot_all_sweeps(all_results_df)
    all_results_df.to_csv(f"{OUTPUT_PATH}/hyperparameter_sweeps.csv")

if flag_test_SD == 1:
    d_misfit, m_misfit, edot_post, edot_posts, Cm_post, resolution, Cm_posts = nonlinear_iteration(A, 
                                                                             cov, 
                                                                             cov_d, 
                                                                             eps_prior, 
                                                                             0.5, 
                                                                             100, 
                                                                             etot,
                                                                             d, 
                                                                             mask_size, 
                                                                             flag_firstprior, 
                                                                             0)

    ratio = np.sqrt(np.diag(Cm_post))/np.sqrt(np.diag(cov))
    plot_true_and_posterior_erosion_maps(R, 
                                         e_true_xy, 
                                         edot_posts, 
                                         resolution, 
                                         ratio, 
                                         mask, ny, nx, 
                                         shape_gdf, 
                                         OUTPUT_PATH, 'SD_test')

    e_post = v_to_xy(edot_posts[-1], mask.flatten(), ny, nx, zero_val=np.nan)

    generate_summary_figure(
        R, e_true_xy, e_true_v, e_post, hillshade_path, glacier_shapefile, geology_shapefile,
        source_data, lithology_colors,
        d_misfit, m_misfit, edot_posts, Cm_posts, resolution, ratio,
        'SD_test')

