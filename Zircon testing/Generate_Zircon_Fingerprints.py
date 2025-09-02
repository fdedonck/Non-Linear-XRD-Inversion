#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 01:23:34 2025

@author: fiendedoncker
"""

import numpy as np
import scipy.io as sio

# Set folders
INPUT_PATH = '/Users/fiendedoncker/Documents/Gorner_new/INPUT/Data Bruno/Data_to_Generate_Fingerprints'
OUTPUT_PATH = '/Users/fiendedoncker/Documents/Gorner_new/INPUT/Data Bruno/Data_to_Generate_Fingerprints/New_Fingerprints'
# === CONFIG ===
use_comb = False
indices_comb = [0, 4]  # Python uses 0-based indexing
bins = np.array([
    [40, 140], [140, 200], [200, 210], [210, 250], [250, 263],
    [263, 270], [270, 340], [340, 430], [430, 475], [475, 555],
    [555, 585], [585, 750], [750, 800]
])
numbins = bins.shape[0]

# === LOAD DATA ===
data_bedrock = sio.loadmat(f'{INPUT_PATH}/data_bedrock.mat')['data_bedrock'][0]
names_bedrock = sio.loadmat(f'{INPUT_PATH}/names_bedrock.mat')['names_bedrock']
data_sediments = sio.loadmat(f'{INPUT_PATH}/data_sediments.mat')['data_sediments'][0]
names_sediments = sio.loadmat(f'{INPUT_PATH}/names_sediments.mat')['names_sediments']
Concentrations = sio.loadmat(f'{INPUT_PATH}/Concentrations.mat')
C = Concentrations['C']
Conc = np.array([float(cell[0][0]) for cell in C[1]])
sedConcentrations = sio.loadmat(f'{INPUT_PATH}/sedConcentrations.mat')
sedC = sedConcentrations['sedC']
sedConc = np.array([float(cell[0][0]) for cell in sedC[1]])

if not use_comb:
    data_sediments = np.delete(data_sediments, indices_comb)
    names_sediments = np.delete(names_sediments, indices_comb)
    sedConc = np.delete(sedConc, indices_comb)

# === BINNING ===
def bin_counts(sample_list, bins):
    num_bins = bins.shape[0]
    num_samples = len(sample_list)
    percentages = np.zeros((num_bins, num_samples))
    for i, sample in enumerate(sample_list):
        sample = np.sort(sample, axis=0)
        for k in range(num_bins):
            lower, upper = bins[k]
            count = np.sum((sample[:, 0] >= lower) & (sample[:, 0] < upper))
            percentages[k, i] = count / len(sample)
    return percentages

# === APPLY BINNING AND SCALE BY CONCENTRATION ===
source_percentage = bin_counts(data_bedrock, bins) * Conc
data_percentage = bin_counts(data_sediments, bins) * sedConc

# === REORDER COLUMNS ACCORDING TO GEOLOGICAL MAP ===
# Original reorder = [4 7 1 5 3 2] in MATLAB -> 0-based: [3, 6, 0, 4, 2, 1]
sp_temp = np.zeros((numbins, 8))  # Add 8 columns to handle GR04 and MRg-g special cases
sp_temp[:, 0] = source_percentage[:, 2]  # 3rd col (index 2)
sp_temp[:, 1] = source_percentage[:, 5]  # 6th col
sp_temp[:, 2] = source_percentage[:, 4]  # 5th col
sp_temp[:, 3] = source_percentage[:, 0]  # 1st col
sp_temp[:, 4] = 0                         # GR04 -> no zircons
sp_temp[:, 5] = source_percentage[:, 0]  # MR g-g = granite
sp_temp[:, 6] = source_percentage[:, 1]  # 2nd col
source_percentage = sp_temp[:, :7]

# === CROSS-SAMPLE NORMALIZATION ===
max_vals = np.max(source_percentage, axis=1, keepdims=True)
max_vals[max_vals == 0] = 1e-5  # avoid divide-by-zero

source_crossnorm = source_percentage / max_vals
data_crossnorm = data_percentage / max_vals

# === STATS ===
data_sigma = np.sqrt(np.var(data_crossnorm, axis=1))

# === SAVE IF NEEDED ===
sio.savemat(f'{OUTPUT_PATH}/source_B.mat', {'source_percentage': source_crossnorm})
sio.savemat(f'{OUTPUT_PATH}/DataSamples_B.mat', {'data_percentage': data_crossnorm, 'data_sigma': data_sigma})
sio.savemat(f'{OUTPUT_PATH}/datenames.mat', {'datenames': names_sediments})
