#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 18:54:03 2025

@author: fiendedoncker
"""
# 1. Imports, pathnames and constants
# 1.1. Imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pickle

from pathlib import Path

import os

# 1.2. Paths
INPUT_FOLDER = '/Users/fiendedoncker/Documents/Gorner_new/INPUT/XRD data/RUFF Files'  # Path to the input data folder
OUTPUT_FOLDER = '/Users/fiendedoncker/Documents/Gorner_new/OUTPUT' 

# 1.3. List of mineral names
names_minerals = [
    'Tremolite', 'Talc', 'Quartz', 'Pyroxene', 'Phlogopite',
    'Paragonite', 'Omphacite', 'Muscovite', 'Microcline', 'Magnesite',
    'Glaucophane', 'Fayalite', 'Dolomite', 'Diopside', 'Chrysotile',
    'Calcite', 'Brucite', 'Antigorite', 'Anorthite', 'Annite',
    'Ankerite', 'Almandine', 'Albite'
]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2. Functions
def create_output_path(base_output_folder, savename):
    full_output_path = os.path.join(base_output_folder, savename)
    os.makedirs(full_output_path, exist_ok=True)
    return full_output_path

def get_mineral_data(mineral_name,i):
    th2_list = []
    mineral_idx_list = []
    mineral_list = []
    
    # Get RUFF file of this mineral
    filepath = Path(f"{INPUT_FOLDER}/{mineral_name}.txt")
    
    if not filepath.exists():
        print(f"File not found: {filepath}")

    # Load mineral data
    mineral = np.loadtxt(filepath)
    th2data = mineral[:, 0]
    intensity = mineral[:, 1]

    # Find 20 highest intensities with their corresponding 2 theta and intensity values
    indx = np.argsort(intensity)
    intpeaks = intensity[indx][-20:][::-1]  # reverse for descending
    th2peaks = th2data[indx][-20:][::-1]

    # Store the main peak
    th2_list.append(th2peaks[0])
    mineral_idx_list.append(i)
    mineral_list.append(mineral_name)

    # Process additional peaks: only store them if they are at least 2° away from other peak
    rounded_th2 = np.round(th2peaks)
    for unique_val in np.unique(rounded_th2):
        matches = np.where(rounded_th2 == unique_val)[0]
        th2_mp = th2peaks[matches]
        int_mp = intpeaks[matches]

        th2peak_i = th2_mp[0]
        intpeak_i = int_mp[0]

        if abs(th2peak_i - th2peaks[0]) > 2 and intpeak_i > intpeaks[0] / 4:
            th2_list.append(th2peak_i)
            mineral_idx_list.append(i)
            mineral_list.append(mineral_name)

    # Plot XRD spectrum
    plt.figure()
    plt.plot(th2data, intensity, label='Spectrum')
    plt.plot(th2peaks, intpeaks, 'o', label='Top Peaks')
    plt.title(mineral_name)
    plt.xlabel("2θ")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return th2_list, mineral_idx_list, mineral_list

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Set output path
OUTPUT_PATH = create_output_path(OUTPUT_FOLDER, 'MineralData')

# Initialize a list to collect rows of data
th2_list_full = []
mineral_idx_list_full = []
mineral_list_full = []

for i, mineral_name in enumerate(names_minerals):
    th2_i, min_idx_i, min_list_i = get_mineral_data(mineral_name,i)
    
    th2_list_full.extend(th2_i)
    mineral_idx_list_full.extend(min_idx_i)
    mineral_list_full.extend(min_list_i)
    
# Convert to DataFrame
elements_df = pd.DataFrame({'2theta': th2_list_full, 
                           'Mineral_Index':mineral_idx_list_full,
                           'Mineral_Name':mineral_list_full})

# Remove rows with 0 2-theta (if any)
elements_df = elements_df[elements_df['2theta'] != 0]

# Save everything to a pickle file
output = {
    'names_minerals': names_minerals,
    'elements': elements_df
}

with open(f'{OUTPUT_PATH}/minerals_clean.pkl', 'wb') as f:
    pickle.dump(output, f)

print("Data saved to 'minerals_clean.pkl'")

# Do the same for silver
# Initialize a list to collect rows of data

th2_i, min_idx_i, min_list_i = get_mineral_data('Silver',i)

# Convert to DataFrame
elements_df = pd.DataFrame({'2theta': th2_i, 
                           'Mineral_Index': min_idx_i,
                           'Mineral_Name':min_list_i})

# Remove rows with 0 2-theta (if any)
elements_df = elements_df[elements_df['2theta'] != 0]

# Save everything to a pickle file
output = {
    'elements': elements_df
}

with open(f'{OUTPUT_PATH}/silver.pkl', 'wb') as f:
    pickle.dump(output, f)

print("Data saved to 'silver.pkl'")