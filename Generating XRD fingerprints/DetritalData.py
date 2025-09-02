#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 18:11:15 2025

@author: fiendedoncker
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 20:25:56 2025

@author: fiendedoncker
"""
# 1. Imports, pathnames and constants
# 1.1. Imports
import numpy as np

import pandas as pd

import os

import re

import pickle

import matplotlib.pyplot as plt

from glob import glob

# 1.2. Paths
INPUT_FOLDER = '/Users/fiendedoncker/Documents/Gorner_new/INPUT/XRD data/Detrital data'  # Path to the input data folder
OUTPUT_FOLDER = '/Users/fiendedoncker/Documents/Gorner_new/OUTPUT' 
MINERAL_FOLDER = '/Users/fiendedoncker/Documents/Gorner_new/OUTPUT/MineralData'
SOURCE_FOLDER = '/Users/fiendedoncker/Documents/Gorner_new/OUTPUT/SourceData'
SAMPLE_INFO_PATH = '/Users/fiendedoncker/Documents/Gorner_new/INPUT/XRD data/SampleName_Load_AreaXRD.csv'

# 1.3. Parameters etc.
searchradius = 0.25 # degrees ± window

with open(f'{MINERAL_FOLDER}/silver.pkl', 'rb') as f:
    silver_data = pickle.load(f)
silver_df = silver_data['elements']   
silver_2theta = silver_df['2theta'].iloc[0] # 2 theta angle used for peak alignment (silver alignment)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2. Functions
def create_output_path(base_output_folder, savename):
    full_output_path = os.path.join(base_output_folder, savename)
    os.makedirs(full_output_path, exist_ok=True)
    return full_output_path

def get_filenames(INPUT_FOLDER):
    all_files = os.listdir(INPUT_FOLDER)
    matching_files = [f for f in all_files 
                      if re.search(r'pf', f, re.IGNORECASE) and f.lower().endswith('.txt')
                      ]
    cleansample_names = [re.sub(r'_pf_.*$', '', f, flags=re.IGNORECASE) 
                         for f in matching_files]
    
    return matching_files, cleansample_names


def load_xrd_data_and_get_column_indices(text):
    lines = text.strip().splitlines()

    # Locate header line
    for i, line in enumerate(lines):
        if line.strip().startswith("Position"):
            header_line = lines[i]
            data_start_index = i + 2
            break
    else:
        raise ValueError("Header line not found")

    # Parse header
    header = [h.strip() for h in header_line.split() if h.strip()]
    col_map = {name: idx for idx, name in enumerate(header)}
    num_columns = len(header)

    # Required columns
    required = ['Position', 'Intensity', 'Rel.Int.', 'FWHM(L)', 'Area']
    for name in required:
        if name not in col_map:
            raise ValueError(f"Column '{name}' not found in header")

    # Parse data
    data = []
    for line in lines[data_start_index:]:
        if not line.strip():
            continue

        tokens = line.strip().split()
        if len(tokens) < num_columns:
            continue

        row = []
        for i in range(num_columns):
            token = tokens[i]
            try:
                val = float(token) if token.lower() != "none" else np.nan
            except ValueError:
                val = np.nan  # safely fill non-numeric values with NaN
            row.append(val)
        data.append(row)

    data_array = np.array(data)
    return data_array, col_map


def align_peaks_to_silver(sample_data, silver_ref=silver_2theta, searchrad=0.25, theta_col=0, area_col=8):
    # Find the silver peak in sample_data and calculate 2theta shift
    # Align all sample 2theta values by subtracting the silver shift.

    th2 = sample_data[:, theta_col]
    area = sample_data[:, area_col]  


    # Search for peaks near silver_ref
    silver_mask = (th2 >= silver_ref - searchrad) & (th2 <= silver_ref + searchrad)
    if not np.any(silver_mask):
        # No silver peak found, no alignment
        shift = 0
    else:
        silver_peak_pos = th2[silver_mask][np.argmax(area[silver_mask])]
        shift = silver_peak_pos - silver_ref

    # Apply shift
    sample_data[:, theta_col] = th2 - shift
    return sample_data, shift

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 3. Analysis
# 3.1. Load mineral reference data and get source maximal areas
with open(f'{MINERAL_FOLDER}/minerals_clean.pkl', 'rb') as f:
    mineral_data = pickle.load(f)

elements_df = mineral_data['elements']  # DataFrame with '2theta', 'Mineral_Index', 'Mineral_Name'
names_minerals = mineral_data['names_minerals']

with open(f'{SOURCE_FOLDER}/SourceData.pkl', 'rb') as f:
    source_data = pickle.load(f)
max_area_per_mineral = source_data['max_area_per_mineral'] # to normalize area data


# 3.2. Initialize
num_minerals = max(elements_df['Mineral_Index']) + 1

df_sample_info       = pd.read_csv(SAMPLE_INFO_PATH) # csv sorted by date containing filenames, dates, load
detritalsample_names = df_sample_info['Full name'] # txt file name
dates                = df_sample_info['Date time'] # sampling date time
clean_names          = df_sample_info['Name'] # sample name
load                 = df_sample_info['Load'] # load of sample (mg)
    
num_samples = len(detritalsample_names)

area_matrix = np.zeros((num_minerals, num_samples))  # to store final normalized data

# 3.3. Process each sample
for sample_i, sample_name in enumerate(detritalsample_names):
    # 3.3.1. Load sample data
    with open(f"{INPUT_FOLDER}/{sample_name}", "r") as f:
        text = f.read()

    sample_data, columns = load_xrd_data_and_get_column_indices(text)

    # 3.3.2. Align sample data 2theta with silver peak
    sample_data, silver_shift = align_peaks_to_silver(sample_data, 
                                                      silver_ref=silver_2theta, 
                                                      searchrad=searchradius,
                                                      theta_col=columns['Position'],
                                                      area_col=columns['Area'])

    # 3.3.3. Fill dictionary with area for all different 2theta angles
    areas_by_mineral = {}

    for _, row in elements_df.iterrows():
        # Get 2 theta of this mineral where its intensity peaks (some minerals have multiple angles, so will span multiple rows)
        th2_peak = row['2theta']
        mineral_idx = row['Mineral_Index']
        
        # Look in window of +- search radius around the 2 theta angle
        rounded_target = round(th2_peak / searchradius) * searchradius
        th2_source = sample_data[:, columns['Position']]
        round_down = np.floor(th2_source / searchradius) * searchradius
        round_up   = np.ceil(th2_source / searchradius) * searchradius
        
        # Get the index/indices where the angle matches the 2 theta angle of the mineral
        matches = np.where((round_down == rounded_target) | (round_up == rounded_target))[0]
        if len(matches) == 0:
            continue
        matched = sample_data[matches]
        
        # Get the peak area and the FWMH for this 2 theta
        area_vals = matched[:, columns['Area']]     
        
        # Pick the peak with the highest corrected area
        best_idx = np.argmax(area_vals)

        area_best = area_vals[best_idx]

        areas_by_mineral.setdefault(mineral_idx, []).append(area_best)

    # Average area per mineral (some minerals have multiple angles, so take avg area for different peaks)
    for mineral_idx, area_list in areas_by_mineral.items():
        area_matrix[mineral_idx, sample_i] = np.mean(area_list)


# 3.4. Divide by maximal area per mineral to scale from 0 to 1
# For each mineral, get the max area
rel_area_detrital = area_matrix / max_area_per_mineral[:, np.newaxis]

max_area_per_sample = np.max(area_matrix, axis=0) 
rel_area_sample_detrital = area_matrix / max_area_per_sample

# Identify indices for brucite and calcite
idx_brucite = names_minerals.index('Brucite')
idx_calcite = names_minerals.index('Calcite')

# Cap all values > 1 to 1 except brucite and calcite
for i in range(rel_area_detrital.shape[0]):
    if i not in (idx_brucite, idx_calcite):
        rel_area_detrital[i] = np.clip(rel_area_detrital[i], 0, 1)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 4. Saving
# Set output path
OUTPUT_PATH = create_output_path(OUTPUT_FOLDER, 'DetritalData')

# Save everything to a pickle file
output = {
    'file_names': detritalsample_names,
    'names': clean_names,
    'dates': dates,
    'load': load,
    'Detrital: XRD peak area per mineral per detrital sample (not normalised)': area_matrix,
    'data_percentage': rel_area_detrital,
    'data_percentage_per_sample': rel_area_sample_detrital,
    'Mineral names (rows)': names_minerals
}

with open(f'{OUTPUT_PATH}/DetritalData.pkl', 'wb') as f:
    pickle.dump(output, f)

print("Data saved to 'DetritalData.pkl'")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Remove brucite and calcite from both source and detrital
remove_indices = sorted([idx_brucite, idx_calcite], reverse=True)

# Remove from detrital data
for idx in remove_indices:
    rel_area_detrital = np.delete(rel_area_detrital, idx, axis=0)
    rel_area_sample_detrital = np.delete(rel_area_sample_detrital, idx, axis=0)
    area_matrix = np.delete(area_matrix, idx, axis=0)
    names_minerals.pop(idx)

# Remove from source data
source = source_data['source']
source_rel = source_data['source_rel_sample']
max_area = source_data['max_area_per_mineral']

for idx in remove_indices:
    source = np.delete(source, idx, axis=0)
    source_rel = np.delete(source_rel, idx, axis=0)
    max_area = np.delete(max_area, idx)
    source_data['Mineral names (rows)'].pop(idx)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save cleaned source and detrital data
# Save cleaned source
source_cleaned = source_data.copy()
source_cleaned['source'] = source
source_cleaned['source_rel_sample'] = source_rel
source_cleaned['max_area_per_mineral'] = max_area

with open(f'{SOURCE_FOLDER}/source_cleaned.pkl', 'wb') as f:
    pickle.dump(source_cleaned, f)

# Save cleaned detrital
detrital_cleaned = output.copy()
detrital_cleaned['data_percentage'] = rel_area_detrital
detrital_cleaned['data_percentage_per_sample'] = rel_area_sample_detrital
detrital_cleaned['Detrital: XRD peak area per mineral per detrital sample (not normalised)'] = area_matrix
detrital_cleaned['Mineral names (rows)'] = names_minerals

with open(f'{OUTPUT_FOLDER}/DetritalData/detrital_cleaned.pkl', 'wb') as f:
    pickle.dump(detrital_cleaned, f)

print("Cleaned source and detrital data saved.")
