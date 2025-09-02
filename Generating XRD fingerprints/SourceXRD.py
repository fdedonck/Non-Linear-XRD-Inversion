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
import pickle
import matplotlib.pyplot as plt
from glob import glob

# 1.2. Paths
INPUT_FOLDER = '/Users/fiendedoncker/Documents/Gorner_new/INPUT/XRD data/Source data'  # Path to the input data folder
OUTPUT_FOLDER = '/Users/fiendedoncker/Documents/Gorner_new/OUTPUT' 
MINERAL_FOLDER = '/Users/fiendedoncker/Documents/Gorner_new/OUTPUT/MineralData'

# 1.3. Parameters etc.
searchradius = 0.25 # degrees ± window
quartz_2theta = 26.65 # 2 theta angle used for peak alignment (quartz alignment)

# Names of samples representative of different source areas
sourcesample_names = [
    'GS_01', 'GS_02', 'GS_03', 'GS_05', 'GS_06',
    'GS_08', 'GR_02', 'GR_03', 'GR_04', 'GR_07', 'GR_mica'
]

# Corresponding geology for every source sample
geology_index = [7, 4, 4, 5, 3, 2, 4, 4, 5, 1, 6]

# Corresponding name to geology index
geology_names = {1:"ZSF ophiolites (serpentinites)",
                 2:"ZSF sediments",
                 3: "Stockhorn, Tuftgrat, Gornergrat",
                 4: "Monte Rosa (granite)",
                 5: "ZSF ophiolites (metabasites, eclogites)",
                 6:"Monte Rosa (gneiss, micaschist)",
                 7: "Furgg series"}

# Colors for each lithology
lithology_colors = {
    "ZSF ophiolites (serpentinites)": "#e3dd19",
    "ZSF sediments": "#809847",
    "Stockhorn, Tuftgrat, Gornergrat": "#ff5001",
    "Monte Rosa (granite)": "#fa9b9a",
    "ZSF ophiolites (metabasites, eclogites)": "#32a02d",
    "Monte Rosa (gneiss, micaschist)": "#ff7f41",
    "Furgg series": "#badd68"
}


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2. Functions
def create_output_path(base_output_folder, savename):
    full_output_path = os.path.join(base_output_folder, savename)
    os.makedirs(full_output_path, exist_ok=True)
    return full_output_path

def load_xrd_data_and_get_column_indices(text):
    lines = text.strip().splitlines()

    # Locate the header line (starts with 'Position')
    for i, line in enumerate(lines):
        if line.strip().startswith("Position"):
            header_line = lines[i]
            data_start_index = i + 2  # skip units line
            break
    else:
        raise ValueError("Header line not found")

    # Parse header columns (ignoring empty fields)
    header = [h.strip() for h in header_line.split(",") if h.strip()]

    # Parse data lines: collect rows of floats
    data = []
    for line in lines[data_start_index:]:
        if not line.strip():
            continue
        values = [v.strip() for v in line.split(",") if v.strip()]
        try:
            float_values = list(map(float, values))
            data.append(float_values)
        except ValueError:
            continue  # Skip any non-numeric line

    # Convert to numpy array
    data_array = np.array(data)

    # Map column names to indices
    col_map = {}
    for name in ['Position', 'Intensity', 'Rel.Int.', 'FWHM(L)', 'Area']:
        for idx, col in enumerate(header):
            if col == name:
                col_map[name] = idx
                break
        else:
            raise ValueError(f"Column '{name}' not found in header")

    return data_array, col_map

def align_peaks_to_quartz(sample_data, quartz_ref=quartz_2theta, searchrad=0.25, theta_col=0, area_col=8):
    # Find the quartz peak in sample_data and calculate 2theta shift
    # Align all sample 2theta values by subtracting the quartz shift.

    th2 = sample_data[:, theta_col]
    area = sample_data[:, area_col]  


    # Search for peaks near quartz_ref
    quartz_mask = (th2 >= quartz_ref - searchrad) & (th2 <= quartz_ref + searchrad)
    if not np.any(quartz_mask):
        # No quartz peak found, no alignment
        shift = 0
    else:
        quartz_peak_pos = th2[quartz_mask][np.argmax(area[quartz_mask])]
        shift = quartz_peak_pos - quartz_ref

    # Apply shift
    sample_data[:, theta_col] = th2 - shift
    return sample_data, shift

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 3. Analysis
# 3.1. Load mineral reference data 
with open(f'{MINERAL_FOLDER}/minerals_clean.pkl', 'rb') as f:
    mineral_data = pickle.load(f)

elements_df = mineral_data['elements']  # DataFrame with '2theta', 'Mineral_Index', 'Mineral_Name'
names_minerals = mineral_data['names_minerals']


# 3.2. Initialize
num_minerals = max(elements_df['Mineral_Index']) + 1
num_samples = len(sourcesample_names)
source_data_corrected = np.zeros((num_minerals, num_samples))  # store final normalized data


# 3.3. Process each sample
for sample_i, sample_name in enumerate(sourcesample_names):
    # 3.3.1. Load sample data
    with open(f"{INPUT_FOLDER}/{sample_name}_RT_PF.txt", "r") as f:
        text = f.read()

    sample_data, columns = load_xrd_data_and_get_column_indices(text)

    # 3.3.2. Align sample data 2theta with quartz peak
    sample_data, quartz_shift = align_peaks_to_quartz(sample_data, 
                                                      quartz_ref=quartz_2theta, 
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
        round_up = np.ceil(th2_source / searchradius) * searchradius
        
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

    # Average area per mineral
    for mineral_idx, area_list in areas_by_mineral.items():
        source_data_corrected[mineral_idx, sample_i] = np.mean(area_list)

    # Optional: Plot aligned peaks
    plt.figure()
    plt.plot(sample_data[:, columns['Position']], sample_data[:, columns['Area']], 'x', label='Area')
    plt.plot(elements_df['2theta'], [max(sample_data[:, columns['Area']])] * len(elements_df),
             'k|', label='Mineral 2θ Peaks')
    plt.title(f"{sample_name} (Quartz shift applied: {quartz_shift:.3f} deg)")
    plt.xlabel('2θ (deg)')
    plt.ylabel('Area')
    plt.legend()
    plt.grid(True)
    plt.show()

# 3.4. Merge by geology/lithology (multiple samples per litho)
max_area_per_mineral = np.max(source_data_corrected, axis=1)  # shape: (num_minerals,)

num_geologies = len(set(geology_index))
source = np.zeros((num_minerals, num_geologies))

for geo_idx in set(geology_index):
    sample_indices = [i for i, val in enumerate(geology_index) if val == geo_idx]
    if len(sample_indices) == 1:
        source[:, geo_idx - 1] = source_data_corrected[:, sample_indices[0]]
    else:
        source[:, geo_idx - 1] = np.mean(source_data_corrected[:, sample_indices], axis=1)

# 3.4. Divide by maximal area per mineral to scale from 0 to 1
# For each mineral, get the max area
rel_area_source = source / max_area_per_mineral[:, np.newaxis]

max_area_per_sample = np.max(source, axis=0) 
rel_area_sample_source = source / max_area_per_sample

# 3.5 Plot
# Indices of spotlight samples in data_percentage
spotlight_indices = [58]  # Change as needed

for i, lith_idx in enumerate(geology_names):
    lith = geology_names[lith_idx]
    color = lithology_colors[lith]
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(range(num_minerals), rel_area_source[:, i], color=color, label=f"{lith} (source)")
    
    ax.set_ylabel('Source Mineral Concentration (%)', color=color)
    ax.set_ylim(0,1)
    ax.tick_params(axis='y', labelcolor=color)
    
    ax.set_xticks(range(23))
    ax.set_xticklabels(names_minerals, rotation=90)
    
    plt.title(f'Mineral Signature: {lith}')
    
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()
    
for i, lith_idx in enumerate(geology_names):
    lith = geology_names[lith_idx]
    color = lithology_colors[lith]
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(range(num_minerals), rel_area_sample_source[:, i], color=color, label=f"{lith} (source)")
    
    ax.set_ylabel('Source Mineral Concentration (%)', color=color)
    ax.set_ylim(0,1)
    ax.tick_params(axis='y', labelcolor=color)
    
    ax.set_xticks(range(23))
    ax.set_xticklabels(names_minerals, rotation=90)
    
    plt.title(f'Mineral Signature (relative per sample): {lith}')
    
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 4. Saving
# Set output path
OUTPUT_PATH = create_output_path(OUTPUT_FOLDER, 'SourceData')

# Save everything to a pickle file
output = {
    'geology_index': geology_index,
    'sample_names': sourcesample_names,
    'Source: XRD peak area per mineral per source sample (not normalised)': source_data_corrected,
    'source': rel_area_source,
    'source_rel_sample': rel_area_sample_source, # values scaled between min-max area per sample (instead of per mineral over different samples)
    'max_area_per_mineral': max_area_per_mineral,
    'Geology names (columns)': geology_names,
    'Mineral names (rows)': names_minerals
}

with open(f'{OUTPUT_PATH}/SourceData.pkl', 'wb') as f:
    pickle.dump(output, f)

print("Data saved to 'SourceData.pkl'")
