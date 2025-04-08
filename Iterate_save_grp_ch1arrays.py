#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:27:08 2025

1. Create GRP variations to assess output  (see input parameters)
2. Saving with desired keys 
3. Slicing and compute postive cells per variation

@author: rajeevkumar
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def csv_toarray(fpath):   # skip_blank_lines = False
    array = pd.read_csv(fpath, header=None).to_numpy()
    print(f"Loaded array from {fpath} with shape: {array.shape}")
    return array

# Define the function for calculating fish growth
def calculate_growth_fdos(DO_array, Temp_array, mass_coeff, P_coeff, AD_arr, slope_val, intercept_val):
    DOcrit = slope_val * Temp_array + intercept_val # vary - this is the highest so far 
    fDO = DO_array / DOcrit
    #fDO = np.minimum(fDO, 1.0)  # Cap values greater than 1 to 1
    fDO = np.clip(fDO, 0,1)
    
    t_lethal =  Temp_array #np.minimum(Temp_array, 22.9999999) # cap values to asymptote where all normoxia is lethal
    DO_lethal = 0.4 + 0.000006*(np.exp(0.59*t_lethal))
    fdo_lethal = (DO_array- DO_lethal)/ DOcrit
    fdo_lethal = np.clip(fdo_lethal, 0, 1)
    
    mass = np.full_like(Temp_array, 1 * mass_coeff)  # mass sensitivity test
    # Parameters
    CA = 1.61
    CB = -0.538
    CQ = 3.53
    CTO = 16.8
    CTM = 26.0

    # Respiration parameters
    RA = 0.0018
    RB = -0.12
    RQ = 0.047
    RTO = 0.025
    RTM = 0.0
    RTL = 0.0
    RK1 = 7.23

    Vel = RK1 * np.power(mass, 0.025) #was 0.25 (Rustam cm/s- changed to R4= 0.025 (Hanson 3.0 derived from rudstam))
    ACT = np.exp(RTO * Vel)
    SDA = 0.17

    # Egestion and excretion factors
    FA = 0.25
    UA = 0.1

    # Predator energy density and OCC
    ED = 6500
    OCC = 13556.0
    # Define AD values
    AD = np.full_like(Temp_array, AD_arr, dtype=np.float64)  # Ensure AD is a 2D array
    AD_benthos = 3138  # Schaeffer et al. 1999 - Arend 2011

    # Apply AD_benthos to the bottom row (last depth layer)
    AD[-1, :] = AD_benthos  # Now correctly modifying the last row

    # Consumption calculation with variable coefficient
    P = P_coeff * fDO
    P_lethal = P_coeff * fdo_lethal

    V = (CTM - Temp_array) / (CTM - CTO)
    V = np.maximum(V, 0.0)
    Z = np.log(CQ) * (CTM - CTO)
    Y = np.log(CQ) * (CTM - CTO + 2.0)
    X = (Z ** 2.0) * (1.0 + ((1.0 + 40.0) / Y) ** 0.5) ** 2 / 400.0
    Ft = (V ** X) * np.exp(X * (1.0 - V))
    Cmax = CA * (mass ** CB)
    C = Cmax * Ft * P
    C_lethal = Cmax * Ft * P_lethal

    F = FA * C
    S = SDA * (C - F)
    Ftr = np.exp(RQ * Temp_array)
    R = RA * (mass ** RB) * Ftr * ACT
    U = UA * (C - F)

    GRP = C - (S + F + U) * (AD / ED) - (R * OCC) / ED
    GRP_lethal = C_lethal - (S + F + U) * (AD / ED) - (R * OCC) / ED
    return GRP, GRP_lethal

# Define mass and P coefficient values to analyze
mass_coefficients = np.array([200, 400, 600])  # Example mass values
P_coefficients = np.array([0.2, 0.4, 0.6, 0.8])  # Example P values
AD_values = np.array([2000])  # AD values
slope_vals = np.array([0.168, 0.138])
intercept_vals = np.array([1.63, 2.09])



# Define base file path
####### STRAT BELOW
#base_path = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/strat_values" #strat

####### PRETURN BELOW
base_path = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/preturn_values"   # preturn

# Define locations and years
#locations = ["crook_day", "fail_day", "crook_hr", "fail_hr"]
#locations = ["crooked_daily", "failing_daily", "crooked_hourly", "failing_hourly"]
#years = ["2021", "2022"]

# Redoing Failing 2022
locations = ["failing_daily", "failing_hourly"]
years = ["2022"]

# Dictionary to store GRP and GRP_lethal arrays
GRP_results = {}
GRP_lethal_results = {}

# Loop through all locations and years
for location in locations:
    for year in years:
        # Construct file paths dynamically
        DO_file = f"{base_path}/{location}_DO_{year}.csv"
        Temp_file = f"{base_path}/{location}_Temp_{year}.csv"

        # Check if files exist before processing
        if not os.path.exists(DO_file) or not os.path.exists(Temp_file):
            print(f"⚠️ Skipping {location}_var_{year}: Missing files")
            continue

        # Load data
        DO_array = csv_toarray(DO_file)
        Temp_array = csv_toarray(Temp_file)
        print(f"shape of array from file {location}")

        # Compute GRP and GRP_lethal for all parameter combinations efficiently
        for mass_coeff in mass_coefficients:
            for P_coeff in P_coefficients:
                for AD_arr in AD_values:
                    for slope in slope_vals:
                        for intercept in intercept_vals:
                            # Compute GRP and GRP_lethal
                            GRP_array, GRP_lethal_array = calculate_growth_fdos(DO_array, Temp_array, mass_coeff, P_coeff, AD_arr, slope, intercept)

                            # Generate dictionary keys
                            key = f"GRP_P{P_coeff}_mass{mass_coeff}_slope{slope}_intercept{intercept}_{location}_{year}"
                            key_lethal = f"GRP_lethal_P{P_coeff}_mass{mass_coeff}_slope{slope}_intercept{intercept}_{location}_{year}"

                            # Store in dictionaries
                            GRP_results[key] = GRP_array
                            GRP_lethal_results[key_lethal] = GRP_lethal_array

#%%
"SAVING - F PATHS HAVE BEEN ALTERED SINCE COMPLETION"

# Define a mapping from shorthand to full names
replacements = {
    # Time scale normalization
    "hrs": "hr", "hourly": "hr", "hours": "hr",

    # Variable normalization
    "temperature": "temp", "temps": "temp", "tmp": "temp",
    "dissolved_oxygen": "do", "oxygen": "do", "dos": "do",
    "failing": "fail", "failinging":"fail", 
    "crooked": "crook", "crookeded": "crook", "crookeday":"crook",
    
    # Miscellaneous normalization
    "daily": "day", "days": "day", "dayy": "day", "dayay": "day",
    "grp": "grp"  # Keep "grp" the same
}

# Function to normalize dictionary keys
def normalize_keys(data_dict):
    normalized_dict = {}
    for key, value in data_dict.items():
        new_key = key.lower()
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)  # Apply replacements
        normalized_dict[new_key] = value
    return normalized_dict

def save_arrays_to_csv(data_dict, save_dir):
    """
    Saves each NumPy array in a dictionary to a CSV file.
    
    Parameters:
        data_dict (dict): Dictionary where keys are filenames, values are NumPy arrays.
        save_dir (str): Directory to save the CSV files.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    for key, array in data_dict.items():
        save_path = os.path.join(save_dir, f"{key}.csv")
        np.savetxt(save_path, array, delimiter=",")
        print(f"✅ Saved: {save_path}")

"======= normalizing key names ============="

GRP_results = normalize_keys(GRP_results)
GRP_lethal_results = normalize_keys(GRP_lethal_results)


"======= saving ============="

#save_dir_full = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/GRPvals_alldates_02152025'
# dirs for 2D, sliced value arrays 
save_dir_strat = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/stratified_lakesyears/GRPvals_strat'
save_dir_preturn = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/preturn_lakesyears/GRPvals_preturn'

# Saving function
#save_arrays_to_csv(GRP_results, save_dir_preturn)
#save_arrays_to_csv(GRP_lethal_results, save_dir_preturn)

#save_arrays_to_csv(GRP_results, save_dir_strat)
#save_arrays_to_csv(GRP_lethal_results, save_dir_strat)

#save_arrays_to_csv(GRP_results, save_dir_full)
#save_arrays_to_csv(GRP_lethal_results, save_dir_full)




#%%

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# ✅ Convert all arrays to binary and Save each GRP and GRP_lethal array to CSV files
def csv_toarray(fpath):   # skip_blank_lines = False
    array = pd.read_csv(fpath, header=None).to_numpy()
    print(f"Loaded array from {fpath} with shape: {array.shape}")
    return array

def load_csv_files(directory):
    "Reads into dictionary and takes fname as key without extension"
    return {os.path.splitext(f)[0]: csv_toarray(os.path.join(directory, f)) 
            for f in os.listdir(directory) if f.endswith('.csv')}

def column_sums_with_interval_dict(data_dict, interval):
    """
    Calculate column sums in intervals for each 2D array in a dictionary.
    Parameters:
        data_dict (dict): Dictionary where keys are dataset names and values are 2D numpy arrays.
        interval (int): The number of columns to sum together in each interval.
    Returns:
        dict: A dictionary with the same keys and 1D numpy arrays containing the sums for each interval.
    """
    summed_dict = {}
    
    for key, data in data_dict.items():
        num_rows, num_cols = data.shape
        num_intervals = num_cols // interval
        remainder_cols = num_cols % interval
        sums = np.zeros(num_intervals)        
        for i in range(num_intervals):
            start_col = i * interval
            end_col = start_col + interval
            sums[i] = np.sum(data[:, start_col:end_col])       
        if remainder_cols != 0:
            sums = np.append(sums, np.sum(data[:, -remainder_cols:]))        
        summed_dict[key] = sums   
    return summed_dict

def save_arrays_to_csv(data_dict, save_dir):
    """
    Saves each NumPy array in a dictionary to a CSV file.
    
    Parameters:
        data_dict (dict): Dictionary where keys are filenames, values are NumPy arrays.
        save_dir (str): Directory to save the CSV files.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    for key, array in data_dict.items():
        save_path = os.path.join(save_dir, f"{key}.csv")
        np.savetxt(save_path, array, delimiter=",")
        print(f"✅ Saved: {save_path}")


# dirs for 2D value arrays 
save_dir_strat = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/stratified_lakesyears/GRPvals_strat'
save_dir_preturn = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/preturn_lakesyears/GRPvals_preturn'
# 2D binary slice dirs
save_dir_binarystrat = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/stratified_lakesyears/GRPbinary_strat'
save_dir_binarypturn = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/preturn_lakesyears/GRPbinary_preturn'
# summed 1D save dirs
save_dir_binary_summedstrat = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/stratified_lakesyears/GRPbinary_strat/GRPsummed_strat'
save_dir_binary_summedpturn = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/preturn_lakesyears/GRPbinary_preturn/GRPsummed_preturn'


# Load these if working with a full directory
#grp_all = load_csv_files(save_dir_strat)
#grp_all = load_csv_files(save_dir_preturn)


# Dictionary to store binary arrays
binary_grp_results = {k:(np.where(arr > 0, 1, 0)) for k,arr in GRP_results.items()}
binary_grp_lethalresults = {k:(np.where(arr > 0, 1, 0)) for k,arr in GRP_lethal_results.items()}



#save_arrays_to_csv(binary_grp_results, save_dir_binarypturn)
#save_arrays_to_csv(binary_grp_lethalresults, save_dir_binarypturn)
#save_arrays_to_csv(binary_grp_results, save_dir_binarystrat)
#save_arrays_to_csv(binary_grp_lethalresults, save_dir_binarystrat) #These are correct


# sum binary columns
#binary_col_sums_results = column_sums_with_interval_dict(binary_grp_results, 1)
#binary_col_sums_lethalresults = column_sums_with_interval_dict(binary_grp_lethalresults, 1)


#save_arrays_to_csv(binary_col_sums_results, save_dir_binary_summedstrat) 
#save_arrays_to_csv(binary_col_sums_lethalresults, save_dir_binary_summedstrat)
"Preturn values got saved here (non_lethal preturn sums went in strat sums) - deleted thanks to file delete code, but needs a resave- DONE, but barely slept and needs double check"
#save_arrays_to_csv(binary_col_sums_results, save_dir_binary_summedpturn)
#save_arrays_to_csv(binary_col_sums_lethalresults, save_dir_binary_summedpturn)

"End all model variation array creation - Raw values, 2D binary, and 1D sums"


#%%

def generate_timesteps(start_date, end_date, timescale):
    if timescale == "hr":
        delta = timedelta(hours=1)
    elif timescale == "day":
        delta = timedelta(days=1)
    else:
        raise ValueError(f"Unsupported timescale: {timescale}")
    
    timesteps = []
    current_date = start_date
    while current_date <= end_date:
        timesteps.append(current_date)
        current_date += delta
    return timesteps

# Function to split location into lake and temporal scale
def split_location(location):
    parts = location.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]  # lake, temporal_scale
    else:
        return "unknown", "unknown"  # Default values for invalid format

# Parse filenames and calculate positive cells per timestep
results = []
for file in os.listdir(binary_folder):
    if file.endswith(".csv"):
        # Extract parameters from filename
        parts = file.split("_")
        
        # Check if the filename contains "GRP_lethal" or "GRP"
        if parts[0] == "GRP" and parts[1][0] == "P":
            P_index = 1
            mass_index = 2
            slope_index = 3
            intercept_index = 4
            location_index = -3
            timescale_index =-2
            year_index = -1
            lethal_used = False  # GRP file, so fdo_lethal was not used
        elif parts[0] == "GRP" and parts[1] == "lethal":
            P_index = 2
            mass_index = 3
            slope_index = 4
            intercept_index = 5
            location_index = -3
            timescale_index = -2
            year_index = -1
            lethal_used = True  # GRP_lethal file, so fdo_lethal was used
        else:
            print(f"⚠️ Skipping file {file}: Unexpected filename format")
            continue
        # Extract values
        try:
            P = float(parts[P_index][1:])  # P0.2 → 0.2
            mass = int(parts[mass_index][4:])  # mass200 → 200
            slope = float(parts[slope_index][5:])  # slope0.168 → 0.168
            intercept = float(parts[intercept_index][9:])  # intercept1.63 → 1.63
            location = f"{parts[location_index]}_{parts[location_index + 1]}"  # crook_day
            year = parts[year_index].split(".")[0]  # 2021
        except (IndexError, ValueError) as e:
            print(f"⚠️ Skipping file {file}: Error parsing filename - {e}")
            continue
        
        # Load binary array
        binary_array = np.loadtxt(os.path.join(binary_folder, file), delimiter=",")
        
        # Define start and end dates for each lake, temporal scale, and year
        start_end_dates = {
            "crook": {
                "day": {
                    "2021": ("2021-06-24", "2021-11-11"),
                    "2022": ("2022-06-07", "2022-12-01"),
                },
                "hr": {
                    "2021": ("2021-06-24", "2021-11-11"),
                    "2022": ("2022-06-07", "2022-12-01"),
                },
            },
            "fail": {
                "day": {
                    "2021": ("2021-06-05", "2021-12-01"),
                    "2022": ("2022-04-12", "2022-12-01"),
                },
                "hr": {
                    "2021": ("2021-06-05", "2021-12-01"),
                    "2022": ("2022-04-12", "2022-12-01"),
                },
            },
        }
        
        # Access start and end dates correctly
        lake, temporal_scale = split_location(location)
        if lake not in start_end_dates or temporal_scale not in start_end_dates[lake] or year not in start_end_dates[lake][temporal_scale]:
            print(f"⚠️ Skipping file {file}: No start/end dates defined for {lake} {temporal_scale} {year}")
            continue
        
        start_date, end_date = start_end_dates[lake][temporal_scale][year]
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Generate timesteps
        timesteps = generate_timesteps(start_date, end_date, temporal_scale)
        
        # Ensure timesteps match the length of the binary array
        if len(timesteps) > binary_array.shape[0]:
            # If timesteps are longer than the binary array, adjust the end date
            end_date = start_date + (len(binary_array) - 1) * (timedelta(hours=1) if temporal_scale == "hr" else timedelta(days=1))
            timesteps = generate_timesteps(start_date, end_date, temporal_scale)
        elif len(timesteps) < binary_array.shape[0]:
            # If binary array is longer than the timesteps, slice the binary array
            binary_array = binary_array[:len(timesteps), :]
        
        # Ensure timesteps and binary array have the same length
        if len(timesteps) != binary_array.shape[0]:
            print(f"⚠️ Skipping file {file}: Timesteps length does not match binary array after adjustment")
            continue
        
        # Calculate positive cells for each timestep
        for timestep, binary_row in zip(timesteps, binary_array):
            positive_cells = np.sum(binary_row)  # Sum positive cells for the current timestep
            results.append([P, mass, slope, intercept, location, year, timestep, positive_cells, lethal_used])

# Create DataFrame
df = pd.DataFrame(
    results,
    columns=["P", "mass", "slope", "intercept", "location", "year", "timestep", "positive_cells", "lethal_used"]
)

# Split location into lake and temporal scale
df[["lake", "temporal_scale"]] = df["location"].apply(split_location).apply(pd.Series)

# Drop rows where lake or temporal_scale is "unknown" (invalid format)
df = df[~df["lake"].isin(["unknown"]) & ~df["temporal_scale"].isin(["unknown"])]

# Display the DataFrame
pd.set_option("display.max_columns", None)
print(df.head())


df.to_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/comp_GRP_variation_positive_attempt/notcomplete_allGRPcellpostive.csv', index = False)


#%% 
"Checking dataframe values and combinations to see if all values make sense before visual evaluation"
"They are not correct date ranges - likely incorrect elsewhere "
# crook day 2021
crook_day_2021 = df[(df["location"] == "crook_day") & (df["year"] == "2021")]
print(crook_day_2021)

# Generate expected timesteps for a specific combination
start_date = datetime.strptime("2021-06-24", "%Y-%m-%d")
end_date = datetime.strptime("2021-12-01", "%Y-%m-%d")
expected_timesteps = pd.date_range(start_date, end_date, freq="D")  # Daily timesteps

# Get actual timesteps for the specific combination
actual_timesteps = crook_day_2021["timestep"]

# Find missing timesteps
missing_timesteps = expected_timesteps.difference(actual_timesteps)
print(f"Missing timesteps for crook_day in 2021: {missing_timesteps}")






fail_day_2021 = df[(df["location"] == "fail_day") & (df["year"] == "2021")]
print(fail_day_2021)

# 2022
crook_day_2022 = df[(df["location"] == "crook_day") & (df["year"] == "2022")]
print(crook_day_2022)

fail_day_2022 = df[(df["location"] == "fail_day") & (df["year"] == "2022")]
print(fail_day_2022)






#%%

"""
Plotting from chat 
"""


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

def plot_dynamic_heatmap(GRP_results, key, start_date=None, color_range=None):
    """
    Plots a heatmap from a 2D GRP array with dynamically chosen titles, date-based x-axis, 
    and customizable color scaling.

    Parameters:
    - GRP_results: Dictionary containing GRP arrays
    - key: The key corresponding to the GRP array in the dictionary
    - start_date: (Optional) Manually set start date as 'YYYY-MM-DD'
    - color_range: Tuple (vmin, vmax) to set the color scale (optional)
    """
    if key not in GRP_results:
        print(f"❌ Error: Key '{key}' not found in GRP_results.")
        return
    
    # Extract the GRP array
    GRP_array = GRP_results[key]
    
    # Extract metadata from key
    key_parts = key.split("_")  # Example: "GRP_P0.4_mass400_crook_day_2021"
    P_val = key_parts[1].replace("P", "")  # Extract P value
    mass = key_parts[2].replace("mass", "")  # Extract mass
    location = key_parts[3]  # Extract location
    year = key_parts[-1]  # Extract last element as year

    # Dynamically extract start_date if not provided
    if start_date is None:
        if "2021" in key:
            start_date = f"{year}-01-01"  # Default to Jan 1st, 2021
        elif "2022" in key:
            start_date = f"{year}-06-01"  # Example: Default to June 1st, 2022
        else:
            start_date = f"{year}-01-01"  # Fallback default

    # Convert start_date to datetime object
    start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    # Determine x-axis format (hourly or daily)
    if "hr" in key:
        time_format = "Hourly"
        num_timestamps = GRP_array.shape[1]
        x_labels = [start_datetime + datetime.timedelta(hours=i) for i in range(num_timestamps)]
        date_formatter = mdates.DateFormatter("%H:%M")
    else:  # Assume daily format
        time_format = "Daily"
        num_timestamps = GRP_array.shape[1]
        x_labels = [start_datetime + datetime.timedelta(days=i) for i in range(num_timestamps)]
        date_formatter = mdates.DateFormatter("%b %d")

    # Define title dynamically
    title = f"GRP Heatmap: P={P_val}, Mass={mass}g, {location.capitalize()} ({year}, {time_format})"

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define color range dynamically
    vmin, vmax = color_range if color_range else (np.min(GRP_array), np.max(GRP_array))

    # Plot heatmap
    cax = ax.imshow(GRP_array, aspect='auto', cmap="coolwarm", vmin=vmin, vmax=vmax)

    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Depth/Grid Cell", fontsize=12)
    
    # trying to format y ticks

    # Format x-axis with dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto spacing
    ax.xaxis.set_major_formatter(date_formatter)  # Format as time-based

    # Set correct x-tick labels
    ax.set_xticks(np.linspace(0, num_timestamps-1, min(num_timestamps, 10)))
    ax.set_xticklabels([x_labels[int(i)].strftime('%Y-%m-%d') for i in np.linspace(0, num_timestamps-1, min(num_timestamps, 10))], rotation=45)
    
    # Default: Evenly spaced y-ticks
    num_y_ticks = min(GRP_array.shape[0], 6)  # Max 10 labels
    y_ticks = np.linspace(0, GRP_array.shape[0] - 1, num_y_ticks)  # Get default row indices

    # ✅ divide by 2 (scaling depth values)
    y_tick_values = (y_ticks/2).astype(int)

    # ✅ Add one more tick at the bottom
    y_tick_values = np.append(y_tick_values, y_tick_values[-1])  #-takes away cbar for some reason

    # ✅ Convert to string format with "m"
    y_tick_labels = [f"{d}m" for d in y_tick_values]

    # ✅ Set the new ticks & labels
    ax.set_yticks(np.append(y_ticks, y_ticks[-1]) ) # Append one extra tick
    ax.set_yticklabels(y_tick_labels)
    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label("GRP Value", fontsize=12)

    # Show plot
    plt.show()





plot_dynamic_heatmap(GRP_results, key="GRP_P0.4_mass400_crook_day_2021", start_date="2021-06-24")
plot_dynamic_heatmap(GRP_results, key="GRP_P0.6_mass600_fail_hr_2022", color_range=(-0.02, 0.02))

