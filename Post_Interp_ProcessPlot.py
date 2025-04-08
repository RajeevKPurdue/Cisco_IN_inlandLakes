#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:55:39 2025

@author: rajeevkumar

* Key flaws/ User actions 
    1. File naming conventions - Underscore splitting, Manual change for normalize key names replacements 
        2. With new GRP iterations- not all splitting logic for filenames is transferable

(Arrays)
Load interpolated arrays (Temp, DO, GRP (12- 4 P, 3 Mass)) for Periods (1,2), Timescales (H, D), Lakes (Crook, Fail)
summarize into 0.5m cells 
Make binaries
Make lineplot binaries  


Organize by period and time scale 
Get summary outputs to ... (Determine summary outputs for each variable)

(Plots/Figures)
Make summary tables (T, DO, GRP (12- 4 P, 3 Mass)) for Periods (1,2), Timescales (H, D), Lakes (Crook, Fail)
Heatmaps (T, DO, GRP (1)) for Periods (1,2), Timescales (H, D), Lakes (Crook, Fail)
Lineplots (Binary, GRP (1)) for Periods (1,2), Timescales (H, D), Lakes (Crook, Fail)

"""

#%%



#imports 
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date, timedelta
from operator import itemgetter


# new code - debugging test successful

def csv_toarray(fpath):   # skip_blank_lines = False
    array = pd.read_csv(fpath, header=None).to_numpy()
    print(f"Loaded array from {fpath} with shape: {array.shape}")
    return array

"load_csv_files is unchanged by deepseek"
def load_csv_files(directory):
    "Reads into dictionary and takes fname as key without extension"
    return {os.path.splitext(f)[0]: csv_toarray(os.path.join(directory, f)) 
            for f in os.listdir(directory) if f.endswith('.csv')}


def save_arrays_to_csv(data_dict, save_dir):
    """
    Saves each NumPy array in a dictionary to a CSV file.
    
    Parameters:
        data_dict (dict): Dictionary where keys are filenames, values are NumPy arrays.
        save_dir (str): Directory to save the CSV files.
    """
    os.makedirs(save_dir, exist_ok=True)
    for key, array in data_dict.items():
        save_path = os.path.join(save_dir, f"{key}.csv")
        np.savetxt(save_path, array, delimiter=",", fmt="%s")  # Use consistent formatting
        print(f"✅ Saved: {save_path} with shape: {array.shape}")




def create_binary_arrays_dict(data_dict, condition):
    """
    Convert 2D arrays in a dictionary to binary arrays based on a condition.
    
    Parameters:
        data_dict (dict): Dictionary where keys are dataset names and values are 2D numpy arrays.
        condition (callable or numpy.ndarray): Condition applied element-wise.
    
    Returns:
        dict: A dictionary with the same keys and binary 2D arrays as values.
    """
    binary_dict = {}
    
    for key, data in data_dict.items():
        binary_dict[key] = np.where(condition(data), 1, 0) if callable(condition) else np.where(condition, 1, 0)
    
    return binary_dict


def average_rows_dict(data_dict, window_size):
    """
    Calculate the average of every 'window_size' rows in each column of the input 2D arrays 
    stored in a dictionary, while preserving key-value mapping.

    Parameters:
        data_dict (dict): Dictionary where keys are dataset names and values are 2D numpy arrays.
        window_size (int): The number of rows to average together.

    Returns:
        dict: A new dictionary with the same keys and downsampled 2D arrays as values.
    """
    averaged_dict = {}

    for key, data in data_dict.items():
        num_rows, num_cols = data.shape
        result_rows = num_rows // window_size  # New row count after averaging
        print(f"num_rows, {num_rows}: result_rows, {result_rows}")
        # Compute the averaged array
        averaged_data = np.array([
            np.mean(data[i * window_size : (i + 1) * window_size], axis=0)
            for i in range(result_rows)
        ])

        averaged_dict[key] = averaged_data  # Preserve key-value mapping

    return averaged_dict




def replicate_cols_dict(data_dict, num_replications=24):
    """
    Replicate each column of the input 2D array 'num_replications' times.
    """
    rep_data_dict = {}

    for key, data in data_dict.items():  # Corrected dictionary iteration
        num_rows, num_cols = data.shape

        # Use broadcasting to replicate columns efficiently
        replicated_data = np.repeat(data, num_replications, axis=1)

        # Store the replicated array in the new dictionary
        rep_data_dict[key] = replicated_data

    return rep_data_dict



# sum rows for all columns in 2D array

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



def match_datetime_to_indices(time_index, target_dates):
    """
    Matches a list of target DateTime values to the closest indices in a DateTime index.

    Parameters:
        time_index (pd.Index): A Pandas DatetimeIndex representing the x-axis time steps.
        target_dates (list of str or list of pd.Timestamp): List of dates to locate in the index.

    Returns:
        dict: A dictionary mapping target dates to their closest index positions in time_index.
    """
    # Ensure the DateTime index is in NumPy format
    time_array = time_index.to_numpy()

    # Convert target dates to NumPy datetime64 if they are in string format
    target_dates = np.array(pd.to_datetime(target_dates))

    # Find the closest indices using np.searchsorted
    indices = np.searchsorted(time_array, target_dates, side="left")

    # Ensure indices stay within bounds
    indices = np.clip(indices, 0, len(time_index) - 1)

    # Return dictionary mapping target dates to array indices
    return {str(date): index for date, index in zip(target_dates, indices)}





def generate_date_ranges(years, lakes, variables=['do', 'temp', 'grp']):
    """Generate date ranges for all years, lakes, and variables."""
    base_dates = {
        'failing': {2021: date(2021, 6, 5), 2022: date(2022, 4, 12)},
        'crooked': {2021: date(2021, 6, 24), 2022: date(2022, 6, 7)}
    }
    
    date_ranges = {}
    for lake in lakes:
        for year in years:
            for var in variables:
                for scale in ['daily', 'hourly']:
                    key = f"{lake}_{scale}_{var}_{year}"
                    date_ranges[key] = (base_dates[lake][year], date(year, 12, 1))

                    # Add preturn period keys
                    preturn_key = f"{lake}_{scale}_{var}_{year}_preturn"
                    date_ranges[preturn_key] = (date(year, 9, 30), date(year, 11, 3))
                    #print('date ranges')
    
    return date_ranges






def generate_date_arrays(start_dates, num_intervals):
    """Generate daily and hourly date arrays for all lakes, years, and variables."""
    date_arrays = {}

    for (lake, year), start_date in start_dates.items():
        for scale in ['daily', 'hourly']:
            key = (lake, year, scale)

            if scale == 'daily':
                dates = np.array([start_date + timedelta(days=i) for i in range(num_intervals)])
            else:
                dates = np.array([start_date + timedelta(hours=i) for i in range(num_intervals)])

            date_arrays[key] = dates  # No `.date()` call needed

    return date_arrays


def dates_for_key(start_dates, data_dict):
    """
    Generate date arrays for each key in the dataset based on matching start dates.
    
    Parameters:
        start_dates (dict): Dictionary {('lake', year): start_date}.
        data_dict (dict): Dictionary {key: 2D NumPy array}.

    Returns:
        dict: Dictionary {key: NumPy array of dates}.
    """
    date_arrays = {}

    for key, arr in data_dict.items():
        num_intervals = arr.shape[1]  # Get column count for this specific key

        # Extract lake, timescale, var, year from key
        parts = key.split('_')
        try:
            lake, timescale, var, year = parts[0], parts[1], parts[2], int(parts[-1])
        except ValueError:
            print(f"⚠️ Skipping {key}: Unexpected key format")
            continue

        # Match start date
        start_date = start_dates.get((lake, year), None)
        if start_date is None:
            print(f"⚠️ Warning: No start date found for {key}")
            continue

        # Generate date array
        if 'daily' in timescale:
            dates = np.array([start_date + timedelta(days=i) for i in range(num_intervals)])
        else:  # Assume hourly if not daily
            dates = np.array([start_date + timedelta(hours=i) for i in range(num_intervals)])

        date_arrays[key] = dates  # Store unique key-date pair

    return date_arrays


def extract_time_slices2(data_dict, date_dict, date_ranges):
    """
    Extracts subsets of data for given date ranges for multiple datasets.

    Parameters:
        data_dict (dict): Dictionary of {array_name: 2D NumPy array}.
        date_dict (dict): Dictionary of {lake_scale_var_year: date_values}.
        date_ranges (dict): Dictionary of {lake_scale_var_year: (start_date, end_date)}.

    Returns:
        tuple: (full_period_dict, preturn_period_dict)
            - full_period_dict: Dictionary with full-period slices.
            - preturn_period_dict: Dictionary with preturn-period slices.
    """
    full_slices = {}
    preturn_slices = {}

    for array_name, data in data_dict.items():
        parts = array_name.split('_')
        lake, scale, var, year = parts[0], parts[1], parts[2], int(parts[3])
        key = f"{lake}_{scale}_{var}_{year}"
        
        print(f"\n🔍 Processing array: {array_name}")

        if key in date_dict:
            date_vals = date_dict[key]
            start_range, end_range = date_ranges.get(key, (None, None))

            print(f"  🔹 Found date range: {start_range} to {end_range}")
            print(f"  📆 Date values (first 5): {date_vals[:5]}")

            if start_range and end_range:
                # Ensure format consistency
                if isinstance(date_vals[0], datetime):
                    start_range = datetime(start_range.year, start_range.month, start_range.day)
                    end_range = datetime(end_range.year, end_range.month, end_range.day) + timedelta(days=1)

                indices = np.where((date_vals >= start_range) & (date_vals <= end_range))[0]

                # Prevent out-of-bounds indices
                max_columns = data.shape[1]
                indices = indices[indices < max_columns]

                if len(indices) == 0:
                    print(f"⚠️ Warning: No valid indices found for {array_name}")

                full_slices[array_name] = data[:, indices]  # Store full-period slice

        # Handle preturn period separately
        preturn_key = f"{key}_preturn"
        if preturn_key in date_ranges and key in date_dict:
            print(f"  🔎 Checking preturn key: {preturn_key}")
            pre_start, pre_end = date_ranges[preturn_key]

            if isinstance(date_vals[0], datetime):
                pre_start = datetime(pre_start.year, pre_start.month, pre_start.day)
                pre_end = datetime(pre_end.year, pre_end.month, pre_end.day) + timedelta(days=1)

            pre_indices = np.where((date_vals >= pre_start) & (date_vals <= pre_end))[0]
            pre_indices = pre_indices[pre_indices < max_columns]
            

            if len(pre_indices) == 0:
                print(f"⚠️ Warning: No valid indices found for preturn {array_name}")

            preturn_slices[array_name] = data[:, pre_indices]  # Store preturn-period slice

    return full_slices, preturn_slices



def normalize_key_names(data_dict, replace_indices=None):
    """
    Standardizes key names by replacing multiple variations of terms with a single standard term.
    Parameters:
        data_dict (dict): Dictionary where keys are formatted as "lake_timescale_var_year".
        replace_indices (list or None): List of indices to apply replacements to.
            - If None, apply replacements to **all** parts.
    Returns:
        dict: A new dictionary with standardized keys.
    """
    # Define mapping rules for standardizing key parts
    replacements = {
        # Time scale normalization
        "hrs": "hr", "hourly": "hr", "hours": "hr",

        # Variable normalization
        "temperature": "temp", "temps": "temp", "tmp": "temp",
        "dissolved_oxygen": "do", "oxygen": "do", "dos": "do",
        
        # Miscellaneous normalization
        "daily": "day", "d": "day", "days": "day", "dayy": "day",
        "grp": "grp"  # Keep "grp" the same
    }

    normalized_dict = {}

    for key, value in data_dict.items():
        parts = key.lower().split('_')  # Split key into parts

        # Apply replacements dynamically based on replace_indices
        normalized_parts = [
            replacements.get(part, part) if (replace_indices is None or i in replace_indices) else part
            for i, part in enumerate(parts)
        ]

        normalized_key = "_".join(normalized_parts)  # Reconstruct key
        normalized_dict[normalized_key] = value  # Store with updated key

    return normalized_dict

def lyons_cond(data_dict, match_indices):
    """
    Matches temp and DO arrays based on specified indices in key names, 
    and multiplies them together.

    Parameters:
        data_dict (dict): Dictionary where keys are formatted as "lake_timescale_var_year".
        match_indices (list): Indices specifying which parts of the key should be used for matching.

    Returns:
        dict: New dictionary with multiplied arrays, stored as "lyons_<matched_parts>".
    """
    lyons_dict = {}  # To store results
    temp_arrays = {}
    do_arrays = {}

    for key, df in data_dict.items():
        parts = key.lower().split('_')

        # Ensure key has enough parts to be indexed
        if max(match_indices) >= len(parts):
            print(f"Skipping key {key} due to insufficient parts.")
            continue

        key_match = tuple(itemgetter(*match_indices)(parts))  # Extract match-relevant parts

        if "temp" in parts:
            temp_arrays[key_match] = df  # Store temp array under matching key
        elif "do" in parts:
            do_arrays[key_match] = df  # Store DO array under matching key

    # Multiply matching temp and DO arrays
    for key_match in temp_arrays:
        if key_match in do_arrays:  # Only multiply if a matching DO array exists
            lyons_key = f"lyons_{'_'.join(key_match)}"  # Generate new key
            lyons_dict[lyons_key] = temp_arrays[key_match] * do_arrays[key_match]  # Element-wise multiplication

    return lyons_dict

"End initial packages and functions"

#%%
" ==========READING AND SORTING FILES FROM DIRS==============="
# load dirs read .csv files to array using list comprehension - function read array to csv

#fail_p1dir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/Failing_2yr/fail_Comp_Arrays_2021'
fail_p2dir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/Failing_2yr/fail_Comp_Arrays_2022'
#crook_p1dir = '//Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/Crook_2yr/Crook_Comp_Arrays_2021/Need_halfm_cells_etc'
#crook_p2dir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/Crook_2yr/Crook_Comp_Arrays_2022'

#preemptively segregate arrays by dimensions

#failP1 = load_csv_files(fail_p1dir)
failP2 = load_csv_files(fail_p2dir)
#crookP1 = load_csv_files(crook_p1dir)
#crookP2 = load_csv_files(crook_p2dir)

#all_list = [failP1, failP2, crookP1, crookP2]
all_list = [failP2]


"""
KEYS FOR COPYING

fp1 = dict_keys(['fail_DO_hr_2021', 'fail_temp_hr_2021', 'fail_GRP_hr_2021', 
                 'fail_DO_day_2021', 'fail_GRP_day_2021', 'fail_temp_day_2021'])
    
fp2 = dict_keys(['fail_hr_DO_2022', 'fail_day_DO_2022', 'fail_day_Temp_2022', 'fail_hr_Temp_2022'])

crP1 = dict_keys(['crook_hr_GRP_2021', 'crook_day_Temp_2021', 'crook_hr_DO_2021', 
                  'crook_day_DO_2021', 'crook_day_GRP_2021', 'crook_hr_Temp_2021'])
    
crP2 = dict_keys(['crook_hrs_temp_2022', 'crook_day_Temp_2022', 'crook_hrs_GRP_2022', 
                  'crook_day_DO_2022', 'crook_day_GRP_2022', 'crook_hrs_DO_2022'])

"""

def sort_dict(data):
    "Sorts data into GRP-related and Temp/DO-related dictionaries based on key names."
    grp_dict = {}
    tempDO_dict = {}
    
    for key, df in data.items():  # Use .items() to iterate over dict key-value pairs
        parts = key.lower().split('_')  # Ensure lowercase and split key into parts
        
        if "grp" in parts:
            grp_dict[key] = df
        else:
            tempDO_dict[key] = df

    return grp_dict, tempDO_dict



# Initialize empty dictionaries to store combined results
allgrp = {}
alltempdo = {}


# Iterate over all dictionaries in all_list and merge the results
for data in all_list:
    grp, tempDO = sort_dict(data)
    allgrp.update(grp)      # Merge GRP-related keys and values
    alltempdo.update(tempDO)  # Merge Temp/DO-related keys and values


alldicts = {}

for data in all_list:
    alldicts.update(data)


print("All GRP keys:", allgrp.keys())
print("All Temp/DO keys:", alltempdo.keys())    
print("All keys:", alldicts.keys())




" SUMMARIZING ALL TO HALF METER CELLS - AVE ROWS ALL ARRAYS"
#average_rows_dict(data_dict, window_size) 

halfm_all = average_rows_dict(alldicts, window_size=5) 




" CREATING BINARY ARRAYS BY CONDTION - T, DO IS LYONS (<=22.8, >=6) ; GRP IS > 0 "

lyons_t = 22.8
lyons_do = 6.0
grp_thresh = 0.0

# Dictionary to store binary arrays
binarydict = {}

for key, df in halfm_all.items():  # Iterate over the averaged arrays
    parts = key.lower().split('_')  # Split key for variable detection
    
    if "temp" in parts:
        condition = df < lyons_t  # 1 if good, 0 if not
    elif "do" in parts:
        condition = df >= lyons_do  # 1 if DO is good, 0 otherwise
    elif "grp" in parts:
        condition = df > grp_thresh  # 1 if GRP is above threshold, 0 otherwise
    else:
        print(f'No variable recognized from key: {key}')
        continue  # Skip this iteration if no variable is recognized

    # Apply binary conversion and store in the dictionary
    binarydict[key] = np.where(condition, 1, 0)
    




" REPLICATE COLS FOR ALL DAILY ARRAYS - might be able to apply conditonal for key easily without func below"
#replicate_cols_dict(data_dict, num_replications=24)
" sort and make daily files"
repdaily = {}
repdailybinary = {}

for key, df in halfm_all.items():  # Iterate over the averaged arrays
    parts = key.lower().split('_')  # Split key for variable detection
    if "day" in parts:  
        repdaily[key] = replicate_cols_dict({key: df}, num_replications=24)[key]  # Store result
        repdailybinary[key] = replicate_cols_dict({key: binarydict[key]}, num_replications=24)[key]
        


"Normalizing Key Names For Binaries - Artifact of Lyons dict logic"

normbinary = normalize_key_names(binarydict, replace_indices=[1])

# Custom match criteria (lake, timescale, year) - This will not hold
match_indices = [0, 1, 3]  

lyonsdict = lyons_cond(normbinary, match_indices)
for key, val in lyonsdict.items():
    print(f"{key}: \n{val}")
    

    

"adding failing 2021 pairs due to file naming inconsistency that needs fix- *FIXED"

#lyonsdict['lyons_fail_hr_2021'] = binarydict['fail_DO_hr_2021'] * binarydict['fail_temp_hr_2021']
#lyonsdict['lyons_fail_day_2021'] = binarydict['fail_DO_day_2021'] * binarydict['fail_temp_day_2021']



    
# Define save paths
#save_dir_main = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Processed_da
#save_dir_main = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data"
#save_dir_main = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/fail2022_redone"

#save_arrays_to_csv(alldicts, os.path.join(save_dir_main, "all_dates"))
#save_arrays_to_csv(halfm_all, os.path.join(save_dir_main, "half_meter"))
#save_arrays_to_csv(binarydict, os.path.join(save_dir_main, "binary"))
#save_arrays_to_csv(repdaily, os.path.join(save_dir_main, "replicated_day"))
#save_arrays_to_csv(repdailybinary, os.path.join(save_dir_main, "replicated_binary"))
#save_arrays_to_csv(lyonsdict, os.path.join(save_dir_main, "lyons_cond"))




"Slicing into Nightmare amount of groups - Review this"
# Define a mapping from shorthand to full names
timekey_replacements = {
    "fail": "failing",
    "crook": "crooked",
    "hr": "hourly",
    "hrs": "hourly",
    "hourlys": "hourly",
    "day": "daily",
    "Temp": "temp",
    "DO": "do",
    "GRP": "grp"
}

# Function to normalize dictionary keys
def normalize_keys(data_dict):
    normalized_dict = {}
    for key, value in data_dict.items():
        new_key = key.lower()
        for old, new in timekey_replacements.items():
            new_key = new_key.replace(old, new)  # Apply replacements
        normalized_dict[new_key] = value
    return normalized_dict

# Normalize keys in the dataset
halfm_all_norm = normalize_keys(halfm_all)
print('half norm keys:', halfm_all_norm.keys())

# Normalize keys in the dataset
bin_norm = normalize_keys(binarydict)

#lyonsnorm = normalize_keys(lyonsdict) # "not possible atm"

repdailynorm = normalize_keys(repdaily)

repdailybinarynorm = normalize_keys(repdailybinary)

"All non time slicing array creations complete"

#%%

"========================= ALL ABOVE = code for full observation; ALL BELOW = sliced by time interval =========="


"temportary working/testing time funcs here"
years = [2021, 2022]
lakes = ['failing', 'crooked']
genranges = generate_date_ranges(years, lakes)

years_fp2 = [2022]
lakes_fp2 = ['failing']
genranges_fp2 = generate_date_ranges(years_fp2, lakes_fp2)

# Extract start dates for each lake and year
start_dates = {
    (lake, year): genranges_fp2[f"{lake}_daily_do_{year}"][0] for lake in lakes_fp2 for year in years_fp2
}

    
dationaryfull = dates_for_key(start_dates, halfm_all_norm)     
                    
for key, val in dationaryfull.items():
    print('key and shape:', list(dationaryfull[key].shape))
    print(f"{key}: {val[:5]}")
                    
                



full_slices, preturn_slices = extract_time_slices2(halfm_all_norm, dationaryfull, genranges_fp2)


bin_strat, bin_turn = extract_time_slices2(bin_norm, dationaryfull, genranges_fp2)

#lyons_strat, lyons_turn = extract_time_slices2(lyonsnorm, dationaryfull, genranges)



"========== making lyons dictionarries with binary slices to work around naming logic issue with dict keys ========"
    
lyonsdict_strat = lyons_cond(bin_strat, match_indices)
lyonsdict_preturn = lyons_cond(bin_turn, match_indices)



print("\n✅ Extracted Full Period Slices:")
for key in full_slices:
    print(f"  {key}: Shape {full_slices[key].shape}")

print("\n✅ Extracted Preturn Period Slices:")
for key in preturn_slices:
    print(f"  {key}: Shape {preturn_slices[key].shape}")


# Pick a key to check
key_to_check = "failing_daily_do_2021"  # Change this to any array of interest
preturn_key_to_check = f"{key_to_check}_preturn"

if key_to_check in dationaryfull:
    print(f"\n📌 Checking preturn slicing for {key_to_check}")

    # Get original date values before slicing
    original_dates = dationaryfull[key_to_check]
    preturn_start, preturn_end = genranges[preturn_key_to_check]  # Get preturn range

    # Print first and last 5 values before slicing
    print(f"  📅 Full Date Range (first 5): {original_dates[:5]}")
    print(f"  📅 Full Date Range (last 5): {original_dates[-5:]}")
    print(f"  🔎 Expected Preturn Range: {preturn_start} to {preturn_end}")

    # Extract the sliced preturn dates
    pre_indices = np.where((original_dates >= preturn_start) & (original_dates <= preturn_end))[0]
    preturn_dates = original_dates[pre_indices]

    print(f"  ✅ Extracted Preturn Dates (first 5): {preturn_dates[:5]}")
    print(f"  ✅ Extracted Preturn Dates (last 5): {preturn_dates[-5:]}")
    
    # Verify shape of the extracted data
    if preturn_key_to_check in preturn_slices:
        print(f"  ✅ Preturn Data Shape: {preturn_slices[preturn_key_to_check].shape}")
    else:
        print(f"  ❌ Warning: No preturn slice found for {preturn_key_to_check}")
        
#save_dir_main = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/fail2022_redone"

#save_arrays_to_csv(preturn_slices, os.path.join(save_dir_main, "preturn_slices"))
#save_arrays_to_csv(full_slices, os.path.join(save_dir_main, "full_slices"))
#save_arrays_to_csv(bin_strat, os.path.join(save_dir_main, "binary_strat"))
#save_arrays_to_csv(bin_turn, os.path.join(save_dir_main, "binary_preturn"))

save_arrays_to_csv(preturn_slices, "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/preturn_values")
save_arrays_to_csv(full_slices, "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/strat_values")
save_arrays_to_csv(bin_strat, "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/strat_binary")
save_arrays_to_csv(bin_turn, "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/preturn_binary")


# Lyons strat and preturn

#save_arrays_to_csv(lyonsdict_strat, os.path.join(save_dir_main, "lyons_strat"))
#save_arrays_to_csv(lyonsdict_preturn, os.path.join(save_dir_main, "lyons_preturn"))

save_arrays_to_csv(lyonsdict_strat, "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/lyons_strat")
save_arrays_to_csv(lyonsdict_preturn, "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/lyons_preturn")

# rep daily for lyons periods

repdaily_stratlyons = {}
repdaily_preturnlyons = {}

for key, df in lyonsdict_strat.items():  # Iterate over the averaged arrays
    parts = key.lower().split('_')  # Split key for variable detection
    if "daily" in parts:  
        repdaily_stratlyons[key] = replicate_cols_dict({key: df}, num_replications=24)[key]  # Store result


for key, df in lyonsdict_preturn.items():  # Iterate over the averaged arrays
    parts = key.lower().split('_')  # Split key for variable detection
    if "daily" in parts:  
        repdaily_preturnlyons[key] = replicate_cols_dict({key: df}, num_replications=24)[key]  # Store result


#save_arrays_to_csv(repdaily_stratlyons, os.path.join(save_dir_main, "lyons_strat_repdaily"))
#save_arrays_to_csv(repdaily_preturnlyons, os.path.join(save_dir_main, "lyons_preturn_repdaily"))

save_arrays_to_csv(repdaily_stratlyons, "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/lyons_strat/lyons_strat_repdaily")
save_arrays_to_csv(repdaily_preturnlyons, "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/lyons_preturn/lyons_preturn_repdaily")

#%%

"Summing and saving all the lyons files - repdaily, hourly"


# lyons_rep_daily = load_csv_diles(" ")
# lyons_ = load_csv_diles(" ")

lyons_rep_daily_sum = column_sums_with_interval_dict(lyons_rep_daily, 1) #summing binary cols

#repdaily = {}
#repdailybinary = {}

for key, df in halfm_all.items():  # Iterate over the averaged arrays
    parts = key.lower().split('_')  # Split key for variable detection
    if "day" in parts:  
        repdaily[key] = replicate_cols_dict({key: df}, num_replications=24)[key]  # Store result
        repdailybinary[key] = replicate_cols_dict({key: binarydict[key]}, num_replications=24)[key]
        

lyons_hourly_sum = column_sums_with_interval_dict(lyons_hourly, 1) #summing binary cols


"replicating and summing all the cols for just the GRP preturn summed for data usage"

grp_preturn_repdaily_sum = replicate_cols_dict(grp_preturn_binary_sum) # replicating 1D daily array

#%%
#import numpy as np
#import pandas as pd

"Removed class code to define a dataset and query from a directory = See 'classes_ch1_draft"




#%%
"========== Creating metadata file for all these files and folders =========="

import json
#import os

# Define the metadata as a Python dictionary
metadata = {
    "folder_name": "Processed Chapter 1 Folder",
    "description": "This folder contains all 2D arrays for failing and crooked lake vars for 2021 and 2022.",
    "created_by": "Rajeev Kumar",
    "re-created_on": "2025-02-10",
    "folder_explanations": {
        "main": "All folders are summarized half meter cells - except {original} arrays.",
        "GRP_arrays": "All GRP arrays in non {GRP_varied/variation} naming convention folders are mass 200, p=0.4, fDO Arend modified (slope, int).",
        "strat": "observation period of interest",
        "preturn": "preturnover",
        "lyons": "binary using T & DO",
        "binary": "(T, DO separated w. lyons) and grp > 0",
        "repdaily...": "days replicated to match respective hourly file- useful for visualization"
    },
    "tags": ["project_Ch1", "data_analysis", "macbook", "google drive", "SSD", "place in SharedDrive and/or depot"]
}

# Define the folder path where you want to save the JSON file
#folder_path = "/path/to/your/folder"  # Replace with the actual path to your folder

# Ensure the folder exists
os.makedirs(save_dir_main, exist_ok=True)

# Define the path for the JSON file
json_file_path = os.path.join(save_dir_main, "metadata.json")

# Save the metadata as a JSON file
with open(json_file_path, "w") as json_file:
    json.dump(metadata, json_file, indent=4)  # `indent=4` makes the JSON file human-readable

print(f"✅ Metadata saved to {json_file_path}")


#%%

"Use to slice full binary GRP arrays or calculate them from fucntion and corresponding T, DO directories"

"Modifying fragile time funcs"


"loading into dictionaries"


save_folder = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Processed_data_Ch1/GRP_variation_p_mass_fdo_notsliced"
#binary_folder = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Processed_data_Ch1/GRP_variation_binary_notsliced"

grp_dict = load_csv_files(save_folder)
#grp_binary_dict = load_csv_files(binary_folder)



def pls_make_dates(data_dict):
    
    date_ranges_full = {}
    date_ranges_preturn = {}
    
    base_dates = {
        'fail': {2021: date(2021, 6, 5), 2022: date(2022, 4, 12)},
        'crook': {2021: date(2021, 6, 24), 2022: date(2022, 6, 7)}
    }
    
    preturn_dates = {
        'fail': {2021: date(2021, 9, 30), 2022: date(2022, 9, 30)},
        'crook': {2021: date(2021, 9, 30), 2022: date(2022, 9, 30)}
    }
    
    for key,array in data_dict.items():
        parts = key.lower().split('_')
        print(f'parts {parts}')
        #parts_preturn = parts + ['preturn']
        lake, year = parts[-3], int(parts[-1])
        #date_keymatch = f"{lake}_{year}_{scale}"
        date_ranges_full[key] = (base_dates[lake][year], date(year, 12, 1))
        print(f'date_ranges_full {date_ranges_full}')
        #key_preturn = f"{key.lower()}_preturn" # f"{key.lower()}_preturn" f"preturn_{key.lower()}"
        date_ranges_preturn[key] = (preturn_dates[lake][year], date(year, 11,3))
        
    return date_ranges_full, date_ranges_preturn
    

def date_arrays_for_data_start_all(daterange_dict, data_dict):
    """
    Generate date arrays for each key in multiple datasets based on matching start dates.
    
    Parameters:
        date_dict (dict): Dictionary {key: (start_date, end_date)}.
        *data_dicts (dict): Multiple dictionaries {key: 2D NumPy array}.
    
    Returns:
        dict: Dictionary {key: NumPy array of dates}.
    """
    date_arrays = {}
    
    for key, data in data_dict.items(): #date_dict[date_key]   # using date dictionary for conditional date assingment 
        parts_key = key.lower().split('_')
        scale = parts_key[-2] 
        num_intervals = data.shape[1]
        #for key in date_dict():
        start_date, end_date = daterange_dict.get(key, (None, None))
            #parts_data = key.lower().split('_')
            
            #start_date = date_dict[date_key][0]
            
        if scale == 'day':
            dates = np.array([start_date + timedelta(days=i) for i in range(num_intervals)])
        else:
            dates = np.array([start_date + timedelta(hours=i) for i in range(num_intervals)])
            #if date_key == f"{key}_preturn":
        date_arrays[key] = dates
        
    return date_arrays


def extract_time_slices_singledict(data_dict, date_dict, daterange_dict):
    """
    Extracts subsets of data for given date ranges for multiple datasets.

    Parameters:
        data_dicts (list): List of dictionaries of {array_name: 2D NumPy array}.
        date_dict (dict): Dictionary of {lake_scale_var_year: date_values}.
        date_ranges (dict): Dictionary of {lake_scale_var_year: (start_date, end_date)}.

    Returns:
        list: [(full_period_dict1, preturn_period_dict1), (full_period_dict2, preturn_period_dict2)]
    """
    #results = []

    date_slices = {}
    #preturn_slices = {}

    for key, data in data_dict.items():
        #parts = key.split('_')
        #lake, scale, year = parts[-3], parts[-2], int(parts[-1])
            #key = f"{lake}_{timescale}_{var}_{year}"
            
        print(f"\n🔍 Processing array: {key} in data_dict")
        #for date_key in daterange_dict:
        if key in date_dict:
            date_vals = date_dict[key]
            start_range, end_range = daterange_dict.get(key, (None, None))

            print(f"  🔹 Found date range: {start_range} to {end_range}")
            print(f"  📆 Date values (first 5): {date_vals[:5]}")

            if start_range and end_range:
                # Ensure format consistency
                if isinstance(date_vals[0], datetime):
                    start_range = datetime(start_range.year, start_range.month, start_range.day)
                    end_range = datetime(end_range.year, end_range.month, end_range.day) + timedelta(days=1)

                indices = np.where((date_vals >= start_range) & (date_vals <= end_range))[0]

                    # Prevent out-of-bounds indices
                max_columns = data.shape[1]
                indices = indices[indices < max_columns]

                if len(indices) == 0:
                    print(f"⚠️ Warning: No valid indices found for {key}")

                date_slices[key] = data[:, indices]  # Store full-period slice

    return date_slices

# a,b = daterange_full.get('GRP_lethal_P0.2_mass200_slope0.138_intercept1.63_crook_day_2021',(None, None))

#%%



"""
save_arrays_to_csv(preturn_slices, os.path.join(save_dir_main, "preturn_slices"))

save_arrays_to_csv(full_slices, os.path.join(save_dir_main, "full_slices"))

save_arrays_to_csv(bin_strat, os.path.join(save_dir_main, "binary_strat"))

save_arrays_to_csv(bin_turn, os.path.join(save_dir_main, "binary_preturn"))
        

######### Above are saved arrays that have been sliced - lyons arrays have to be recreated from sliced DO, T or function has to be modified (key logic too rigid)


"Running and saving"

"creating date ranges for period and preturn"
daterange_full, daterange_pturn = pls_make_dates(grp_dict)   

"creating date_dict"
date_arrays_full = date_arrays_for_data_start_all(daterange_full, grp_dict)

"running"
datetestkeys = extract_time_slices_singledict(grp_dict, date_arrays_full, daterange_full)
datetestkeys_pturn = extract_time_slices_singledict(grp_dict, date_arrays_full, daterange_pturn)

#binary - same keys as values 
grp_var_binary_full = extract_time_slices_singledict(grp_binary_dict, date_arrays_full, daterange_full)
grp_var_binary_preturn = extract_time_slices_singledict(grp_binary_dict, date_arrays_full, daterange_pturn)


"saving all files"
# Define save paths
#save_dir_main = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Processed_data_Ch1/"

# GRP values full period
#save_arrays_to_csv(datetestkeys, os.path.join(save_dir_main, "GRP_variation_sliced_rawvals_fullperiods"))
# GRP values preturnover 
#save_arrays_to_csv(datetestkeys_pturn, os.path.join(save_dir_main, "GRP_variation_sliced_fullperiods"))


# GRP binary values full period
#save_arrays_to_csv(grp_var_binary_full, os.path.join(save_dir_main, "GRP_varied_sliced_fullperiods_binaryall_full"))
# GRP binary values preturnover 
#save_arrays_to_csv(grp_var_binary_preturn, os.path.join(save_dir_main, "GRP_varied_sliced_fullperiods_binaryall_preturn"))




######### Above are more code for running and saved arrays that have been sliced - GRP variations are massive and nearly shut down ol spyder

"""

#%%

"------------------- This Cell onwards = tabular/matrix summaries, plotting GRP variation metrics ------------------"


# tasks - can apply to all GRP models 


##   summing positive cells > Export sums 

print(grp_binary_summed)
##   Use sums to calculate proportion of postive cells > split by lake (num_rows)
##       .. e.g. num_pos/num_rows

grp_prop_pos = {} #{k:(column_sums_with_interval_dict(v)/v.shape[0]) for k,v in grp_binary_all}
for k, v in grp_binary_all.items():
    grp_prop_pos = grp_binary_all[k]:(column_sums_with_interval_dict(grp_binary_all[v], interval = 1)/v.shape[0])
    



print(grp_prop_pos)


" ######### defining dictionary with key value pairs for parsing complex dictionaries for this project "

ch1_key_parser = {'lakes': ['crook', 'fail'], 'variables': ['do', 'temp', 'grp', 'lyons'], 'timescales': ['day', 'hr'], 'years': ['2021', '2022', '2023'],
                  'slopes': ['slope0.138','slope0.168'], 'intercepts': ['intercept1.63','intercept2.09'], 'fDO_lethal': ['lethal'],
                  'masses': ['mass200', 'mass400', 'mass600'], 'p_vals': ['p0.2', 'p0.4', 'p0.6','p0.8']}




def select_arrays(data_dictionary, ch1_key_parser, key_conditions):
    """
    Select arrays from a data dictionary based on key conditions and a key parser.
    
    Parameters:
        data_dictionary (dict): Dictionary of {filename: array}.
        ch1_key_parser (dict): Dictionary of {variable_type: [possible_values]}.
        key_conditions (list): List of strings representing variable names or key patterns.
    
    Returns:
        dict: Dictionary of {filename: array} containing only the selected arrays.
    """
    output_dict = {}
    
    # Generate all possible key patterns based on key_conditions
    possible_patterns = []
    for condition in key_conditions:
        if condition in ch1_key_parser:
            # If the condition is a variable type in ch1_key_parser, get all possible values
            possible_patterns.extend(ch1_key_parser[condition])
        else:
            # Treat the condition as a direct key pattern
            possible_patterns.append(condition)
    
    # Match possible patterns against data_dictionary keys
    for key in data_dictionary:
        for pattern in possible_patterns:
            if pattern in key:  # Check if the pattern matches the key
                output_dict[key] = data_dictionary[key]
                break  # Stop checking other patterns for this key
    
    return output_dict



""" RUNNING TEST FUNCTION EXAMPLE TO RETURN DICT ITEMS FLEXIBLY

selected_arrays = select_arrays(data_dictionary, ch1_key_parser, key_conditions)

# Print the selected arrays
for key, array in selected_arrays.items():
    print(f"Key: {key}")
    print(f"Array Shape: {array.shape}")
"""




#   Export percent_pos








































