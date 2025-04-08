#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:28:47 2025


Taking all the plotting code from the dataframe ("Classes Ch1") script because it needs organization


@author: rajeevkumar
"""


import os
import re
import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import style 
#style.use()    #



class Dataset:
    def __init__(self, filepath, metadata=None):
        """
        Initialize a dataset object.

        Parameters:
            filepath (str): Path to the dataset file.
            metadata (dict): Metadata about the dataset (e.g., keys, variables, etc.).
        """
        self.filepath = filepath
        self.metadata = metadata or {}
        self.data = None  # Data will be loaded lazily

    def load(self):
        """
        Load the dataset into memory.
        """
        if self.data is None:
            # Load data from file (e.g., CSV, NPY, etc.)
            if self.filepath.endswith(".csv"):
                self.data = pd.read_csv(self.filepath, header = None).to_numpy()
                print((f"Loaded csv to array from {self.filepath} with shape: {self.data.shape}"))
            elif self.filepath.endswith(".npy"):
                self.data = np.load(self.filepath)
            else:
                raise ValueError(f"Unsupported file format: {self.filepath}")
        return self.data

    def sort(self, by=None):
        """
        Sort the dataset by a specific column or key.

        Parameters:
            by (str): Column or key to sort by.
        """
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data.sort_values(by=by)
        elif isinstance(self.data, np.ndarray):
            self.data = self.data[np.argsort(self.data[:, by])]
        else:
            raise ValueError("Sorting is only supported for DataFrames and NumPy arrays.")

    def filter(self, condition):
        """
        Filter the dataset based on a condition.

        Parameters:
            condition (callable): A function that returns a boolean mask.
        """
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data[condition(self.data)]
        elif isinstance(self.data, np.ndarray):
            self.data = self.data[condition(self.data)]
        else:
            raise ValueError("Filtering is only supported for DataFrames and NumPy arrays.")

    def __repr__(self):
        return f"Dataset(filepath={self.filepath}, metadata={self.metadata})"


# Define metadata parser
ch1_metadata = {
    'lakes': ['crook', 'fail', 'crooked', 'failing'],
    'variables': ['do', 'temp', 'grp', 'lyons'],
    'timescales': ['day', 'hr', 'daily', 'hourly'],
    'years': ['2021', '2022', '2023'],
    'periods': ['strat', 'pturn', 'stratified', 'preturn'],
    'slopes': ['slope0.138', 'slope0.168'],
    'intercepts': ['intercept1.63', 'intercept2.09'],
    'fDO_lethal': ['lethal'],
    'masses': ['mass200', 'mass400', 'mass600'],
    'p_vals': ['p0.2', 'p0.4', 'p0.6', 'p0.8'],
}



# new code - debugging test successful

def extract_metadata_from_filename(filename):
    """
    Extract metadata from a filename based on ch1_metadata.

    Parameters:
        filename (str): The filename to parse.

    Returns:
        dict: Extracted metadata.
    """
    metadata = {}
    for key, values in ch1_metadata.items():
        for value in values:
            if value in filename:
                metadata[key] = value
                break
    return metadata


def load_datasets_from_directory(directory):
    """
    Load datasets from a directory and extract metadata from filenames.

    Parameters:
        directory (str): Path to the directory containing dataset files.

    Returns:
        dict: Dictionary of {filename: Dataset object}.
    """
    datasets = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv") or filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            metadata = extract_metadata_from_filename(filename)
            datasets[filename] = Dataset(filepath, metadata)
    return datasets


def query_datasets(datasets, **conditions):
    """
    Query datasets based on metadata conditions.

    Parameters:
        datasets (dict): Dictionary of {filename: Dataset object}.
        **conditions: Key-value pairs for filtering (e.g., lakes="crook", years="2021").

    Returns:
        dict: Dictionary of {filename: Dataset object} matching the conditions.
    """
    results = {}
    for filename, dataset in datasets.items():
        match = True
        for key, value in conditions.items():
            if key not in dataset.metadata or dataset.metadata[key] != value:
                match = False
                break
        if match:
            results[filename] = dataset
    return results


"loading datasets with queries into dictionary of keys and arrays (key, val)"
# Ensure datasets are loaded into NumPy arrays
def load_datasets_asdict(dataset_dict):
    data_dict = {}
    for filename, dataset in dataset_dict.items():
        #dataset.load()  # Load the data from CSV or NPY file
        data_dict[filename] = dataset.load()  # Extract only the NumPy array
    return data_dict


def extract_first_datetime(nested_dict):
    """
    Extracts the first datetime value from a nested dictionary structured as:
    {outer_key: {model_key: DatetimeIndex (as a 1D array)}}

    Args:
        nested_dict (dict): A dictionary where the innermost values are Pandas DatetimeIndex objects.

    Returns:
        dict: A new dictionary mapping each outer key to its first datetime value.
    """
    first_datetime_values = {}

    for outer_key, middle_dict in nested_dict.items():
        if isinstance(middle_dict, dict):  # Ensure it's a dictionary
            for model_key, datetime_index in middle_dict.items():
                if isinstance(datetime_index, pd.DatetimeIndex) and len(datetime_index) > 0:
                    first_datetime_values[outer_key] = datetime_index # include [0] at end if you want to Extract first timestamp
                    break  # Stop after first valid datetime found

    return first_datetime_values

        
def time_index_by_lakeyear(nested_dict, start_end_dates): # num_intervals,
    """create dict of time indices based on key name conditions from nested dictionary"""
    num_intervals ={}
    time_indices = {}
    
    datekeys = {datekey:datekey for datekey in start_end_dates}
    lakekeys = {lakekey:lakekey for lakekey in nested_dict}
    #datekey_info  = f"{start_end_dates.keys()}".lower().split('_')
    #dates_period, dates_year, dates_lake = datekey_info[datekey]
    print(f"date_dict info, {datekeys}")
    print(f"lakekeys info, {lakekeys}")
    for key, data_dict in nested_dict.items():
        if datekeys[key] == lakekeys[key]:
            #for k,v in data_dict.items():
            num_intervals[key] = {k:v.shape[0] for k,v in data_dict.items()}
            
            #for k,v in num_intervals.items():
            time_indices[key] = {k: pd.date_range(start=start_end_dates[key]['start_date'], periods=num_intervals[key][k], freq="h", tz="America/New_York") for k,v in data_dict.items()}
    
    return time_indices

def extract_metadata(column_name, pattern):
    """Extract metadata from a column name using regex."""
    match = re.search(pattern, column_name)
    return match.groupdict() if match else {}

def extract_matching_columns(df, pattern, user_inputs):
    """
    Extracts column names that match user-specified variable values,
    allowing selection of lethal/non-lethal columns.
    """
    matching_columns = []
    
    for col in df.columns:
        metadata = extract_metadata(col, pattern)
        if metadata:
            is_lethal = "lethal" in col  # Check if the column contains "lethal"
            if user_inputs.get("lethal", None) is True and not is_lethal:
                continue  # Skip non-lethal columns
            if user_inputs.get("lethal", None) is False and is_lethal:
                continue  # Skip lethal columns
            if all(str(user_inputs.get(k, '')) == metadata.get(k, '') for k in user_inputs if k != "lethal"):
                matching_columns.append(col)
    return matching_columns


def save_dfs_to_csv(df_dict, save_dir):
    """Saves each DataFrame in df_dict to CSV using its key as the filename."""
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    for key, df in df_dict.items():
        file_path = os.path.join(save_dir, f"{key}.csv")  # Construct filename
        df.to_csv(file_path, index=True)  # Save with index
        print(f"Saved: {file_path}")  # Debugging message

def simple_ndict_to_df(nested_dict, time_dict):
    df_dict = {}  # Dictionary to store DataFrames

    for lk_yr in nested_dict:
        lk_yr_dict = nested_dict[lk_yr]
        df = pd.DataFrame.from_dict(lk_yr_dict)
        df.index = time_dict[lk_yr]
        df.index.name = "EST_DateTime"

        # Store DataFrame in dictionary with the key name
        df_dict[lk_yr] = df

        print(f"Created DataFrame for: {lk_yr}, Shape: {df.shape}")  # Optional print for debugging

    return df_dict  # Return


    
def read_df_dir_tocsv(directory):
    return {os.path.splitext(f)[0]: pd.read_csv(os.path.join(directory, f)) 
        for f in os.listdir(directory) if f.endswith('.csv')}


def clean_column_name(col_name):
    """Removes '.csv' and underscores from column names for better legend readability."""
    return col_name.replace('.csv', '').replace('_', ' ')


    
def plot_time_series(df, lake_year_period, pattern, user_inputs, user_title = None, start_date=None, end_date=None):
    """
    Plots a time series for selected variables from the dataframe within a date range.
    """
    if lake_year_period not in df:
        raise ValueError(f"Key '{lake_year_period}' not found in df.")

    df = df[lake_year_period]
    selected_columns = extract_matching_columns(df, pattern, user_inputs)
    if not selected_columns:
        raise ValueError("No matching columns found for the selected criteria.")

    df['EST_DateTime'] = pd.to_datetime(df['EST_DateTime'], utc=True)
    if start_date and end_date:
        df = df[(df['EST_DateTime'] >= start_date) & (df['EST_DateTime'] <= end_date)]
    
    plt.figure(figsize=(12, 8))

    #for col in selected_columns:
        #color = 'royalblue' if 'hr' in col else 'darkorange' if 'day' in col else 'black'
        #plt.plot(df['EST_DateTime'], df[col], label=clean_column_name(col), color=color, linewidth=2)
    for col in selected_columns:
        if 'day' in col:
            color = 'darkorange'
            z = 3  # Higher zorder: plotted on top
        elif 'hr' in col:
            color = 'royalblue'
            z = 2  # Lower zorder: plotted first
        else:
            color = 'black'
            z = 2

        plt.plot(df['EST_DateTime'], df[col], label=clean_column_name(col), color=color, linewidth=2, zorder=z)


    # Set a fixed y-axis based on the overall range of the data
    y_min = 0
    #y_max = df[selected_columns].max().max()
    y_max = 60 if df[selected_columns].max().max() > 30 else 30
    plt.ylim(y_min, y_max)
    # Set the y-axis limits: minimum fixed at 0, maximum dynamically based on column lengths.
    # We use the maximum length among all selected columns.
    # Set y-axis top limit to the number of rows in the selected column (or overall DataFrame)
    #y_max = len(df[selected_columns[0]])
    #plt.ylim(plt.ylim()[0], y_max)

    # Extract lake, year, and period for a separate title line
    lake, year, period = lake_year_period.split('_')  # Assuming underscore-separated format
    
    plt.xlabel("DOY", fontsize=13, fontweight='bold')
    plt.ylabel("Positive GRP Cell Count", fontsize=13, fontweight='bold')

    # Separate lake-year-period from the title - DYNAMIC TITLE NAMING
    plt.suptitle(f"{lake} {year} {period}", fontsize=16, fontweight='bold', fontname = 'Times')  # First line
    #plt.title(f"Time Series Plot for {', '.join(selected_columns)}", fontsize=16, fontweight='bold', pad=15, fontname = 'TImes New Roman')  # Second line

    # STATIC USER TITLE
    plt.title(user_title, fontsize=16, fontweight='bold', fontname = 'Times')

    plt.legend(fontsize=12, frameon=False)
    plt.xticks(fontsize=12, fontname = 'Times New Roman', rotation = 45)
    plt.yticks(fontsize=12, fontname = 'Times New Roman')
    plt.grid(False)  # Remove grid lines
    plt.tight_layout()  # Improve spacing for better publication quality
    plt.show()
    

"""
loading in global variables 
"""
pturn_dates_2021 = {'start_date': '2021-09-30', 'end_date': '2021-11-03'}
pturn_dates_2022 = {'start_date': '2022-09-30', 'end_date': '2022-11-03'}

strat_dates_2021_crook = {'start_date': '2021-06-24', 'end_date': '2021-11-11'}
strat_dates_2021_fail = {'start_date': '2021-06-05', 'end_date': '2021-12-01'}

strat_dates_2022_crook = {'start_date': '2022-06-07', 'end_date': '2022-12-01'}
strat_dates_2022_fail = {'start_date': '2022-04-12', 'end_date': '2022-12-01'}

start_end_dates = {'crook_2021_pturn':{'start_date': '2021-09-30', 'end_date': '2021-11-03'}, 'crook_2022_pturn': {'start_date': '2022-09-30', 'end_date': '2022-11-03'},
                   'fail_2021_pturn':{'start_date': '2021-09-30', 'end_date': '2021-11-03'}, 'fail_2022_pturn': {'start_date': '2022-09-30', 'end_date': '2022-11-03'},
                  'crook_2021_strat': {'start_date': '2021-06-24', 'end_date': '2021-11-11'}, 'fail_2021_strat': {'start_date': '2021-06-05', 'end_date': '2021-12-01'},
                 'crook_2022_strat': {'start_date': '2022-06-07', 'end_date': '2022-12-01'}, 'fail_2022_strat': {'start_date': '2022-04-12', 'end_date': '2022-12-01'}
                 }

# Regex pattern to extract type (GRP vs. GRP_lethal)
pattern = r"(?i)(?P<type>grp|grp_lethal)_p(?P<P>[0-9.]+)_mass(?P<mass>[0-9]+)_slope(?P<slope>[0-9.]+)_intercept(?P<intercept>[0-9.]+)"

# Regex pattern to extract type lyons_lake_timescale_year .csv files
pattern_lyons = "lyons_(crooked|failing)_(daily|hourly)_(2021|2022)\.csv"

# Regex pattern to extract type lake__ (temp OR DO)_timescale_year .csv files
pattern_TempDO = "(crooked|failing)_(daily|hourly)_(do|temp)_(2021|2022)\.csv"

"end loading packages, defining functions, metadata, defining classes, and globals for loading, sorting, and preprocessing arrays to dataframes"
    
#%%
" ----================================== PLOTTING CODE- LINEPLOTS  ==========================================-----"
" ----------------- LINEPLOTS HERE - Need removal of underscores and title modifications ------------"


dfscount = read_df_dir_tocsv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Redo_pos_summaries_dfs/NumCells')
dfsprop = read_df_dir_tocsv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Redo_pos_summaries_dfs/Proportions')

# Example Usage:
# Define regex pattern for extracting variable values from column names
pattern = r"grp(?:_lethal)?_p(?P<P>[0-9.]+)_mass(?P<mass>[0-9]+)_slope(?P<slope>[0-9.]+)_intercept(?P<intercept>[0-9.]+)"




user_inputs_fail = {"P":0.4, "mass": 200, "slope": 0.168, "intercept": 1.63, "lethal": False}
user_inputs_fail_lethal = {"P":0.4, "mass": 200, "slope": 0.168, "intercept": 1.63, "lethal": True}

user_inputs_crook = {"P":0.4, "mass": 400, "slope": 0.168, "intercept": 1.63, "lethal": False}
user_inputs_crook_lethal = {"P":0.4, "mass": 400, "slope": 0.168, "intercept": 1.63, "lethal": True}



strats = [key for key in dfscount if "strat" in key]
#print(strats)
pturns = [key for key in dfscount if "pturn" in key]
#print(pturns)

#run the time series plot function
for strat in strats:
    if "fail" in strat:
        user_input = user_inputs_fail_lethal
        user_title = 'Sub-daily vs. Daily temporal resolution for Failing Lake Positive GRP cell counts - (fDO lethal)'
    else:
        user_input = user_inputs_crook
        user_title = 'Sub-daily vs. Daily temporal resolution for Crooked Lake Positive GRP cell counts' #- (fDO lethal)'
    plot_time_series(dfscount, strat, pattern, user_input, user_title)
    #plot_heatmap1D(dfscount, strat, pattern, user_input)
        

for pturn in pturns:
    if "fail" in pturn:
        user_input = user_inputs_fail_lethal
        user_title = 'Sub-daily vs. Daily temporal resolution for Failing Lake Positive GRP cell counts - (fDO lethal)' #- (fDO lethal)'
    else:
        user_input = user_inputs_crook_lethal
        user_title = 'Sub-daily vs. Daily temporal resolution for Crooked Lake Positive GRP cell counts - (fDO lethal)' #- (fDO lethal)'
    plot_time_series(dfscount, pturn, pattern, user_input, user_title)
    

for pturn in pturns:
    if "fail" in pturn:
        user_input = user_inputs_crook
        user_title = 'Sub-daily vs. Daily temporal resolution for Failing Lake Positive GRP cell counts' #- (fDO lethal)'
    #elif "fail" in pturn and "lethal" not in pturn:
        #user_input = user_inputs_crook
        #user_title = 'Sub-daily vs. Daily temporal resolution for Failing Lake Positive GRP cell counts' 
    plot_time_series(dfscount, pturn, pattern, user_input, user_title)
        

#%%
" ----================================== FUNCTIONS PLOTTING CODE- 2D ARRAYS   ==========================================-----"
" ----------------- ARRAYS HERE - Need removal of underscores and title modifications ------------"

"---------------------------- Plotting 2D heatmaps  ------------------------"

from datetime import datetime
from datetime import date, timedelta
from operator import itemgetter


def extract_metadata(obj_name, pattern):
    """Extract metadata from a loaded object's name using regex."""
    match = re.search(pattern, obj_name)
    return match.groupdict() if match else {}

def extract_matching_arrays(arr_dict, lake_yr_period, pattern, user_inputs):
    """
    Extracts column names that match user-specified variable values,
    allowing selection of lethal/non-lethal columns.
    """
    matching_arrays = []
    
    for arrayname in arr_dict[lake_yr_period]:
        metadata = extract_metadata(arrayname, pattern)
        if metadata:
            is_lethal = "lethal" in arrayname  # Check if the column contains "lethal"
            if user_inputs.get("lethal", None) is True and not is_lethal:
                continue  # Skip non-lethal columns
            if user_inputs.get("lethal", None) is False and is_lethal:
                continue  # Skip lethal columns
            if all(str(user_inputs.get(k, '')) == metadata.get(k, '') for k in user_inputs if k != "lethal"):
                matching_arrays.append(arrayname)
    return matching_arrays


# Step 1: Preprocess `start_end_dates` to include both hourly and daily keys
def preprocess_start_end_dates(start_end_dates):
    expanded_dates = {}
    for key, value in start_end_dates.items():
        expanded_dates[f"{key}_hr"] = value  # Hourly version
        expanded_dates[f"{key}_day"] = value  # Daily version
    return expanded_dates

def time_index_by_lakeyear_2D(nested_dict, start_end_dates):
    """
    Create a dictionary of time indices based on preprocessed start_end_dates.
    Handles both hourly and daily intervals efficiently.
    """
    num_intervals = {}
    time_indices = {}

    # Expand start_end_dates to include both `hr` and `day` versions
    start_end_dates = preprocess_start_end_dates(start_end_dates)

    print(f"Expanded date dictionary keys: {list(start_end_dates.keys())}")
    
    for key, data_dict in nested_dict.items():
        if key not in start_end_dates:
            print(f"⚠️ Warning: No start/end date found for {key}. Skipping.")
            continue  # Skip if the key is not in the expanded dictionary

        # Compute number of time intervals
        num_intervals[key] = {k: v.shape[1] for k, v in data_dict.items()}

        # Determine frequency based on `hr` (hourly) or `day` (daily)
        freq = "h" if "hr" in key or "hourly" in key else "d"

        # Generate time indices
        time_indices[key] = {
            k: pd.date_range(
                start=start_end_dates[key]['start_date'],
                periods=num_intervals[key][k],
                freq=freq,  
                tz="America/New_York"
            ) for k, v in data_dict.items()
        }

    return time_indices



def time_index_by_lakeyear_2D_nongrp(flat_dict, start_end_dates):
    """
    Create a dictionary of time indices based on preprocessed start_end_dates.
    Handles both hourly and daily intervals efficiently.
    """
    num_intervals = {}
    time_indices = {}

    # Expand start_end_dates to include both `hr` and `day` versions
    start_end_dates = preprocess_start_end_dates(start_end_dates)

    print(f"Expanded date dictionary keys: {list(start_end_dates.keys())}")
    
    for key, array in flat_dict.items():
        parts = key.split()
        if 'lyons' in parts:
            _, lake, timescale, year = parts[0], parts[1], parts[2], parts[3]
        else:
            lake, var, timescale, year = parts[0], parts[1], parts[2], parts[3]
        # Compute number of time intervals
        num_time_steps = array.shape[1]

        # Determine frequency based on `hr` (hourly) or `day` (daily)
        freq = "h" if "hr" in key or "hourly" in key else "d"

        # Generate datetime index
        time_indices[key] = pd.date_range(
            start=start_end_dates[key]['start_date'],
            periods=num_time_steps,
            freq=freq,
            tz="America/New_York"
        )
        # Debugging prints
        print(f"✅ Created time index for {key}: {len(time_indices[key])} timestamps ({freq})")
        
    return time_indices


def plot_heatmapnew(arr_dict, lake_yr_period, pattern, user_inputs, date_dict, start_date=None, end_date=None):
    """
    Plots heatmaps for selected 2D arrays from a nested dictionary structure 
    {lake_yr_period: {array_name: 2D array}}, using corresponding time from date_dict as the x-axis.
    Uses `imshow()` instead of `seaborn.heatmap()`.
    """
    if lake_yr_period not in arr_dict:
        raise ValueError(f"Key '{lake_yr_period}' not found in arr_dict.")

    selected_arrays = extract_matching_arrays(arr_dict, lake_yr_period, pattern, user_inputs)
    if not selected_arrays:
        raise ValueError("No matching arrays found for the selected criteria.")

    # Ensure datetime index is formatted correctly
    date_dict[lake_yr_period] = pd.to_datetime(date_dict[lake_yr_period], utc=True)

    # Filter date range if provided
    if start_date and end_date:
        mask = (date_dict[lake_yr_period] >= start_date) & (date_dict[lake_yr_period] <= end_date)
        date_labels = date_dict[lake_yr_period][mask]
    else:
        date_labels = date_dict[lake_yr_period]  # Use full time range

    for selection in selected_arrays:
        array_data = arr_dict[lake_yr_period][selection]  # Get the 2D array
        
        # ✅ Detect and adjust time steps automatically
        num_time_steps = array_data.shape[1]
        num_dates = len(date_labels)

        if num_time_steps == num_dates * 24:  # Dataset is hourly, but dates are daily
            date_labels = np.repeat(date_labels, 24)  # Expand daily timestamps into hourly
        elif num_time_steps == num_dates / 24:  # Dataset is daily, but dates are hourly
            date_labels = date_labels[::24]  # Downsample hourly timestamps to daily

        # ✅ Final check: Ensure time matches the number of columns in the array
        if array_data.shape[1] != len(date_labels):
            raise ValueError(f"Mismatch: {selection} has {array_data.shape[1]} time steps, but date_dict has {len(date_labels)}.")

        plt.figure(figsize=(12, 8))
        plt.imshow((array_data), aspect='auto', cmap='seismic', origin='lower') #np.flipud(array_data),...

        # Add colorbar
        plt.colorbar(label='Value')

        # Set axis labels
        plt.xlabel("DOY")
        plt.ylabel("Depth (m)")
        plt.title(f"Heatmap for {selection} in {lake_yr_period}")

        # Set x-axis ticks
        x_ticks = np.linspace(0, array_data.shape[1] - 1, min(10, len(date_labels))).astype(int)
        plt.xticks(x_ticks, date_labels[x_ticks].strftime('%Y-%m-%d'), rotation=45, ha='right')

        # Flip y-axis so depth increases downward
        plt.gca().invert_yaxis()

        plt.show()


" End loading time series visualization and plotting functions "

#%%



" GRP base dirs are the lakes and years"
#vals_strats = ''
base_dirs = {
    'grp_vals_pturn':'/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/preturn_lakesyears/GRPvals_preturn',
    'grp_vals_strat':'/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/stratified_lakesyears/GRPvals_strat',

    'grp_bi_pturn': '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/preturn_lakesyears/GRPbinary_preturn',
    'grp_bi_strat': '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/stratified_lakesyears/GRPbinary_strat',

    }


"  ========  # Steps - info : Query datasets based on metadata and generate ========="
# Example usage
datasets_vals = {k:load_datasets_from_directory(v) for k,v in base_dirs.items() if 'vals' in k}
datasets_binary = {k:load_datasets_from_directory(v) for k,v in base_dirs.items() if 'bi' in k}


# Flatten the datasets into a single dictionary for easier querying
#all_datasets_vals = {filename: dataset for dset in datasets_vals.values() for filename, dataset in dset.items()}

# Step 2: Query datasets dynamically - these list iterations will select and plot all 
lake_names = ["crook", "fail"]
years = ["2021", "2022"]
t_scales = ["hr", "day"]
#periods = ["strat", "preturn"]

######--------------------------------- VALUES GRP ----------------------------------###############
# Dictionary to store query results
query_results_vals = {}

# Iterate through lake name,tscale, years, and dataset types - can do all individually using metadata above
for lake in lake_names:
    for year in years:
        for t_scale in t_scales:
            for dataset_key, dataset_dict in datasets_vals.items():
                # Determine if this dataset type belongs to stratified or preturn
                if "strat" in dataset_key:
                    query_results_vals[f"{lake}_{year}_strat_{t_scale}"] = query_datasets(
                        dataset_dict, lakes=lake, timescales=t_scale, years=year, variables="grp"
                        )
                elif "pturn" in dataset_key:
                    query_results_vals[f"{lake}_{year}_pturn_{t_scale}"] = query_datasets(
                        dataset_dict, lakes=lake, timescales=t_scale, years=year, variables="grp"
                        )

######--------------------------------- BINARY GRP ----------------------------------###############
# Dictionary to store query results
query_results_binary = {}  

# Iterate through lake name,tscale, years, and dataset types - can do all individually using metadata above
for lake in lake_names:
    for year in years:
        for t_scale in t_scales:
            for dataset_key, dataset_dict in datasets_binary.items():
                # Determine if this dataset type belongs to stratified or preturn
                if "strat" in dataset_key:
                    query_results_binary[f"{lake}_{year}_strat_{t_scale}"] = query_datasets(
                        dataset_dict, lakes=lake, timescales=t_scale, years=year, variables="grp"
                        )
                elif "pturn" in dataset_key:
                    query_results_binary[f"{lake}_{year}_pturn_{t_scale}"] = query_datasets(
                        dataset_dict, lakes=lake, timescales=t_scale, years=year, variables="grp"
                        )                   
                    

# Convert query results to NumPy arrays -------Vals
query_data_vals = {key: load_datasets_asdict(value) for key, value in query_results_vals.items()}

# Convert query results to NumPy arrays ------Binary
query_data_binary = {key: load_datasets_asdict(value) for key, value in query_results_binary.items()}

# set user inputs to query datasets
user_inputs_fail = {"P":0.4, "mass": 200, "slope": 0.168, "intercept": 1.63, "lethal": False}
user_inputs_fail_lethal = {"P":0.4, "mass": 200, "slope": 0.168, "intercept": 1.63, "lethal": True}
user_inputs_crook = {"P":0.4, "mass": 400, "slope": 0.168, "intercept": 1.63, "lethal": False}
user_inputs_crook_lethal = {"P":0.4, "mass": 400, "slope": 0.168, "intercept": 1.63, "lethal": True}

# creating time indices
period_dates = time_index_by_lakeyear_2D(query_data_vals, start_end_dates)
# Overwrite if all datetime indices are the same per lake_yr_whatever you parse by 
period_dates = extract_first_datetime(period_dates)       

# defining the outer keys for -inefficient and hardcoded- loop through nested dictionary (check 'simple ndict..' func in cell 0 )
#outer_keys_grp = ['crook_2021_strat', 'crook_2021_pturn', 'crook_2022_strat', 'crook_2022_pturn', 'fail_2021_strat', 'fail_2021_pturn', 'fail_2022_strat', 'fail_2022_pturn']

#for outer_key in outer_keys_grp:
    #plot_heatmapnew(query_data, outer_key, pattern, user_inputs_crook, period_dates)
    
"---- Plotting GRPs Values- Failing 2021 is upside down - use flipud array data"

for outer_key, array_dict in query_data_vals.items():
    if 'crook' in outer_key:
        plot_heatmapnew(query_data_vals, outer_key, pattern, user_inputs_crook, period_dates)
    else:
        plot_heatmapnew(query_data_vals, outer_key, pattern, user_inputs_fail, period_dates)
        

for outer_key, array_dict in query_data_vals.items():
    if 'crook' in outer_key:
        plot_heatmapnew(query_data_vals, outer_key, pattern, user_inputs_crook_lethal, period_dates)
    else:
        plot_heatmapnew(query_data_vals, outer_key, pattern, user_inputs_fail_lethal, period_dates)
        
  
        
 

"""
for outer_key, array_dict in query_data_binary.items():
    if 'crook' in outer_key:
        plot_heatmapnew(query_data_binary, outer_key, pattern, user_inputs_crook, period_dates)
    else:
        plot_heatmapnew(query_data_binary, outer_key, pattern, user_inputs_fail, period_dates)

 """   
    
    
    