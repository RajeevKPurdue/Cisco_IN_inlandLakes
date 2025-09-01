#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:00:26 2025

@author: rajeevkumar

GRP analyses on lyons data
"""


import os
import re
import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#from itertools import itemgetter
#from matplotlib import style 
#style.use()    #

# Regex pattern to extract type (GRP vs. GRP_lethal)
pattern = r"(?i)(?P<type>grp|grp_lethal)_p(?P<P>[0-9.]+)_mass(?P<mass>[0-9]+)_slope(?P<slope>[0-9.]+)_intercept(?P<intercept>[0-9.]+)"

# Regex pattern to extract type lyons_lake_timescale_year .csv files
pattern_lyons = "lyons_(crooked|failing)_(daily|hourly)_(2021|2022)\.csv"

# Regex pattern to extract type lake__ (temp OR DO)_timescale_year .csv files
pattern_TempDO = "(crooked|failing)_(daily|hourly)_(do|temp)_(2021|2022)\.csv"

"end loading packages, defining functions, metadata, defining classes, and globals for loading, sorting, and preprocessing arrays to dataframes"

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


def csv_toarray(fpath):   # skip_blank_lines = False
    array = pd.read_csv(fpath, header=None).to_numpy()
    print(f"Loaded array from {fpath} with shape: {array.shape}")
    return array

def load_csv_files(directory):
    "Reads into dictionary and takes fname as key without extension"
    return {os.path.splitext(f)[0]: csv_toarray(os.path.join(directory, f)) 
            for f in os.listdir(directory) if f.endswith('.csv')}


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


# Step 1: Preprocess `start_end_dates` to include both hourly and daily keys
def preprocess_start_end_dates(start_end_dates):
    expanded_dates = {}
    for key, value in start_end_dates.items():
        expanded_dates[f"{key}_hr"] = value  # Hourly version
        expanded_dates[f"{key}_day"] = value  # Daily version
    return expanded_dates

        
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




def replace_at_timestamps_with_window_avg(df, timestamps, num_before, num_after):
    """
    Replace values at user-specified timestamp(s) with an average computed 
    over a window of data from a number of time steps before and after 
    the target rows, but only for columns that include '_hr_' in the name.
    
    Parameters:
      df (pd.DataFrame): The input DataFrame containing a column 'EST_DateTime'
                         with datetime values.
      timestamps (str, datetime, or list-like): A single timestamp or a list of timestamps.
      num_before (int): The number of rows (time steps) before the target row.
      num_after (int): The number of rows (time steps) after the target row.
      
    Returns:
      pd.DataFrame: The modified DataFrame with the updated values.
    """
    
    # Make sure the EST_DateTime column is parsed as timezone-aware datetime.
    df['EST_DateTime'] = pd.to_datetime(df['EST_DateTime'], utc=True)
    
    # Ensure timestamps is a list; if not, encapsulate it in a list.
    if not isinstance(timestamps, (list, tuple, pd.Series)):
        timestamps = [timestamps]
    
    # Convert each timestamp to a timezone-aware datetime.
    timestamps = [pd.to_datetime(ts, utc=True) for ts in timestamps]
    
    # Process each provided timestamp.
    for ts in timestamps:
        # Locate matching rows; adjust this if you want to use approximate matching.
        matching_rows = df.index[df['EST_DateTime'] == ts]
        
        if matching_rows.empty:
            print(f"Warning: The timestamp {ts} was not found in the DataFrame. Skipping.")
            continue
        
        for row_idx in matching_rows:
            # Process only columns that include '_hr_' in their names.
            for col in df.columns:
                if '_hr_' in col:
                    # Determine window boundaries, ensuring we don't go out-of-bounds.
                    start_idx = max(0, row_idx - num_before)
                    end_idx = row_idx + num_after + 1  # +1 because the end index is exclusive
                    
                    # Obtain values before and after the target row, excluding the target row.
                    before_window = df.iloc[start_idx:row_idx][col]
                    after_window = df.iloc[row_idx+1:end_idx][col]
                    
                    # Combine both windows.
                    window_vals = pd.concat([before_window, after_window])
                    
                    if not window_vals.empty:
                        # Compute the average and replace the target value.
                        avg_value = window_vals.mean()
                        df.at[row_idx, col] = avg_value
                    else:
                        print(f"Warning: No available data around row {row_idx} in column '{col}'.")
    
    return df




def compare_hourly_daily_timeseries(df_dict, lake_year_period, pattern, user_inputs,
                                      user_title=None, start_date=None, end_date=None, ymax = None):
    """
    For a specific lake_year_period key from df_dict:
      - Filters the DataFrame based on a user-selected date interval.
      - Extracts matching columns using a regex pattern and user inputs.
      - Separates the results into hourly and daily columns.
      - Prints descriptive statistics for each.
      - Plots the hourly and daily series on the same axes for direct comparison.
      
    Parameters:
      df_dict (dict): Dictionary of DataFrames (e.g., loaded via read_df_dir_tocsv).
      lake_year_period (str): Key identifier to select a specific DataFrame.
      pattern (str): Regex pattern used to extract variable metadata from column names.
      user_inputs (dict): Dictionary of filtering criteria (e.g., {"P": 0.4, "mass": 400, ...}).
      user_title (str, optional): Title for the plot.
      start_date (str, optional): Start date (e.g., "2022-06-08").
      end_date (str, optional): End date (e.g., "2022-12-01").
    """
    if lake_year_period not in df_dict:
        raise ValueError(f"Key '{lake_year_period}' not found in data dictionary.")

    # Extract the DataFrame for the given key and convert EST_DateTime to timezone-aware datetime.
    df = df_dict[lake_year_period]
    df['EST_DateTime'] = pd.to_datetime(df['EST_DateTime'], utc=True)
    
    # Optionally, filter by the selected date interval.
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        df = df[(df['EST_DateTime'] >= start_dt) & (df['EST_DateTime'] <= end_dt)]

    # Extract matching columns based on the regex and user inputs.
    selected_columns = extract_matching_columns(df, pattern, user_inputs)
    if not selected_columns:
        raise ValueError("No matching columns found for the selected criteria.")

    # --- New Step: Separate matching columns by timescale (hourly vs daily) ---
    hourly_columns = [col for col in selected_columns if 'hourly' in col.lower()]
    daily_columns  = [col for col in selected_columns if 'daily' in col.lower()]
    
    if not hourly_columns or not daily_columns:
        raise ValueError("Both hourly and daily columns must be present to perform comparison.")

    # --- New Step: Print Descriptive Statistics ---
    print("== Hourly Columns Descriptive Statistics ==")
    print(df[hourly_columns].describe())
    print("\n== Daily Columns Descriptive Statistics ==")
    print(df[daily_columns].describe())

    # --- New Step: Plot both sets of time series on the same figure ---
    plt.figure(figsize=(12, 8))
    
    for col in hourly_columns:
        plt.plot(df['EST_DateTime'], df[col],
                 label=clean_column_name(col) + " - Hourly",
                 color='royalblue', linewidth=2, zorder=2)
    for col in daily_columns:
        plt.plot(df['EST_DateTime'], df[col],
                 label=clean_column_name(col) + " - Daily",
                 color='darkorange', linewidth=2, zorder=3)
    # Set a fixed y-axis based on the overall range of the data
    y_min = 0
    #y_max = df[selected_columns].max().max()
    y_max = ymax
    plt.ylim(y_min, y_max)

    plt.xlabel("DOY", fontsize=13, fontweight='bold')
    plt.ylabel("Water-Column Cells", fontsize=13, fontweight='bold')
    plt.title(user_title if user_title else "Hourly vs Daily Counts Comparison", 
              fontsize=16, fontweight='bold', fontname='Times')
    plt.xticks(rotation=45, fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.legend(fontsize=12, frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    


# --- Functions for variance and distribution analysis ---
def find_high_variance_period(df, col, window='14D'):
    """
    Identifies the time window with the highest variance for a given column.
    
    Parameters:
      df (pd.DataFrame): The DataFrame containing the data. Must include an 'EST_DateTime' column.
      col (str): The column name (e.g., a daily mean column) to assess.
      window (str): The time window (e.g., '14D' for 14 days, '1M' for one month) for grouping.
    
    Returns:
      tuple: (period_start, period_end, max_variance) of the window with the highest variance.
    """
    df = df.copy()
    df['EST_DateTime'] = pd.to_datetime(df['EST_DateTime'], utc=True)
    df.set_index('EST_DateTime', inplace=True)
    
    # Group by the specified window and compute the variance for the selected column.
    grouped = df[col].groupby(pd.Grouper(freq=window))
    variance_by_window = grouped.var()
    
    # Identify which window has the highest variance.
    max_variance_period = variance_by_window.idxmax()
    max_variance = variance_by_window.max()
    
    # Determine the start and end of the window.
    period_start = max_variance_period
    period_end = period_start + pd.Timedelta(window)
    
    print(f"Highest variance window for column '{col}':")
    print(f"  Start: {period_start}, End: {period_end}, Variance: {max_variance:.3f}")
    
    return period_start, period_end, max_variance

def analyze_distribution_in_period(df, col, period_start, period_end):
    """
    Computes and plots the distribution of hourly values and daily mean values for a given period.
    Also computes the distribution of the deviations of hourly values from their daily means.
    
    Parameters:
      df (pd.DataFrame): The DataFrame with data (including 'EST_DateTime').
      col (str): The column name to analyze.
      period_start (datetime-like): Start of the time period.
      period_end (datetime-like): End of the time period.
    """
    # Ensure EST_DateTime is datetime and set as index.
    df = df.copy()
    df['EST_DateTime'] = pd.to_datetime(df['EST_DateTime'], utc=True)
    df.set_index('EST_DateTime', inplace=True)
    
    # Filter the DataFrame for the selected period.
    df_period = df.loc[period_start:period_end]
    
    # Compute daily means for the period.
    daily_mean = df_period[col].resample('D').mean()
    
    # Calculate deviations: each hourly value minus its corresponding day's mean.
    # The transform('mean') repeats the daily mean for each hourly record.
    daily_mean_expanded = df_period[col].resample('D').transform('mean')
    df_period['deviation'] = df_period[col] - daily_mean_expanded
    
    # Create subplots for the distribution comparisons.
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram for hourly values.
    axs[0].hist(df_period[col], bins=20, color='royalblue', alpha=0.7)
    axs[0].set_title("Hourly Values Distribution")
    axs[0].set_xlabel("Hourly Value")
    axs[0].set_ylabel("Frequency")
    
    # Histogram for daily means.
    axs[1].hist(daily_mean.dropna(), bins=20, color='darkorange', alpha=0.7)
    axs[1].set_title("Daily Means Distribution")
    axs[1].set_xlabel("Daily Mean Value")
    axs[1].set_ylabel("Frequency")
    
    # Histogram for deviations from daily mean.
    axs[2].hist(df_period['deviation'].dropna(), bins=20, color='green', alpha=0.7)
    axs[2].set_title("Deviation from Daily Mean")
    axs[2].set_xlabel("Deviation")
    axs[2].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def compare_hourly_daily_distributions_lyons(df, hourly_col, daily_col, start_date=None, end_date=None, lake_year_time=None):
    """
    Compare the distribution of the hourly column values to the corresponding daily column values.
    
    Parameters:
      df (pd.DataFrame): DataFrame containing the data, including an 'EST_DateTime' column.
      hourly_col (str): Name of the column containing hourly values.
      daily_col (str): Name of the column containing daily values.
      start_date (str or datetime, optional): Start date for filtering (e.g., '2022-06-01').
      end_date (str or datetime, optional): End date for filtering (e.g., '2022-12-01').
    """
    # Create a copy and ensure the datetime column is parsed properly.
    df = df.copy()
    df['EST_DateTime'] = pd.to_datetime(df['EST_DateTime'], utc=True)
    
    # Optionally filter the DataFrame for a specific date range.
    if start_date is not None and end_date is not None:
        start_date = pd.to_datetime(start_date, utc=True)
        end_date = pd.to_datetime(end_date, utc=True)
        df = df[(df['EST_DateTime'] >= start_date) & (df['EST_DateTime'] <= end_date)]
    
    # For clarity, sort values by time (optional).
    df = df.sort_values('EST_DateTime')
    
    # Compute basic variation metrics.
    hourly_std = df[hourly_col].std()
    daily_std = df[daily_col].std()
    print(f"Standard Deviation for Hourly Values ({hourly_col}): {hourly_std:.2f}")
    print(f"Standard Deviation for Daily Values ({daily_col}): {daily_std:.2f}")
    
    # Plot overlaid histograms to compare distributions.
    plt.figure(figsize=(12, 6))
    plt.hist(df[hourly_col].dropna(), bins=30, alpha=0.5, color='royalblue', label='Hourly')
    plt.hist(df[daily_col].dropna(), bins=30, alpha=0.5, color='darkorange', label='Daily')
    
    plt.xlabel("Suitable Proportion of the Water-Column", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Distribution Comparison: Hourly vs Daily in {lake_year_time}", fontsize=14)
    plt.legend()
    #plt.xlim(0, 1)  # Fix x-axis between 0 and 1
    plt.show()
    
    # Optionally, you could also compare the kernel density estimates:
    plt.figure(figsize=(12, 6))
    df[hourly_col].dropna().plot(kind='kde', color='royalblue', label='Hourly', linewidth=2)
    df[daily_col].dropna().plot(kind='kde', color='darkorange', label='Daily', linewidth=2)
    plt.xlabel("Suitable Proportion of the Water-Column", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"KDE for Suitable Cells: Hourly vs Daily in {lake_year_time}", fontsize=14)
    plt.legend()
    plt.show()
    
    
def compute_descriptive_stats_for_iter(df, hourly_col, daily_col, start_date, end_date, lake_year_time):
    """
    Computes descriptive statistics for a given iteration (a single DataFrame) on the specified 
    hourly and daily columns over a user-defined date range. Returns a DataFrame (one row) with
    statistics that can be combined for multiple iterations.
    
    Parameters:
      df (pd.DataFrame): DataFrame containing the data (must include an 'EST_DateTime' column).
      hourly_col (str): Name of the column with hourly values.
      daily_col (str): Name of the column with daily values.
      start_date (str or datetime-like): The start date of the period (e.g., '2022-10-01').
      end_date (str or datetime-like): The end date of the period (e.g., '2022-10-20').
      lake_year_time (str): A label identifying the lake, year, and period (e.g., "Crooked Lake 2022-10-01 to 2022-10-20").
      
    Returns:
      pd.DataFrame: A single-row DataFrame with descriptive statistics.
    """
    # Make a copy and ensure the datetime column is set correctly.
    df_copy = df.copy()
    df_copy['EST_DateTime'] = pd.to_datetime(df_copy['EST_DateTime'], utc=True)
    start = pd.to_datetime(start_date, utc=True)
    end = pd.to_datetime(end_date, utc=True)
    
    # Filter for the specified time period.
    df_period = df_copy[(df_copy['EST_DateTime'] >= start) & (df_copy['EST_DateTime'] <= end)]
    
    # Get descriptive statistics using pandas' describe() method.
    stats_hourly = df_period[hourly_col].describe()
    stats_daily  = df_period[daily_col].describe()
    
    # Compute the interquartile range (IQR) for each column.
    iqr_hourly = stats_hourly['75%'] - stats_hourly['25%']
    iqr_daily  = stats_daily['75%'] - stats_daily['25%']
    
    # Prepare a dictionary with the stats.
    stats_dict = {
         'Lake_Year_Time': lake_year_time,
         'Hourly_Mean': stats_hourly['mean'],
         'Daily_Mean': stats_daily['mean'],
         'Hourly_Median': stats_hourly['50%'],
         'Daily_Median': stats_daily['50%'],
         'Hourly_STD': stats_hourly['std'],
         'Daily_STD': stats_daily['std'],
         'Hourly_IQR': iqr_hourly,
         'Daily_IQR': iqr_daily,
         'Hourly_Min': stats_hourly['min'],
         'Daily_Min': stats_daily['min'],
         'Hourly_Max': stats_hourly['max'],
         'Daily_Max': stats_daily['max'],
         'Count': stats_hourly['count']  # This count should be the same for both if derived from the same filtering.
    }
    
    # Return as a one-row DataFrame.
    return pd.DataFrame([stats_dict])


"End loading Distribution functions and descriptive stats functions"

#%%

"lyons dfs"

# Regex pattern to extract type lyons_lake_timescale_year .csv files
pattern_lyons_dfs = "lyons_(crook|fail)_(2021|2022).csv"
# pattern for df cols
pattern_lyons_df_cols = r"(?P<lakes>failing|crooked)_(?P<timescales>hourly|daily)_(?P<variables>lyons)_(?P<years>2021|2022)"


dfscount = read_df_dir_tocsv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Lyons_summary_dfs/Counts')
dfsprop = read_df_dir_tocsv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Lyons_summary_dfs/Props')

strats = [key for key in dfscount]
#pturns = [key for key in dfscount if "pturn" in key]

# times left to average out
timestamps_crook2021 = ['2021-07-26 10:00:00-04:00', '2021-08-13 15:00:00-04:00', '2021-09-01 15:00:00-04:00', '2021-10-08 12:00:00-04:00']
timestamps_fail2022 = ['2022-07-21 12:00:00-04:00', '2022-09-16 12:00:00-04:00']
# Averaging out removal hours and saving 
for strat in strats:
    if "fail" in strat and "2022" in strat:
        print(dfscount[strat]['EST_DateTime'])
        replace_at_timestamps_with_window_avg(dfscount[strat], timestamps_fail2022, num_before=1, num_after=1)
        replace_at_timestamps_with_window_avg(dfsprop[strat], timestamps_fail2022, num_before=1, num_after=1)
    elif "crook" in strat and "2021" in strat:
        replace_at_timestamps_with_window_avg(dfscount[strat], timestamps_crook2021, num_before=1, num_after=1)
        replace_at_timestamps_with_window_avg(dfsprop[strat], timestamps_crook2021, num_before=1, num_after=1)





# pattern for df cols
# Regex pattern to extract type lake__ (temp OR DO)_timescale_year .csv files
pattern_TempDO = "(crooked|failing)_(daily|hourly)_(do|temp)_(2021|2022)\.csv"
#pattern_lyons_df_cols = r"(?P<lakes>failing|crooked)_(?P<timescales>hourly|daily)_(?P<variables>lyons)_(?P<years>2021|2022)"
user_input = {'variables':'lyons'}
pattern = pattern_lyons_df_cols
#pattern = pattern_lyons_dfs


for strat in strats:
    if "crook" in strat and "2022" in strat:
        #print(dfscount[strat]['EST_DateTime'])
        #ymax=60 
        user_title = 'Sub-daily vs. Daily Counts of Suitable Oxythermal Cells in Crooked Lake, 2022'
        compare_hourly_daily_timeseries(dfscount, strat, pattern, user_input, user_title, start_date='2022-06-08', end_date='2022-12-01', ymax=60)
        compare_hourly_daily_timeseries(dfscount, strat, pattern, user_input, user_title, start_date='2022-09-01', end_date='2022-11-10', ymax=60)
    elif "crook" in strat and "2021" in strat:
        #ymax=60 
        user_title = 'Sub-daily vs. Daily Counts of Suitable Oxythermal Cells in Crooked Lake, 2021' #- (fDO lethal)'
        compare_hourly_daily_timeseries(dfscount, strat, pattern, user_input, user_title, ymax=30)
        compare_hourly_daily_timeseries(dfscount, strat, pattern, user_input, user_title, start_date='2021-09-15', end_date='2021-11-10', ymax=30)
    elif "fail" in strat and "2022" in strat:
        #ymax=30 
        user_title = 'Sub-daily vs. Daily Counts of Suitable Oxythermal Cells in Failing Lake, 2022 ' 
        compare_hourly_daily_timeseries(dfscount, strat, pattern, user_input, user_title, start_date='2022-04-13', end_date='2022-12-01', ymax=30)
        compare_hourly_daily_timeseries(dfscount, strat, pattern, user_input, user_title, start_date='2022-09-01', end_date='2022-11-10', ymax=30) 
        compare_hourly_daily_timeseries(dfscount, strat, pattern, user_input, user_title, start_date='2022-05-05', end_date='2022-06-05', ymax=30) 
    elif "fail" in strat and "2021" in strat:
        #ymax=30 
        user_title = 'Sub-daily vs. Daily Counts of Suitable Oxythermal Cells in Failing Lake, 2021' 
        compare_hourly_daily_timeseries(dfscount, strat, pattern, user_input, user_title, start_date='2021-06-06', end_date='2022-12-01', ymax=30)
        compare_hourly_daily_timeseries(dfscount, strat, pattern, user_input, user_title, start_date='2021-09-15', end_date='2021-11-10' , ymax=30)
    

#%%
# df = pd.read_csv("your_data.csv")


"""
############### CROOK 22
"""
col_to_analyze = extract_matching_columns(dfsprop["lyons_crook_2022"], pattern, user_input) 

hourly_cols = [col for col in col_to_analyze if 'hourly' in col.lower()]
daily_cols  = [col for col in col_to_analyze if 'daily' in col.lower()]


# For this example, select the first found column from each.
hourly_col = hourly_cols[0]
daily_col  = daily_cols[0]

# Compare the distributions over a specified date range.
compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_crook_2022"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2022-10-01',    # Optional start date
    end_date='2022-10-20',      # Optional end date
    lake_year_time='Crooked Lake 2022-10-01 to 2022-10-20'
)

# Compare the distributions over a specified date range.
compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_crook_2022"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2022-09-25',    # Optional start date
    end_date='2022-11-01',      # Optional end date
    lake_year_time='Crooked Lake 2022-09-20 to 2022-11-01'
)

# Compare the distributions over a specified date range.
compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_crook_2022"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2022-06-08',    # Optional start date
    end_date='2022-12-01',      # Optional end date
    lake_year_time='Crooked Lake 2022-06-08 to 2022-12-01'
)

# formation

# Compare the distributions over a specified date range.
compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_crook_2022"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2022-06-08',    # Optional start date
    end_date='2022-07-15',      # Optional end date
    lake_year_time='Crooked Lake 2022-06-08 to 2022-08-01'
)

#################


"""
############### CROOK 21
"""
# df = pd.read_csv("your_data.csv")
col_to_analyze = extract_matching_columns(dfsprop["lyons_crook_2021"], pattern, user_input) 

hourly_cols = [col for col in col_to_analyze if 'hourly' in col.lower()]
daily_cols  = [col for col in col_to_analyze if 'daily' in col.lower()]


# For this example, select the first found column from each.
hourly_col = hourly_cols[0]
daily_col  = daily_cols[0]





# Compare the distributions over a specified date range.
compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_crook_2021"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2021-06-25',    # Optional start date
    end_date='2021-07-12',      # Optional end date
    lake_year_time='Crooked Lake 2021-06-25 to 2021-07-12'
)


#################
"""
############### FAIL 22
"""

# df = pd.read_csv("your_data.csv")
col_to_analyze = extract_matching_columns(dfsprop["lyons_fail_2022"], pattern, user_input) 

hourly_cols = [col for col in col_to_analyze if 'hourly' in col.lower()]
daily_cols  = [col for col in col_to_analyze if 'daily' in col.lower()]

# For this example, select the first found column from each.
hourly_col = hourly_cols[0]
daily_col  = daily_cols[0]

# Compare the distributions over a specified date range.
compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_fail_2022"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2022-09-20',    # Optional start date
    end_date='2022-11-05',      # Optional end date
    lake_year_time='Failing Lake 2022-09-20 to 2022-11-05'
)


compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_fail_2022"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2022-05-05',
    end_date='2022-06-05',
    lake_year_time='Failing Lake 2022-05-05 to 2022-06-05'
)


compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_fail_2022"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2022-04-18',
    end_date='2022-06-05',
    lake_year_time='Failing Lake 2022-04-25 to 2022-06-10'
)

compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_fail_2022"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2022-04-13',
    end_date='2022-12-01',
    lake_year_time='Failing Lake 2022-04-13 to 2022-12-01'
)



#################

"""
############### FAIL 21
"""


# df = pd.read_csv("your_data.csv")
col_to_analyze = extract_matching_columns(dfsprop["lyons_fail_2021"], pattern, user_input) 

hourly_cols = [col for col in col_to_analyze if 'hourly' in col.lower()]
daily_cols  = [col for col in col_to_analyze if 'daily' in col.lower()]



# For this example, select the first found column from each.
hourly_col = hourly_cols[0]
daily_col  = daily_cols[0]

# Compare the distributions over a specified date range.
compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_fail_2021"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2021-09-20',    # Optional start date
    end_date='2021-11-05',      # Optional end date
    lake_year_time='Failing Lake 2021-09-20 to 2021-11-05'
)

# Compare the distributions over a specified date range.
compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_fail_2021"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2021-09-20',    # Optional start date
    end_date='2021-12-01',      # Optional end date
    lake_year_time='Failing Lake 2021-09-20 to 2021-12-01'
)

# Compare the distributions over a specified date range.
compare_hourly_daily_distributions_lyons(
    dfsprop["lyons_fail_2021"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2021-06-06',    # Optional start date
    end_date='2021-12-01',      # Optional end date
    lake_year_time='Failing Lake 2021-06-06 to 2021-12-01'
)


#%%
"-------------------- summary data -------------------------------"







# For Crooked Lake 2022:
col_to_analyze = extract_matching_columns(dfsprop["lyons_crook_2022"], pattern, user_input) 
hourly_cols = [col for col in col_to_analyze if 'hourly' in col.lower()]
daily_cols  = [col for col in col_to_analyze if 'daily' in col.lower()]

if not hourly_cols or not daily_cols:
    raise ValueError("Missing one of the required column types (hourly or daily).")

# For this example, select the first found column from each.
hourly_col = hourly_cols[0]
daily_col  = daily_cols[0]

# Compute the stats DataFrame for a specified period.
stats_crook2022 = compute_descriptive_stats_for_iter(
    dfsprop["lyons_crook_2022"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2022-09-25',
    end_date='2022-11-01',
    lake_year_time='Crooked Lake 2022-10-01 to 2022-10-20'
)

# Similarly, for other iterations (e.g., Crooked Lake 2021, Failing Lake 2022, Failing Lake 2021),
# compute individual stats DataFrames:
col_to_analyze = extract_matching_columns(dfsprop["lyons_crook_2021"], pattern, user_input) 
hourly_cols = [col for col in col_to_analyze if 'hourly' in col.lower()]
daily_cols  = [col for col in col_to_analyze if 'daily' in col.lower()]
hourly_col = hourly_cols[0]
daily_col  = daily_cols[0]
stats_crook2021 = compute_descriptive_stats_for_iter(
    dfsprop["lyons_crook_2021"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2021-09-20',
    end_date='2021-10-25',
    lake_year_time='Crooked Lake 2021-10-01 to 2021-10-20'
)

col_to_analyze = extract_matching_columns(dfsprop["lyons_fail_2022"], pattern, user_input) 
hourly_cols = [col for col in col_to_analyze if 'hourly' in col.lower()]
daily_cols  = [col for col in col_to_analyze if 'daily' in col.lower()]
hourly_col = hourly_cols[0]
daily_col  = daily_cols[0]
stats_fail2022fall = compute_descriptive_stats_for_iter(
    dfsprop["lyons_fail_2022"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2022-09-20',
    end_date='2022-11-05',
    lake_year_time='Failing Lake 2022-09-20 to 2022-11-05'
)

"""
col_to_analyze = extract_matching_columns(dfsprop["lyons_fail_2022"], pattern, user_input) 
hourly_cols = [col for col in col_to_analyze if 'hourly' in col.lower()]
daily_cols  = [col for col in col_to_analyze if 'daily' in col.lower()]
hourly_col = hourly_cols[0]
daily_col  = daily_cols[0]
stats_fail2022spring = compute_descriptive_stats_for_iter(
    dfsprop["lyons_fail_2022"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2022-05-05',
    end_date='2022-06-05',
    lake_year_time='Failing Lake 2022-05-05 to 2022-06-05'
)

"""

col_to_analyze = extract_matching_columns(dfsprop["lyons_fail_2021"], pattern, user_input) 

hourly_cols = [col for col in col_to_analyze if 'hourly' in col.lower()]
daily_cols  = [col for col in col_to_analyze if 'daily' in col.lower()]
hourly_col = hourly_cols[0]
daily_col  = daily_cols[0]
stats_fail2021 = compute_descriptive_stats_for_iter(
    dfsprop["lyons_fail_2021"],
    hourly_col=hourly_col,
    daily_col=daily_col,
    start_date='2021-09-20',
    end_date='2021-11-05',
    lake_year_time='Failing Lake 2021-09-20 to 2021-11-05'
)

# Concatenate all statistics into one DataFrame.
all_stats = pd.concat([stats_crook2022, stats_crook2021, stats_fail2022fall, stats_fail2021], ignore_index=True)
print(all_stats)

output_dir = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/summary_dfs_hrvsDaycols"
output_file = os.path.join(output_dir, "descriptive_statistics_lyons_moreconst.csv")
all_stats.to_csv(output_file, index=False)



#%%

from itertools import groupby
from operator import itemgetter
" ================= 2D array plotting =================="


def extract_matching_arrays(arr_dict, lake_yr_period, pattern, user_inputs):
    """
    Inputs: arr_dict = nested dictionary of arrays {outer_key: {key: array, ...}}
            lake_yr_period = outer key of nested dicitonary
            pattern = regex pattern to match inner key structure
            user_inputs = dictionary to match metadata dict (key) to inner_key (value)
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



def time_index_by_lakeyear_2D_v2(nested_dict, start_end_dates):
    """
    Create a dictionary of time indices keyed by outer_key → { inner_key → DatetimeIndex }.
    Uses the filename (inner_key) to decide 'H' vs 'D' frequency.
    """
    time_indices = {}
    print("Available outer keys:", list(start_end_dates.keys()))

    for outer_key, data_dict in nested_dict.items():
        if outer_key not in start_end_dates:
            print(f"⚠️  No start/end for {outer_key}, skipping.")
            continue

        se = start_end_dates[outer_key]
        start = pd.to_datetime(se['start_date'], utc=True)
        end   = pd.to_datetime(se['end_date'],   utc=True)

        # Build inner dict
        idxs = {}
        for inner_key, arr in data_dict.items():
            ncols = arr.shape[1]

            # decide freq on the fly:
            if 'hr' in inner_key or 'hourly' in inner_key:
                freq = 'H'
            else:
                freq = 'D'

            # now build a date_range that has exactly ncols points
            idxs[inner_key] = pd.date_range(
                start=start,
                periods=ncols,
                freq=freq,
                #tz="America/New_York"
            )

        time_indices[outer_key] = idxs

    return time_indices





def plot_heatmap_v4(arr_dict, lake_yr_period, pattern, user_inputs, date_dict,
                    flip=True, user_title=None, start_date=None, end_date=None, user_label = None):
    # … [initial checks & extract_matching_arrays IDENTICAL] …
    # Ensure the key exists in the data dictionary.
    if lake_yr_period not in arr_dict:
        raise ValueError(f"Key '{lake_yr_period}' not found in arr_dict.")

    # 2) If you used an alternate key logic, keep that here:
    key_used = lake_yr_period
    if key_used not in date_dict:
        alt = lake_yr_period.rsplit('_', 1)[0]
        if alt in date_dict:
            key_used = alt
        else:
            raise KeyError(f"No dates for {lake_yr_period}")
            
    # 3) Loop over each array you want to plot
    selected = extract_matching_arrays(arr_dict, lake_yr_period, pattern, user_inputs)
    for sel in selected:
        arr = arr_dict[lake_yr_period][sel]

        # ── pull the correct DatetimeIndex for *this* array ──
        full_idx = date_dict[key_used][sel]  

        # ── optionally filter by start/end ──
        if start_date and end_date:
            s = pd.to_datetime(start_date, utc=True)
            e = pd.to_datetime(end_date,   utc=True)
            mask = (full_idx >= s) & (full_idx <= e)
            dates = full_idx[mask]
            cols  = np.where(mask)[0]
            arr2  = arr[:, cols]
        else:
            dates = full_idx
            arr2  = arr

        # ── sanity check ──
        if arr2.shape[1] != len(dates):
            raise ValueError(
                f"{sel} has {arr2.shape[1]} cols but {len(dates)} dates")

        # ── flip or not ──
        data = np.flipud(arr2) if flip else arr2

        # ── plotting ──
        fig, ax = plt.subplots(figsize=(12,8))

        # Choose origin based on flip flag:
        #origin = 'upper' if flip else 'lower'
        im = ax.imshow(arr2,
                       aspect='auto',
                       cmap='seismic',
                       origin='lower')

        fig.colorbar(im, label= user_label)
        ax.set_xlabel("DOY")
        ax.set_ylabel("Depth (m)")
        if user_title:
            ax.set_title(user_title, fontweight='bold')

        # x‑ticks from your real dates
        xt = np.linspace(0, arr2.shape[1]-1,
                         min(10, len(dates))).astype(int)
        ax.set_xticks(xt)
        ax.set_xticklabels(dates[xt].strftime('%Y-%m-%d'),
                           rotation=45, ha='right')

        # ── y‑axis ──
        # Force y from 0 to last row
        ax.set_ylim(0, arr2.shape[0]-1)

        # Now pick some nice tick locations (e.g., every 2 rows → 1 m):
        nrows = arr2.shape[0]
        ylocs = np.linspace(0, nrows, min(7, nrows)).astype(int)
        ax.set_yticks(ylocs)
        # Divide by 2 to convert “half‑meters” to meters
        ax.set_yticklabels((ylocs/2).round(2))
        ax.invert_yaxis()
        #if not flip:
            #ax.invert_yaxis()
        #ax.invert_yaxis()
        

        plt.tight_layout()
        plt.show()



def replace_groups_with_avg(array, groups):
    rows, cols = array.shape
    for start, end in groups:
        # only replace if we have both left & right neighbors
        if start > 0 and end < cols - 1:
            avg_values = (array[:, start - 1] + array[:, end + 1]) / 2
            array[:, start:end + 1] = np.tile(avg_values[:, None], end - start + 1)
        else:
            print(f"Group {(start,end)} cannot be replaced (no neighbors).")
    return array

# 2. a helper to turn any boolean mask over columns into contiguous (start,end) groups
def find_column_groups(mask):
    """
    mask: 1D boolean array of length n_cols
    returns: list of (start_idx, end_idx) for each contiguous True run
    """
    idxs = np.nonzero(mask)[0]
    groups = []
    for _, grp in groupby(enumerate(idxs), key=lambda x: x[0] - x[1]):
        run = [i for _, i in grp]
        groups.append((run[0], run[-1]))
    return groups

# 3. define your “bad column” condition separately for hourly vs daily
#    here as an *example* we’ll say “drop ANY column where all values are NaN”,
#    but you can swap in any df → 1D‐bool‑mask logic you like.
def bad_cols_daily(df):
    # e.g. remove days before 2021‑07‑01
    return (df.columns < pd.Timestamp("2021-07-01")).values

def bad_cols_hourly(df):
    # e.g. remove any hour where temperature == 0 for all depths
    return (df == 0).all(axis=0).values


#%%

" Getting Correct 2D plots for directories of arrays that contain (Lake, Year) ------"

# start end dates for outer_keys
start_end_dates = {
                  'crooked_2021': {'start_date': '2021-06-24', 'end_date': '2021-11-12'}, 'failing_2021': {'start_date': '2021-06-05', 'end_date': '2021-12-01'},
                 'crooked_2022': {'start_date': '2022-06-07', 'end_date': '2022-12-01'}, 'failing_2022': {'start_date': '2022-04-12', 'end_date': '2022-12-01'}
                 }

# Regex pattern to extract type lake__ (temp OR DO)_timescale_year .csv files
pattern_parplot = "(?P<lakes>crooked|failing)_(daily|hourly)_(?P<variables>lyons|do|temp|grp)_(?P<years>2021|2022)"

#lyons_base = r"(?P<lakes>failing|crooked)_(?P<timescales>hourly|daily)_(?P<variables>lyons)_(?P<years>2021|2022)"

#master_dir = {'all_lakes_years':'/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Par_plot_groups'}
#all_lakes_years = {k:load_datasets_from_directory(v) for k,v in master_dir.items()}

#split each into nested - potentially optional
base_dirs = {
    'crooked_2021': '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Par_plot_groups/crooked_2021',
    'crooked_2022':'/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Par_plot_groups/crooked_2022',

    'failing_2021': '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Par_plot_groups/failing_2021',
    'failing_2022': '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Par_plot_groups/failing_2022',

    }


"  ========  # Steps - info : Query datasets based on metadata and generate ========="
# Example usage
crooked_2021 = {k:load_csv_files(v) for k,v in base_dirs.items() if 'crooked_2021' in k}
crooked_2022 = {k:load_csv_files(v) for k,v in base_dirs.items() if 'crooked_2022' in k}
failing_2021 = {k:load_csv_files(v) for k,v in base_dirs.items() if 'failing_2021' in k}
failing_2022 = {k:load_csv_files(v) for k,v in base_dirs.items() if 'failing_2022' in k}


# making date dictionary for each nested lk_yr dict

# creating time indices
crooked_2021_dates = time_index_by_lakeyear_2D_v2(crooked_2021, start_end_dates)
crooked_2022_dates = time_index_by_lakeyear_2D_v2(crooked_2022, start_end_dates)
failing_2021_dates = time_index_by_lakeyear_2D_v2(failing_2021, start_end_dates)
failing_2022_dates = time_index_by_lakeyear_2D_v2(failing_2022, start_end_dates)

# replaceing time stamps of removal
timestamps_crook2021 = ['2021-07-26 10:00:00-04:00', '2021-08-13 15:00:00-04:00', '2021-09-01 15:00:00-04:00', '2021-10-08 12:00:00-04:00']
timestamps_fail2022 = ['2022-07-21 12:00:00-04:00', '2022-09-16 12:00:00-04:00']



grp_label = r' GRP ($g\,g^{-1}\,d^{-1}$)'
do_label = ''
temp_label= ''

crook21_inputs = {"lakes":"crooked", "variables":"temp", "years": "2021"}

check ={"lakes":"crooked"}

check_f = {"lakes":"failing"}
# replaceing time stamps of removal
#timestamps_to_replace = ['2021-06-05 04:00:00-04:00', '2021-06-05 05:00:00-04:00']
#before_day, after_day = 1, 1
before, after = 24, 24



smoothed = apply_window_avg_to_nested_arrays(
    crooked_2021,
    crooked_2021_dates,
    timestamps_crook2021,
    num_before=before,
    num_after=after
)




# testing sincle lk_yr
#extract_matching_arrays(arr_dict, lake_yr_period, pattern, user_inputs)
#plot_heatmap_v3(arr_dict, lake_yr_period, pattern, user_inputs, date_dict, flip=True, user_title=None, start_date=None, end_date=None)

for outer_key, array_dict in crooked_2021.items():
    plot_heatmap_v4(crooked_2021, outer_key, pattern_parplot, check, crooked_2021_dates, user_title=None, 
                       start_date=None, end_date=None)
    plot_heatmap_v4(crooked_2021, outer_key, pattern_parplot, check, crooked_2021_dates, flip=True, user_title=None, 
                       start_date=None, end_date=None)
    
    
for outer_key, array_dict in failing_2022.items():
    plot_heatmap_v4(failing_2022, outer_key, pattern_parplot, check_f, failing_2022_dates, user_title=None, 
                       start_date=None, end_date=None)
    plot_heatmap_v4(failing_2022, outer_key, pattern_parplot, check_f, failing_2022_dates, flip=True, user_title=None, 
                       start_date=None, end_date=None)



"""

# Step 2: Query datasets dynamically - these list iterations will select and plot all 
lake_names = ["crooked", "failing"]
years = ["2021", "2022"]
time_scales = ["hourly", "daily"]
variables = ["temp", "do", "grp", "lyons"]


for variable in variables:
    for time_scale in time_scales:
    
        pass
    pass

    
"""



#%%

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




def replace_groups_with_avg(array, groups):
    rows, cols = array.shape
    for start, end in groups:
        # only replace if we have both left & right neighbors
        if start > 0 and end < cols - 1:
            avg_values = (array[:, start - 1] + array[:, end + 1]) / 2
            array[:, start:end + 1] = np.tile(avg_values[:, None], end - start + 1)
        else:
            print(f"Group {(start,end)} cannot be replaced (no neighbors).")
    return array


for key, value in crooked_2021.items():
    match_datetime_to_indices(time_index, target_dates)


#%%

import numpy as np
import pandas as pd
import re
import os
from datetime import datetime

def replace_groups_with_avg(array, groups):
    rows, cols = array.shape
    for group in groups:
        start, end = group
        # Ensure the group is within the bounds of the array
        if start > 0 and end < cols - 1:
            # Calculate the average of the column before and after the group
            avg_values = (array[:, start - 1] + array[:, end + 1]) / 2
            # Replace the values in the group of columns with the average values
            array[:, start:end + 1] = np.tile(avg_values[:, None], end - start + 1)
        else:
            print(f"Group {group} cannot be replaced because it doesn't have both left and right neighbors.")
    return array

def find_timestamps_indices(timestamps, target_timestamps):
    """
    Find the indices of target timestamps in a list of timestamps.
    
    Args:
        timestamps: List of timestamp objects (pandas Timestamp or datetime)
        target_timestamps: List of timestamp strings to find
    
    Returns:
        Dictionary mapping each target timestamp to its index in the timestamps list
    """
    # Convert target timestamps to pandas datetime objects
    target_ts_pd = pd.to_datetime(target_timestamps)
    
    # Find indices
    indices = {}
    for target_ts in target_ts_pd:
        # Find the timestamp or closest match
        closest_idx = None
        min_delta = None
        
        for i, ts in enumerate(timestamps):
            delta = abs((ts - target_ts).total_seconds())
            if min_delta is None or delta < min_delta:
                min_delta = delta
                closest_idx = i
        
        indices[str(target_ts)] = closest_idx
    
    return indices

def find_and_replace_timestamps(data_dict, dates_dict, target_timestamps, window_size=1):
    """
    Find and replace columns around target timestamps in data arrays.
    
    Args:
        data_dict: Dictionary containing data arrays or sub-dictionaries
        dates_dict: Dictionary with same structure containing timestamps for each array
        target_timestamps: List of timestamps to find and replace
        window_size: Number of columns to replace on each side of the target timestamp
    
    Returns:
        Dictionary with same structure containing processed arrays
    """
    processed_data = {}
    
    # Check if data_dict is a dictionary or an array
    if isinstance(data_dict, dict):
        # Process each key in the dictionary
        for key, value in data_dict.items():
            if key in dates_dict:
                # Recursive call for nested dictionaries
                if isinstance(value, dict):
                    processed_data[key] = find_and_replace_timestamps(value, dates_dict[key], target_timestamps, window_size)
                # Process array directly
                elif isinstance(value, np.ndarray):
                    timestamps = dates_dict[key]
                    # Find indices of target timestamps
                    timestamp_indices = find_timestamps_indices(timestamps, target_timestamps)
                    
                    # Create groups to replace (target index +/- window_size)
                    groups = []
                    for ts, idx in timestamp_indices.items():
                        if idx is not None:
                            start = max(0, idx - window_size)
                            end = min(value.shape[1] - 1, idx + window_size)
                            groups.append((start, end))
                    
                    # Apply replacement function
                    if groups:
                        processed_array = replace_groups_with_avg(value.copy(), groups)
                        processed_data[key] = processed_array
                    else:
                        processed_data[key] = value.copy()
                else:
                    processed_data[key] = value
            else:
                processed_data[key] = value
    elif isinstance(data_dict, np.ndarray):
        # If data_dict is already an array (leaf node), process it directly
        timestamps = dates_dict  # Assume dates_dict is directly the timestamps array at this level
        
        # Find indices of target timestamps
        timestamp_indices = find_timestamps_indices(timestamps, target_timestamps)
        
        # Create groups to replace
        groups = []
        for ts, idx in timestamp_indices.items():
            if idx is not None:
                start = max(0, idx - window_size)
                end = min(data_dict.shape[1] - 1, idx + window_size)
                groups.append((start, end))
        
        # Apply replacement function
        if groups:
            processed_data = replace_groups_with_avg(data_dict.copy(), groups)
        else:
            processed_data = data_dict.copy()
    else:
        # For other types, just return as is
        processed_data = data_dict
    
    return processed_data

# Function to apply specific timestamp replacements based on lake and year
def process_lake_year_data(lake_year_dicts, dates_dicts):
    """
    Process data for different lakes and years with their specific timestamp replacements.
    
    Args:
        lake_year_dicts: Dictionary of lake_year data dictionaries
        dates_dicts: Dictionary of lake_year dates dictionaries
    
    Returns:
        Dictionary of processed lake_year data dictionaries
    """
    processed_dicts = {}
    
    # Process crooked_2021 with its specific timestamps
    if 'crooked_2021' in lake_year_dicts:
        timestamps_crook2021 = [
            '2021-07-26 10:00:00-04:00', 
            '2021-08-13 15:00:00-04:00', 
            '2021-09-01 15:00:00-04:00', 
            '2021-10-08 12:00:00-04:00'
        ]
        processed_dicts['crooked_2021'] = find_and_replace_timestamps(
            lake_year_dicts['crooked_2021'], 
            dates_dicts['crooked_2021'],
            timestamps_crook2021,
            window_size=1
        )
    
    # Process failing_2022 with its specific timestamps
    if 'failing_2022' in lake_year_dicts:
        timestamps_fail2022 = [
            '2022-07-21 12:00:00-04:00', 
            '2022-09-16 12:00:00-04:00'
        ]
        processed_dicts['failing_2022'] = find_and_replace_timestamps(
            lake_year_dicts['failing_2022'], 
            dates_dicts['failing_2022'],
            timestamps_fail2022,
            window_size=1
        )
    
    # For other lake_years, just copy the original data
    for lake_year in lake_year_dicts:
        if lake_year not in processed_dicts:
            processed_dicts[lake_year] = lake_year_dicts[lake_year].copy()
    
    return processed_dicts

# Function to visualize the effect of replacements
def plot_comparison(original_array, processed_array, timestamps, target_timestamps, filename, depth_idx=0):
    """
    Plot a comparison of original vs processed data for a specific depth.
    
    Args:
        original_array: Original 2D array
        processed_array: Processed 2D array
        timestamps: Array of timestamps for columns
        target_timestamps: List of timestamps that were targeted for replacement
        filename: Name of the file being plotted
        depth_idx: Index of depth to plot (default: surface layer)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    plt.figure(figsize=(14, 6))
    
    # Convert target_timestamps to datetime objects if they're strings
    if isinstance(target_timestamps[0], str):
        target_timestamps = pd.to_datetime(target_timestamps)
    
    # Plot original data
    plt.plot(timestamps, original_array[depth_idx, :], 'b-', label='Original', alpha=0.7)
    
    # Plot processed data
    plt.plot(timestamps, processed_array[depth_idx, :], 'r-', label='Processed')
    
    # Highlight the replaced timestamps
    for ts in target_timestamps:
        plt.axvline(x=ts, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f"Comparison for {filename} at depth index {depth_idx}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    
    # Format the x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))  # Show date every 14 days
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

# Print the structure of a nested dictionary to understand its format
def print_dict_structure(d, depth=0, max_depth=3):
    """Helper function to print the structure of a nested dictionary."""
    if depth >= max_depth:
        print("  " * depth + "...")
        return
    
    if isinstance(d, dict):
        for k, v in d.items():
            print("  " * depth + f"{k}: {type(v)}")
            if isinstance(v, (dict, list)):
                print_dict_structure(v, depth + 1, max_depth)
    elif isinstance(d, list) and len(d) > 0:
        print("  " * depth + f"[0]: {type(d[0])}")
        if isinstance(d[0], (dict, list)):
            print_dict_structure(d[0], depth + 1, max_depth)

# Example usage - main execution block
def main():
    # Print structure of dictionaries to understand the format
    print("Structure of crooked_2021:")
    print_dict_structure(crooked_2021)
    print("\nStructure of crooked_2021_dates:")
    print_dict_structure(crooked_2021_dates)
    
    # Combine all lake_year dictionaries
    lake_year_dicts = {
        'crooked_2021': crooked_2021['crooked_2021'] if isinstance(crooked_2021, dict) and 'crooked_2021' in crooked_2021 else crooked_2021,
        'crooked_2022': crooked_2022['crooked_2022'] if isinstance(crooked_2022, dict) and 'crooked_2022' in crooked_2022 else crooked_2022,
        'failing_2021': failing_2021['failing_2021'] if isinstance(failing_2021, dict) and 'failing_2021' in failing_2021 else failing_2021,
        'failing_2022': failing_2022['failing_2022'] if isinstance(failing_2022, dict) and 'failing_2022' in failing_2022 else failing_2022
    }
    
    # Combine all dates dictionaries
    dates_dicts = {
        'crooked_2021': crooked_2021_dates,
        'crooked_2022': crooked_2022_dates,
        'failing_2021': failing_2021_dates,
        'failing_2022': failing_2022_dates
    }
    
    # Process the data
    processed_dicts = process_lake_year_data(lake_year_dicts, dates_dicts)
    
    # Return the processed data
    return processed_dicts

# If this script is run directly, execute the main function
if __name__ == "__main__":
    processed_data = main()











