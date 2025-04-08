#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:12:19 2025

@author: rajeevkumar
"""

"============= Creating systemic way to work with datasets efficiently by creating array and df classes w. filename metadata ============"
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#"main packages loaded"


#def csv_toarray(fpath):   # skip_blank_lines = False
    #array = pd.read_csv(fpath, header=None).to_numpy()
    #print(f"Loaded array from {fpath} with shape: {array.shape}")
    #return array



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

class CompareDailyHourly:
    def __init__(self, data, interval_colsum, num_replications):
        """
        Initialize the dataset object.

        Parameters:
            data (dict): Dictionary of datasets where keys are dataset names.
            interval_colsum (int): Number of columns to sum.
            num_replications (int): Number of times to replicate each summed 1D array.
        """
        self.data_dict = data  # Properly assigning
        self.interval_colsum = interval_colsum
        self.num_replications = num_replications
        self.summed_dict = {}  # To store summed column data
        self.rep_data_dict = {}  # To store replicated data
        self.prop_pos_dict = {}  # To store proportion data

    def sum_cols_with_interval(self):
        """Sum columns within specified intervals."""
        if self.data_dict is None:
            raise ValueError("Data dictionary is empty. Please provide data.")

        for key, value in self.data_dict.items():
            num_rows, num_cols = value.shape
            num_intervals = num_cols // self.interval_colsum
            remainder_cols = num_cols % self.interval_colsum

            sums = np.zeros(num_intervals)

            for i in range(num_intervals):
                start_col = i * self.interval_colsum
                end_col = start_col + self.interval_colsum
                sums[i] = np.sum(value[:, start_col:end_col])

            if remainder_cols != 0:
                sums = np.append(sums, np.sum(value[:, -remainder_cols:]))

            self.summed_dict[key] = sums  # Store results

        return self.summed_dict

    def replicate_match_dims_dict(self):
        """Replicate each value in the summed arrays 24 times, maintaining order."""
        self.rep_data_dict = {}  # Reset dictionary

        for key, value in self.summed_dict.items():
            # Ensure it's a 1D array
            if value.ndim > 1:
                value = value.flatten()
                
            if "day" in key.lower() or "daily" in key.lower():
            # Replicate each value `num_replications` times (without altering order)
                replicated_data = np.repeat(value, self.num_replications)  
                self.rep_data_dict[key] = replicated_data  # Store replicated data
            else:
                self.rep_data_dict[key] = value
                
        return self.rep_data_dict

    def proportion_pos(self):
        """Compute the proportion of positive values for both replicated daily and summed hourly data."""
    
        self.prop_pos_dict = {}  # Flat dictionary with all proportions

        # Process Replicated Daily Data
        for key, value in self.rep_data_dict.items():
            if "day" in key.lower() or "daily" in key.lower():
                lakedepth = 30 if "fail" in key.lower() or "failing" in key.lower() else 60
                self.prop_pos_dict[key] = value / lakedepth  # Compute and store proportions

        # Process Summed Hourly Data
        for sumkey, sumvalue in self.summed_dict.items():
            if "hr" in sumkey.lower() or "hourly" in sumkey.lower():
                lakedepth = 30 if "fail" in sumkey.lower() or "failing" in sumkey.lower() else 60
                self.prop_pos_dict[sumkey] = sumvalue / lakedepth  # Compute and store proportions

        return self.prop_pos_dict  
    
    
class RepDayGetProps:
    def __init__(self, data, num_replications):
        """
        Initialize the dataset object.
        INPUT = summed binary columns 
        Parameters:
            data (dict): Dictionary of datasets where keys are dataset names.
            num_replications (int): Number of times to replicate each column.
        """
        self.summed_dict = data  # Store summed column data
        self.num_replications = num_replications
        self.rep_data_dict = {}  # To store replicated data
        self.prop_pos_dict = {}  # To store proportion data
        
    def replicate_match_dims_dict(self):
        """Replicate each value in the summed arrays 24 times, maintaining order."""
        self.rep_data_dict = {}  # Reset dictionary

        for key, value in self.summed_dict.items():
            # Ensure it's a 1D array
            if value.ndim > 1:
                value = value.flatten()
                
            if "day" in key.lower() or "daily" in key.lower():
            # Replicate each value `num_replications` times (without altering order)
                replicated_data = np.repeat(value, self.num_replications)  
                self.rep_data_dict[key] = replicated_data  # Store replicated data
            else:
                self.rep_data_dict[key] = value
                
        return self.rep_data_dict
    
    def proportion_pos(self):
       """Compute the proportion of positive values for both replicated daily and summed hourly data."""
    
       self.prop_pos_dict = {}  # Flat dictionary with all proportions

        # Process Replicated Daily Data
       for key, value in self.rep_data_dict.items():
            if "day" in key.lower() or "daily" in key.lower():
                lakedepth = 30 if "fail" in key.lower() or "failing" in key.lower() else 60
                self.prop_pos_dict[key] = value.flatten() / lakedepth  # Ensure 1D before division

        # Process Summed Hourly Data
       for sumkey, sumvalue in self.summed_dict.items():
            if self.summed_dict[sumkey].ndim > 1:
                sumvalue = sumvalue.flatten()  # Ensure 1D before division
            if "hr" in sumkey.lower() or "hourly" in sumkey.lower():
                lakedepth = 30 if "fail" in sumkey.lower() or "failing" in sumkey.lower() else 60
                self.prop_pos_dict[sumkey] = sumvalue / lakedepth  # Compute and store proportions

       return self.prop_pos_dict

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


def plot_heatmap1D(dfscount, lake_year_period, pattern, user_inputs, start_date=None, end_date=None):
    """
    Plots a heatmap for selected variables from the dataframe within a date range.
    """
    if lake_year_period not in dfscount:
        raise ValueError(f"Key '{lake_year_period}' not found in dfscount.")
    
    df = dfscount[lake_year_period]
    selected_columns = extract_matching_columns(df, pattern, user_inputs)
    if not selected_columns:
        raise ValueError("No matching columns found for the selected criteria.")
    
    df['EST_DateTime'] = pd.to_datetime(df['EST_DateTime'], utc=True)
    if start_date and end_date:
        df = df[(df['EST_DateTime'] >= start_date) & (df['EST_DateTime'] <= end_date)]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[selected_columns].T, cmap='coolwarm', xticklabels=False)
    plt.xlabel("Time")
    plt.ylabel("Variables")
    plt.title(f"Heatmap for {', '.join(selected_columns)} in {lake_year_period}")
    plt.show()
    
    
def read_df_dir_tocsv(directory):
    return {os.path.splitext(f)[0]: pd.read_csv(os.path.join(directory, f)) 
        for f in os.listdir(directory) if f.endswith('.csv')}


def clean_column_name(col_name):
    """Removes '.csv' and underscores from column names for better legend readability."""
    return col_name.replace('.csv', '').replace('_', ' ')


    
def plot_time_series(df, lake_year_period, pattern, user_inputs, start_date=None, end_date=None):
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

    for col in selected_columns:
        color = 'royalblue' if 'hr' in col else 'darkorange' if 'day' in col else 'black'
        plt.plot(df['EST_DateTime'], df[col], label=clean_column_name(col), color=color, linewidth=2)

    # Extract lake, year, and period for a separate title line
    lake, year, period = lake_year_period.split('_')  # Assuming underscore-separated format
    
    plt.xlabel("DateTime", fontsize=15, fontweight='bold')
    plt.ylabel("Value", fontsize=15, fontweight='bold')

    # Separate lake-year-period from the title
    plt.suptitle(f"{lake} {year} {period}", fontsize=16, fontweight='bold', fontname = 'Calibri')  # First line
    plt.title(f"Time Series Plot for {', '.join(selected_columns)}", fontsize=16, fontweight='bold', pad=15, fontname = 'Calibri')  # Second line

    plt.legend(fontsize=12, frameon=False)
    plt.xticks(fontsize=12, fontname = 'Times New Roman')
    plt.yticks(fontsize=12, fontname = 'Times New Roman')
    plt.grid(False)  # Remove grid lines
    plt.tight_layout()  # Improve spacing for better publication quality
    plt.show()


"end loading packages, defining functions, metadata and classes for loading, sorting, and preprocessing arrays to dataframes"
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
"end loading packages, defining functions, metadata and classes for loading, sorting, and preprocessing arrays to dataframes"
"end loading globals"

#%%
"""
Make dataframes for each lake, year, period: (grp models + lyons), with daily_replicated and hourly

1. load and name directories (2*2*2) = 6 (name = lake_year_period)
2. query by (lake, year, period) > make dictionary
3. replicate daily files > make dataframes = 1. sums (count); 2. proportion positive (proportion)
4. Load correct Lyons hourly and rep daily files > process into proportion and sums (see above)
4. add processed lyons to columns
"""
#####
"==================  1. Loading directories (see above 'class' to use if using binary summed or not) =================="

#grp_binary_summed = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_allRuns_binary/GRP_allRuns_strat_binary_summed'
grp_pturn_sum = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/preturn_lakesyears/GRPbinary_preturn/GRPsummed_preturn'
grp_strat_sum ='/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRP_all_021425/stratified_lakesyears/GRPbinary_strat/GRPsummed_strat'

# Create variables from dirs
datasets_pturn = load_datasets_from_directory(grp_pturn_sum)
datasets_strat = load_datasets_from_directory(grp_strat_sum)

# list
base_dirs = [grp_pturn_sum, grp_strat_sum]

# itearate
datasets = [load_datasets_from_directory(i) for i in base_dirs] #list of dictionaries


"  ========  # Steps - info : Query datasets based on metadata and generate ========="


# Load datasets for each directory - nested
datasets = {dir_path: load_datasets_from_directory(dir_path) for dir_path in base_dirs}

# Flatten the datasets into a single dictionary for easier querying
all_datasets = {filename: dataset for dset in datasets.values() for filename, dataset in dset.items()}


# Step 2: Query datasets dynamically
lake_names = ["crook", "fail"]
years = ["2021", "2022"]
query_results = {}

for lake in lake_names:
    for year in years:
        query_results[f"{lake}_{year}_pturn"] = query_datasets(datasets[grp_pturn_sum], lakes=lake, years=year, variables="grp")
        query_results[f"{lake}_{year}_strat"] = query_datasets(datasets[grp_strat_sum], lakes=lake, years=year, variables="grp")



# Convert query results to NumPy arrays
query_data = {key: load_datasets_asdict(value) for key, value in query_results.items()}

# Ensure all data are stored as 1D NumPy arrays
for lk_yr, models in query_data.items():  # Get dictionary corresponding to each lake-year
    for model in models:  # Iterate over datasets within that lake-year
        models[model] = models[model].flatten()  # Ensure 1D array


"1D array sums and proportions - See classes "

count_obj = {}  # Dictionary to store objects
rep_days = {}  # Dictionary to store replicated datasets
prop_dict = {}  # Dictionary to store proportion results


count_obj = {key: RepDayGetProps(data_dict, num_replications=24) for key, data_dict in query_data.items()} 
print(count_obj)# Pass dictionary

rep_days = {key: count_obj[key].replicate_match_dims_dict() for key, data_dict in query_data.items()}  # No arguments

prop_dict = {key: count_obj[key].proportion_pos() for key, data_dict in query_data.items()} # No arguments


"""
Examples
  # Crook   
    # Crook pturn
crook_2021_pturn = query_datasets(datasets_pturn, lakes="crook", years="2021", variables="grp")
    
    # fail strat
fail_2022_strat = query_datasets(datasets_strat, lakes="fail", years="2022", variables="grp")

# Step #: Perform operations on selected datasets
    
#for dataset in datasets:
#for filename, dataset in datasets.items():
#    print(f"Processing {filename}...")
 #   dataset.load()
        #print(dataset.metadata.keys())
        #print(dataset.data.shape)  # Print the first few rows of the processed data

# Step 4: Efficiently Sort & Process Data
#def sort_datasets(dataset_dict, by_col=0):
#    
#    Sort each dataset in a dictionary by a given column (default: first column).
#    
#    for filename, dataset in dataset_dict.items():
#        dataset.sort(by=by_col)
#    return dataset_dict


# Apply sorting if needed
#sorted_data = {key: sort_datasets(value) for key, value in query_results.items()}        
        
"""
"end making dictionaries {lake_yr_period: {model_run fname: 1D array data}} for prop pos and num cells replicated/match dims"

#%%
" ------------------------ Making dataframes Updating cell below below-------------------------------#"

# Step 2: Convert to DataFrame (Column Names = Array Names)
#crook_grp_numpos_2021 = pd.DataFrame(replicated_results1)
#crook_grp_proppos_2021 = pd.DataFrame(proportion_results1)


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

    
# CHECK
#datekeys = {datekey:datekey.lower().split('_') for datekey in start_end_dates}
#print(datekeys)


# trying time indices - used for the replicated files so no conditional needed for hourly or daily frequency
period_dates = time_index_by_lakeyear(prop_dict,start_end_dates )

# Overwrite if all datetime indices are the same per lake_yr_whatever you parse by 
period_dates = extract_first_datetime(period_dates)

# Print extracted dictionary
#print(period_dates)



# Generate a time index with EST timezone and a fixed number of intervals
#time_index = pd.date_range(start=start_date, periods=num_intervals, freq="H", tz="America/New_York")

#%%
"Creating dataframes for all lakes, year, period using nested dicts"
"dataframes = proppos= prop pos, rep days = num pos, --- cols = model, "

#pd.set_option("display.max_columns", None)

"--------------------- 'Main' cell ---------------------------------"
    
count_dfs = simple_ndict_to_df(rep_days, period_dates)
prop_dfs = simple_ndict_to_df(prop_dict, period_dates)
    
# Base dir
df_counts_savedir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Redo_pos_summaries_dfs/NumCells'
df_props_savedir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Redo_pos_summaries_dfs/Proportions'


#save_dfs_to_csv(count_dfs, df_counts_savedir)
#save_dfs_to_csv(prop_dfs, df_props_savedir)


#################### REMAKING FAIL 2022
fail_2022_counts = {k:v for k, v in count_dfs.items() if "fail" in k.lower() and "2022" in k.lower()}
fail_2022_props = {k:v for k, v in prop_dfs.items() if "fail" in k.lower() and "2022" in k.lower()}


#save_dfs_to_csv(fail_2022_counts, df_counts_savedir)
#save_dfs_to_csv(fail_2022_props, df_props_savedir)



" DataFrames for Num Cell Counts and Proportions per lake_year_period saved in directory"
#%%

" ----================================== PLOTTING CODE- LINEPLOTS & ARRAYS  ==========================================-----"
" --------Clear variables and restart to avoid memory issues and to run code to visualize -------"
" ----------------- Re-run cell(s) for essential packages, functions, and globals, which is the intial cell(s) ------------"
import os
import re
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style 
#style.use()    #


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
        user_input = user_inputs_fail
    else:
        user_input = user_inputs_crook
    plot_time_series(dfscount, strat, pattern, user_input)
    #plot_heatmap1D(dfscount, strat, pattern, user_input)
        

for pturn in pturns:
    if "fail" in strat:
        user_input = user_inputs_fail
    else:
        user_input = user_inputs_crook
    plot_time_series(dfscount, pturn, pattern, user_input)
    
    


#%%

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


def plot_heatmap(arr_dict, lake_yr_period, pattern, user_inputs, date_dict, start_date=None, end_date=None):
    """
    Plots heatmaps for selected 2D arrays from a nested dictionary structure 
    {lake_yr_period: {array_name: 2D array}}, using corresponding time from date_dict as the x-axis.
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
        
        # Expand date_labels if needed (for hourly data)
        if array_data.shape[1] == len(date_labels) * 24:  # If hourly data exists
            date_labels = np.repeat(date_labels, 24)  # Convert daily timestamps into hourly

        # Ensure time matches the number of columns in the array
        if array_data.shape[1] != len(date_labels):
            raise ValueError(f"Mismatch: {selection} has {array_data.shape[1]} time steps, but date_dict has {len(date_labels)}.")

        plt.figure(figsize=(12, 8))
        sns.heatmap(array_data, cmap='seismic', xticklabels=date_labels.strftime('%Y-%m-%d %H:%M')[::100], cbar=True)
        plt.xlabel("Time")
        plt.ylabel("Depth / Variables")
        plt.title(f"Heatmap for {selection} in {lake_yr_period}")

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45, ha='right')

        # Flip y-axis so depth increases downward
        #plt.gca().invert_yaxis()

        plt.show()

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
        plt.imshow(np.flipud(array_data), aspect='auto', cmap='seismic', origin='lower') #np.flipud(array_data),...

        # Add colorbar
        plt.colorbar(label='Value')

        # Set axis labels
        plt.xlabel("Time")
        plt.ylabel("Depth / Variables")
        plt.title(f"Heatmap for {selection} in {lake_yr_period}")

        # Set x-axis ticks
        x_ticks = np.linspace(0, array_data.shape[1] - 1, min(10, len(date_labels))).astype(int)
        plt.xticks(x_ticks, date_labels[x_ticks].strftime('%Y-%m-%d'), rotation=45, ha='right')

        # Flip y-axis so depth increases downward
        plt.gca().invert_yaxis()

        plt.show()


" End loading time series visualization and plotting functions "


#%%


"""
-----------========= Heatmaps but using the values to inspect the models not the binary ============--------------

-Current issues with the heatmaps_new function include: 
    The time plotting is incorrect for the time intervals and dates 
    There seems to be another issue with the 'outer keys' single loop (or the function) that only plots some and not all of the base dirs 

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

"""

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
        
        

for outer_key, array_dict in query_data_binary.items():
    if 'crook' in outer_key:
        plot_heatmapnew(query_data_binary, outer_key, pattern, user_inputs_crook, period_dates)
    else:
        plot_heatmapnew(query_data_binary, outer_key, pattern, user_inputs_fail, period_dates)



"""
Trying to isolate other days dynamically but who cares that isnt my job rn
for outer_key, array_dict in query_data.items():
    if 'crook_2021_strat' in outer_key:
        plot_heatmapnew_er(query_data, outer_key, pattern, user_inputs_crook, period_dates, start_date='2021-09-15', end_date= '2021-11-03')
    elif 'crook_2022_strat' in outer_key:
        plot_heatmapnew_er(query_data, outer_key, pattern, user_inputs_crook, period_dates, start_date='2022-09-15', end_date= '2022-11-03')
    elif 'fail_2021_strat' in outer_key:
        plot_heatmapnew_er(query_data, outer_key, pattern, user_inputs_crook, period_dates, start_date='2021-10-15', end_date= '2021-12-01')
    elif 'fail_2022_strat' in outer_key:
        plot_heatmapnew_er(query_data, outer_key, pattern, user_inputs_crook, period_dates, start_date='2021-10-15', end_date= '2021-12-01')

"""
#%%
"----- Step # ==================== Plotting Lyons Arrays - diff pattern and no query ==============-----"

import matplotlib.colors as mcolors

# Function to convert CSV to NumPy array
def csv_toarray(fpath):   
    array = pd.read_csv(fpath, header=None).to_numpy()
    print(f"Loaded array from {fpath} with shape: {array.shape}")
    return array

# Load all CSV files from a directory into a dictionary
def load_csv_files(directory):
    return {
        os.path.splitext(f)[0]: csv_toarray(os.path.join(directory, f))
        for f in os.listdir(directory) if f.endswith('.csv')
    }

def time_index_by_lakeyear_flatdict(flat_dict, period, start_end_dates):
    """
    Create a dictionary of time indices based on preprocessed start_end_dates.
    Handles both hourly and daily intervals efficiently.

    Parameters:
        flat_dict (dict): Dictionary where keys contain lake, variable, time scale, and year.
        period (str): Either 'strat' or 'pturn' (inferred from dictionary name).
        start_end_dates (dict): Dictionary containing start and end dates.

    Returns:
        dict: Dictionary with datetime indices matching the dataset's time steps.
    """
    time_indices = {}

    # Expand start_end_dates to include both `hr` and `day` versions
    start_end_dates = preprocess_start_end_dates(start_end_dates)

    print(f"Expanded date dictionary keys: {list(start_end_dates.keys())}")

    for key, array in flat_dict.items():
        # ✅ Ensure proper key splitting
        parts = key.split('_')

        if len(parts) == 4:
            # Case: Lyons dictionary (e.g., "lyons_crook_hr_2021")
            _, lake, timescale, year = parts
            variable = "lyons"  # Default variable for Lyons data
        elif len(parts) == 5:
            # Case: Preturn dictionary (e.g., "crooked_hourly_grp_2021")
            lake, timescale, variable, year = parts
        else:
            print(f"⚠️ Skipping {key} (Unrecognized format)")
            continue

        # ✅ Construct `date_key` with period included
        date_key = f"{lake}_{year}_{period}_{timescale}"
        if date_key not in start_end_dates:
            print(f"⚠️ Warning: No start/end date found for {date_key}. Skipping.")
            continue

        # ✅ Determine number of time steps in the dataset
        num_time_steps = array.shape[1]

        # ✅ Determine frequency based on `hr` (hourly) or `day` (daily)
        freq = "h" if "hr" in timescale or "hourly" in timescale else "d"

        # ✅ Generate datetime index
        time_indices[key] = pd.date_range(
            start=start_end_dates[date_key]['start_date'],
            periods=num_time_steps,
            freq=freq,
            tz="America/New_York"
        )

        # Debugging prints
        print(f"✅ Created time index for {key}: {len(time_indices[key])} timestamps ({freq})")

    return time_indices


# Load strat and preturn datasets
lyons_strat = load_csv_files('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/lyons_strat')
lyons_pturn = load_csv_files('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/lyons_preturn')

# Define a mapping for replacements
replacements = {
    "hrs": "hr", "hourly": "hr", "hours": "hr",
    "temperature": "temp", "temps": "temp", "tmp": "temp",
    "dissolved_oxygen": "do", "oxygen": "do", "dos": "do",
    "failing": "fail", "crooked": "crook",
    "daily": "day", "days": "day",
}

# Function to normalize dictionary keys
def normalize_keys(data_dict):
    normalized_dict = {}
    for key, value in data_dict.items():
        new_key = key.lower()
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)
        normalized_dict[new_key] = value
    return normalized_dict

# Normalize dataset keys
lyons_strat = normalize_keys(lyons_strat)
lyons_pturn = normalize_keys(lyons_pturn)

# Generate time indices
# ✅ Call function for 'strat' datasets
period_dates_strat = time_index_by_lakeyear_flatdict(lyons_strat, "strat", start_end_dates)
# ✅ Call function for 'pturn' datasets
period_dates_pturn = time_index_by_lakeyear_flatdict(lyons_pturn, "pturn", start_end_dates)
# ✅ Merge both into a single dictionary
period_dates = {**period_dates_strat, **period_dates_pturn}


def plot_lyons_heatmaps(data_dict, period, period_dates, start_end_dates):
    """
    Plots heatmaps for all datasets in the directory, adjusting for time scale (hourly or daily).
    
    Parameters:
        data_dict (dict): Dictionary of {filename: NumPy array}.
        period (str): Either 'strat' or 'pturn' (extracted from dictionary name).
        period_dates (dict): Dictionary of datetime indices for each dataset.
        start_end_dates (dict): Dictionary with start/end date ranges for filtering.
    """
    # ✅ Define binary colormap (white for 0, red for 1)
    cmap = mcolors.ListedColormap(["midnightblue", "maroon"])
    norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)

    for key, array in data_dict.items():
        # ✅ Extract lake, time scale, and year from the key
        parts = key.split('_')
        if len(parts) < 4:  # Ensure at least 'lyons', lake, time scale, and year exist
            print(f"⚠️ Skipping {key} (Unrecognized format: expected at least 4 parts)")
            continue

        _, lake, t_scale, year = parts  # Ignore 'lyons'

        # Debugging print
        print(f"🔹 Processing: {lake}, {year}, {period}, {t_scale}")

        # ✅ Generate start/end date key
        date_key = f"{lake}_{year}_{period}"
        if date_key in start_end_dates:
            start_date = start_end_dates[date_key]['start_date']
            end_date = start_end_dates[date_key]['end_date']
        else:
            print(f"⚠️ Warning: No start/end date found for {date_key}. Using full range.")
            start_date, end_date = None, None

        # ✅ Check if the dataset exists in period_dates
        if key not in period_dates:
            print(f"⚠️ Warning: No date index found for {key}. Skipping.")
            continue

        date_labels = period_dates[key]

        # ✅ Adjust date labels to match dataset time steps
        num_time_steps = array.shape[1]
        num_dates = len(date_labels)

        if t_scale == "hr":
            if num_dates == num_time_steps // 24:
                print(f"🔄 Expanding daily timestamps to hourly for {key}")
                date_labels = np.repeat(date_labels, 24)
        elif t_scale == "day":
            if num_dates * 24 == num_time_steps:
                print(f"🔄 Downsampling hourly timestamps to daily for {key}")
                date_labels = date_labels[::24]

        # ✅ Final check: Ensure time matches the number of columns in the array
        if array.shape[1] != len(date_labels):
            print(f"❌ Mismatch for {key}: Array time steps ({array.shape[1]}) ≠ Date labels ({len(date_labels)})")
            continue
        # ✅ Plot heatmap with binary colormap
        plt.figure(figsize=(12, 8))
        plt.imshow(np.flipud(array), aspect='auto', cmap=cmap, norm=norm, origin='lower') #np.flipud(array),...

        # ✅ Add discrete colorbar with 0 and 1 labels
        cbar = plt.colorbar(ticks=[0, 1])
        cbar.set_label("Value")
        cbar.ax.set_yticklabels(["0", "1"])

        # Set axis labels
        plt.xlabel("Time")
        plt.ylabel("Depth / Variables")
        plt.title(f"Heatmap for TD06 {lake} {year} {period} ({t_scale})")  # ✅ Added `t_scale` to title

        # Set x-axis ticks
        x_ticks = np.linspace(0, array.shape[1] - 1, min(10, len(date_labels))).astype(int)
        plt.xticks(x_ticks, date_labels[x_ticks].strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')

        # Flip y-axis so depth increases downward
        plt.gca().invert_yaxis()

        plt.show()


# ✅ Call function for both strat and pturn datasets
plot_lyons_heatmaps(lyons_strat, "strat", period_dates_strat, start_end_dates)
plot_lyons_heatmaps(lyons_pturn, "pturn", period_dates_pturn, start_end_dates)
#%%
"----- Step # ==================== Plotting T & DO Arrays - diff pattern and no query ==============-----"

import matplotlib.colors as mcolors

# Function to convert CSV to NumPy array
def csv_toarray(fpath):   
    array = pd.read_csv(fpath, header=None).to_numpy()
    print(f"Loaded array from {fpath} with shape: {array.shape}")
    return array

# Load all CSV files from a directory into a dictionary
def load_csv_files(directory):
    return {
        os.path.splitext(f)[0]: csv_toarray(os.path.join(directory, f))
        for f in os.listdir(directory) if f.endswith('.csv')
    }

def time_index_by_lakeyear_flatdict(flat_dict, period, start_end_dates):
    """
    Create a dictionary of time indices based on preprocessed start_end_dates.
    Handles both hourly and daily intervals efficiently.

    Parameters:
        flat_dict (dict): Dictionary where keys contain lake, variable, time scale, and year.
        period (str): Either 'strat' or 'pturn' (inferred from dictionary name).
        start_end_dates (dict): Dictionary containing start and end dates.

    Returns:
        dict: Dictionary with datetime indices matching the dataset's time steps.
    """
    time_indices = {}

    # Expand start_end_dates to include both `hr` and `day` versions
    start_end_dates = preprocess_start_end_dates(start_end_dates)

    print(f"Expanded date dictionary keys: {list(start_end_dates.keys())}")

    for key, array in flat_dict.items():
        # ✅ Ensure proper key splitting
        parts = key.split('_')

        if 'lyons' in parts:
            # Case: Lyons dictionary (e.g., "lyons_crook_hr_2021")
            _, lake, timescale, year = parts
            variable = "lyons"  # Default variable for Lyons data
        elif 'lyons' not in parts:
            # Case: Preturn dictionary (e.g., "crooked_hourly_grp_2021")
            lake, timescale, variable, year = parts
        else:
            print(f"⚠️ Skipping {key} (Unrecognized format)")
            continue

        # ✅ Construct `date_key` with period included
        date_key = f"{lake}_{year}_{period}_{timescale}"
        if date_key not in start_end_dates:
            print(f"⚠️ Warning: No start/end date found for {date_key}. Skipping.")
            continue

        # ✅ Determine number of time steps in the dataset
        num_time_steps = array.shape[1]

        # ✅ Determine frequency based on `hr` (hourly) or `day` (daily)
        freq = "h" if "hr" in timescale or "hourly" in timescale else "d"

        # ✅ Generate datetime index
        time_indices[key] = pd.date_range(
            start=start_end_dates[date_key]['start_date'],
            periods=num_time_steps,
            freq=freq,
            tz="America/New_York"
        )

        # Debugging prints
        print(f"✅ Created time index for {key}: {len(time_indices[key])} timestamps ({freq})")

    return time_indices

# Define a mapping for replacements
replacements = {
    "hrs": "hr", "hourly": "hr", "hours": "hr",
    "temperature": "temp", "temps": "temp", "tmp": "temp",
    "dissolved_oxygen": "do", "oxygen": "do", "dos": "do",
    "failing": "fail", "crooked": "crook",
    "daily": "day", "days": "day",
}

# Function to normalize dictionary keys
def normalize_keys(data_dict):
    normalized_dict = {}
    for key, value in data_dict.items():
        new_key = key.lower()
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)
        normalized_dict[new_key] = value
    return normalized_dict

def plot_nongrp_heatmaps(data_dict, period, period_dates, start_end_dates):
    """
    Plots heatmaps for all datasets in the directory, adjusting for time scale (hourly or daily).

    Parameters:
        data_dict (dict): Dictionary of {filename: NumPy array}.
        period (str): Either 'strat' or 'pturn' (extracted from dictionary name).
        period_dates (dict): Dictionary of datetime indices for each dataset.
        start_end_dates (dict): Dictionary with start/end date ranges for filtering.
    """

    for key, array in data_dict.items():
        parts = key.split('_')
        if len(parts) < 3:
            print(f"⚠️ Skipping {key} (Unrecognized format)")
            continue

        lake, t_scale, variable, year = parts
        date_key = f"{lake}_{year}_{period}"

        if date_key not in start_end_dates:
            print(f"⚠️ Warning: No start/end date found for {date_key}. Using full range.")
            start_date, end_date = None, None

        if key not in period_dates:
            print(f"⚠️ Warning: No date index found for {key}. Skipping.")
            continue

        date_labels = period_dates[key]

        # ✅ Auto-detect if dataset is binary
        unique_vals = np.unique(array)
        is_binary = np.array_equal(unique_vals, [0, 1])

        if is_binary:
            cmap = mcolors.ListedColormap(["darkblue", "maroon"])
            norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)
        else:
            # ✅ Use `TwoSlopeNorm` to center at 0
            vmin, vmax = np.nanmin(array), np.nanmax(array)
            midpoint = 0  # Center the colormap at zero
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)
            cmap = "seismic"  # Diverging colormap for balanced values

        # ✅ Ensure date labels match dataset time steps
        num_time_steps = array.shape[1]
        if len(date_labels) != num_time_steps:
            print(f"❌ Mismatch for {key}: Array time steps ({num_time_steps}) ≠ Date labels ({len(date_labels)})")
            continue

        # ✅ Plot heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(np.flipud(array), aspect='auto', cmap=cmap, norm=norm, origin='lower') #np.flipud(array),...

        # ✅ Add discrete colorbar for binary, normal for continuous
        cbar = plt.colorbar()
        cbar.set_label("Value")
        if is_binary:
            cbar.set_ticks([0, 1])
            cbar.ax.set_yticklabels(["0", "1"])

        # ✅ Set axis labels
        plt.xlabel("Time")
        plt.ylabel("Depth / Variables")
        plt.title(f"Heatmap for {lake} {variable} {year} {period} ({t_scale})")

        # ✅ Set x-axis ticks
        x_ticks = np.linspace(0, num_time_steps - 1, min(10, len(date_labels))).astype(int)
        plt.xticks(x_ticks, date_labels[x_ticks].strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')

        # Flip y-axis so depth increases downward
        plt.gca().invert_yaxis()

        plt.show()

#-------------================---------------================---------------=-=-=-=-

# ✅ Load datasets
preturn_values = load_csv_files('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/preturn_values')
strat_values = load_csv_files('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/strat_values')

# ✅ Normalize dataset keys
preturn_values = normalize_keys(preturn_values)
strat_values = normalize_keys(strat_values)

# ✅ Generate time indices
period_dates_preturn = time_index_by_lakeyear_flatdict(preturn_values, "pturn", start_end_dates)
period_dates_strat = time_index_by_lakeyear_flatdict(strat_values, "strat", start_end_dates)

# ✅ Plot
plot_nongrp_heatmaps(strat_values, "strat", period_dates_strat, start_end_dates)
plot_nongrp_heatmaps(preturn_values, "pturn", period_dates_preturn, start_end_dates)




#%%
"Testing simply to evaluate why all cells are suitable for failing 2022  GRP models"
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date, timedelta
from operator import itemgetter


def csv_toarray(fpath):   # skip_blank_lines = False
    array = pd.read_csv(fpath, header=None).to_numpy()
    print(f"Loaded array from {fpath} with shape: {array.shape}")
    return array

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


def flip_and_plot(arrays, time_index, plot=True, figsize=(12, 8), time_format ="%Y-%m-%d"):
    """
    Flips a list of arrays about the y axis, plots time from date index array on the x-axis labels.
    
    Parameters:
        data_dict (dict): Dictionary where keys are filenames, values are NumPy arrays.
        save_dir (str): Directory to save the CSV files.
    """
    flipped_arrays = [(arr) for arr in arrays]              #np.flipud(arr)...
    #time_index = day_indx
    if isinstance(plot, bool) and plot:
        time_labels = time_index.to_numpy()  # Convert index to numpy array
        num_ticks = min(10, len(time_labels))  # Limit number of x-axis ticks
        tick_positions = np.linspace(0, len(time_labels) - 1, num_ticks, dtype=int)
        tick_labels = [pd.to_datetime(time_labels[i]).strftime(time_format) for i in tick_positions]

        for i, arr in enumerate(flipped_arrays):
            plt.figure(figsize=figsize)  # Set figure size
            plt.imshow(arr, cmap='seismic', origin='lower', aspect='auto')
            plt.colorbar(label=f'Array {i+1}')  # Label arrays numerically
            plt.xlabel('Time')
            plt.ylabel('Depth')
            plt.title(f'Flipped Array {i+1}')
            
            # Set x-axis labels
            plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)

            # Flip y-axis so depth increases downward
            plt.gca().invert_yaxis()
            plt.show()

    return flipped_arrays  

# Define GRP with variable params
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
    return GRP, GRP_lethal, fDO, fdo_lethal

# Define mass and P coefficient values to analyze
mass_coefficients = np.array([200])  # Example mass values
P_coefficients = np.array([0.4])  # Example P values
AD_values = np.array([2000])  # AD values
slope_vals = np.array([0.168, 0.138])
intercept_vals = np.array([1.63, 2.09])


fail22_do = csv_toarray('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/strat_values/failing_daily_do_2022.csv')
fail22_temp = csv_toarray('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/strat_values/failing_daily_temp_2022.csv')
fail22_do_hr = csv_toarray('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/strat_values/failing_hourly_do_2022.csv')
fail22_temp_hr = csv_toarray('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/strat_values/failing_hourly_temp_2022.csv')

# making output dictionaries 
fail22_grp = {}
fail22_grp_lethal ={}
fail22_fDO = {}
fail22_fdo_lethal = {}

for slope in slope_vals:
    for intercept in intercept_vals:
        fail22_grp, fail22_grp_lethal, fail22_fDO, fail22_fdo_lethal = calculate_growth_fdos(fail22_do, fail22_temp, mass_coefficients, P_coefficients, AD_values, slope, intercept)


fail22_arrlist = [fail22_temp, fail22_do, fail22_grp, fail22_grp_lethal]



plt.figure(figsize =(12,8))  # Set figure size
plt.imshow(fail22_do_hr, cmap='seismic', origin='lower', aspect='auto')
plt.colorbar(label='DO fail 22')  # Label arrays numerically
plt.xlabel('Time')
plt.ylabel('Depth')
plt.title('Flipped fail DO')
# Flip y-axis so depth increases downward
plt.gca().invert_yaxis()
plt.show()




plt.figure(figsize =(12,8))  # Set figure size
plt.imshow(fail22_temp_hr, cmap='seismic', origin='lower', aspect='auto')
plt.colorbar(label='Temp fail 22')  # Label arrays numerically
plt.xlabel('Time')
plt.ylabel('Depth')
plt.title('Flipped fail temp')
# Flip y-axis so depth increases downward
plt.gca().invert_yaxis()
plt.show()





plt.figure(figsize =(12,8))  # Set figure size
plt.imshow(fail_22_ogDOday, cmap='seismic', origin='lower', aspect='auto')
plt.colorbar(label='og DO fail 22')  # Label arrays numerically
plt.xlabel('Time')
plt.ylabel('Depth')
plt.title('Flipped fail temp')
# Flip y-axis so depth increases downward
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize =(12,8))  # Set figure size
plt.imshow(fail_22_ogTempday, cmap='seismic', origin='lower', aspect='auto')
plt.colorbar(label='og Temp fail 22')  # Label arrays numerically
plt.xlabel('Time')
plt.ylabel('Depth')
plt.title('Flipped fail temp')
# Flip y-axis so depth increases downward
plt.gca().invert_yaxis()
plt.show()
#%%
"================== REMAKING FAILING GRP's, LYONS ================================="


"------- DELTEING WRONG ONES FROM GRP DICT (LYONS ALREADY SEPARATED) ---------"



#strat values fail 2022 save dir
strat_valsdir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/strat_values'

#strat binary fail 2022 save dir
strat_binarydir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/strat_binary'
#preturn values fail 2022 save dir
preturn_valsdir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/preturn_values'

#preturn binary fail 2022 save dir
preturn_binarydir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/preturn_binary'


lyonsstrat_dir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/lyons_strat'

lyonspturn_dir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/lyons_preturn'


base_fail2022_dir = ''












#%%
"""
classmethod DataFrame.from_dict(data, orient='columns', dtype=None, columns=None)[source]

"""

def nested_dicts_to_df(count_dict, prop_dict, time_dict, outer_key):
    """
    Converts nested count and proportion dictionaries into a MultiIndex DataFrame
    with timestamps properly aligned as a column, ensuring correct CSV formatting.
    """

    # Merge count and proportion dictionaries under 'metric' keys
    data = {
        "count": count_dict.get(outer_key, {}), 
        "proportion": prop_dict.get(outer_key, {})
    }

    # Define metric mapping for column names
    dict_metrics = {
        "count": "count",
        "proportion": "proportion"
    }

    default_metric = "unknown"

    # Create structured dictionary
    structured_data = {
        (model, dict_metrics.get(metric, default_metric)): values
        for metric, models in data.items()  # First level: metric
        for model, values in models.items()  # Second level: model
    }

    # Convert the structured dictionary to a DataFrame
    df = pd.DataFrame(structured_data)

    # Ensure the time index is properly set and aligned
    if outer_key in time_dict:
        df.index = pd.to_datetime(time_dict[outer_key])  # Convert to datetime index
        df.index.name = "EST_DateTime"
    else:
        raise KeyError(f"outer_key '{outer_key}' not found in time_dict.")

    # Set MultiIndex for columns (model, metric)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["model", "metric"])

    # **Key Fix**: Reset index so that "EST_DateTime" is a column, not an index
    df.reset_index(inplace=True)
    

    return df

# If i wasnt so lame I would automate this and the save code

crook2021strat = nested_dicts_to_df(rep_days, prop_dict, period_dates, outer_key = 'crook_2021_strat')
print(crook2021strat.shape)
crook2022strat = nested_dicts_to_df(rep_days, prop_dict, period_dates, outer_key = 'crook_2022_strat')
print(crook2022strat.shape)
crook2021pturn = nested_dicts_to_df(rep_days, prop_dict, period_dates, outer_key = 'crook_2021_pturn')
print(crook2021pturn.shape)
crook2022pturn = nested_dicts_to_df(rep_days, prop_dict, period_dates, outer_key = 'crook_2022_pturn')
print(crook2022pturn.shape)



fail2021strat = nested_dicts_to_df(rep_days, prop_dict, period_dates, outer_key = 'fail_2021_strat')
print(fail2021strat.shape)
fail2022strat = nested_dicts_to_df(rep_days, prop_dict, period_dates, outer_key = 'fail_2022_strat')
print(fail2022strat.shape)
fail2021pturn = nested_dicts_to_df(rep_days, prop_dict, period_dates, outer_key = 'fail_2021_pturn')
print(fail2021pturn.shape)
fail2022pturn = nested_dicts_to_df(rep_days, prop_dict, period_dates, outer_key = 'fail_2021_pturn')
print(fail2022pturn.shape)


# Base dir
#df_savedir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Positive_cell_summaries_dfs'
df_savedir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Positive_cell_summaries_dfs/combined_oneidx'
# Crooked
#crook2021stratpath = 'crook_2021_strat_df.csv' # was the test so already saved
df_paths ={
'crook_2021_strat_df.csv': crook2021strat,
'crook_2022_strat_df.csv': crook2022strat,
'crook_2021_pturn_df.csv': crook2021pturn,
'crook_2022_pturn_df.csv': crook2022pturn ,

# Failing
'fail_2021_strat_df.csv': fail2021strat,
'fail_2022_strat_df.csv': fail2022strat,
'fail_2021_pturn_df.csv': fail2021pturn,
'fail_2022_pturn_df.csv': fail2022pturn }


#testcrook.to_csv(os.path.join(df_dir, 'crook_2021_strat_df.csv'))

for df_path, df in df_paths.items():
    df.to_csv(os.path.join(df_savedir, df_path))
    
" Incorrectly Formatted Multi-Index Dataframes saved succesfully -- useful for storage kinda..."

#%%
"testing the combined dataframes"

def plot_time_series(df, selected_columns=None, start_time=None, end_time=None):
    """
    Plots time series data for selected model columns over a specified time range.

    Parameters:
    df (pd.DataFrame): DataFrame containing time series data.
    selected_columns (list, optional): List of column names to plot. If None, plots the first numeric column.
    start_time (str, optional): Start time for filtering data (format: 'YYYY-MM-DD HH:MM:SS').
    end_time (str, optional): End time for filtering data (format: 'YYYY-MM-DD HH:MM:SS').
    """

    # Step 1: Rename the datetime column if necessary
    df.rename(columns={df.columns[0]: "EST_DateTime"}, inplace=True)

    # Step 2: Convert 'EST_DateTime' to datetime format
    #df["EST_DateTime"] = pd.to_datetime(df["EST_DateTime"], errors='coerce')
    #df.columns[0] = pd.to_datetime(df["EST_DateTime"], errors='coerce')

    # Step 3: Convert all numeric columns to float
    df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

    # Step 4: Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # **Fix: Convert MultiIndex to single-level column names if necessary**
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, col)) for col in df.columns]

    # Step 5: Select columns to plot
    if selected_columns is None:
        selected_columns = [numeric_cols[0]] if len(numeric_cols) > 0 else []
    else:
        # Convert column names to match flattened format
        selected_columns = [
            "_".join(map(str, col)) if isinstance(col, tuple) else col
            for col in selected_columns
        ]

    # **Fix: Check if all selected columns exist in DataFrame**
    selected_columns = [col for col in selected_columns if col in df.columns]

    if not selected_columns:
        print("No valid numeric columns found for plotting. Check column names.")
        return

    # Step 6: Filter data based on time range
    if start_time or end_time:
        mask = (df["EST_DateTime"] >= start_time) & (df["EST_DateTime"] <= end_time)
        df = df.loc[mask]

    # Step 7: Plot selected columns
    plt.figure(figsize=(12, 6))

    for col in selected_columns:
        plt.plot(df["EST_DateTime"], df[col], label=col, linestyle='-', marker='o')

    # **Fix: Ensure column names are strings for plot title**
    plt.title(f"Time Series Plot of {', '.join(selected_columns)}")

    # Formatting the plot
    plt.xlabel("Date Time")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()

    # Show plot
    plt.show()



# Example usage: Plot first numeric column over all available data
plot_time_series(fail_2021_strat)

# Example: Plot specific model columns
plot_time_series(df, selected_columns=["grp_lethal_p0.8_mass200_slope0.168_intercept1.63_crook_day_2021_csv_count"])


# Example usage: Plot specific model columns over all available data
plot_time_series(crook2021strat, selected_columns=["model1", "model2"])

# Example usage: Plot specific model columns over a selected time range
plot_time_series(crook2021strat, selected_columns=["grp_lethal_p0.8_mass200_slope0.168_intercept1.63_crook_day_2021_csv_count"], start_time="2021-10-01 00:00:00", end_time="2021-10-10 23:59:59")


#%%
    
def query_by_keywords(data_dict, *keywords):
    """
    Query a dictionary of NumPy arrays by checking if the key (filename) contains all given keywords.

    Parameters:
        data_dict (dict): Dictionary where keys are filenames (strings) and values are NumPy arrays.
        *keywords: Keywords to filter filenames by (e.g., "crook", "2021", "grp", "day").
    
    Returns:
        dict: Dictionary containing only key-value pairs where the key contains all specified keywords.
    """
    matched_dict = {
        k: v for k, v in data_dict.items() if all(keyword in k for keyword in keywords)
    }

    if not matched_dict:
        print(f"No matches found. Available keys: {list(data_dict.keys())[:5]}...")  # Show only first 5 keys
    
    return matched_dict



    # Step 2: Query datasets based on metadata
select_dictionary = query_by_keywords(data_dict,"crook", "2021", "grp")


#%%
#import pandas as pd 
#import matplotlib.pyplot as plt

"""
    base_dates = {
        'fail': {2021: date(2021, 6, 5), 2022: date(2022, 4, 12)},
        'crook': {2021: date(2021, 6, 24), 2022: date(2022, 6, 7)}
    }
    
    preturn_dates = {
        'fail': {2021: date(2021, 9, 30), 2022: date(2022, 9, 30)},
        'crook': {2021: date(2021, 9, 30), 2022: date(2022, 9, 30)}

"""

# Step 2: Convert to DataFrame (Column Names = Array Names)
crook_grp_numpos_2021 = pd.DataFrame(replicated_results1)
crook_grp_proppos_2021 = pd.DataFrame(proportion_results1)


start_date = '2021-09-30'
end_date = '2021-11-03'

num_intervals = 840  # Example: 10 time points

# Generate a time index with EST timezone and a fixed number of intervals
time_index = pd.date_range(start=start_date, periods=num_intervals, freq="H", tz="America/New_York")

"""

crook_grp_numpos_2021["time_EST"] = time_index  # Attach time index
crook_grp_proppos_2021["time_EST"] = time_index

#df_data = pd.DataFrame({"value": range(num_intervals)})  # Example data

print(crook_grp_numpos_2021)
print(crook_grp_proppos_2021)

# saving 

# Save to CSV
crook_grp_numpos_2021.to_csv("/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/1D_binaryarray_dfs/crook_grp_numpos_2021_preturn.csv", index=False)  

crook_grp_proppos_2021.to_csv("/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/1D_binaryarray_dfs/crook_grp_proppos_2021.csv", index=False)  
"""


df_prop = pd.read_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/1D_binaryarray_dfs/crook_grp_proppos_2021_preturn.csv')

df_num = pd.read_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/1D_binaryarray_dfs/crook_grp_numpos_2021_preturn.csv')



# Convert time column to datetime (if not already)
df_prop["time_EST"] = pd.to_datetime(df_prop["time_EST"]).dt.tz_convert("America/New_York")

# Extract year-month
df_prop["year_month"] = df_prop["time_EST"].dt.to_period("M")

# Compute mean for each month for all numeric columns
monthly_summary = df_prop.groupby("year_month").mean(numeric_only=True)

monthly_summary.to_csv("/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/1D_binaryarray_dfs/crook_grp_pro_pos_2021_preturn_MonthlySummary.csv", index=False)






# Convert time column to datetime (if not already)
crook_grp_proppos_2021["time_EST"] = pd.to_datetime(crook_grp_proppos_2021["time_EST"]).dt.tz_convert("America/New_York")

# Extract year-month
crook_grp_proppos_2021["year_month"] = crook_grp_proppos_2021["time_EST"].dt.to_period("M")

# Compute mean for each month for all numeric columns
monthly_summary = crook_grp_proppos_2021.groupby("year_month").mean(numeric_only=True)



monthly_summary_prop_2021_preturn = crook_grp_proppos_2021.groupby("year_month").agg(["mean", "median", "std"])


monthly_summary_prop_2021_preturn.to_csv("/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/1D_binaryarray_dfs/crook_grp_numpos_2021_preturn_MonthlySummary.csv", index=False)











# Create a DataFrame with the EST time column
#crook_grp_numpos_2021 = pd.DataFrame({"EST_DateTime": time_index})

#crook_grp_proppos_2021 = pd.DataFrame({"EST_DateTime": time_index})


# Step 3: Add a Time Index (Numeric or Datetime)
# df["time"] = pd.date_range(start="2024-01-01", periods=time_steps, freq="D")  # Optional: Datetime index

# Step 4: Set Time Column as Index
crook_grp_numpos_2021.set_index("time_EST", inplace=True)

# Step 5: Plot All Columns
crook_grp_numpos_2021.plot(figsize=(10, 5), marker="o", linestyle="-")
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Time Series of Multiple Arrays")
plt.legend(title="Variables")
plt.show()

