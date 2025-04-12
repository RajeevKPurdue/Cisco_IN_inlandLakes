#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:29:25 2025

@author: rajeevkumar
"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def pyproj_yrs_flatdict(all_files, input_dir):
    data_dict = {}
    
    for file_name in all_files:
        # Parse the filename
        lakename, depth, period, timescale = file_name.replace('.csv', '').split('_')
        depth_value = float(depth)

        # Build the composite key
        key = f"{lakename}_{depth_value:.1f}_{period}_{timescale}"

        # Read the file and store it in the dictionary
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path, parse_dates=["EST_DateTime"], index_col="EST_DateTime")
        data_dict[key] = df

    for key, df in data_dict.items():
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        df.columns = [col.strip() for col in df.columns]
        data_dict[key] = df
    
    return data_dict


def separate_by_period(data_dict):
    """
    Separate the flat_dict into period-specific dictionaries.

    Args:
        flat_dict (dict): Dictionary with composite keys.

    Returns:
        tuple: Three dictionaries (period1_dict, period2_dict, period3_dict).
    """
    period1_dict = {}
    period2_dict = {}
    period3_dict = {}

    for key, df in data_dict.items():
        # Extract period from the key
        _, _, period, timescale = key.split('_')

        # Determine the target dictionary
        target_dict = None
        if period == "period1":
            target_dict = period1_dict
        elif period == "period2":
            target_dict = period2_dict
        elif period == "period3":
            target_dict = period3_dict
        else:
            print(f"Unknown period in key: {key}. Skipping.")
            continue

        # Store the DataFrame in the appropriate dictionary
        target_dict[key] = df

    return period1_dict, period2_dict, period3_dict


def separate_by_timescale(data_dict):
    """
    Separate data_dict into hourly and daily dictionaries.

    Args:
        data_dict (dict): A dictionary with composite keys and DataFrames as values. str- {key:value, ..., }

    Returns:
        tuple: Two dictionaries (hourly_dict, daily_dict) separated by timescale.
    """
    hourly_dict = {}
    daily_dict = {}

    for key, df in data_dict.items():
        if "hourly" in key:
            hourly_dict[key] = df
        elif "daily" in key:
            daily_dict[key] = df
        else:
            print(f"Unknown timescale in key: {key}")

    return hourly_dict, daily_dict

def sep_T_DO_cols(data_dict, t_col, do_col):
    """
    Separates miniDOT T & DO cols for separate array based interpolation
    
    """
    t_dict = {}
    do_dict = {}
    
    for key, df in data_dict.items():
        t_dict[key] = df[t_col]
        do_dict[key] = df[do_col]
    
    return t_dict, do_dict


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

def setup_lakearr(data_dict, depth):
    timesteps = len(data_dict[list(data_dict.keys())[0]]) # get first len since all should be same post slice
    
    lake_temp_arr = np.empty((depth,  timesteps ), dtype = float)
    lake_do_arr = np.empty((depth,  timesteps ), dtype = float)
    
    return lake_temp_arr, lake_do_arr

def convert_dict_dfs_to_arrays(data_dict, reshape="row"):
    """
    Convert DataFrames in a dictionary to NumPy arrays and reshape.
    """
    if reshape == "row":
        return {key: df.values.reshape(1, -1) for key, df in data_dict.items()}  # (1, x)
    elif reshape == "col":
        return {key: df.values.reshape(-1, 1) for key, df in data_dict.items()}  # (x, 1)
    else:
        raise ValueError("reshape must be 'row' or 'col'")
        
#freq - inputted
def slice_dates(data_dict, start_date, end_date):
    "Returns new dictionary sliced from input dict by date bounds"
    return {key: df.loc[start_date:end_date] for key, df in data_dict.items()}

def populate_arrs(t_data_dict, array): #modify to crank out at least t and do ?
    depth_inds = []
    #reshaped_dict = {}
    for key, df in t_data_dict.items():
        # Extract depth from the key
        depth = key.split('_')[1]
        depth_inds.append(int((float(depth))*10)) #indices for filling in

        for val, row_index in enumerate(depth_inds):
            if depth_inds[val] == (float(depth)*10):
                data = t_data_dict[key]
                print('t_data_dict[key]:', data)
                array[row_index,:] = data
    print('depth_inds:', depth_inds)
    pop_arr = array
                
    return pop_arr

        
def replace_groups_with_avg(array, groups):
    rows, cols = array.shape
    for group in groups:
        start, end = group
        # Ensure the group is within the bounds of the array
        if start > 0 and end < cols - 1:
            # Calculate the average of the column before and after the group
            avg_values = (array[:, start - 1] + array[:, end + 1]) / 2

            # Replace the values in the group of columns with the average values
            array[:, start:end +
                  1] = np.tile(avg_values[:, None], end - start + 1)
        else:
            print(
                f"Group {group} cannot be replaced because it doesn't have both left and right neighbors.")
    return array


def presorted_resam_read(all_files, input_dir):
    data_dict = {}
    
    for file_name in all_files:
        # Parse the filename
        lakename, depth, timescale = file_name.replace('.csv', '').split('_')
        depth_value = float(depth)

        # Build the composite key
        key = f"{lakename}_{depth_value:.1f}_{timescale}"

        # Read the file and store it in the dictionary
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path, parse_dates=["EST_DateTime"], index_col="EST_DateTime")
        data_dict[key] = df

    for key, df in data_dict.items():
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        df.columns = [col.strip() for col in df.columns]
        data_dict[key] = df
    
    return data_dict

def csv_toarray(fpath):   # skip_blank_lines = False
    """
    loads csv file as an array without interpreting the first row as a header
    Keeping .csv files and not .npy in case data are needed for R programmers
    Parameters:
        filepath (str)
    """
    array = pd.read_csv(fpath, header=None).to_numpy()
    print(f"Loaded array from {fpath} with shape: {array.shape}")
    return array

def load_csv_files(directory):
    """
    Reads a directory of .csv arrays into dictionary and takes fname as key without extension
    Parameters: 
        directory (str): directory containing arrays stored as .csv files
    """
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

def rep_abovebelow0(Temp_array, DO_array, indices):
    """
    Setting depths above the shallowest sensor equal to the shallowest sensor, and deepest below equal to deepest sensor
    Parameters:
        Temp_array (array): 2D array filled with sensor data at given rows (depths)
        DO_array (array): 2D array filled with sensor data at given rows (depths)
        indices (list or array): numeric values where rows with sensor data are in your array (from least to greatest)
    """
    for j in range(Temp_array.shape[1]):
        for a in range(indices[0]):
            Temp_array[a, j] = Temp_array[indices[0], j]
            DO_array[a, j] = DO_array[indices[0], j]
            
    # Setting depths below the deepest sensor equal to the deepest sensor
        for b in range(indices[-1] + 1, Temp_array.shape[0]):
            DO_array[b, j] = DO_array[indices[-1], j]
            Temp_array[b, j] = Temp_array[indices[-1], j]
    return Temp_array, DO_array

def interp_locs(indices):
    """
    Defines indices where to interpolate rows given the depth (row) in a 2D array
    Parameters:
        indices (list or array): numeric values where rows with sensor data are in your array (from least to greatest)
    """
    interp_row1 =np.array(indices[:-1])+1
    interp_row2 =np.array(indices[1:])-1
    interp_points = interp_points = np.sort(np.concatenate([interp_row1, interp_row2]))
    interp_dist = interp_row2 - interp_row1
    
    sensor_depth = np.array(indices[:-1])
    sensor_nextdepth = np.array(indices[1:])
    
    return interp_row1, interp_row2, interp_points, interp_dist, sensor_depth,sensor_nextdepth 


def interp_ugly(Temp_array, DO_array):
    """
    Interpolates rows (depths) for each column (time) linearly using a weighted average
    """
    for m in range(len(interp_row1)):
        j_start = interp_row1[m]
        j_end = interp_row2[m]
        for j in range(j_start, j_end + 1):
            weight1 = abs(j - j_end) / interp_dist[m]
            weight2 = abs(j - j_start) / interp_dist[m]
            for i in range(len(Temp_array[1])):
                Temp_array[j,i] = (Temp_array[depth_value[m],i] * weight1) + (Temp_array[next_value[m],i] * weight2)
                DO_array[j,i] = (DO_array[depth_value[m],i] * weight1) + (DO_array[next_value[m],i] * weight2)

    return Temp_array, DO_array




# Set lake depths (num rows) - 0.1 meters here - Failing = 15m & Crooked = 30m
fail_depth = 150
crook_depth = 300




# ============ Define Input directory - If reading in multiple Lakes and Years =============== #

#input_dir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/crook_test/Pycharm_test_resample'

#all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and len(f.split('_'))==4]

#allfiles_dict = pyproj_yrs_flatdict(all_files, input_dir)

# Separate into period-specific dictionaries
#period1_dict, period2_dict, period3_dict = separate_by_period(allfiles_dict)
#print('p1 keys:', period1_dict.keys())


# Separate each flattened period#_dict into hourly and daily dictionaries
# Period 1
#period1_hourly, period1_daily = separate_by_timescale(period1_dict)
#print('p1 hourly keys:', period1_hourly.keys())

# ============ Define Input directory - If reading in single Lake x Year =============== #
#input_dirday = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/Failing_2yr/Raw_cat_files/Files_for_analysis/Split_periods/Periods/Period_2/Daily'
#input_dirhour = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/Failing_2yr/Raw_cat_files/Files_for_analysis/Split_periods/Periods/Period_2/Hourly'

failday_files = [f for f in os.listdir(input_dirday) if f.endswith('.csv') and len(f.split('_'))==3]
failhour_files = [f for f in os.listdir(input_dirhour) if f.endswith('.csv') and len(f.split('_'))==3]

faildays = presorted_resam_read(failday_files, input_dirday)
failhrs = presorted_resam_read(failhour_files, input_dirhour)

    
# Hourly
t_hourly_dict, do_hourly_dict = sep_T_DO_cols(failhrs, 'Temp_C', 'DO_mg_L')

# Daily
t_daily_dict, do_daily_dict = sep_T_DO_cols(faildays, 'Temp_C', 'DO_mg_L')

#================ Checks (shape) =======================#
for key, df in t_daily_dict.items():
    print(f"Key: {key}")
    print(df.head())
    print(df.shape)
#================ End Checks =======================#

#%%

#================ Interpolating =======================#
"""
# Define parameters - 2021 (p1) Crook (Crooked lake)
    p1_start_date = "2021-06-24"
    p1_end_date = "2021-11-11"
# Parameters for pre-turnover 
    p1_crookturn_start = "2021-09-20"
    p1_crookturn_end = "2021-10-20"
"""
# Define your start and end dates  - 'Failing (fail) Lake 2022'
p1_start_date = "2022-4-12"
p1_end_date = "2023-06-22"

# create a list of your array dictionaries 
p1_dict_list = [t_hourly_dict, do_hourly_dict, t_daily_dict, do_daily_dict]

# get date slices of arrays
p1_sliced_dicts = [slice_dates(d, p1_start_date, p1_end_date) for d in p1_dict_list]

# unpack each variable (Temp, DO) and timescale (hr, day)
t_hour_dict, do_hour_dict,t_day_dict, do_day_dict = p1_sliced_dicts


#================ Checks (shape) =======================#
#for key, df in t_hour_dict.items():
#    print(f"Key: {key}")
#   print(df.head())
#   print(df.shape)

#for key, df in t_day_dict.items():
#    print(f"Key: {key}")
#    print(df.head())
#    print(df.shape)
#================ End Checks =======================#
   


# Convert columns to rows for input of individual files into a dataframe  (daily)   
reshape_tdays = convert_dict_dfs_to_arrays(t_day_dict, reshape="row")
reshape_dodays = convert_dict_dfs_to_arrays(do_day_dict, reshape="row")

# Convert columns to rows for input of individual files into a dataframe  (hourly)
reshape_thrs = convert_dict_dfs_to_arrays(t_hour_dict, reshape="row")
reshape_dohrs = convert_dict_dfs_to_arrays(do_hour_dict, reshape="row")





# Create 2D array with correct dimensions 
    # num rows = depth & num cols = time steps -  (failing lake here)

t_day_arr, do_day_arr = setup_lakearr(t_day_dict, fail_depth)
t_hours_arr, do_hours_arr = setup_lakearr(t_hour_dict, fail_depth)

tdays = populate_arrs(reshape_tdays, t_day_arr)
dodays = populate_arrs(reshape_dodays, do_day_arr)

thours = populate_arrs(reshape_thrs, t_hours_arr)
dohours = populate_arrs(reshape_dohrs, do_hours_arr)

#def checkpop()
#nonzero_rows = np.where(tdays.any(axis=1))[0]

# Depth (row) indices of sensor locations
fail_indices = [20, 37, 53, 69, 76, 91, 130]
crook_indices = [23, 46, 62, 76, 89, 107, 122, 137, 183, 213, 244, 274]

 # generate interpolation indices and order
interp_row1, interp_row2, interp_points, interp_dist, depth_value, next_value = interp_locs(fail_indices)      

# fill depths above and below
tdaysfill, dodaysfill = rep_abovebelow0(tdays, dodays, fail_indices) # Hourly data - replicate shallow to surface and deepest to bottom 
thoursfill, dohoursfill = rep_abovebelow0(thours, dohours, fail_indices) # Daily data - replicate shallow to surface and deepest to bottom 

# interpolate 
Temp_arrayday, DO_arraydayold = interp_ugly(tdaysfill, dodaysfill) # Daily data - interpolate all Failing indices
Temp_arrayhr, DO_arrayhrold = interp_ugly(thoursfill, dohoursfill) # Hourly data - interpolate all Failing indices


# ============ Optional - LEAVE OUT Sensor at 7.6m (Row # = 76)  ======================#


fail_indices_2022DO = [20, 37, 53, 69, 91, 130]

interp_row1, interp_row2, interp_points, interp_dist, depth_value, next_value = interp_locs(fail_indices_2022DO)


def interp_ugly_failno76DO(DO_array, Temp_array):
    for m in range(len(interp_row1)):
        j_start = interp_row1[m]
        j_end = interp_row2[m]
        for j in range(j_start, j_end + 1):
            weight1 = abs(j - j_end) / interp_dist[m]
            weight2 = abs(j - j_start) / interp_dist[m]
            for i in range(len(DO_array[1])):
                Temp_array[j,i] = (Temp_array[depth_value[m],i] * weight1) + (Temp_array[next_value[m],i] * weight2)
                DO_array[j,i] = (DO_array[depth_value[m],i] * weight1) + (DO_array[next_value[m],i] * weight2)

    return DO_array, Temp_array


#DO_arrayday, Temp_daynew = interp_ugly_failno76DO(dodaysfill, tdaysfill)  # leave out 7.6m sensor (daily data)
#DO_arrayhr, Temp_hrnew = interp_ugly_failno76DO(dohoursfill, thoursfill) # leave out 7.6m sensor (hourly data)

#DOdayflp=np.flipud(DO_arrayday) # flip the array for visualizaion (lake surface to bottom)
#DOhrflp=np.flipud(DO_arrayhr)   # flip the array for visualizaion (lake surface to bottom)
# ============ LEAVE OUT ======================#



#%%
# ============ Plotting ======================#
"""
Visual Check here
                  Visual Check here
                  Visual Check here
Visual Check here
"""



# Use respective time columns from any singular sensor used ('EST_DateTime') to index the x axis for plotting
day_indx = t_day_dict[ 'fail_5.3_daily'].index     
hr_indx = t_hour_dict[ 'fail_5.3_hourly'].index

def flip_and_plot(arrays, time_index, plot=True, figsize=(12, 8), time_format ="%Y-%m-%d"):
    """
    Takes input of a list of 2D arrays, a time index of equal size for the x-axis, and plots a heatmap with a y-axis from lake surface to bottom
    incorporate the # np.flipud(arr)... # in the first line if necessary
    """
    flipped_arrays = [(arr) for arr in arrays]              # include to offset the invert axis command if array is reflected about y-axis --- np.flipud(arr)...
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

# Sensor at 7.6m left out - '{timescale}new'
#    Create lists of however many arrays based on need
leave76outdays = [DO_arrayday,Temp_daynew]
leave76outhours = [DO_arrayhr, Temp_hrnew]
#    Plot
flip_and_plot(leave76outdays, day_indx)
flip_and_plot(leave76outhours, hr_indx)

# All sensors - '{timescale}old'
#    Create lists of however many arrays based on need
old_days = [Temp_arrayday, DO_arraydayold]
old_hrs = [Temp_arrayhr, DO_arrayhrold]
#    Plot
flip_and_plot(old_days, day_indx)
flip_and_plot(old_hrs, hr_indx)



#%%
# ======== Save selected outputs ===========# 
"""
 Save selected outputs
 Save selected outputs 
"""



# Dictionary of filenames (key) and arrays (value) to save as .csv files
fail_76_out_tdo = {'fail_day_Temp_array_2022': DO_arrayday, 'fail_day_DO_array_2022': Temp_daynew,
                 'fail_hr_Temp_array_2022': DO_arrayhr, 'fail_hr_DO_array_2022': Temp_hrnew }

# Save directory
outdir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/Failing_2yr/fail_Comp_Arrays_2022'
# Save to desired path
save_arrays_to_csv(fail_76_out_tdo, outdir)




#%%
"""
This cell is for checking data, averaging out days of removal in the field 
"""

# Load the array back
test_fail2022 = load_csv_files(outdir)
print("Loaded array:", test_fail2022["fail_hr_DO_array_2022"])


fail_old_test = csv_toarray('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/Failing_P1_noheader/Fail_P1_daily_noheader/Fail_P1_daily/DO_array.csv')

outdir2021 = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/Failing_P1_noheader/Fail_P1_daily_noheader/Fail_P1_daily'

test_fail2021 = load_csv_files(outdir2021)

outdir2021hr = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/Failing_P1_noheader/Failing_hrly_P1_noheader/FailP1_Raw_arrays'

test_fail2021hr = load_csv_files(outdir2021hr)



"checking visually"

from datetime import datetime
from datetime import date, timedelta
#day_indx = test_fail2021[ 'DO_array'].index
#hr_indx = test_fail2021hr[ 'DO_array'].index

start_date = datetime(2021,6,3)

num_days = test_fail2021['DO_array'].shape[1]
num_hrs = test_fail2021hr['DO_array'].shape[1]

days = np.array([start_date + timedelta(days=i) for i in range(num_days)])
hrs = np.array([start_date + timedelta(hours=i) for i in range(num_hrs)])

day_df = pd.to_datetime(days, format = "%Y-%m-%d")
hr_df = pd.to_datetime(hrs, format = "%Y-%m-%d")


#end_date = start_date + 

def flip_and_plot(arrays, time_index, plot=True, figsize=(12, 8), time_format ="%Y-%m-%d"):
    flipped_arrays = [np.flipud(arr) for arr in arrays]              #np.flipud(arr)...
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


list_daily = []
for k in test_fail2021:
    list_daily.append(test_fail2021[k])
    
list_hourly = []
for k in test_fail2021hr:
    list_hourly.append(test_fail2021hr[k])

flip_and_plot(list_daily,day_df)

flip_and_plot(list_hourly,hr_df)

def replace_groups_with_avg(array, groups):
    rows, cols = array.shape

    for group in groups:
        start, end = group

        # Ensure the group is within the bounds of the array
        if start > 0 and end < cols - 1:
            # Calculate the average of the column before and after the group
            avg_values = (array[:, start - 1] + array[:, end + 1]) / 2

            # Replace the values in the group of columns with the average values
            array[:, start:end +
                  1] = np.tile(avg_values[:, None], end - start + 1)
        else:
            print(
                f"Group {group} cannot be replaced because it doesn't have both left and right neighbors.")

    return array


def replace_nans_with_adjacent_avg(array, x):
    rows, cols = array.shape

    def get_nearest_values(r, c, x):
        values = []

        # Collect values from the left
        for i in range(1, x+1):
            if c - i >= 0 and not np.isnan(array[r, c - i]):
                values.append(array[r, c - i])

        # Collect values from the right
        for i in range(1, x+1):
            if c + i < cols and not np.isnan(array[r, c + i]):
                values.append(array[r, c + i])

        return values

    for r in range(rows):
        for c in range(cols):
            if np.isnan(array[r, c]):
                nearest_values = get_nearest_values(r, c, x)
                if nearest_values:
                    array[r, c] = np.mean(nearest_values)

    return array







# Groups of columns to be replaced (start and end inclusive)
groups_to_replace_day = [(35,36),(91,93)]         #possible fail day 2021      #[(850, 875), (1701,1703), (2190, 2230)] - Failing Hourly Arrays 2021
groups_to_replace_hr = [(850, 875), (1701,1703), (2190, 2230)] #- Failing Hourly Arrays 2021

# Replace specified groups of columns with the average of the column before and the column after the group
result_day = {k:(replace_groups_with_avg(arr, groups_to_replace_day)) for k,arr in test_fail2021.items()}
result_daylist = [result_day[k] for k in result_day]

# Replace specified groups of columns with the average of the column before and the column after the group
result_hr = {k:(replace_groups_with_avg(arr, groups_to_replace_hr)) for k,arr in test_fail2021hr.items()}
result_hr_list = [result_hr[k] for k in result_hr]

flip_and_plot(result_daylist,day_df)

flip_and_plot(result_hr_list,hr_df)

####
# -------- NAN REPLACE
####

# Replace NaNs with the average of the nearest 2 values from adjacent columns
result_hrnonan = {k:(replace_nans_with_adjacent_avg(arr, 2)) for k,arr in result_hr.items()}
result_hr_listnonan = [result_hrnonan[k] for k in result_hrnonan]

flip_and_plot(result_hr_listnonan,hr_df)

"saving fail 2021 and checking shape"
fail_2021dir = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Data/2yr_datasets/Failing_2yr/fail_Comp_Arrays_2021'
save_arrays_to_csv(result_day, fail_2021dir)





# Display the original array
print("Original array:")
print(array)

# Display the modified array
print("Array after replacing NaNs:")
print(result)


#%%



