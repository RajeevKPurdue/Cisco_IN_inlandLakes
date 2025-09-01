#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:16:33 2025

@author: rajeevkumar
"""


"""
Corrected line plot framework for comparing daily vs hourly metrics
Properly structured with all imports and functions organized correctly

Rajeev Kumar, 08/06/2025 (Line Plot Extension - Corrected Structure)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def replace_at_timestamps_with_window_avg(df, timestamps, num_before, num_after, target_columns=None):
    """
    Replace values at user-specified timestamp(s) with an average computed 
    over a window of data from a number of time steps before and after 
    the target rows, for specified columns or all columns containing '_hr_'.
    
    Parameters:
      df (pd.DataFrame): The input DataFrame containing a column 'EST_DateTime'
                         with datetime values.
      timestamps (str, datetime, or list-like): A single timestamp or a list of timestamps.
      num_before (int): The number of rows (time steps) before the target row.
      num_after (int): The number of rows (time steps) after the target row.
      target_columns (list, optional): Specific columns to process. If None, 
                                      processes all columns containing '_hr_'.
      
    Returns:
      pd.DataFrame: The modified DataFrame with the updated values.
    """
    
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Make sure the EST_DateTime column is parsed as timezone-aware datetime.
    df['EST_DateTime'] = pd.to_datetime(df['EST_DateTime'], utc=True)
    
    # Ensure timestamps is a list; if not, encapsulate it in a list.
    if not isinstance(timestamps, (list, tuple, pd.Series)):
        timestamps = [timestamps]
    
    # Convert each timestamp to a timezone-aware datetime.
    timestamps = [pd.to_datetime(ts, utc=True) for ts in timestamps]
    
    # Determine which columns to process
    if target_columns is None:
        # Default: process columns that include '_hr_' in their names
        target_columns = [col for col in df.columns if '_hr_' in col]
    else:
        # Use specified columns, but only if they exist in the DataFrame
        target_columns = [col for col in target_columns if col in df.columns]
    
    print(f"Processing {len(target_columns)} target columns for timestamp replacement")
    if target_columns:
        print(f"Target columns: {target_columns[:3]}..." if len(target_columns) > 3 else f"Target columns: {target_columns}")
    
    # Process each provided timestamp.
    for ts in timestamps:
        # Locate matching rows; adjust this if you want to use approximate matching.
        matching_rows = df.index[df['EST_DateTime'] == ts]
        
        if matching_rows.empty:
            print(f"Warning: The timestamp {ts} was not found in the DataFrame. Skipping.")
            continue
        
        print(f"Processing timestamp {ts}: found {len(matching_rows)} matching rows")
        
        for row_idx in matching_rows:
            # Process only the specified target columns
            for col in target_columns:
                # Determine window boundaries, ensuring we don't go out-of-bounds.
                start_idx = max(0, row_idx - num_before)
                end_idx = min(len(df), row_idx + num_after + 1)  # +1 because the end index is exclusive
                
                # Obtain values before and after the target row, excluding the target row.
                before_window = df.iloc[start_idx:row_idx][col]
                after_window = df.iloc[row_idx+1:end_idx][col]
                
                # Combine both windows.
                window_vals = pd.concat([before_window, after_window])
                
                if not window_vals.empty:
                    # Compute the average and replace the target value.
                    original_value = df.at[row_idx, col]
                    avg_value = window_vals.mean()
                    df.at[row_idx, col] = avg_value
                    print(f"  Column '{col}': {original_value:.4f} → {avg_value:.4f}")
                else:
                    print(f"Warning: No available data around row {row_idx} in column '{col}'.")
    
    return df

def apply_timestamp_corrections(df, lake, year, start_date=None, end_date= None):
    """
    Apply timestamp-specific corrections based on lake and year combinations.
    
    Parameters:
    df (pd.DataFrame): DataFrame with time series data
    lake (str): Lake name ('Crooked' or 'Failing')
    year (str): Year ('2021' or '2022')
    
    Returns:
    pd.DataFrame: DataFrame with corrected values
    """
    
    # Define timestamps that need correction for each lake-year combination
    correction_timestamps = {
        ('Crooked', '2021'): [
            '2021-07-26 10:00:00-04:00', 
            '2021-08-13 15:00:00-04:00', 
            '2021-09-01 15:00:00-04:00', 
            '2021-10-08 12:00:00-04:00'
        ],
        ('Failing', '2022'): [
            '2022-07-21 12:00:00-04:00', 
            '2022-09-16 12:00:00-04:00'
        ]
        # Add more combinations as needed:
        # ('Failing', '2021'): [...],
        # ('Crooked', '2022'): [...],
    }
    
    timestamps = correction_timestamps.get((lake, year), [])
    
    if timestamps:
        print(f"\nApplying timestamp corrections for {lake} {year}")
        print(f"Correcting {len(timestamps)} timestamps")
        
        # Apply corrections with window averaging
        df = replace_at_timestamps_with_window_avg(
            df, 
            timestamps, 
            num_before=1, 
            num_after=1,
            target_columns=None  # Will default to columns with '_hr_'
        )
    else:
        print(f"\nNo timestamp corrections defined for {lake} {year}")
    
    return df
    """
    Filter dataframe by date range using the EST_DateTime column
    Handles timezone-aware EST_DateTime properly
    """
    df = df.copy()
    
    # Reset index to make EST_DateTime a regular column if it's the index
    if 'EST_DateTime' not in df.columns:
        df = df.reset_index()
    
    # Convert EST_DateTime to datetime if it's not already
    df['EST_DateTime'] = pd.to_datetime(df['EST_DateTime'], utc=True)
    
    # Convert string dates to datetime objects
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)
    
    print(f"    Target date range: {start_dt.date()} to {end_dt.date()}")
    
    try:
        # Show actual date range in data
        index_min = df['EST_DateTime'].min()
        index_max = df['EST_DateTime'].max()
        print(f"    Data date range: {index_min.date()} to {index_max.date()}")
        
        # Apply the filter
        original_count = len(df)
        df_filtered = df[(df['EST_DateTime'] >= start_dt) & (df['EST_DateTime'] <= end_dt)].copy()
        filtered_count = len(df_filtered)
        
        print(f"    Filtered: {original_count} → {filtered_count} rows ({original_count - filtered_count} removed)")
        
        if filtered_count == 0:
            print(f"    WARNING: No data in specified date range!")
            print(f"    Check if your date range overlaps with data range")
        
        # Sort by datetime for clarity
        df_filtered = df_filtered.sort_values('EST_DateTime')
        
        return df_filtered
        
    except Exception as e:
        print(f"    ERROR in date filtering: {e}")
        import traceback
        traceback.print_exc()
        print("    RETURNING UNFILTERED DATA")
        return df

def load_and_process_data_for_timeseries(data_directory=None, date_ranges=None):
    """
    Load CSV files and prepare data for time series plotting (preserving datetime info)
    
    Parameters:
    data_directory (str): Path to directory containing both GRP and TDO6 CSV files
    date_ranges (dict): Dictionary specifying date ranges for each lake-year combination
    """
    
    # Set default directory
    if data_directory is None:
        data_directory = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Redo_pos_summaries_dfs_GRP/Proportions/partial_plot_all_dfs'
    
    print(f"Data directory: {data_directory}")
    
    if date_ranges:
        print("Date ranges specified:")
        for (lake, year), (start, end) in date_ranges.items():
            print(f"  {lake} {year}: {start} to {end}")
    
    all_data = []
    original_dir = os.getcwd()
    
    if not os.path.exists(data_directory):
        print(f"ERROR: Data directory not found: {data_directory}")
        return pd.DataFrame()
    
    os.chdir(data_directory)
    print(f"\nProcessing files in {data_directory}...")
    
    # Get all relevant CSV files (both GRP and TDO6)
    all_files = []
    for pattern in ['*crook*2021*.csv', '*crook*2022*.csv', '*fail*2021*.csv', '*fail*2022*.csv', 
                   'lyons_crook_2021.csv', 'lyons_crook_2022.csv', 'lyons_fail_2021.csv', 'lyons_fail_2022.csv']:
        all_files.extend(glob.glob(pattern))
    
    print(f"Found {len(all_files)} files to process")
    
    for file_path in all_files:
        print(f"\n  Processing: {os.path.basename(file_path)}")
        
        # Determine file type and extract lake/year
        filename = os.path.basename(file_path).lower()
        
        # Parse lake and year consistently
        if 'crook' in filename:
            lake = 'Crooked'
        elif 'fail' in filename:
            lake = 'Failing'  
        else:
            print("    Skipping - could not determine lake")
            continue
            
        if '2021' in filename:
            year = '2021'
        elif '2022' in filename:
            year = '2022'
        else:
            print("    Skipping - could not determine year")
            continue
        
        # Determine file type
        is_tdo6 = 'lyons' in filename
        file_type = 'TDO6' if is_tdo6 else 'GRP'
        
        print(f"    File type: {file_type}, Lake: {lake}, Year: {year}")
        
        try:
            # Read CSV - try with EST_DateTime as index first, then as column
            try:
                df = pd.read_csv(file_path, index_col='EST_DateTime')
                # Reset index to make EST_DateTime a column
                df = df.reset_index()
            except (KeyError, ValueError):
                # EST_DateTime might already be a column
                df = pd.read_csv(file_path)
            
            print(f"    Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            
            if 'EST_DateTime' in df.columns:
                print(f"    DateTime range: {df['EST_DateTime'].iloc[0]} to {df['EST_DateTime'].iloc[-1]}")
            else:
                print("    WARNING: No EST_DateTime column found!")
                available_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                print(f"    Available time columns: {available_cols}")
                continue
            
            # Apply timestamp corrections before date filtering if specified
            if date_ranges and (lake, year) in date_ranges:
                # Apply corrections first, then filter by date
                df = apply_timestamp_corrections(df, lake, year)
                
                start_date, end_date = date_ranges[(lake, year)]
                df = filter_by_date_range(df, start_date, end_date)
                print(f"    After corrections and filtering: {df.shape[0]} rows")
                
                if len(df) == 0:
                    print("    Skipping - no data after filtering")
                    continue
            else:
                # Just apply corrections without date filtering
                df = apply_timestamp_corrections(df, lake, year)
                print(f"    After corrections (no date filtering): {df.shape[0]} rows")
        
        except Exception as e:
            print(f"    ERROR loading file: {e}")
            continue
        
        # Process based on file type
        if is_tdo6:
            # TDO6 file processing
            daily_cols = [col for col in df.columns if 'daily' in col.lower() and 'lyons' in col.lower()]
            hourly_cols = [col for col in df.columns if 'hourly' in col.lower() and 'lyons' in col.lower()]
            
            print(f"    TDO6 columns - Daily: {len(daily_cols)}, Hourly: {len(hourly_cols)}")
            
            # Process daily and hourly columns - preserve datetime
            for col_list, timescale in [(daily_cols, 'Daily'), (hourly_cols, 'Hourly')]:
                for column in col_list:
                    if column in df.columns:
                        subset = df[['EST_DateTime', column]].dropna()
                        print(f"    Processing TDO6 {timescale}: {len(subset)} values from {column}")
                        for _, row in subset.iterrows():
                            all_data.append({
                                'Lake': lake,
                                'Year': year,
                                'Metric': 'TDO6',
                                'Timescale': timescale,
                                'EST_DateTime': row['EST_DateTime'],
                                'Value': row[column]
                            })
        else:
            # GRP file processing
            print("    Searching for target columns with your specific parameters...")
            
            # Your original target column criteria
            target_columns = [col for col in df.columns if all(param in col for param in 
                            ['mass400', 'p0.4', 'slope0.168', 'intercept1.63']) and
                            ('grp_p0.4' in col or 'grp_lethal_p0.4' in col)]
            
            print(f"    Target columns found: {len(target_columns)}")
            if target_columns:
                print("    Target columns:")
                for col in target_columns:
                    print(f"      {col}")
            
            # Process each target column - preserve datetime
            for col in target_columns:
                # Determine metric type
                if 'grp_lethal' in col:
                    metric = 'GRPE'
                elif 'grp_p0.4' in col:
                    metric = 'GRP'
                else:
                    metric = None
                    
                # Determine timescale  
                if '_hr_' in col:
                    timescale = 'Hourly'
                elif '_day_' in col:
                    timescale = 'Daily'
                else:
                    timescale = None
                
                if metric and timescale:
                    subset = df[['EST_DateTime', col]].dropna()
                    print(f"    Processing {metric} {timescale}: {len(subset)} values from {col}")
                    for _, row in subset.iterrows():
                        all_data.append({
                            'Lake': lake,
                            'Year': year,
                            'Metric': metric,
                            'Timescale': timescale,
                            'EST_DateTime': row['EST_DateTime'],
                            'Value': row[col]
                        })
    
    # Return to original directory
    os.chdir(original_dir)
    
    df_result = pd.DataFrame(all_data)
    print(f"\nFINAL RESULTS:")
    print(f"Total data points loaded: {len(df_result)}")
    if len(df_result) > 0:
        # Convert EST_DateTime to proper datetime
        df_result['EST_DateTime'] = pd.to_datetime(df_result['EST_DateTime'])
        
        print(f"Metrics: {df_result['Metric'].unique()}")
        print(f"Lakes: {df_result['Lake'].unique()}")  
        print(f"Years: {df_result['Year'].unique()}")
        print(f"Timescales: {df_result['Timescale'].unique()}")
        
        # Show breakdown by metric
        print("\nData point breakdown:")
        for metric in df_result['Metric'].unique():
            count = len(df_result[df_result['Metric'] == metric])
            print(f"  {metric}: {count} points")
    else:
        print("NO DATA LOADED - Check if your target columns exist with specified parameters")
    
    return df_result

def create_daily_vs_hourly_line_plots(df):
    """
    Create line plots comparing daily vs hourly for each metric
    Separate subplots for each lake-year-metric combination
    Hourly (blue) plotted first, then daily (orange)
    """
    
    if len(df) == 0:
        print("No data available for line plotting")
        return
    
    # Get unique combinations of lake, year, and metric
    combinations = df.groupby(['Lake', 'Year', 'Metric']).size().reset_index()[['Lake', 'Year', 'Metric']]
    
    print(f"Found {len(combinations)} lake-year-metric combinations")
    
    # Determine subplot layout
    n_combos = len(combinations)
    if n_combos == 0:
        print("No valid combinations found")
        return
    
    # Calculate subplot grid (try to make it roughly square)
    n_cols = min(3, n_combos)  # Max 3 columns
    n_rows = (n_combos + n_cols - 1) // n_cols  # Ceiling division
    
    # Increased figure size for better interpretability
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    
    # Handle single subplot case
    if n_combos == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Color scheme: hourly=blue, daily=orange
    colors = {'Hourly': '#1f77b4', 'Daily': '#ff7f0e'}
    
    # Track if we've added legend elements
    legend_added = False
    legend_handles = []
    legend_labels = []
    
    for i, (_, combo) in enumerate(combinations.iterrows()):
        lake, year, metric = combo['Lake'], combo['Year'], combo['Metric']
        ax = axes[i]
        
        # Filter data for this combination
        subset = df[(df['Lake'] == lake) & (df['Year'] == year) & (df['Metric'] == metric)].copy()
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, f'No data\n{lake} {year}\n{metric}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{lake} {year} - {metric}')
            continue
        
        # Sort by datetime
        subset = subset.sort_values('EST_DateTime')
        
        # Plot hourly first (blue), then daily (orange)
        for timescale in ['Hourly', 'Daily']:
            timescale_data = subset[subset['Timescale'] == timescale]
            
            if len(timescale_data) > 0:
                line = ax.plot(timescale_data['EST_DateTime'], timescale_data['Value'], 
                       color=colors[timescale], label=timescale, linewidth=2,
                       marker='o' if timescale == 'Daily' else '.', markersize=4)
                
                # Collect legend elements from first subplot only
                if not legend_added and len(timescale_data) > 0:
                    legend_handles.append(line[0])
                    legend_labels.append(timescale)
        
        legend_added = True
        
        # Format subplot
        ax.set_ylim(0, 1)
        
        # Remove individual y-axis labels (will add single one later)
        ax.set_ylabel('')
        ax.set_xlabel('')
        
        # Format title with subscripts
        if metric == 'GRPE':
            title_metric = 'GRP$_E$'
        elif metric == 'GRP':
            title_metric = 'GRP$_L$'
        else:
            title_metric = metric
            
        ax.set_title(f'{lake} {year} - {title_metric}', 
                    fontfamily='Times New Roman', fontweight='bold')
        
        # Format x-axis with biweekly ticks
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Remove gridlines
        ax.grid(False)
    
    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Add single legend outside the subplots
    if legend_handles:
        fig.legend(legend_handles, legend_labels, 
                  bbox_to_anchor=(0.98, 0.98), loc='upper right', 
                  prop={'family': 'Times New Roman', 'size': 12})
    
    # Add single y-axis label on the outer left side
    fig.text(0.04, 0.5, 'Proportion of Positive Cells', va='center', rotation='vertical', 
             fontsize=14, fontweight='bold', family='Times New Roman')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.95)  # Make room for y-label and legend
    plt.show()

def create_combined_metrics_line_plot(df):
    """
    Create a single figure with subplots for each lake-year combination
    Each subplot shows all metrics (GRP, GRPE, TDO6) with daily vs hourly
    """
    
    if len(df) == 0:
        print("No data available for combined line plotting")
        return
    
    # Get unique lake-year combinations
    lake_year_combos = df.groupby(['Lake', 'Year']).size().reset_index()[['Lake', 'Year']]
    
    print(f"Found {len(lake_year_combos)} lake-year combinations")
    
    if len(lake_year_combos) == 0:
        print("No valid lake-year combinations found")
        return
    
    # Determine subplot layout
    n_combos = len(lake_year_combos)
    n_cols = min(2, n_combos)  # Max 2 columns for readability
    n_rows = (n_combos + n_cols - 1) // n_cols
    
    # Increased figure size for better interpretability
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 8*n_rows))
    
    # Handle single subplot case
    if n_combos == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Color scheme for metrics and timescales
    metric_colors = {'GRP': '#2ca02c', 'GRPE': '#d62728', 'TDO6': '#9467bd'}
    linestyles = {'Hourly': '-', 'Daily': '--'}
    
    # Track legend elements
    legend_handles = []
    legend_labels = []
    legend_added = False
    
    for i, (_, combo) in enumerate(lake_year_combos.iterrows()):
        lake, year = combo['Lake'], combo['Year']
        ax = axes[i]
        
        # Filter data for this lake-year combination
        subset = df[(df['Lake'] == lake) & (df['Year'] == year)].copy()
        subset = subset.sort_values('EST_DateTime')
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, f'No data\n{lake} {year}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{lake} {year}')
            continue
        
        # Plot each metric-timescale combination
        for metric in ['GRP', 'GRPE', 'TDO6']:
            for timescale in ['Hourly', 'Daily']:
                metric_timescale_data = subset[
                    (subset['Metric'] == metric) & (subset['Timescale'] == timescale)
                ]
                
                if len(metric_timescale_data) > 0:
                    color = metric_colors.get(metric, '#000000')
                    linestyle = linestyles[timescale]
                    
                    # Format label with subscripts
                    if metric == 'GRPE':
                        label_metric = 'GRP$_E$'
                    elif metric == 'GRP':
                        label_metric = 'GRP$_L$'
                    else:
                        label_metric = metric
                    
                    label = f'{label_metric} {timescale}'
                    
                    line = ax.plot(metric_timescale_data['EST_DateTime'], 
                           metric_timescale_data['Value'],
                           color=color, linestyle=linestyle, label=label, 
                           linewidth=2, alpha=0.8)
                    
                    # Collect legend elements from first subplot only
                    if not legend_added and len(metric_timescale_data) > 0:
                        legend_handles.append(line[0])
                        legend_labels.append(label)
        
        legend_added = True
        
        # Format subplot
        ax.set_ylim(0, 1)
        
        # Remove individual axis labels (will add single ones later)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(f'{lake} {year}', fontfamily='Times New Roman', fontweight='bold')
        
        # Format x-axis with biweekly ticks
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Remove gridlines
        ax.grid(False)
    
    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Add single legend outside the subplots
    if legend_handles:
        fig.legend(legend_handles, legend_labels, 
                  bbox_to_anchor=(0.98, 0.98), loc='upper right', 
                  prop={'family': 'Times New Roman', 'size': 12})
    
    # Add single y-axis label on the outer left side
    fig.text(0.04, 0.5, 'Proportion of Positive Cells', va='center', rotation='vertical', 
             fontsize=14, fontweight='bold', family='Times New Roman')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.90)  # Make room for y-label and legend
    plt.show()

def main_line_plots(data_directory=None, date_ranges=None, outpath=None, apply_corrections=True):
    """
    Main function for creating line plots comparing daily vs hourly timescales
    
    Parameters:
    data_directory (str): Path to directory containing all data files
    date_ranges (dict): Dictionary specifying date ranges for each lake-year combination
    outpath (str): Path for saving output files
    apply_corrections (bool): Whether to apply timestamp corrections (default: True)
    """
    
    print("=== Line Plot Analysis - Daily vs Hourly Comparison ===")
    
    if apply_corrections:
        print("Timestamp corrections will be applied where defined")
    else:
        print("Timestamp corrections disabled")
    
    # Load data with datetime preservation
    df = load_and_process_data_for_timeseries(data_directory, date_ranges)
    
    if len(df) == 0:
        print("\nNo data loaded! Check:")
        print("1. Directory paths are correct")
        print("2. CSV files exist in specified directories") 
        print("3. Files contain expected column patterns")
        print("4. Date ranges (if specified) are valid")
        print("5. EST_DateTime column exists in files")
        return None
    
    print("\nData overview:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['EST_DateTime'].min()} to {df['EST_DateTime'].max()}")
    
    # Create individual metric plots (one subplot per lake-year-metric)
    print("\n=== Creating Individual Metric Line Plots ===")
    create_daily_vs_hourly_line_plots(df)
    
    # Create combined metrics plots (one subplot per lake-year, all metrics together)
    print("\n=== Creating Combined Metrics Line Plots ===")
    create_combined_metrics_line_plot(df)
    
    return df

# Run line plot analysis
if __name__ == "__main__":
    # Same date ranges and paths as before
    date_ranges = {
         ('Crooked', '2021'): ('2021-09-20', '2021-10-25'),
         ('Crooked', '2022'): ('2022-09-25', '2022-11-01'),
         ('Failing', '2021'): ('2021-09-20', '2021-11-05'),
         ('Failing', '2022'): ('2022-09-20', '2022-11-05')
     }
    
    data_directory = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Redo_pos_summaries_dfs_GRP/Proportions/partial_plot_all_dfs'
    
    outpath = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Aug25_lineplots_dailyVhourly'
    
    df = main_line_plots(data_directory=data_directory, date_ranges=date_ranges, outpath=outpath)




#%%

import pandas as pd
import os
import glob

def diagnose_date_columns(file_path):
    """
    Analyze date columns in a CSV file to understand the timezone and format issues
    """
    print(f"\n=== ANALYZING: {os.path.basename(file_path)} ===")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Look for potential date columns
        potential_date_cols = []
        for col in df.columns:
            if any(word in col.lower() for word in ['date', 'time', 'day', 'datetime', 'timestamp']):
                potential_date_cols.append(col)
        
        print(f"Potential date columns: {potential_date_cols}")
        
        # Analyze each potential date column
        for col in potential_date_cols:
            print(f"\n--- Analyzing column: {col} ---")
            print(f"Data type: {df[col].dtype}")
            print(f"Non-null count: {df[col].notna().sum()}")
            print(f"Sample values:")
            sample_values = df[col].dropna().head(5)
            for i, val in enumerate(sample_values):
                print(f"  [{i}]: {val} (type: {type(val)})")
            
            # Try to convert to datetime and check timezone info
            try:
                dt_series = pd.to_datetime(df[col].dropna())
                print(f"Datetime conversion successful")
                print(f"Timezone info: {dt_series.dt.tz}")
                print(f"Min date: {dt_series.min()}")
                print(f"Max date: {dt_series.max()}")
                
                # Show timezone-aware vs timezone-naive comparison
                if dt_series.dt.tz is not None:
                    print(f"This is TIMEZONE-AWARE data")
                    naive_comparison = pd.to_datetime('2021-09-20')
                    print(f"Comparison with naive timestamp will fail: {naive_comparison}")
                else:
                    print(f"This is TIMEZONE-NAIVE data")
                    
            except Exception as e:
                print(f"Datetime conversion failed: {e}")
        
        # If no obvious date columns, show all columns for manual inspection
        if not potential_date_cols:
            print(f"\nNo obvious date columns found. All columns:")
            for col in df.columns:
                print(f"  {col}: {df[col].dtype}")
                
    except Exception as e:
        print(f"Error reading file: {e}")

# Run diagnostics on your files
def run_full_diagnosis():
    """
    Run diagnosis on all your data files
    """
    
    # GRP files
    grp_directory = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Redo_pos_summaries_dfs_GRP/Proportions'
    tdo6_directory = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Lyons_summary_dfs/Props'
    
    print("="*60)
    print("DIAGNOSING GRP FILES")
    print("="*60)
    
    if os.path.exists(grp_directory):
        os.chdir(grp_directory)
        grp_files = []
        for pattern in ['*crook*2021*.csv', '*crook*2022*.csv', '*fail*2021*.csv', '*fail*2022*.csv']:
            grp_files.extend(glob.glob(pattern))
        
        for file_path in grp_files[:2]:  # Just check first 2 files
            diagnose_date_columns(file_path)
    
    print("\n" + "="*60)
    print("DIAGNOSING TDO6 FILES")
    print("="*60)
    
    if os.path.exists(tdo6_directory):
        os.chdir(tdo6_directory)
        tdo6_files = []
        for pattern in ['*crook*2021*.csv', '*crook*2022*.csv', '*fail*2021*.csv', '*fail*2022*.csv']:
            tdo6_files.extend(glob.glob(pattern))
            
        for file_path in tdo6_files[:2]:  # Just check first 2 files
            diagnose_date_columns(file_path)

if __name__ == "__main__":
    run_full_diagnosis()
    
    
    
#%%

import pandas as pd
import os
import glob

def diagnose_index_structure(file_path):
    """
    Analyze the index structure and column selection in your CSV files
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    try:
        # Read without any index specification first
        df_no_index = pd.read_csv(file_path)
        print(f"Without index specification:")
        print(f"  Shape: {df_no_index.shape}")
        print(f"  Columns: {df_no_index.columns.tolist()[:5]}...")  # First 5 columns
        print(f"  Index: {df_no_index.index}")
        
        # Check if first column looks like datetime
        first_col = df_no_index.columns[0]
        print(f"\nFirst column analysis: {first_col}")
        print(f"  Sample values: {df_no_index[first_col].head(3).tolist()}")
        print(f"  Data type: {df_no_index[first_col].dtype}")
        
        # Try reading with first column as index
        df_with_index = pd.read_csv(file_path, index_col=0)
        print(f"\nWith first column as index:")
        print(f"  Shape: {df_with_index.shape}")
        print(f"  Index name: {df_with_index.index.name}")
        print(f"  Index type: {type(df_with_index.index)}")
        print(f"  Sample index values: {df_with_index.index[:3].tolist()}")
        
        # Try to convert index to datetime
        try:
            datetime_index = pd.to_datetime(df_with_index.index)
            print(f"  Index datetime conversion: SUCCESS")
            print(f"  Index timezone: {datetime_index.tz}")
            print(f"  Index date range: {datetime_index.min()} to {datetime_index.max()}")
        except Exception as e:
            print(f"  Index datetime conversion: FAILED - {e}")
        
        # Analyze target columns for GRP files
        if 'grp' in file_path.lower():
            print(f"\nGRP TARGET COLUMN ANALYSIS:")
            target_columns = [col for col in df_with_index.columns if all(param in col for param in 
                            ['mass400', 'p0.4', 'slope0.168', 'intercept1.63']) and
                            ('grp_p0.4' in col or 'grp_lethal_p0.4' in col)]
            
            print(f"  Total columns: {len(df_with_index.columns)}")
            print(f"  Target columns found: {len(target_columns)}")
            print(f"  Target columns: {target_columns}")
            
            # Show some non-target columns for comparison
            non_target = [col for col in df_with_index.columns if col not in target_columns][:5]
            print(f"  Sample non-target columns: {non_target}")
            
        # Analyze TDO6 columns
        elif 'lyons' in file_path.lower():
            print(f"\nTDO6 TARGET COLUMN ANALYSIS:")
            daily_cols = [col for col in df_with_index.columns if 'daily' in col.lower() and 'lyons' in col.lower()]
            hourly_cols = [col for col in df_with_index.columns if 'hourly' in col.lower() and 'lyons' in col.lower()]
            
            print(f"  Total columns: {len(df_with_index.columns)}")
            print(f"  Daily columns: {daily_cols}")
            print(f"  Hourly columns: {hourly_cols}")
            
    except Exception as e:
        print(f"ERROR reading file: {e}")

def run_index_diagnosis():
    """Run diagnosis focusing on index structure"""
    
    grp_directory = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Redo_pos_summaries_dfs_GRP/Proportions'
    tdo6_directory = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Lyons_summary_dfs/Props'
    
    print("DIAGNOSING INDEX STRUCTURE AND COLUMN SELECTION")
    
    # Check one GRP file
    if os.path.exists(grp_directory):
        os.chdir(grp_directory)
        grp_files = glob.glob('*crook*2021*.csv')
        if grp_files:
            diagnose_index_structure(grp_files[0])
    
    # Check one TDO6 file
    if os.path.exists(tdo6_directory):
        os.chdir(tdo6_directory)
        tdo6_files = glob.glob('*crook*2021*.csv')
        if tdo6_files:
            diagnose_index_structure(tdo6_files[0])

if __name__ == "__main__":
    run_index_diagnosis()