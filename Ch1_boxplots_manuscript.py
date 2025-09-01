"""
Corrected boxplotting code with proper date filtering, time ranges, and single directory handling
Fixed major issues with date filtering logic and CSV reading

Rajeev Kumar, 08/06/2025 (Corrected)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from datetime import datetime, timedelta

def filter_by_date_range(df, start_date, end_date):
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

def load_and_process_data(data_directory=None, date_ranges=None):
    """
    Load CSV files from single directory with consistent Eastern Time handling
    
    Parameters:
    data_directory (str): Path to directory containing both GRP and TDO6 CSV files
    date_ranges (dict): Dictionary specifying date ranges for each lake-year combination
    """
    
    # Set default directory - UPDATE THIS PATH
    if data_directory is None:
        data_directory = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Redo_pos_summaries_dfs_GRP/Proportions/partial_plot_all_dfs'  # UPDATE THIS
    
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
            
            # Apply date filtering if specified
            if date_ranges and (lake, year) in date_ranges:
                start_date, end_date = date_ranges[(lake, year)]
                df = filter_by_date_range(df, start_date, end_date)
                print(f"    After filtering: {df.shape[0]} rows")
                
                if len(df) == 0:
                    print("    Skipping - no data after filtering")
                    continue
        
        except Exception as e:
            print(f"    ERROR loading file: {e}")
            continue
        
        # Process based on file type
        if is_tdo6:
            # TDO6 file processing
            daily_cols = [col for col in df.columns if 'daily' in col.lower() and 'lyons' in col.lower()]
            hourly_cols = [col for col in df.columns if 'hourly' in col.lower() and 'lyons' in col.lower()]
            
            print(f"    TDO6 columns - Daily: {len(daily_cols)}, Hourly: {len(hourly_cols)}")
            
            # Process daily and hourly columns
            for col_list, timescale in [(daily_cols, 'Daily'), (hourly_cols, 'Hourly')]:
                for column in col_list:
                    if column in df.columns:
                        values = df[column].dropna().values
                        print(f"    Processing TDO6 {timescale}: {len(values)} values from {column}")
                        for value in values:
                            all_data.append({
                                'Lake': lake,
                                'Year': year,
                                'Metric': 'TDO6',
                                'Timescale': timescale,
                                'Value': value
                            })
        else:
            # GRP file processing
            print("    Searching for target columns with your specific parameters...")
            
            # Debug each parameter separately
            mass400_cols = [col for col in df.columns if 'mass400' in col]
            p04_cols = [col for col in df.columns if 'p0.4' in col]
            slope_cols = [col for col in df.columns if 'slope0.168' in col]
            intercept_cols = [col for col in df.columns if 'intercept1.63' in col]
            grp_p04_cols = [col for col in df.columns if 'grp_p0.4' in col]
            grp_lethal_cols = [col for col in df.columns if 'grp_lethal_p0.4' in col]
            
            print(f"    Columns with 'mass400': {len(mass400_cols)}")
            print(f"    Columns with 'p0.4': {len(p04_cols)}")  
            print(f"    Columns with 'slope0.168': {len(slope_cols)}")
            print(f"    Columns with 'intercept1.63': {len(intercept_cols)}")
            print(f"    Columns with 'grp_p0.4': {len(grp_p04_cols)}")
            print(f"    Columns with 'grp_lethal_p0.4': {len(grp_lethal_cols)}")
            
            # Your original target column criteria
            target_columns = [col for col in df.columns if all(param in col for param in 
                            ['mass400', 'p0.4', 'slope0.168', 'intercept1.63']) and
                            ('grp_p0.4' in col or 'grp_lethal_p0.4' in col)]
            
            print(f"    Target columns found: {len(target_columns)}")
            if target_columns:
                print("    Target columns:")
                for col in target_columns:
                    print(f"      {col}")
            else:
                print("    NO TARGET COLUMNS FOUND with your criteria!")
                print("    Sample column names:")
                for col in df.columns[:5]:  # Show first 5 columns
                    print(f"      {col}")
            
            # Process each target column
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
                    values = df[col].dropna().values
                    print(f"    Processing {metric} {timescale}: {len(values)} values from {col}")
                    for value in values:
                        all_data.append({
                            'Lake': lake,
                            'Year': year,
                            'Metric': metric,
                            'Timescale': timescale,
                            'Value': value
                        })
                else:
                    print(f"    Skipping {col}: could not determine metric ({metric}) or timescale ({timescale})")
    
    # Return to original directory
    os.chdir(original_dir)
    
    df_result = pd.DataFrame(all_data)
    print(f"\nFINAL RESULTS:")
    print(f"Total data points loaded: {len(df_result)}")
    if len(df_result) > 0:
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

def create_main_box_plots(df):
    """
    Create main box plots: metrics (rows) x years (columns)
    Format: No titles, full lake names, GRPE with subscript, parallel labels
    """
    
    metric_order = ['GRP', 'TDO6', 'GRPE']
    available_metrics = [m for m in metric_order if m in df['Metric'].unique()]
    years = sorted(df['Year'].unique())
    
    if len(available_metrics) == 0 or len(years) == 0:
        print("No data available for plotting")
        return
    
    # Create figure
    n_metrics = len(available_metrics)
    n_years = len(years)
    fig, axes = plt.subplots(n_metrics, n_years, figsize=(6*n_years, 4*n_metrics))
    
    # Handle single row/column cases
    if n_metrics == 1 and n_years == 1:
        axes = np.array([[axes]])
    elif n_metrics == 1:
        axes = axes.reshape(1, -1)
    elif n_years == 1:
        axes = axes.reshape(-1, 1)
    
    # Create plots
    for i, metric in enumerate(available_metrics):
        for j, year in enumerate(years):
            ax = axes[i, j]
            
            # Filter data
            subset = df[(df['Metric'] == metric) & (df['Year'] == year)].copy()
            
            if len(subset) > 0:
                # Combine Lake and Timescale for x-axis
                subset['Treatment_Timescale'] = subset['Lake'] + '\n' + subset['Timescale']
                
                # Create box plot
                sns.boxplot(data=subset, x='Treatment_Timescale', y='Value', ax=ax)
                
                # Add year label on top row only
                if i == 0:
                    ax.text(0.5, 1.02, f'{year}', transform=ax.transAxes, 
                           ha='center', va='bottom', fontsize=14, fontweight='bold',
                           family='Times New Roman')
                    
                # Add metric label on left column only
                if j == 0:
                    if metric == 'GRPE':
                        ax.text(-0.2, 0.5, 'GRP$_E$', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=14, fontweight='bold', 
                               rotation=90, family='Times New Roman')
                    elif metric == 'GRP':
                        ax.text(-0.2, 0.5, 'GRP$_L$', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=14, fontweight='bold', 
                               rotation=90, family='Times New Roman')
                    else:
                        ax.text(-0.2, 0.5, metric, transform=ax.transAxes, 
                               ha='center', va='center', fontsize=14, fontweight='bold', 
                               rotation=90, family='Times New Roman')
                
                # Format axes - parallel labels (rotation=0)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.tick_params(axis='x', rotation=0)  # Parallel labels
                ax.set_ylim(0, 1)
                
            else:
                ax.text(0.5, 0.5, f'No data\n{metric} {year}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel('')
                ax.set_ylabel('')
    
    # Add overall y-axis label with more spacing
    fig.text(0.05, 0.5, 'Proportion of positive cells', va='center', rotation='vertical', 
             fontsize=16, fontweight='bold', family='Times New Roman')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.18)  # Increased left margin for better spacing
    plt.show()

def create_comprehensive_plot(df):
    """
    Create comprehensive plot with scientific colors, grouped legend, and grouped metrics
    """
    
    if len(df) == 0:
        print("No data available for comprehensive plot")
        return
    
    # Prepare data
    df = df.copy()
    df['Lake_Year'] = df['Lake'] + ' ' + df['Year']
    
    # Create a combined category for proper grouping
    df['Metric_Timescale'] = df['Metric'] + ' ' + df['Timescale']
    
    # Scientific publication colorblind-friendly colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    plt.figure(figsize=(14, 8))
    
    # Get unique combinations and sort them to group like metrics together
    metric_order = ['GRP', 'TDO6', 'GRPE']
    timescale_order = ['Daily', 'Hourly']
    
    # Create ordered list of metric-timescale combinations
    ordered_combinations = []
    for metric in metric_order:
        for timescale in timescale_order:
            combo = f'{metric} {timescale}'
            if combo in df['Metric_Timescale'].unique():
                ordered_combinations.append(combo)
    
    # Filter dataframe to only include ordered combinations
    df_filtered = df[df['Metric_Timescale'].isin(ordered_combinations)].copy()
    
    # Create the plot with custom order
    lake_year_order = sorted(df_filtered['Lake_Year'].unique())
    
    # Create box plot with grouped metrics
    ax = sns.boxplot(data=df_filtered, x='Lake_Year', y='Value', hue='Metric_Timescale', 
                     hue_order=ordered_combinations, order=lake_year_order,
                     palette=colors[:len(ordered_combinations)])
    
    # Format plot
    plt.title('')  # No title
    plt.ylabel('Proportion of Positive Cells', fontfamily='Times New Roman', fontweight='bold', fontsize=16)
    plt.xlabel('')  # No x-axis label
    
    # Set x-axis labels to be parallel (horizontal)
    plt.xticks(rotation=0, fontfamily='Times New Roman', fontsize=16)  # Parallel labels
    
    # Organize legend with proper formatting
    handles, labels = ax.get_legend_handles_labels()
    
    # Format legend labels (add subscript for GRPE)
    formatted_labels = []
    for label in labels:
        if label.startswith('GRPE'):
            formatted_labels.append(label.replace('GRPE', 'GRP$_E$'))
        elif label.startswith('GRP'):
            formatted_labels.append(label.replace('GRP', 'GRP$_L$'))
        else:
            formatted_labels.append(label)
    
    plt.legend(handles, formatted_labels, 
              title='Model', 
              bbox_to_anchor=(1.025, 1), 
              loc='upper left',
              prop={'family': 'Times New Roman'})
    
    # Set y-axis tick labels font
    plt.yticks(fontfamily='Times New Roman')
    
    plt.tight_layout()
    plt.show()

def create_summary_statistics(df):
    """
    Generate summary statistics table
    """
    if len(df) == 0:
        print("No data available for summary statistics")
        return None
    
    summary = df.groupby(['Lake', 'Year', 'Metric', 'Timescale'])['Value'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(3)
    
    print("\nSummary Statistics:")
    print(summary)
    return summary

def save_summary_stats(summary_df, outpath):
    """
    Save summary statistics to CSV file
    """
    if summary_df is None or len(summary_df) == 0:
        print("No summary data to save")
        return None
    
    # Create output directory if it doesn't exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        print(f"Created output directory: {outpath}")
    
    summary_stats_path = os.path.join(outpath, "boxplot_models_yrs_pturn.csv")
    
    try:
        summary_df.to_csv(summary_stats_path, index=True)
        print(f"Summary statistics saved to: {summary_stats_path}")
        return summary_stats_path
    except Exception as e:
        print(f"Error saving summary statistics: {e}")
        return None

def main(data_directory=None, grp_directory=None, tdo6_directory=None, date_ranges=None, outpath=None):
    """
    Main analysis function with date range filtering
    
    Parameters:
    data_directory (str): Path to directory containing all data files
    grp_directory (str): Legacy parameter (not used)
    tdo6_directory (str): Legacy parameter (not used)
    date_ranges (dict): Dictionary specifying date ranges for each lake-year combination
                       Format: {('Lake', 'Year'): ('start_date', 'end_date')}
    outpath (str): Path for saving output files
    
    Usage examples:
    - main()  # Use default paths, no date filtering
    - main(data_directory='/path/to/data')  # Specify path, no date filtering
    - main(date_ranges={('Crooked', '2021'): ('2021-06-01', '2021-08-31')})  # With date filtering
    """
    
    print("=== Box Plot Analysis with Date Range Filtering ===")
    
    # Set default output path
    if outpath is None:
        outpath = '/tmp/boxplot_output'  # Default output path
    
    # Load data
    df = load_and_process_data(data_directory, date_ranges)
    
    if len(df) == 0:
        print("\nNo data loaded! Check:")
        print("1. Directory paths are correct")
        print("2. CSV files exist in specified directories")
        print("3. Files contain expected column patterns")
        print("4. Date ranges (if specified) are valid")
        print("5. EST_DateTime column exists in files")
        return None, None
    
    print("\nData overview:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"Unique values:")
    for col in ['Lake', 'Year', 'Metric', 'Timescale']:
        print(f"  {col}: {sorted(df[col].unique())}")
    
    # Create main box plots
    print("\n=== Creating Main Box Plots ===")
    create_main_box_plots(df)
    
    # Create comprehensive plot
    print("\n=== Creating Comprehensive Plot ===")
    create_comprehensive_plot(df)
    
    # Generate summary statistics
    summary = create_summary_statistics(df)
    
    # Save summary statistics
    if summary is not None:
        save_summary_stats(summary, outpath)
    
    return df, summary

# Run analysis
if __name__ == "__main__":
    # Option 3: With date range filtering and the outpath
    date_ranges = {
         ('Crooked', '2021'): ('2021-09-20', '2021-10-25'),
         ('Crooked', '2022'): ('2022-09-25', '2022-11-01'),
         ('Failing', '2021'): ('2021-09-20', '2021-11-05'),
         ('Failing', '2022'): ('2022-09-20', '2022-11-05')
     }
    
    data_directory = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Redo_pos_summaries_dfs_GRP/Proportions/partial_plot_all_dfs'
    
    outpath = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/Aug25_bxplt_yrsModels_df'
    
    df, summary = main(data_directory=data_directory, date_ranges=date_ranges, outpath=outpath)



#%%

"""
Line plot framework for comparing daily vs hourly metrics
Uses same data loading framework as box plots but creates time series line plots

Rajeev Kumar, 08/06/2025 (Line Plot Extension)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from datetime import datetime, timedelta

def filter_by_date_range(df, start_date, end_date):
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
            
            # Apply date filtering if specified
            if date_ranges and (lake, year) in date_ranges:
                start_date, end_date = date_ranges[(lake, year)]
                df = filter_by_date_range(df, start_date, end_date)
                print(f"    After filtering: {df.shape[0]} rows")
                
                if len(df) == 0:
                    print("    Skipping - no data after filtering")
                    continue
        
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
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    
    # Handle single subplot case
    if n_combos == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Color scheme: hourly=blue, daily=orange
    colors = {'Hourly': '#1f77b4', 'Daily': '#ff7f0e'}
    
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
                ax.plot(timescale_data['EST_DateTime'], timescale_data['Value'], 
                       color=colors[timescale], label=timescale, linewidth=2,
                       marker='o' if timescale == 'Daily' else '.', markersize=4)
        
        # Format subplot
        ax.set_ylim(0, 1)
        ax.set_ylabel('Proportion of positive cells', fontfamily='Times New Roman')
        ax.set_xlabel('Date', fontfamily='Times New Roman')
        
        # Format title with subscripts
        if metric == 'GRPE':
            title_metric = 'GRP$_E$'
        elif metric == 'GRP':
            title_metric = 'GRP$_L$'
        else:
            title_metric = metric
            
        ax.set_title(f'{lake} {year} - {title_metric}', 
                    fontfamily='Times New Roman', fontweight='bold')
        
        # Add legend
        ax.legend(loc='upper right', prop={'family': 'Times New Roman'})
        
        # Format x-axis dates
        ax.tick_params(axis='x', rotation=45)
        
        # Grid for better readability
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
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
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 6*n_rows))
    
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
        legend_elements = []
        
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
                    
                    ax.plot(metric_timescale_data['EST_DateTime'], 
                           metric_timescale_data['Value'],
                           color=color, linestyle=linestyle, label=label, 
                           linewidth=2, alpha=0.8)
        
        # Format subplot
        ax.set_ylim(0, 1)
        ax.set_ylabel('Proportion of positive cells', fontfamily='Times New Roman')
        ax.set_xlabel('Date', fontfamily='Times New Roman')
        ax.set_title(f'{lake} {year}', fontfamily='Times New Roman', fontweight='bold')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 prop={'family': 'Times New Roman'})
        
        # Format x-axis dates
        ax.tick_params(axis='x', rotation=45)
        
        # Grid for better readability
        ax.grid(False, alpha=0.3)
    
    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def main_line_plots(data_directory=None, date_ranges=None, outpath=None):
    """
    Main function for creating line plots comparing daily vs hourly timescales
    
    Parameters:
    data_directory (str): Path to directory containing all data files
    date_ranges (dict): Dictionary specifying date ranges for each lake-year combination
    outpath (str): Path for saving output files
    """
    
    print("=== Line Plot Analysis - Daily vs Hourly Comparison ===")
    
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

def filter_by_date_range(df, start_date, end_date):
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
            
            # Apply date filtering if specified
            if date_ranges and (lake, year) in date_ranges:
                start_date, end_date = date_ranges[(lake, year)]
                df = filter_by_date_range(df, start_date, end_date)
                print(f"    After filtering: {df.shape[0]} rows")
                
                if len(df) == 0:
                    print("    Skipping - no data after filtering")
                    continue
        
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

def main_line_plots(data_directory=None, date_ranges=None, outpath=None):
    """
    Main function for creating line plots comparing daily vs hourly timescales
    
    Parameters:
    data_directory (str): Path to directory containing all data files
    date_ranges (dict): Dictionary specifying date ranges for each lake-year combination
    outpath (str): Path for saving output files
    """
    
    print("=== Line Plot Analysis - Daily vs Hourly Comparison ===")
    
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

"""
Corrected line plot framework for comparing daily vs hourly metrics
Properly structured with all imports and functions organized correctly

Rajeev Kumar, 08/11/2025 (Line Plot Extension - Corrected Structure)
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

def apply_timestamp_corrections(df, lake, year):
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