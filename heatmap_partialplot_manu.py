import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import json
from pathlib import Path

# Set Times New Roman font globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

class TimestampAverager:
    def __init__(self, data_directory=None):
        """
        Initialize the TimestampAverager with data directory
        
        Parameters:
        data_directory (str): Path to directory containing 2D array files
        """
        self.data_directory = data_directory or os.getcwd()
        self.arrays = {}
        self.timestamps = {}
        self.timescales = {}
        self.modified_arrays = {}
        
        # Default timestamps for replacement
        self.default_timestamps = {
            ('Crooked', '2021'): ['2021-07-26 10:00:00-04:00', '2021-08-13 15:00:00-04:00', 
                                 '2021-09-01 15:00:00-04:00', '2021-10-08 12:00:00-04:00'],
            ('Failing', '2022'): ['2022-07-21 12:00:00-04:00', '2022-09-16 12:00:00-04:00']
        }
    
    def load_arrays(self, file_pattern="*.csv"):
        """
        Load 2D arrays from CSV files in specified directory
        
        Parameters:
        file_pattern (str): Pattern to match CSV files
        """
        print(f"Loading arrays from: {self.data_directory}")
        print(f"Looking for pattern: {file_pattern}")
        
        # Check if directory exists
        if not os.path.exists(self.data_directory):
            print(f"ERROR: Directory does not exist: {self.data_directory}")
            return
        
        # Look for CSV files
        search_path = os.path.join(self.data_directory, file_pattern)
        print(f"Full search path: {search_path}")
        
        array_files = glob.glob(search_path)
        print(f"Found {len(array_files)} files matching pattern")
        
        if not array_files:
            # Debug: show what files ARE in the directory
            all_files = os.listdir(self.data_directory)
            csv_files = [f for f in all_files if f.endswith('.csv')]
            print(f"Available CSV files in directory:")
            for f in csv_files[:10]:  # Show first 10
                print(f"  {f}")
            if len(csv_files) > 10:
                print(f"  ... and {len(csv_files)-10} more")
            return
        
        for file_path in array_files:
            filename = os.path.basename(file_path)
            print(f"\nLoading: {filename}")
            
            # Extract lake, year, and timescale from filename
            lake, year, timescale = self._parse_filename(filename)
            
            if lake and year and timescale:
                key = (lake, year, timescale)
                try:
                    # Load CSV file as numpy array
                    df = pd.read_csv(file_path, header=None)  # No header for pure numeric data
                    self.arrays[key] = df.values  # Convert to numpy array
                    print(f"  ✓ Loaded array shape: {self.arrays[key].shape}")
                    
                    # Generate timestamps for this array
                    self.timestamps[key] = self._generate_timestamps(key)
                    
                except Exception as e:
                    print(f"  ✗ Error loading {filename}: {e}")
                    # Try with different CSV loading options if first attempt fails
                    try:
                        df = pd.read_csv(file_path, header=0)  # Try with header
                        self.arrays[key] = df.values
                        print(f"  ✓ Loaded array shape (with header): {self.arrays[key].shape}")
                        self.timestamps[key] = self._generate_timestamps(key)
                    except Exception as e2:
                        print(f"  ✗ Failed with both loading methods: {e2}")
            else:
                print(f"  ✗ Skipped - could not parse filename components")
    
    def _parse_filename(self, filename):
        """
        Parse filename to extract lake, year, and timescale
        Updated to handle flexible structure: {lake}_{variable}_{timescale}_{year}.csv
        """
        filename_lower = filename.lower().replace('.csv', '')  # Remove extension
        parts = filename_lower.split('_')
        
        print(f"    Parsing filename: {filename}")
        print(f"    Split parts: {parts}")
        
        # Initialize variables
        lake = None
        year = None
        timescale = None
        
        # Extract lake (can be first part or part of first part)
        for part in parts:
            if 'crook' in part:  # matches 'crook' or 'crooked'
                lake = 'Crooked'
                break
            elif 'fail' in part:  # matches 'fail' or 'failing'
                lake = 'Failing'
                break
        
        # Extract year (look for 2021 or 2022 in any part)
        for part in parts:
            if '2021' in part:
                year = '2021'
                break
            elif '2022' in part:
                year = '2022'
                break
        
        # Extract timescale (look for daily or hourly in any part)
        for part in parts:
            if 'daily' in part:
                timescale = 'Daily'
                break
            elif 'hourly' in part:
                timescale = 'Hourly'
                break
        
        # Debug output
        print(f"    Extracted: Lake={lake}, Year={year}, Timescale={timescale}")
        
        # Check if we got all required components
        if not lake:
            print(f"    ERROR: Could not identify lake from: {parts}")
            return None, None, None
        if not year:
            print(f"    ERROR: Could not identify year from: {parts}")
            return None, None, None
        if not timescale:
            print(f"    ERROR: Could not identify timescale from: {parts}")
            return None, None, None
        
        return lake, year, timescale
    
    def _generate_timestamps(self, key):
        """
        Generate timestamps for the array based on timescale
        """
        lake, year, timescale = key
        array_shape = self.arrays[key].shape
        n_timepoints = array_shape[1]  # Assuming horizontal axis is time
        
        # Define date ranges for each lake-year combination
        # Fixed: Corrected years in date strings to match the year keys
        date_ranges = {
            ('Crooked', '2021'): ('2021-06-24', '2021-11-11'),
            ('Crooked', '2022'): ('2022-06-05', '2022-12-01'),  # Fixed: 2022 dates
            ('Failing', '2021'): ('2021-06-07', '2021-12-01'),  # Fixed: 2021 dates
            ('Failing', '2022'): ('2022-04-12', '2022-12-01')
        }
        
        start_date, end_date = date_ranges.get((lake, year), ('2021-01-01', '2021-12-31'))
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate timestamps based on timescale
        # Option 1: Use both start and end dates (recommended)
        if timescale == 'Daily':
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='D')
        elif timescale == 'Hourly':
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='H')
        else:
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        # Check if generated timestamps match array dimensions
        if len(timestamps) != n_timepoints:
            print(f"Warning: Generated {len(timestamps)} timestamps but array has {n_timepoints} time points for {key}")
            print(f"  Date range: {start_date} to {end_date}")
            print(f"  Adjusting to match array dimensions...")
            
            # Option 2: If mismatch, use the original approach with periods
            if timescale == 'Daily':
                timestamps = pd.date_range(start=start_dt, periods=n_timepoints, freq='D')
            elif timescale == 'Hourly':
                timestamps = pd.date_range(start=start_dt, periods=n_timepoints, freq='H')
            else:
                timestamps = pd.date_range(start=start_dt, periods=n_timepoints, freq='D')
        
        return timestamps
    
    def set_replacement_timestamps(self, lake_year_timestamps):
        """
        Set custom timestamps for replacement
        
        Parameters:
        lake_year_timestamps (dict): Dictionary with (lake, year) keys and list of timestamp strings
        """
        self.replacement_timestamps = lake_year_timestamps
        print("Replacement timestamps set:")
        for (lake, year), timestamps in lake_year_timestamps.items():
            print(f"  {lake} {year}: {len(timestamps)} timestamps")
    
    def find_timestamp_indices(self, key, target_timestamps):
        """
        Find indices in the array corresponding to target timestamps
        """
        lake, year, timescale = key
        array_timestamps = self.timestamps[key]
        
        indices = []
        for target_ts in target_timestamps:
            target_dt = pd.to_datetime(target_ts)
            
            # Ensure both timestamps are timezone-naive for comparison
            if target_dt.tz is not None:
                target_dt = target_dt.tz_localize(None)
            
            if hasattr(array_timestamps, 'tz') and array_timestamps.tz is not None:
                array_timestamps_compare = array_timestamps.tz_localize(None)
            else:
                array_timestamps_compare = array_timestamps
            
            # Find closest timestamp
            time_diffs = np.abs(array_timestamps_compare - target_dt)
            closest_idx = np.argmin(time_diffs)
            indices.append(closest_idx)
            
            print(f"    Target: {target_ts}")
            print(f"    Closest array timestamp: {array_timestamps[closest_idx]}")
            print(f"    Array index: {closest_idx}")
        
        return indices
    
    def create_averaged_replacement(self, key, target_indices, window_size=5, method='mean'):
        """
        Create averaged values for replacement
        
        Parameters:
        key: (lake, year, timescale) tuple
        target_indices: List of indices to replace
        window_size: Size of averaging window (hours/days before and after)
        method: Averaging method ('mean', 'median')
        """
        array = self.arrays[key]
        modified_array = array.copy()
        
        for idx in target_indices:
            # Define averaging window
            start_idx = max(0, idx - window_size)
            end_idx = min(array.shape[1], idx + window_size + 1)
            
            # Exclude the target index from averaging
            avg_indices = list(range(start_idx, end_idx))
            if idx in avg_indices:
                avg_indices.remove(idx)
            
            if len(avg_indices) > 0:
                # Calculate average for each depth
                if method == 'mean':
                    avg_values = np.mean(array[:, avg_indices], axis=1)
                elif method == 'median':
                    avg_values = np.median(array[:, avg_indices], axis=1)
                else:
                    avg_values = np.mean(array[:, avg_indices], axis=1)
                
                # Replace the target column
                modified_array[:, idx] = avg_values
        
        return modified_array
    
    def process_replacements(self, replacement_config):
        """
        Process all replacements based on configuration
        
        Parameters:
        replacement_config (dict): Configuration dictionary with parameters
        """
        self.modified_arrays = {}
        
        for key in self.arrays.keys():
            lake, year, timescale = key
            
            # Check if this lake-year combination has replacement timestamps
            if (lake, year) in replacement_config['timestamps']:
                target_timestamps = replacement_config['timestamps'][(lake, year)]
                target_indices = self.find_timestamp_indices(key, target_timestamps)
                
                print(f"Processing {lake} {year} {timescale}:")
                print(f"  Target timestamps: {len(target_timestamps)}")
                print(f"  Target indices: {target_indices}")
                
                # Create averaged replacement
                modified_array = self.create_averaged_replacement(
                    key, target_indices, 
                    window_size=replacement_config['window_size'],
                    method=replacement_config['method']
                )
                
                self.modified_arrays[key] = modified_array
            else:
                # No replacement needed, keep original
                self.modified_arrays[key] = self.arrays[key].copy()
    
    def plot_comparison(self, key, save_path=None):
        """
        Plot comparison between original and modified arrays
        """
        if key not in self.arrays or key not in self.modified_arrays:
            print(f"No data available for {key}")
            return
        
        original = self.arrays[key]
        modified = self.modified_arrays[key]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original array
        im1 = axes[0].imshow(original, aspect='auto', cmap='seismic')
        axes[0].set_title('Original Array')
        axes[0].set_ylabel('Depth')
        axes[0].set_xlabel('Time Index')
        plt.colorbar(im1, ax=axes[0])
        
        # Modified array
        im2 = axes[1].imshow(modified, aspect='auto', cmap='seismic')
        axes[1].set_title('Modified Array')
        axes[1].set_ylabel('Depth')
        axes[1].set_xlabel('Time Index')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = modified - original
        im3 = axes[2].imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        axes[2].set_title('Difference (Modified - Original)')
        axes[2].set_ylabel('Depth')
        axes[2].set_xlabel('Time Index')
        plt.colorbar(im3, ax=axes[2])
        
        plt.suptitle(f'{key[0]} {key[1]} {key[2]}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_all_comparisons(self, save_dir=None):
        """
        Plot comparisons for all arrays
        """
        for key in self.arrays.keys():
            if save_dir:
                save_path = os.path.join(save_dir, f'comparison_{key[0]}_{key[1]}_{key[2]}.png')
            else:
                save_path = None
            
            self.plot_comparison(key, save_path)
    
    def save_modified_arrays(self, output_directory, save_format='csv'):
        """
        Save modified arrays to specified directory
        
        Parameters:
        output_directory (str): Directory to save files
        save_format (str): Format to save ('csv', 'npy', or 'both')
        """
        os.makedirs(output_directory, exist_ok=True)
        
        for key, array in self.modified_arrays.items():
            lake, year, timescale = key
            base_filename = f"{lake}_{year}_{timescale}_modified"
            
            if save_format in ['csv', 'both']:
                # Save as CSV
                csv_filename = f"{base_filename}.csv"
                csv_filepath = os.path.join(output_directory, csv_filename)
                
                # Convert back to DataFrame and save
                df = pd.DataFrame(array)
                df.to_csv(csv_filepath, index=False, header=False)
                print(f"Saved CSV: {csv_filepath}")
            
            if save_format in ['npy', 'both']:
                # Save as numpy array
                npy_filename = f"{base_filename}.npy"
                npy_filepath = os.path.join(output_directory, npy_filename)
                
                np.save(npy_filepath, array)
                print(f"Saved NPY: {npy_filepath}")
    
    def get_interactive_config(self):
        """
        Get configuration parameters interactively
        """
        print("\n=== Interactive Configuration ===")
        
        # Window size
        window_size = int(input("Enter averaging window size (e.g., 5): ") or "5")
        
        # Averaging method
        method = input("Enter averaging method (mean/median) [mean]: ") or "mean"
        
        # Replacement timestamps
        timestamps = {}
        
        # Get unique lake-year combinations
        lake_years = set((key[0], key[1]) for key in self.arrays.keys())
        
        for lake, year in lake_years:
            print(f"\nTimestamps for {lake} {year}:")
            
            # Show default if available
            if (lake, year) in self.default_timestamps:
                print(f"Default: {self.default_timestamps[(lake, year)]}")
                use_default = input("Use default timestamps? (y/n) [y]: ") or "y"
                
                if use_default.lower() == 'y':
                    timestamps[(lake, year)] = self.default_timestamps[(lake, year)]
                    continue
            
            # Get custom timestamps
            n_timestamps = int(input(f"Number of timestamps to replace for {lake} {year}: ") or "0")
            
            if n_timestamps > 0:
                ts_list = []
                for i in range(n_timestamps):
                    ts = input(f"  Timestamp {i+1} (YYYY-MM-DD HH:MM:SS-TZ): ")
                    ts_list.append(ts)
                timestamps[(lake, year)] = ts_list
        
        return {
            'window_size': window_size,
            'method': method,
            'timestamps': timestamps
        }
    
    def run_interactive_session(self):
        """
        Run interactive session for timestamp averaging
        """
        print("=== Timestamp Averaging Tool ===")
        
        # Always ask for data directory to ensure we're looking in the right place
        print(f"Current working directory: {os.getcwd()}")
        data_dir = input("Enter data directory (or press Enter for current directory): ").strip()
        
        if data_dir:
            self.data_directory = data_dir
        else:
            self.data_directory = os.getcwd()
        
        print(f"Using data directory: {os.path.abspath(self.data_directory)}")
        
        # Load arrays
        file_pattern = input("Enter file pattern for arrays (*.csv): ") or "*.csv"
        self.load_arrays(file_pattern)
        
        if not self.arrays:
            print("No arrays loaded. Exiting.")
            return
        
        print(f"\nLoaded {len(self.arrays)} arrays:")
        for key in self.arrays.keys():
            print(f"  {key}: {self.arrays[key].shape}")
        
        # Get configuration
        config = self.get_interactive_config()
        
        # Process replacements
        print("\n=== Processing Replacements ===")
        self.process_replacements(config)
        
        # Plot comparisons
        print("\n=== Plotting Comparisons ===")
        plot_choice = input("Plot all comparisons? (y/n) [y]: ") or "y"
        
        if plot_choice.lower() == 'y':
            self.plot_all_comparisons()
        
        # Save option
        save_choice = input("\nSave modified arrays? (y/n) [n]: ") or "n"
        
        if save_choice.lower() == 'y':
            output_dir = input("Enter output directory: ") or "modified_arrays"
            
            # Ask for save format
            save_format = input("Save format (csv/npy/both) [csv]: ") or "csv"
            self.save_modified_arrays(output_dir, save_format)
            
            # Save configuration
            config_path = os.path.join(output_dir, "processing_config.json")
            with open(config_path, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                json_config = config.copy()
                json.dump(json_config, f, indent=2)
            print(f"Configuration saved to: {config_path}")


def main():
    """
    Main function to run the timestamp averaging tool
    """
    ######
    # Option 1: Interactive session - asks for directory
    ######
    averager = TimestampAverager('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/PartialPlot_Arrays_forManuscript')
    averager.run_interactive_session()
    
    # Option 2: If you know the exact path, uncomment and modify this line:
    # averager = TimestampAverager('/Users/rajeevkumar/path/to/your/csv/files')
    # averager.run_interactive_session()
    
    ########
    # Option 2: Programmatic usage example
    ########
    """
    # Initialize with data directory
    averager = TimestampAverager('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/PartialPlot_Arrays_forManuscript')
    
    # Load arrays
    averager.load_arrays('*.csv')
    
    # Define replacement configuration
    config = {
        'window_size': 5,
        'method': 'mean',
        'timestamps': {
            ('Crooked', '2021'): ['2021-07-26 10:00:00-04:00', '2021-08-13 15:00:00-04:00', 
                                 '2021-09-01 15:00:00-04:00', '2021-10-08 12:00:00-04:00'],
            ('Crooked, ' 2022'): ['2022-07-23 12:00:00-04:00'],
            ('Failing', '2022'): ['2022-07-21 12:00:00-04:00', '2022-09-16 12:00:00-04:00']
        }
    }
    
    # Process replacements
    averager.process_replacements(config)
    
    # Plot comparisons
    averager.plot_all_comparisons()
    
    # Save modified arrays (can choose csv, npy, or both)
    averager.save_modified_arrays('output_directory', save_format='csv')
    """

if __name__ == "__main__":
    main()



#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
from pathlib import Path

class LakeDataPlotter:
    def __init__(self):
        # Set Times New Roman as default font
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 10
        
        # Variable configurations
        self.variables = {
            'Temperature': {
                'unit': 'Temperature (°C)',
                'decimals': 1,
                'norm_type': 'capped',
                #'midpoint': 15.0,
                'vmax': 30.0,
                'cmap': 'seismic'
            },
            'DO': {
                'unit': 'DO (mg L⁻¹)',
                'decimals': 1,
                'norm_type': 'capped',
                #'midpoint': 6.0,
                'vmax': 15.0,
                'cmap': 'seismic'
            },
            'TDO6': {
                'unit': 'TDO6 (DO ≥ 6 mg L$^{-1}$ & T ≤ 22.8°C)',
                'decimals': 0,
                'norm_type': 'binary',
                'labels': ['No', 'Yes'],
                'cmap': ['#2166ac', '#d6604d']  # Blue and red from seismic
            },
            'GRP': {
                'unit': 'GRP$_L$ (g g$^{-1}$ d$^{-1}$)',
                'decimals': 3,
                'norm_type': 'symmetric',
                'cmap': 'seismic'
            },
            'GRP_lethal': {
                'unit': 'GRP$_E$ (g g$^{-1}$ d$^{-1}$)',
                'decimals': 3,
                'norm_type': 'symmetric',
                'cmap': 'seismic'
            }
        }
        
        self.timescales = ['hourly', 'daily']
    
    def load_data(self, data_directory, lakes, years, date_ranges):
        """
        Load 2D arrays from CSV files.
        Expected structure: data_directory/{lake}_{variable}_{timescale}_{year}.csv
        
        Args:
            date_ranges: Dictionary with keys like 'crooked_2021' containing 'start_date' and 'end_date'
        """
        import pandas as pd
        
        self.data = {}
        self.date_ranges = date_ranges
        
        # Map your variable names to the expected variable names
        variable_mapping = {
            'temp': 'Temperature',
            'DO': 'DO', 
            'TDO6': 'TDO6',
            'grp': 'GRP',
            'grplethal': 'GRP_lethal'
        }
        
        for lake in lakes:
            self.data[lake] = {}
            for year in years:
                self.data[lake][year] = {}
                
                # Get date range for this lake-year combination
                date_key = f"{lake}_{year}"
                if date_key in date_ranges:
                    start_date = datetime.strptime(date_ranges[date_key]['start_date'], '%Y-%m-%d')
                    end_date = datetime.strptime(date_ranges[date_key]['end_date'], '%Y-%m-%d')
                    n_days = (end_date - start_date).days + 1
                    print(f"{lake} {year}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({n_days} days)")
                else:
                    print(f"Warning: No date range specified for {lake} {year}")
                    n_days = 365  # Default fallback
                
                for file_var, plot_var in variable_mapping.items():
                    self.data[lake][year][plot_var] = {}
                    for timescale in self.timescales:
                        filename = f"{lake}_{file_var}_{timescale}_{year}.csv"
                        filepath = Path(data_directory) / filename
                        
                        if filepath.exists():
                            try:
                                # Load CSV file
                                df = pd.read_csv(filepath, index_col=0)
                                # Convert to numpy array
                                self.data[lake][year][plot_var][timescale] = df.values
                                print(f"Loaded: {filename} - Shape: {df.values.shape}")
                            except Exception as e:
                                print(f"Error loading {filename}: {e}")
                                # Create dummy data based on actual date range
                                if timescale == 'hourly':
                                    shape = (50, n_days * 24)  # 50 depth levels, hourly
                                else:
                                    shape = (50, n_days)     # 50 depth levels, daily
                                self.data[lake][year][plot_var][timescale] = np.random.randn(*shape)
                        else:
                            print(f"Warning: File not found: {filename}")
                            # Create dummy data based on actual date range
                            if timescale == 'hourly':
                                shape = (50, n_days * 24)  # 50 depth levels, hourly
                            else:
                                shape = (50, n_days)     # 50 depth levels, daily
                            self.data[lake][year][plot_var][timescale] = np.random.randn(*shape)
    
    def create_date_labels(self, lake, year, n_timepoints, timescale='daily'):
        """Create date labels for x-axis based on specific lake-year date range"""
        date_key = f"{lake}_{year}"
        
        if date_key not in self.date_ranges:
            print(f"Warning: No date range for {lake} {year}, using default")
            start = datetime(year, 1, 1)
            end = datetime(year, 12, 31)
        else:
            start = datetime.strptime(self.date_ranges[date_key]['start_date'], '%Y-%m-%d')
            end = datetime.strptime(self.date_ranges[date_key]['end_date'], '%Y-%m-%d')
        
        if timescale == 'daily':
            # Create date range
            n_days = (end - start).days + 1
            dates = [start + timedelta(days=i) for i in range(min(n_days, n_timepoints))]
            
            # Create monthly ticks
            month_starts = []
            month_labels = []
            current_month = None
            
            for i, date in enumerate(dates):
                if date.month != current_month:
                    month_starts.append(i)
                    month_labels.append(date.strftime('%b'))
                    current_month = date.month
                    
        else:  # hourly
            # For hourly data, we need to account for 24 hours per day
            n_days = (end - start).days + 1
            dates = []
            for day in range(n_days):
                day_date = start + timedelta(days=day)
                for hour in range(24):
                    if len(dates) < n_timepoints:
                        dates.append(day_date + timedelta(hours=hour))
            
            # Create monthly ticks for hourly data
            month_starts = []
            month_labels = []
            current_month = None
            
            for i, date in enumerate(dates):
                if date.month != current_month and date.hour == 0:  # Only at start of day
                    month_starts.append(i)
                    month_labels.append(date.strftime('%b'))
                    current_month = date.month
        
        return month_starts, month_labels
    
    def create_depth_labels(self, n_depths, depth_interval=0.5):
        """Create depth labels for y-axis"""
        depths = [i * depth_interval for i in range(n_depths)]
        # Show every 5th depth label to avoid crowding
        tick_positions = list(range(0, n_depths, 10))
        tick_labels = [f"{depths[i]:.1f}" for i in tick_positions]
        return tick_positions, tick_labels
    
    def get_normalization(self, data, variable):
        """Get appropriate normalization for the variable"""
        config = self.variables[variable]
    
        if variable in ['Temperature', 'DO']:
            # Use simple normalization with manual vmax cap
            vmin = np.nanmin(data)
            vmax = config['vmax']
            return plt.Normalize(vmin=vmin, vmax=vmax)
    
        elif config['norm_type'] == 'binary':
            return None  # Will use ListedColormap
    
        elif variable in ['GRP', 'GRP_lethal']:
            vmax = np.nanmax(np.abs(data))
            vmin = np.nanmin(data)
            return plt.Normalize(vmin=vmin, vmax=vmax)
    
        return None

    def plot_lake_data(self, lake, years, figsize=(20, 24)):
        """Create the main plot for a single lake"""
        n_variables = len(self.variables)
        n_timescales = len(self.timescales)
        n_years = len(years)
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            n_variables * n_timescales, n_years, 
            figsize=figsize, 
            constrained_layout=True
        )
        
        if n_years == 1:
            axes = axes.reshape(-1, 1)
        
        # Store colorbars for each variable type
        colorbars = {}
        
        # Plot each combination
        for year_idx, year in enumerate(years):
            # Add year title
            axes[0, year_idx].set_title(f"{year}", fontsize=14, fontweight='bold', pad=20)
            
            plot_row = 0
            
            for var_idx, (variable, config) in enumerate(self.variables.items()):
                for ts_idx, timescale in enumerate(self.timescales):
                    ax = axes[plot_row, year_idx]
                    
                    # Get data
                    if (lake in self.data and year in self.data[lake] and 
                        variable in self.data[lake][year] and 
                        timescale in self.data[lake][year][variable]):
                        
                        data = self.data[lake][year][variable][timescale]
                        
                        if lake == 'crooked' and year == 2022 and timescale == 'hourly':
                            data = np.flipud(data)
                        elif lake == 'failing' and year == 2021:
                            data = np.flipud(data)
                    else:
                        # Create dummy data if not available
                        date_key = f"{lake}_{year}"
                        if date_key in self.date_ranges:
                            start_date = datetime.strptime(self.date_ranges[date_key]['start_date'], '%Y-%m-%d')
                            end_date = datetime.strptime(self.date_ranges[date_key]['end_date'], '%Y-%m-%d')
                            n_days = (end_date - start_date).days + 1
                        else:
                            n_days = 365
                            
                        if timescale == 'hourly':
                            data = np.random.randn(50, n_days * 24)
                        else:
                            data = np.random.randn(50, n_days)
                    
                    # Apply data transformations based on variable
                    if variable == 'Temperature':
                        data = np.clip(data, None, config['vmax'])
                    elif variable == 'DO':
                        data = np.clip(data, None, config['vmax'])
                    elif variable == 'TDO6':
                        data = (data > 0).astype(int)  # Convert to binary
                    
                    # Create plot
                    if config['norm_type'] == 'binary':
                        cmap = ListedColormap(config['cmap'])
                        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=1)
                    else:
                        norm = self.get_normalization(data, variable)
                        im = ax.imshow(data, aspect='auto', cmap=config['cmap'], norm=norm)
                    
                    # Set up axes
                    n_depths, n_times = data.shape
                    
                    # Y-axis (depth) - only on leftmost plots
                    if year_idx == 0:
                        depth_ticks, depth_labels = self.create_depth_labels(n_depths)
                        ax.set_yticks(depth_ticks)
                        ax.set_yticklabels(depth_labels)
                        ax.set_ylabel('Depth (m)', fontsize=10)
                    else:
                        ax.set_yticks([])
                    
                    # X-axis (time) - only on bottom plots, using lake-year specific dates
                    if plot_row == len(axes) - 1:
                        time_ticks, time_labels = self.create_date_labels(lake, year, n_times, timescale)
                        ax.set_xticks(time_ticks)
                        ax.set_xticklabels(time_labels, rotation=0)
                        ax.set_xlabel('Month', fontsize=10)
                    else:
                        ax.set_xticks([])
                    
                    
                    # Add unit as y-label centered between the two timescale rows
                    if year_idx == 0:
                        if ts_idx == 0:  # First timescale for this variable
                            # Position between this row and the next row (spans 2 rows)
                            center_y = (plot_row + 0.5) / len(axes)  # Center between current and next row
                            fig.text(-0.02, 1 - center_y, config['unit'], 
                                     rotation=90, verticalalignment='center', horizontalalignment='center',
                                     fontsize=12, fontweight='bold', transform=fig.transFigure)
                    
                    """
                    OLD- works but aligns var label with first row
                    # Add variable label on the left side
                    if year_idx == 0:
                        # Create row group labels
                        if ts_idx == 0:  # First timescale for this variable
                            ax.text(-0.15, 0.5, config['unit'], #f"{variable} ({config['unit']})",  # f"{variable}\n({config['unit']})", 
                                   transform=ax.transAxes, rotation=90, 
                                   verticalalignment='center', horizontalalignment='center',
                                   fontsize=12, fontweight='bold')
                    """
                    
                    # Add timescale indicator
                    if plot_row < 2:  # Only for first variable
                        ax.text(0.02, 0.98, timescale.capitalize(), 
                               transform=ax.transAxes, verticalalignment='top',
                               fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='white', alpha=0.8))
                    
                    # Store colorbar info
                    if variable not in colorbars:
                        colorbars[variable] = {
                            'im': im,
                            'config': config,
                            'ax': ax,
                            'row': plot_row
                        }
                    
                    plot_row += 1
        
        # Add colorbars on the right side
        for variable, cbar_info in colorbars.items():
            config = cbar_info['config']   # modifyinfwith the correct tick marks for the colorbar coodinator
            
            # Create colorbar
            if config['norm_type'] == 'binary':
                cbar = fig.colorbar(cbar_info['im'], ax=axes[cbar_info['row'], :], 
                                   shrink=0.8, aspect=20, pad=0.02)
                cbar.set_ticks([0.25, 0.75])
                cbar.set_ticklabels(config['labels'])
            else:
                cbar = fig.colorbar(cbar_info['im'], ax=axes[cbar_info['row'], :], 
                                   shrink=0.8, aspect=20, pad=0.02)
                
                # Format colorbar labels
                if config['decimals'] == 0:
                    fmt = '%.0f'
                elif config['decimals'] == 1:
                    fmt = '%.1f'
                else:
                    fmt = '%.3f'
                
                # Set manual ticks for T and DO
                if variable == 'Temperature':
                    manual_ticks = [5, 10, 15, 20, 25, 30]
                    manual_labels = ['5', '10', '15', '20', '25', '>30']
                    cbar.set_ticks(manual_ticks)
                    cbar.set_ticklabels(manual_labels)
                elif variable == 'DO':
                    manual_ticks = [0, 2, 4, 6, 8, 11]
                    manual_labels = ['0', '2', '4', '6', '8', '>10']
                    cbar.set_ticks(manual_ticks)
                    cbar.set_ticklabels(manual_labels)
                elif variable in ['GRP', 'GRP_lethal']:
                    # Adding 5-7 even ticks
                    #auto_ticks = cbar.get_ticks()
                    vmax = np.nanmax(np.abs(data))
                    vmin = np.nanmin(data)
                    
                    manual_ticks = np.linspace(vmin, vmax, 6)
                    cbar.set_ticks(manual_ticks)
                    # Format labels with 3 decimals
                    labels = [f"{tick:.3f}" for tick in manual_ticks]
                    cbar.set_ticklabels(labels)
 
            cbar.set_label(config['unit'], rotation=270, labelpad=20, fontsize=10)
        
        #plt.suptitle(f"Lake {lake.capitalize()} - Multi-variable Heatmaps", fontsize=16, y=0.98)
        return fig
    
    def plot_all_lakes(self, lakes, years, save_dir=None):
        """Plot all lakes and optionally save figures"""
        figures = {}
        if save_dir:
            Path(save_dir).mkdir(parents = True, exist_ok = True)
        
        for lake in lakes:
            print(f"Plotting data for Lake {lake}...")
            fig = self.plot_lake_data(lake, years)
            figures[lake] = fig
            
            if save_dir:
                save_path = Path(save_dir) / f"lake_{lake}_heatmaps.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                print(f"Saved plot to {save_path}")
        
        return figures

# Example usage
def main():
    """Example of how to use the LakeDataPlotter with your specific date ranges"""
    
    # Initialize plotter
    plotter = LakeDataPlotter()
    
    # Define your specific date ranges for each (lake, year) combination
    date_ranges = {
        'crooked_2021': {'start_date': '2021-06-24', 'end_date': '2021-11-11'},
        'failing_2021': {'start_date': '2021-06-05', 'end_date': '2021-12-01'},
        'crooked_2022': {'start_date': '2022-06-07', 'end_date': '2022-12-01'},
        'failing_2022': {'start_date': '2022-04-12', 'end_date': '2022-12-01'}
    }
    
    # Define your data parameters
    data_directory = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/PartialPlot_Arrays_forManuscript"  # Current directory or update to your path
    lakes = ["crooked", "failing"]  # Your lake names from the files
    years = [2021, 2022]  # Your years from the files
    
    # Load data from CSV files with specific date ranges
    plotter.load_data(data_directory, lakes, years, date_ranges)
    
    # Create plots (no need for start_date parameter anymore)
    figures = plotter.plot_all_lakes(lakes, years, save_dir="output_plots")
    
    # Show plots
    plt.show()
    
    return figures

if __name__ == "__main__":
    figures = main()