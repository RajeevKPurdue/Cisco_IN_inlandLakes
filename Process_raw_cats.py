#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:47:12 2025

minidot raw cat processing

test/validation

1. skiplines = first 8 -> start at line ==8 ** OR ** start at the first line with a comma (applies to non-cat files as well)
2. eliminate whitespaces before and after columns
3. save columns with selection logic and renaming logic 

@author: rajeevkumar
"""

" readfile"
import os
import pandas as pd


def process_file(input_filepath, output_folder):
    """
    Process a single file: detect the header row, clean data, convert types, and save to output folder.
    """
    print(f"Processing: {input_filepath}")

    try:
        # Step 1: Read file and detect header row
        with open(input_filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Step 2: Identify the first line that looks like column headers (contains ',')
        header_line_index = next(i for i, line in enumerate(lines) if ',' in line)

        # Step 3: Read the file using detected header row and skip units row
        df = pd.read_csv(input_filepath, skiprows=header_line_index, skipinitialspace=True, dtype=str, encoding="utf-8")

        # Step 4: Drop the units row and reset index
        df = df[1:].reset_index(drop=True)

        # Step 5: Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()

        # Step 6: Set 'Unix Timestamp' as index (if present)
        if 'Unix Timestamp' in df.columns:
            df.set_index('Unix Timestamp', inplace=True)
            df.index.name = "Unix Timestamp"
        else:
            print(f"Warning: 'Unix Timestamp' column not found in {input_filepath}!")

        # Step 7: Strip spaces and remove hidden characters
        #df = df.apply(lambda x: x.astype(str).str.replace(r'[^\d.-]', '', regex=True).str.strip())

        # Step 8: Convert columns to correct data types
        dtype_mapping = {
            'Battery': 'float64',
            'Temperature': 'float64',
            'Dissolved Oxygen': 'float64',
            'Dissolved Oxygen Saturation': 'float64',
            'Q': 'float64'
        }

        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce').astype(dtype)
            else:
                print(f"Warning: Column '{col}' not found in {input_filepath}!")

        # Step 9: Convert time-related columns
        df['UTC_Date_&_Time'] = pd.to_datetime(df['UTC_Date_&_Time'], errors='coerce')
        df['Eastern Standard Time'] = pd.to_datetime(df['Eastern Standard Time'], errors='coerce')

        # Step 10: Save processed file to output folder
        os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
        output_filepath = os.path.join(output_folder, os.path.basename(input_filepath).replace('.TXT', '.csv'))

        df.to_csv(output_filepath, index=True)
        print(f"Saved processed file: {output_filepath}\n")

    except Exception as e:
        print(f"Error processing {input_filepath}: {e}")

def process_folder(input_folder, output_folder):
    """
    Process all files in the input folder and save cleaned versions to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    # Process all TXT files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".txt"):  # Handle case-insensitivity
            process_file(os.path.join(input_folder, filename), output_folder)

# Example usage:
input_folder = "/Volumes/WD Backup/Crooked/Crooked_Lake_2023_cat2025"
output_folder = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Crook_cats_processed_2023"

process_folder(input_folder, output_folder)




def clean_dataset(input_filepath, output_folder, drop_columns=None, rename_columns=None):
    """
    Cleans a dataset by:
    - Dropping selected columns
    - Renaming selected columns

    Parameters:
    - input_filepath (str): Path to the input dataset.
    - output_folder (str): Path to the folder where the cleaned file will be saved.
    - drop_columns (list): List of column names to drop.
    - rename_columns (dict): Dictionary mapping old column names to new names.

    Returns:
    - Saves the cleaned dataset to the output folder.
    """
    print(f"Processing: {input_filepath}")

    try:
        # Step 1: Read file and detect header row
        with open(input_filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Step 2: Identify the first line that looks like column headers (contains ',')
        header_line_index = next(i for i, line in enumerate(lines) if ',' in line)

        # Step 3: Read the file using detected header row and skip units row
        df = pd.read_csv(input_filepath, skiprows=header_line_index, skipinitialspace=True, dtype=str, encoding="utf-8")

        # Step 4: Drop the units row and reset index
        df = df[1:].reset_index(drop=True)

        # Step 5: Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Step 6: Set 'Unix Timestamp' as index (if present)
        if 'Unix Timestamp' in df.columns:
            df.set_index('Unix Timestamp', inplace=True)
            df.index.name = "Unix Timestamp"
        else:
            print(f"Warning: 'Unix Timestamp' column not found in {input_filepath}!")

        # Step 7: Strip spaces and remove hidden characters
        #df = df.apply(lambda x: x.astype(str).str.replace(r'[^\d.-]', '', regex=True).str.strip())

        # Step 8: Convert columns to correct data types
        dtype_mapping = {
            'Battery': 'float64',
            'Temperature': 'float64',
            'Dissolved Oxygen': 'float64',
            'Dissolved Oxygen Saturation': 'float64',
            'Q': 'float64'
        }

        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce').astype(dtype)
            else:
                print(f"Warning: Column '{col}' not found in {input_filepath}!")

        # Step 9: Convert time-related columns
        df['UTC_Date_&_Time'] = pd.to_datetime(df['UTC_Date_&_Time'], errors='coerce')
        df['Eastern Standard Time'] = pd.to_datetime(df['Eastern Standard Time'], errors='coerce')


        # Step 6: Drop selected columns
        if drop_columns:
            df.drop(columns=drop_columns, errors='ignore', inplace=True)

        # Step 7: Rename selected columns
        if rename_columns:
            df.rename(columns=rename_columns, inplace=True)

        # Step 8: Save cleaned file
        os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
        output_filepath = os.path.join(output_folder, os.path.basename(input_filepath).replace('.TXT', '.csv'))

        df.to_csv(output_filepath, index=False)  # Save without index
        print(f"Saved cleaned file: {output_filepath}\n")

    except Exception as e:
        print(f"Error processing {input_filepath}: {e}")

def process_folder(input_folder, output_folder, drop_columns=None, rename_columns=None):
    """
    Processes all files in a folder and applies column dropping/renaming.

    Parameters:
    - input_folder (str): Path to the input folder containing datasets.
    - output_folder (str): Path to save the cleaned files.
    - drop_columns (list): List of column names to drop.
    - rename_columns (dict): Dictionary for renaming columns.

    Returns:
    - Saves all cleaned datasets in the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".txt"):  # Handle case-insensitivity
            clean_dataset(os.path.join(input_folder, filename), output_folder, drop_columns, rename_columns)

# Example usage:
input_folder = "/Volumes/WD Backup/Crooked/Crooked_Lake_2023_cat2025"
#output_folder = "/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Crook_cats_processed_2023"

output_folder = '/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Crook_cats_processed_2023/3_Column_Files'

# Customize which columns to drop and rename
drop_columns = ["Battery", "Q", "UTC_Date_&_Time", "Unix Timestamp", "Dissolved Oxygen Saturation"]  # Example: Remove 'Battery' and 'Q'
rename_columns = {"Eastern Standard Time": "EST_DateTime", "Dissolved Oxygen": "DO_mg_L", "Temperature": "Temp_C"}  # Example: Rename columns

# Process the entire folder
process_folder(input_folder, output_folder, drop_columns, rename_columns)





test = pd.read_csv("/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Crook_cats_processed_2023/3_Column_Files/cat_004373.csv")

print(test.columns)


