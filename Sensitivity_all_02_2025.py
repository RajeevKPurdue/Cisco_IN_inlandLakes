#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:33:52 2025

Refactoring code for filename robustness- underscore split is too fragile

First testing/validating functions - should reference and test pycharm scripts as well

Listing packages and functions with data together during validation in code

@author: rajeevkumar
"""

" === Testing if saving and/or reading array functions truncate the data  ==="



#%%
"======================= GRP (Temp, DO) summary dataframe and postive surface testing =============== "
import os
import numpy as np
import pandas as pd
#import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator  # For adaptive tick formatting
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import itertools
import matplotlib.pyplot as plt
import re

"packages load cell"

#%%
"======================= GRP (Temp, DO) summary dataframe and postive surface testing =============== "


# Define the function for calculating fish growth
def calculate_growth_fdos(DO_array, Temp_array, mass_coeff, P_coeff, AD_arr, slope_val, intercept_val):
    DOcrit = slope_val * Temp_array + intercept_val # vary - this is the highest so far 
    fDO = DO_array / DOcrit
    fDO = np.minimum(fDO, 1.0)  # Cap values greater than 1 to 1
    mass = np.full_like(Temp_array, 1 * mass_coeff)  # mass sensitivity test
    t_lethal = Temp_array         #np.minimum(Temp_array, 22.9999999) # cap values to asymptote where all normoxia is lethal
    DO_lethal = 0.4 + 0.000006*(np.exp(0.59*t_lethal))
    fdo_lethal = (DO_array- DO_lethal)/ DOcrit
    # clipping fdo values for min = 0 and max = 1
    fDO = np.clip(fDO, 0,1)
    fdo_lethal = np.clip(fdo_lethal, 0,1)

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
    AD[-1] = AD_benthos  # Now correctly modifying the last row

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
    return GRP, GRP_lethal #, fDO, fdo_lethal


# Define mass and P coefficient values to analyze
mass_coefficients = np.array([200, 400, 600])  # Example mass values
P_coefficients = np.array([0.2, 0.4, 0.6, 0.8])  # Example P values
AD_values = np.array([2000])  # AD values
slope_vals = np.array([0.168, 0.138])
intercept_vals = np.array([1.63, 2.09])

# Create T, DO grid 

"""
# Define the parameter ranges
DO_array = np.linspace(0, 10, 100)  # Dissolved oxygen range (mg/L)
print(f"DO shape {DO_array.shape}")
Temp_array = np.linspace(5, 30, 100)  # Temperature range (°C)
print(f"Temp shape {Temp_array.shape}")

#DO_grid, Temp_grid = np.meshgrid(DO_array, Temp_array)

# Run 1 used above
"""
# Original
#DO_array = np.linspace(0, 10, 50)
#Temp_array = np.linspace(5, 30, 50)

# Even
DO_array = np.linspace(0, 10, 49)
Temp_array = np.linspace(5, 30, 49)

Temp_grid, DO_grid = np.meshgrid(Temp_array, DO_array)

# run 2 for all combinations == nightmare but needed (truncating array steps in half)
T_DO_pairs = np.array(list(itertools.product(Temp_array, DO_array)))  # (10000, 2)
T_arr = T_DO_pairs[:, 0]  # All T values
DO_arr = T_DO_pairs[:, 1]  # All DO values


# Run Model iterations and load by parameter keys into dictionaries 
GRP_results ={}
GRP_lethal_results = {}

for mass_coeff in mass_coefficients:
    for P_coeff in P_coefficients:
        for AD_arr in AD_values:
            for slope in slope_vals:
                for intercept in intercept_vals:
                    # Compute GRP and GRP_lethal
                    GRP_array, GRP_lethal_array = calculate_growth_fdos(DO_arr, T_arr, mass_coeff, P_coeff, AD_arr, slope, intercept)

                    # Generate dictionary keys
                    key = f"GRP_P{P_coeff}_mass{mass_coeff}_slope{slope}_intercept{intercept}"
                    key_lethal = f"GRP_lethal_P{P_coeff}_mass{mass_coeff}_slope{slope}_intercept{intercept}"

                    # Store in dictionaries
                    GRP_results[key] = GRP_array
                    GRP_lethal_results[key_lethal] = GRP_lethal_array



"==== Commmenting out data frame creation"

# create data frame with T,DO, GRP, GRP_lethal arrays as col values, keys as col names 

df_grpsyn = pd.DataFrame({**GRP_results, **GRP_lethal_results})
df_grpfull = pd.DataFrame({**GRP_results, **GRP_lethal_results})

df_grp = pd.DataFrame({**GRP_results, **GRP_lethal_results})

# Add as new columns with user-defined names
df_grp['Temp'] = T_arr
df_grp['DO'] = DO_arr


# saving dataframe for later use 
df_grp.to_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRPmodel_sensitivity_dfs/grp_variants1D_rawvals_rawcols.csv', index = False)

# check 1D binary array... > synthetic grp eval
"All GRP runs synthetic test should be saved check dir "
#%%

"=========== Testing regex for GRP filenames which are also Table column names =============="
import re

# Example filename with both GRP and GRP_lethal formats
#filenames = [
#    "GRP_P0.2_mass200_slope0.168_intercept1.63",
#    "GRP_lethal_P0.4_mass400_slope0.168_intercept1.63"
#]

#pd.set_option('display.max_columns', None)

# Updated regex pattern to extract type (GRP vs. GRP_lethal)
pattern = r"(?P<type>GRP|GRP_lethal)_P(?P<P>[0-9.]+)_mass(?P<mass>[0-9]+)_slope(?P<slope>[0-9.]+)_intercept(?P<intercept>[0-9.]+)"

# Extract values into a structured format
parsed_data = [re.match(pattern, name).groupdict() for name in filenames]

# Convert to DataFrame
import pandas as pd
df_params = pd.DataFrame(parsed_data)
df_params["original_name"] = filenames  # Add original name for reference

print(df_params)



#%%

"================== CREATING SUMMARY TABLES - REAL (filename) and SYNTH. (colname) data  =============================="

# load grp_full df
#dfgrp = pd.to_csv('')
df_grpfull = df_grp

tempcol = df_grp.filter(like = "Temp")
docol = df_grp.filter(like = "DO")

grp_cols = df_grpfull.filter(like="GRP_").columns  # All GRP model iteration columns

#df_params["Model"] = [name for name in filenames if re.match(pattern, name)]  # Add original column names

# Initialize list to store summary statistics including Temp and DO values at Min/Max GRP
summary_stats = []

# Loop through each GRP model column
for col in df_grpfull.columns:
    match = re.match(pattern, col)  # Match regex to extract parameters
    if match:
        params = match.groupdict()  # Extract parameter dictionary

        grp_values = df_grpfull[col]

        # Find min and max GRP values and corresponding Temp and DO
        min_idx = grp_values.idxmin()
        max_idx = grp_values.idxmax()

        summary_stats.append({
            'Model': col,
            'Type': params['type'],
            'P': params['P'],
            'Mass': params['mass'],
            'Slope': params['slope'],
            'Intercept': params['intercept'],
            'Min GRP': grp_values.min(),
            'Temp Min GRP': df_grpfull.loc[min_idx, "Temp"],
            'DO Min GRP': df_grpfull.loc[min_idx, "DO"],
            'Max GRP': grp_values.max(),
            'Temp Max GRP': df_grpfull.loc[max_idx, "Temp"],
            'DO Max GRP': df_grpfull.loc[max_idx, "DO"],
            'Mean GRP': grp_values.mean(),
            'Median GRP': grp_values.median(),
            'Std Dev GRP': grp_values.std()
        })

# Convert list to DataFrame
summary_df = pd.DataFrame(summary_stats)

# Display the summary DataFrame
print(summary_df)

# Save to CSV
#summary_df.to_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRPmodel_sensitivity_dfs/model_summary_statistics_grpfullsyn.csv', index=False)

# Display as a table
#print(summary_df)

min_do_max_grp_row = summary_df.loc[summary_df["DO Max GRP"].idxmin()]
print(min_do_max_grp_row)

max_temp_max_grp_row = summary_df.loc[summary_df["Temp Max GRP"].idxmax()]
print(max_temp_max_grp_row)

crit_repmod = (0.168 * 16.734694) + 1.63
fdorepmod = 4.489796/(crit_repmod)

print(crit_repmod)
print(fdorepmod)
# Weird mod where: (0.2,200,.138,1.63)
#Temp Max GRP                                     16.22449
#DO Max GRP                                       3.877551





#%%

"================= plotting fDO jac(s) ============================"

DO_array = np.linspace(0, 10, 50)
Temp_array = np.linspace(5, 30, 50)
# truncated and untruncated at jac ctm
t_ltl = Temp_array
t_trunc = np.minimum(Temp_array, 22.9999999)  # Cap values at 22.9999999 # cap values to asymptote where all normoxia is lethal

DO_lethal = 0.4 + 0.000006*(np.exp(0.59*t_ltl))
DO_lethal_trunc = 0.4 + 0.000006*(np.exp(0.59*t_trunc))

# slopes and intercepts
slopes = [0.138, 0.168]
intercepts = [1.63,2.09]
slope_ints = np.array([(0.138,1.63), (0.138,2.09), (0.168,1.63), (0.138,2.09)])

# Initialize DOcrit array for all combinations
DO_crits = np.zeros((len(t_ltl), len(slope_ints)))
print(DO_crits.shape)

for i in range(DO_crits.shape[0]):
    for j in range(DO_crits.shape[1]):
        DO_crits[i,j] = (slope_ints[j][0] * t_ltl[i]) + slope_ints[j][1]


# Compute fdo_ltl for each slope-intercept combination
fdo_ltl = np.zeros_like(DO_crits)

for i in range(DO_crits.shape[0]):
    for j in range(DO_crits.shape[1]):
        fdo_ltl[:,j] = (DO_array[:] - DO_lethal[:])/(DO_crits[:,j])

fdo_ltl = np.clip(fdo_ltl, 0,1)
fdo_ltl = np.clip(fdo_ltl_trunc, 0,1)
#print(fdo_ltl)
        
# Print shape to verify
print("DO_crits shape:", DO_crits.shape)
print("fdo_ltl shape:", fdo_ltl.shape)



# Create figure
plt.figure(figsize=(8, 6))

# Plot each iteration as a separate line
for j in range(fdo_ltl.shape[1]):  # Iterate over columns (4 iterations)
    plt.plot(Temp_array, fdo_ltl[:, j], label=f"Slope: {slope_ints[j][0]}, Intercept: {slope_ints[j][1]}")

# Labels and title
plt.xlabel("Temperature (°C)")
plt.ylabel("fdo_ltl")
plt.title("fdo_ltl vs. Temperature for Different DOcrit Iterations")
plt.legend()  # Show legend with custom labels

# Show plot
plt.show()


# contour plots 

# Create a meshgrid for contour plots
Temp_grid, DO_grid = np.meshgrid(Temp_array, DO_array)

# Plot fdo_lethal comparison as a contour plot (Truncated, Clipped)
plt.figure(figsize=(8, 6))
contour1 = plt.contourf(Temp_grid, DO_grid, fdo_lethal_truncated, levels=np.linspace(0, 1, 20), cmap="coolwarm")
plt.colorbar(label="fdo_lethal Value (0 to 1)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Dissolved Oxygen (mg/L)")
plt.title("Contour Plot of fdo_lethal (Truncated at 23°C, Clipped)")
plt.show()

# Plot fdo_lethal comparison as a contour plot (Untruncated, Clipped)
plt.figure(figsize=(8, 6))
contour2 = plt.contourf(Temp_grid, DO_grid, fdo_lethal_untruncated, levels=np.linspace(0, 1, 20), cmap="coolwarm")
plt.colorbar(label="fdo_lethal Value (0 to 1)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Dissolved Oxygen (mg/L)")
plt.title("Contour Plot of fdo_lethal (Untruncated, Clipped)")
plt.show()

#%%

"================= FDO EVAL================= FDO EVAL================= FDO EVAL================= FDO EVAL"

"++ Lineplot, T,DO + fDO combination df, meshgrid contour, for fDO lethal comparison  ++"


"------------------ Trying my own way instead of Chat - Not bothering with 'trunc' either ---------------"

def calculate_fdos(DO_array, Temp_array, slope_val, intercept_val):
    
    DOcrit = slope_val * Temp_array + intercept_val # slope_ints[i][0]  + slope_ints[i][1]  vary - this is the highest so far 
    
    fDO = DO_array / DOcrit
    fDO = np.minimum(fDO, 1.0)  # Cap values greater than 1 to 1
    #mass = np.full_like(Temp_array, 1 * mass_coeff)  # mass sensitivity test
    
    t_lethal = Temp_array         #np.minimum(Temp_array, 22.9999999) # cap values to asymptote where all normoxia is lethal
    
    DO_lethal = 0.4 + 0.000006*(np.exp(0.59*t_lethal))
    fdo_lethal = (DO_array- DO_lethal)/ DOcrit
    
    # clipping fdo values for min = 0 and max = 1
    fDO = np.clip(fDO, 0,1)
    fdo_lethal = np.clip(fdo_lethal, 0,1)
    return fDO, fdo_lethal


# Define the parameter ranges
#DO_array = np.linspace(0, 7, 8)  # Dissolved oxygen range (mg/L)
#Temp_array = np.linspace(5, 30, 50)  # Temperature range (°C)
slope_vals = np.array([0.168, 0.138])
intercept_vals = np.array([1.63, 2.09])

slope_ints = np.array([(0.138,1.63), (0.138,2.09), (0.168,1.63), (0.138,2.09)])

"-------'1D' dictionary"

# Define the parameter ranges

# run 2 for all combinations == nightmare but needed (truncating array steps in half)
T_DO_pairs = np.array(list(itertools.product(Temp_array, DO_array)))  # (10000, 2)
T_arr = T_DO_pairs[:, 0]  # All T values
DO_arr = T_DO_pairs[:, 1]  # All DO values


# Run Model iterations and load by parameter keys into dictionaries 
fDO1D_dict ={}
fdolethal1D_dict = {}

for slope in slope_vals:
    for intercept in intercept_vals:
        # Compute GRP and GRP_lethal
        fDO_array, fDO_lethal_array = calculate_fdos(DO_arr, T_arr, slope, intercept)

        # Generate dictionary keys
        key = f"fDO_slope{slope}_intercept{intercept}"
        key_lethal = f"fDO_lethal_slope{slope}_intercept{intercept}"

        # Store in dictionaries
        fDO1D_dict[key] = fDO_array
        fdolethal1D_dict[key_lethal] = fDO_lethal_array
        

" ------------ Creating & Saving dataframe -------------"
df_fdos = pd.DataFrame({**fDO1D_dict, **fdolethal1D_dict})

# Add as new columns with user-defined names
df_fdos['Temp'] = T_arr
df_fdos['DO'] = DO_arr


# saving dataframe for later use 
#df_fdos.to_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRPmodel_sensitivity_dfs/fdos_all_1D.csv', index = False)

#df_fdos_simple.to_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRPmodel_sensitivity_dfs/fdos_1by1TDO_1D.csv', index = False)




"------- Array Based -------- LINEPLOT -"
# match the dims - below example from stack exchange - could make a simple function (array, repeats, axis)... if this is needed for many later 
#b = np.repeat(a[:, :, np.newaxis], 3, axis=2)



DO_arrmod = np.repeat(DO_array[:, np.newaxis],4,axis=1)
Temp_arrmod = np.repeat(Temp_array[:, np.newaxis],4,axis=1)

print(DO_arrmod.shape)

# Create storage arrays for iterations 1D
fdo_1D = np.zeros((50,4))
fdo_ltl_1D = np.zeros((50,4))

# calulate 1D arrays fDO and fDO_lethal
#for slope in slope_vals:
    #for intercept in intercept_vals:
for i in range(fdo_1D.shape[1]):
    fdo_1D[:,i], fdo_ltl_1D[:,i] = calculate_fdos(DO_arrmod[:,i], Temp_arrmod[:,i], slope_ints[i][0], slope_ints[i][1])



def plotfdo_lines(fdoarr, var_arr, x_label, y_label, title):
    # Create figure 1D
    plt.figure(figsize=(8, 6))

    # Plot each iteration as a separate line
    for j in range(fdoarr.shape[1]):  # Iterate over columns (4 iterations)
        plt.plot(var_arr, fdoarr[:, j], label=f"Slope: {slope_ints[j][0]}, Intercept: {slope_ints[j][1]}")
        #plt.plot(DO_array, fdoarr[:, j], label=f"Slope: {slope_ints[j][0]}, Intercept: {slope_ints[j][1]}") 
    # Labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()  # Show legend with custom labels

    # Show plot
    plt.show()
    

plotfdo_lines(fdo_1D, Temp_array, x_label = 'Temp', y_label = 'fDO value', title = 'fDO vs Temp for DOcrit iterations')
plotfdo_lines(fdo_1D, DO_array, x_label = 'DO', y_label = 'fDO value', title = 'fDO vs DO for DOcrit iterations')
plotfdo_lines(fdo_ltl_1D, Temp_array, x_label = 'Temp', y_label = 'fDO_lethal value', title = 'fDO_lethal vs Temp for DOcrit iterations')
plotfdo_lines(fdo_ltl_1D, DO_array, x_label = 'DO', y_label = 'fDO_lethal value', title = 'fDO_lethal vs DO for DOcrit iterations')

plots = list(fdo_1D)



" ------- ARRAY and DICTIONARY BASED ------- CONTOUR INDIVDUAL AND PAR/SUB PLOTS------"
# Calc 2D arrays fDO and fDO_lethal

slope_vals = np.array([0.168, 0.138])
intercept_vals = np.array([1.63, 2.09])

slope_ints = np.array([(0.138,1.63), (0.138,2.09), (0.168,1.63), (0.138,2.09)])

DO_array = np.linspace(0, 10, 49)
Temp_array = np.linspace(5, 30, 49)

# Make grid
DO_grid, Temp_grid = np.meshgrid(DO_array, Temp_array)


fDO_results = {}
fDO_lethal_results = {}
# calulate 1D arrays fDO and fDO_lethal
for slope in slope_vals:
    for intercept in intercept_vals:
        
        fDO_array, fDO_ltl_array = calculate_fdos(DO_grid, Temp_grid, slope, intercept)
        # Generate dictionary keys
        key = f"fDO_slope{slope}_intercept{intercept}"
        key_lethal = f"fDO_lethal_slope{slope}_intercept{intercept}"

        # Store in dictionaries
        fDO_results[key] = fDO_array
        fDO_lethal_results[key_lethal] = fDO_ltl_array


# Define the function for calculating fish growth
def calculate_growth_fdos_2D(DO_array, Temp_array, mass_coeff, P_coeff, AD_arr, slope_val, intercept_val):
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
    AD_benthos = 2000  # Schaeffer et al. 1999 - Arend 2011

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




# Run Model iterations and load by parameter keys into dictionaries 
GRP_results ={}
GRP_lethal_results = {}

for mass_coeff in mass_coefficients:
    for P_coeff in P_coefficients:
        for AD_arr in AD_values:
            for slope in slope_vals:
                for intercept in intercept_vals:
                    # Compute GRP and GRP_lethal
                    GRP_array, GRP_lethal_array = calculate_growth_fdos_2D(DO_grid, Temp_grid, mass_coeff, P_coeff, AD_arr, slope, intercept)

                    # Generate dictionary keys
                    key = f"GRP_P{P_coeff}_mass{mass_coeff}_slope{slope}_intercept{intercept}"
                    key_lethal = f"GRP_lethal_P{P_coeff}_mass{mass_coeff}_slope{slope}_intercept{intercept}"

                    # Store in dictionaries
                    GRP_results[key] = GRP_array
                    GRP_lethal_results[key_lethal] = GRP_lethal_array
        
        
"---------- CONTOUR PLOTS FROM DICT _ INDIVIDUAL THEN SUBPLOTS --------------"

def contour_fdos(Temp_grid, DO_grid, fdogrid, cmap = 'seismic', cbar_label = 'label', title = 'title'):
    plt.figure(figsize=(8, 6))
    contour1 = plt.contourf(Temp_grid, DO_grid, fdogrid, levels=np.linspace(0, 1, 20), cmap="coolwarm")
    plt.colorbar(label="fdo_lethal Value (0 to 1)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Dissolved Oxygen (mg/L)")
    plt.title(title)
    plt.show()



for key, value in fDO_results.items():
    contour_fdos(Temp_grid, DO_grid, value, cmap = 'seismic', cbar_label = f'{key} value', title = f'countour of {key}')


#for key, value in fDO_lethal_results.items():
#    contour_fdos(Temp_grid, DO_grid, value, cmap = 'seismic', cbar_label = f'{key} value', title = f'countour of {key}')


######### PLT SUBPLOTS ###############

# Define the contour plotting function
def contour_fdos1(ax, Temp_grid, DO_grid, fdogrid, cmap='seismic', cbar_label='label', title='title'):
    contour1 = ax.contourf(Temp_grid, DO_grid, fdogrid, levels=np.linspace(0, 1, 20), cmap=cmap)
    cbar = plt.colorbar(contour1, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Dissolved Oxygen (mg/L)")
    ax.set_title(title)

# Select the first four keys to plot
keys_to_plot = list(fDO_results.keys())[:4]  # Get the first 4 keys

keys_to_plot1 = ['fDO_slope0.168_intercept1.63','fDO_slope0.138_intercept2.09' ]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots
# Loop through the selected keys and corresponding axes
for ax, key in zip(axes.flat, keys_to_plot):
    contour_fdos1(ax, Temp_grid, DO_grid, fDO_results[key], cmap='seismic', 
                 cbar_label='Value', title=f'{key}'.replace('_',', '))

plt.tight_layout()  # Adjust layout for better spacing
plt.show()

# Select the first four keys to plot
keys_to_plot_lethal = list(fDO_lethal_results.keys())[:4]  # Get the first 4 keys


fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots
# Loop through the selected keys and corresponding axes
for ax, key in zip(axes.flat, keys_to_plot_lethal):
    contour_fdos1(ax, Temp_grid, DO_grid, fDO_lethal_results[key], cmap='seismic', 
                 cbar_label='Value', title=f'{key}'.replace('_',', '))

plt.tight_layout()  # Adjust layout for better spacing
plt.show()




### Checking fDO for failing 2022 pturn odd pattterns - NOne where fDO_lethal is more lenient

# mask is where fDO_lethal is greater which is a smaller penalty on C
mask = fDO_lethal_results['fDO_lethal_slope0.168_intercept1.63'] > fDO_results['fDO_slope0.168_intercept1.63']

check_dict = {'arend': fDO_lethal_results['fDO_lethal_slope0.168_intercept1.63'], 'jac': fDO_results['fDO_slope0.168_intercept1.63']}

for key, value in check_dict.items():
    # Create a new array that has the same shape as value; values where the mask is False become NaN.
    masked_value = np.where(mask, value, np.nan)
    contour_fdos(Temp_grid, DO_grid, masked_value,
                 cmap='seismic', 
                 cbar_label=f'{key} value', 
                 title=f'{key} indices where fDO greater than fDO_lethal')


#33333 Checking GRP for reason above
# mask is where GRP is greater than GRP_lethal
maskgrp = GRP_lethal_results["GRP_lethal_P0.4_mass200_slope0.168_intercept1.63"] > GRP_results["GRP_P0.4_mass200_slope0.168_intercept1.63"]

grp_fail_dict = {
    'grp_fail': GRP_results["GRP_P0.4_mass200_slope0.168_intercept1.63"],
    'grp_lethalfail': GRP_lethal_results["GRP_lethal_P0.4_mass200_slope0.168_intercept1.63"]}


for key, value in grp_fail_dict.items():
    # Create a new array that has the same shape as value; values where the mask is False become NaN.
    #masked_value = np.where(maskgrp, value, np.nan)
    masked_value = (maskgrp)
    contour_fdos(Temp_grid, DO_grid, masked_value,
                 cmap='seismic', 
                 cbar_label=f'{key} value', 
                 title=f'{key} indices where GRP greater than GRP_lethal')





# mask is where GRP is greater than GRP_lethal
maskgrp = GRP_results["GRP_P0.4_mass200_slope0.168_intercept1.63"] >=  GRP_lethal_results["GRP_lethal_P0.4_mass200_slope0.168_intercept1.63"]



masked_value = (maskgrp)
contour_fdos(Temp_grid, DO_grid, masked_value,
              cmap='seismic', 
              cbar_label=f'{key} value', 
              title=f'{key} indices where GRP greater than GRP_lethal')



# mask is where GRP is greater than GRP_lethal
maskgrp = GRP_results["GRP_P0.4_mass200_slope0.168_intercept1.63"] >=  GRP_lethal_results["GRP_lethal_P0.4_mass200_slope0.168_intercept1.63"]


plt.figure(figsize=(8, 6))
contour1 = plt.contour(Temp_grid, DO_grid, maskgrp, levels=(1), cmap="coolwarm")
plt.colorbar(label="where GRP > GRP_lethal (False, True)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Dissolved Oxygen (mg/L)")
plt.title("title")
plt.show()


"""
for k in keys_to_plot_lethal:
    for key in keys_to_plot:
        where = np.where(fDO_lethal_results[k] < fDO_results[key])
"""



#%%
"===========================Plotting Code- Dataframes =========================="


"---------- Basic Single Scatter Plot Test ----------------"

# testing single plot

def plot_data(df, column_name):
    """Plot GRP or GRP_lethal data dynamically."""
    match = re.match(pattern, column_name)
    if match:
        params = match.groupdict()
        title = f"{params['type']} Model: P={params['P']}, Mass={params['mass']}, Slope={params['slope']}, Intercept={params['intercept']}"
    else:
        title = column_name  # Fallback to original name

    plt.figure(figsize=(6, 4))
    plt.plot(df["Temp"], df[column_name], label=column_name)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("GRP Value")
    plt.title(title)
    plt.legend()
    plt.show()

# Example usage
for col in df_grp.columns:
    plot_data(df_grp, col)



" ---------------- Scatter plots  ALL SEPARATE FIGURES --------------"

# Plot scatter plots for each model
for model in df_grp.columns:
    if model.startswith('GRP_'):  # Only consider GRP and GRP_lethal columns
        plt.figure(figsize=(8, 6))
        positive_mask = df_grp[model] > 0
        scatter = plt.scatter(
            df_grpfull.loc[positive_mask, 'Temp'],  # T on x-axis
            df_grpfull.loc[positive_mask, 'DO'],    # DO on y-axis
            c=df_grpfull.loc[positive_mask, model], # GRP values for color
            cmap='viridis', alpha=0.6
        )
        
        # Add colorbar with 3 decimal places
        cbar = plt.colorbar(scatter, format='%.3f')
        cbar.set_label('GRP Value')
        
        # Set axis labels
        plt.xlabel('Temperature (T)')
        plt.ylabel('Dissolved Oxygen (DO)')
        
        # Set ticks
        plt.xticks(np.linspace(df_grpfull['Temp'].min(), df_grpfull['Temp'].max(), 6))  # 6 ticks for T
        plt.yticks(np.linspace(df_grpfull['DO'].min(), df_grpfull['DO'].max(), 5))      # 5 ticks for DO
        
        # Format tick labels to 2 decimal places
        plt.gca().xaxis.set_major_formatter('{x:.2f}')
        plt.gca().yaxis.set_major_formatter('{x:.2f}')
        
        plt.title(f'Positive (T, DO) Pairs for {model}')
        plt.show()


"------------- Box Plot ALL- ONE FIGURE ---------------------"

# Melt the DataFrame for box plot
df_melted = df_grp.melt(id_vars=['Temp', 'DO'], value_vars=df_grp.columns[df_grp.columns.str.startswith('GRP_')],
                            var_name='Model', value_name='GRP')

# Melt the DataFrame for box plot
df_melted = df_grp.melt(
    id_vars=['Temp', 'DO'],
    value_vars=df_grp.columns[df_grp.columns.str.startswith('GRP_')],
    var_name='Model', value_name='GRP'
)

# Plot box plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_melted, x='Model', y='GRP')
# Format GRP values to 3 decimal places
plt.gca().yaxis.set_major_formatter('{x:.3f}')
# Set axis labels
plt.title('Distribution of GRP Values for Each Model')
plt.xlabel('Model')
plt.ylabel('GRP Value')
plt.xticks(rotation=90)
plt.show()


"--------------- Violin Plots - NEED TO SEE AGAIN --------------"


# Plot violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=df_melted, x='Model', y='GRP')
# Format GRP values to 3 decimal places
plt.gca().yaxis.set_major_formatter('{x:.3f}')
# Set axis labels
plt.title('Distribution and Density of GRP Values for Each Model')
plt.xlabel('Model')
plt.ylabel('GRP Value')
plt.xticks(rotation=90)
plt.show()



" ------------ Scatter Plots ALL - 1 Figure with 98 Panels *** Fucked indexing at the moment  ------------"

# Get the list of models
models = [col for col in df_grpfull.columns if col.startswith('GRP_')]

# Determine the number of rows and columns for subplots
n_models = len(models)
n_cols = 3  # Number of columns in the subplot grid
n_rows = (n_models + n_cols - 1) // n_cols  # Calculate number of rows

# Create a figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
fig.suptitle('Positive (T, DO) Pairs for Each Model', fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot scatter plots for each model
for i, model in enumerate(models):
    ax = axes[i]
    positive_mask = df_grpfull[model] > 0
    scatter = ax.scatter(
        df_grpfull.loc[positive_mask, 'Temp'],  # T on x-axis
        df_grpfull.loc[positive_mask, 'DO'],    # DO on y-axis
        c=df_grpfull.loc[positive_mask, model], # GRP values for color
        cmap='viridis', alpha=0.6
    )
    
    # Add colorbar with 3 decimal places
    cbar = plt.colorbar(scatter, ax=ax, format='%.3f')
    cbar.set_label('GRP Value')
    
    # Set axis labels
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Dissolved Oxygen (DO)')
    
    # Set ticks
    ax.set_xticks(np.linspace(df_grpfull['Temp'].min(), df_grpfull['Temp'].max(), 6))  # 6 ticks for T
    ax.set_yticks(np.linspace(df_grpfull['DO'].min(), df_grpfull['DO'].max(), 5))      # 5 ticks for DO
    
    # Format tick labels to 2 decimal places
    ax.xaxis.set_major_formatter('{x:.2f}')
    ax.yaxis.set_major_formatter('{x:.2f}')
    
    # Set title (extract parameter values from model name)
    title_parts = model.split('_')
    title = ', '.join([f"{part.split('=')[0]} = {part.split('=')[1]}" for part in title_parts[1:]])
    ax.set_title(title)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



"------------ Heatmaps ALL - Separate Figures ------------------------"

# Reshape GRP values into a grid
T_grid = df_grpfull['Temp'].values.reshape(len(Temp_array), len(DO_array))
DO_grid = df_grpfull['DO'].values.reshape(len(Temp_array), len(DO_array))

# Plot heatmaps for each model
for model in df_grpfull.columns:
    if model.startswith('GRP_'):  # Only consider GRP and GRP_lethal columns
        grp_grid = df_grpfull[model].values.reshape(len(Temp_array), len(DO_array))
        plt.figure(figsize=(10, 6))
        
        # Create heatmap
        heatmap = sns.heatmap(
            grp_grid,
            xticklabels=np.round(np.linspace(DO_array.min(), DO_array.max(), 5), 2),  # 5 ticks for DO
            yticklabels=np.round(np.linspace(Temp_array.min(), Temp_array.max(), 6), 2),  # 6 ticks for T
            cmap='viridis',
            annot=True, fmt='.3f'  # 3 decimal places for GRP values
        )
        
        # Set axis labels
        plt.xlabel('Dissolved Oxygen (DO)')
        plt.ylabel('Temperature (T)')
        
        plt.title(f'GRP Values for {model}')
        plt.show()




#%%
" ====== getting T,DO that correspond with postive GRP from each model iteration ==="

# Extract GRP columns dynamically
grp_cols = df_grpfull.filter(like="GRP_").columns  # All GRP model iteration columns

# List to store data
corner_data = []

# Loop through each model iteration
for grp_col in grp_cols:
    # Filter only positive values for the current model iteration
    positive_df = df_grpfull[df_grpfull[grp_col] >0]   #>= np.abs(0.00001)]

    if not positive_df.empty:  # Ensure there are positive values
        # Find the row where the smallest positive GRP value occurs
        min_grp_row = positive_df.loc[positive_df[grp_col].idxmin()]

        # Identify the four extreme corners based on T and DO values at this point
        min_temp = min_grp_row["Temp"]
        max_temp = min_grp_row["Temp"]
        min_do = min_grp_row["DO"]
        max_do = min_grp_row["DO"]

        # Store corner points for this model iteration
        corners = {
            "(min_T, min_DO)": df_grpfull[(df_grpfull["Temp"] == min_temp) & (df_grpfull["DO"] == min_do)],
            "(max_T, min_DO)": df_grpfull[(df_grpfull["Temp"] == max_temp) & (df_grpfull["DO"] == min_do)],
            "(max_T, max_DO)": df_grpfull[(df_grpfull["Temp"] == max_temp) & (df_grpfull["DO"] == max_do)],
            "(min_T, max_DO)": df_grpfull[(df_grpfull["Temp"] == min_temp) & (df_grpfull["DO"] == max_do)],
        }

        for corner_name, df_corner in corners.items():
            if not df_corner.empty:
                for _, row in df_corner.iterrows():
                    corner_data.append({
                        "Model": grp_col,
                        "Corner": corner_name,
                        "Temp": row["Temp"],
                        "DO": row["DO"],
                        "GRP": row[grp_col]  # Extract GRP value for this model
                    })

# Convert list to DataFrame
df_corners_min_pos = pd.DataFrame(corner_data)
#df_corners_min_pos.to_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/1D_binaryarray_dfs/synthethic_grp_eval/grp_varianats_allcombinationsRangeTDO_plswork.csv', index = False)



def find_boundary_ranges(df, model_column, tempcol, docol):
    positive_values = df[df[model_column] > 0]
    if positive_values.empty:
        return None
    
    # Find the minimum and maximum values of temp and DO for the positive model
    min_temp = positive_values[tempcol].min()
    max_temp = positive_values[tempcol].max()
    min_DO = positive_values[docol].min()
    max_DO = positive_values[docol].max()
    
    # Define the four corners
    corners = {
        'min_temp_min_DO': (min_temp, min_DO),  # (min_temp, min_DO)
        'max_temp_min_DO': (max_temp, min_DO),  # (max_temp, min_DO)
        'max_temp_max_DO': (max_temp, max_DO),  # (max_temp, max_DO)
        'min_temp_max_DO': (min_temp, max_DO)   # (min_temp, max_DO)
    }
    
    return corners

# Assuming df_grpfull is your main DataFrame
tempcol = df_grpfull.filter(like="Temp").columns[0]  # Take the first matching column
docol = df_grpfull.filter(like="DO").columns[0]      # Take the first matching column

# Ensure df_corners_min_pos is defined and contains the necessary columns
# For example, if df_corners_min_pos is a subset of df_grpfull:
df_corners_min_pos = df_grpfull.copy()  # Replace with your actual logic

# Iterate through each model column and find boundary ranges
boundary_ranges = {}
for col in df_corners_min_pos.columns:
    if col.startswith('GRP_'):
        ranges = find_boundary_ranges(df_corners_min_pos, col, tempcol, docol)
        if ranges:
            boundary_ranges[col] = ranges

# Print the boundary ranges
for model, corners in boundary_ranges.items():
    print(f"Model: {model}")
    print(f"  (min_temp, min_DO): {corners['min_temp_min_DO']}")
    print(f"  (max_temp, min_DO): {corners['max_temp_min_DO']}")
    print(f"  (max_temp, max_DO): {corners['max_temp_max_DO']}")
    print(f"  (min_temp, max_DO): {corners['min_temp_max_DO']}")
    print()
#%%

"================= MULTIMETHOD EQUATION FITTING +++++++++++++++++"

df = df_grp.copy()

# Filter positive regions for each model
positive_regions = {}
for col in df.columns:
    if col.startswith('GRP_'):
        positive_regions[col] = df[df[col] > 0][['Temp', 'DO']]




        
# Extract boundary points for each model
boundary_points = {}
for model, data in positive_regions.items():
    points = data[['Temp', 'DO']].values
    boundary_points[model] = extract_boundary_points(points)
    
pos_hull_areas = pd.DataFrame(positive_region_sizes.items(), columns=["Model", "Positive_Region_Size"])
pos_hull_areas.to_csv("/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRPmodel_sensitivity_dfs/pos_region_areas.csv", index=False)



"--------- Worry bout these in another life ---------------------"
"---------- Consider fitting a spline of T, DO partial derivs"
    
"trying power function - resembles one and is one (see Jacobson et al 2011)"

from scipy.spatial import ConvexHull

# Function to extract boundary points using Convex Hull
def extract_boundary_points(points):
    hull = ConvexHull(points)
    boundary_points = points[hull.vertices]
    return boundary_points

# Define the power function
def power_function(T, a, b, c):
    return a * (T ** b) + c


# Fit a polynomial function of the form DO = a * Temp^b + c
#def growth_zero_func(T, a, b, c):
#    return a * T**b + c

# Fit the curve to the extracted contour points
#params, _ = curve_fit(growth_zero_func, Temp_contour, DO_contour, p0=[0.1, 1, 2])

# Get the explicit equation
#a_fit, b_fit, c_fit = params
#explicit_equation = f"DO = {a_fit:.3f} * Temp^{b_fit:.3f} + {c_fit:.3f}"

# Fit power function to boundary points
power_fits = {}
for model, points in boundary_points.items():
    x = points[:, 0]  # T values
    y = points[:, 1]  # DO values
    # Fit the power function
    params, _ = curve_fit(power_function, x, y, p0=[0.1, 1, 2])  # p0 is the initial guess for a and b
    power_fits[model] = params  # Store the fitted parameters (a, b)

# Generate predictions for plotting
x_range = np.linspace(df['Temp'].min(), df['Temp'].max(), 100)  # Range of T values
predictions = {}
for model, params in power_fits.items():
    a, b = params
    y_pred = power_function(x_range, a, b)
    predictions[model] = (x_range, y_pred)
    
    
    
    
    
    
    
    
# Plot power function boundaries
for model, (x_range, y_pred) in predictions.items():
    plt.figure()
    # Plot positive regions
    plt.scatter(positive_regions[model]['Temp'], positive_regions[model]['DO'], 
                label=f'{model} Positive Regions', color='blue', alpha=0.6)
    # Plot power function boundary
    plt.plot(x_range, y_pred, color='red', label=f'{model} Power Function Boundary')
    # Shade the positive region
    plt.fill_between(x_range, y_pred, df['DO'].max(), color='green', alpha=0.2, label='Positive Region')
    # Add labels and title
    plt.xlabel('Temperature (T)')
    plt.ylabel('Dissolved Oxygen (DO)')
    plt.title(f'Power Function Boundary for {model}')
    plt.legend()
    plt.grid(True)
    plt.ylim(df['DO'].min(), df['DO'].max())  # Set y-axis limits
    plt.show()  





# Plot positive regions for each model
for model, data in positive_regions.items():
    plt.figure()
    plt.scatter(data['Temp'], data['DO'], label=f'{model} Positive Regions')
    plt.xlabel('Temperature (Temp C)')
    plt.ylabel('Dissolved Oxygen (DO)')
    plt.title(f'Positive Regions for {model}')
    plt.legend()
    plt.grid(True)
    plt.show()






from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def fit_polynomial_boundary(x, y, degree=2): #INCREASE DEGREE OF POLYNOmial or make a power function
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly, y)
    return model, poly

# Fit boundary curves for each model
boundary_equations = {}
for model, points in boundary_points.items():
    x = points[:, 0]  # T values
    y = points[:, 1]  # DO values
    model_eq, poly = fit_polynomial_boundary(x, y, degree=2)
    boundary_equations[model] = (model_eq, poly)
    
# Plot boundary curves
for model, (model_eq, poly) in boundary_equations.items():
    x = np.linspace(df['Temp'].min(), df['Temp'].max(), 100)
    x_poly = poly.transform(x.reshape(-1, 1))
    y_pred = model_eq.predict(x_poly)
    
    plt.figure()
    plt.scatter(positive_regions[model]['Temp'], positive_regions[model]['DO'], label=f'{model} Positive Regions')
    plt.plot(x, y_pred, color='red', label=f'{model} Boundary Curve')
    plt.xlabel('Temperature (C)')
    plt.ylabel('Dissolved Oxygen (mg/L)')
    plt.title(f'Boundary Curve for {model}')
    plt.legend()
    plt.grid(True)
    plt.show()



#" fitting machine learning support vector regression model to curve"
#" it performs way worse obviously for this deterministic model - deleting"


# calculating positive areas


# calculate area positive 
#from scipy.integrate import trapezoid



#===============================///// 


#%%
" ======= Substituting col sums with positive region areas LIKE ABOVE - integrated in (T,DO) space ============"

# Define the grid spacing
T_step = Temp_array[1] - Temp_array[0]  # Spacing between T values
DO_step = DO_array[1] - DO_array[0]    # Spacing between DO values
grid_cell_area = T_step * DO_step      # Area of each grid cell

# Calculate the positive area for each model
positive_region_sizes = {}

for model in df.columns:
    if model.startswith('GRP_') or model.startswith('GRP_lethal_'):  # Consider both GRP and GRP_lethal
        positive_mask = df[model] > 0  # Boolean mask for positive GRP values
        positive_count = positive_mask.sum()   # Count of positive (T, DO) points
        positive_area = positive_count * grid_cell_area  # Total positive area
        positive_region_sizes[model] = positive_area

# Convert to DataFrame
pos_hull_areas = pd.DataFrame(positive_region_sizes.items(), columns=["Model", "Positive_Region_Size"])
print(pos_hull_areas)


"-------------- OAT sensitivity analysis ------------"
"global results sensiticity leave one out- the aboce is to base nodel "


# Initialize a list to store results
global_sensitivity_results = []

# Iterate through all parameter combinations
for mass in mass_coefficients:
    for P in P_coefficients:
        for slope in slope_vals:
            for intercept in intercept_vals:
                # Non-lethal model
                grp_col = f"GRP_P{P}_mass{mass}_slope{slope}_intercept{intercept}"
                if grp_col in positive_region_sizes:
                    grp_positive_area = positive_region_sizes[grp_col]
                else:
                    grp_positive_area = 0
                
                # Lethal model
                grp_lethal_col = f"GRP_lethal_P{P}_mass{mass}_slope{slope}_intercept{intercept}"
                if grp_lethal_col in positive_region_sizes:
                    grp_lethal_positive_area = positive_region_sizes[grp_lethal_col]
                else:
                    grp_lethal_positive_area = 0
                
                # Store results
                global_sensitivity_results.append({
                    'mass': mass,
                    'P': P,
                    'slope': slope,
                    'intercept': intercept,
                    'GRP_Positive_Area': grp_positive_area,
                    'GRP_lethal_Positive_Area': grp_lethal_positive_area,
                    'Difference': grp_lethal_positive_area - grp_positive_area
                })

# Convert results to DataFrame
global_sensitivity_df = pd.DataFrame(global_sensitivity_results)
print(global_sensitivity_df)
#
#global_sensitivity_df.to_csv("/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/GRPmodel_sensitivity_dfs/pos_grp_TDO_vals.csv", index=False)

#%%

"-------- PLOTTING OAT SENSITIVITY USING AREAS COMPARISON ---------------"

# Pivot the data for heatmaps
heatmap_data = global_sensitivity_df.pivot_table(
    index='mass', columns='P', values='GRP_Positive_Area'
)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".0f")
plt.title('Impact of Mass and P on GRP Positive Area')
plt.xlabel('P')
plt.ylabel('Mass')
plt.show()



# Pair plot for all parameters
sns.pairplot(
    global_sensitivity_df,
    x_vars=['mass', 'P', 'slope', 'intercept'],
    y_vars=['GRP_Positive_Area', 'GRP_lethal_Positive_Area', 'Difference'],
    kind='scatter'
)
plt.show()



mass = 400
P = 0.4
slope = 0.168
intercept = 1.63

# Extract the GRP values for this combination
grp_col = f"GRP_P{P}_mass{mass}_slope{slope}_intercept{intercept}"
grp_values = df_grpfull[grp_col].values.reshape(len(Temp_array), len(DO_array))

# Create contour plot
plt.figure(figsize=(10, 6))
contour = plt.contourf(DO_array, Temp_array, grp_values, levels=20, cmap='viridis')
plt.colorbar(contour, label='GRP Value')
plt.title(f'GRP Response for mass={mass}, P={P}, slope={slope}, intercept={intercept}')
plt.xlabel('DO')
plt.ylabel('T')
plt.show()



#from mpl_toolkits.mplot3d import Axes3D

# Create 3D surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
DO_grid, Temp_grid = np.meshgrid(DO_array, Temp_array)
surf = ax.plot_surface(DO_grid, Temp_grid, grp_values, cmap='viridis', edgecolor='none')
fig.colorbar(surf, label='GRP Value')
ax.set_title(f'GRP Response for mass={mass}, P={P}, slope={slope}, intercept={intercept}')
ax.set_xlabel('DO')
ax.set_ylabel('T')
ax.set_zlabel('GRP Value')
plt.show()

#%%
"These plots are not great???"

import matplotlib.pyplot as plt
import seaborn as sns

# Plot sensitivity results
for param in sensitivity_df['Parameter'].unique():
    plt.figure()
    subset = sensitivity_df[sensitivity_df['Parameter'] == param]
    sns.barplot(data=subset, x='Value', y='GRP_Positive_Area', color='blue', label='GRP')
    sns.barplot(data=subset, x='Value', y='GRP_lethal_Positive_Area', color='red', label='GRP_lethal', alpha=0.6)
    plt.title(f'Sensitivity to {param}: GRP vs GRP_lethal')
    plt.xlabel(param)
    plt.ylabel('Positive Area')
    plt.legend()
    plt.show()


# Pivot the sensitivity results for heatmap
heatmap_data = sensitivity_df.pivot(index='Parameter', columns='Value', values='Difference')

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0)
plt.title('Difference in Positive Area: GRP_lethal - GRP')
plt.show()


#%%








#%%
"""
This code does not give the min values and instead gives each GRP for model at each corner if positive


# Extract GRP columns dynamically
grp_cols = df_grpfull.filter(like="GRP_").columns  # All GRP model iteration columns

# Dictionary to store results for each model iteration
corner_points_per_model = {}

for grp_col in grp_cols:
    # Filter only positive values for the current model iteration
    positive_df = df_grpfull[df_grpfull[grp_col] > 0]

    if not positive_df.empty:  # Ensure there are positive values
        min_temp = positive_df["Temp"].min()
        max_temp = positive_df["Temp"].max()
        min_do = positive_df["DO"].min()
        max_do = positive_df["DO"].max()

        # Extract four corner points for this iteration
        corners = {
            "(min_T, min_DO)": positive_df[(positive_df["Temp"] == min_temp) & (positive_df["DO"] == min_do)],
            "(max_T, min_DO)": positive_df[(positive_df["Temp"] == max_temp) & (positive_df["DO"] == min_do)],
            "(max_T, max_DO)": positive_df[(positive_df["Temp"] == max_temp) & (positive_df["DO"] == max_do)],
            "(min_T, max_DO)": positive_df[(positive_df["Temp"] == min_temp) & (positive_df["DO"] == max_do)],
        }

        corner_points_per_model[grp_col] = corners  # Store results

# Print summary
for grp_col, corners in corner_points_per_model.items():
    print(f"Model: {grp_col}")
    for key, value in corners.items():
        print(f"{key}:\n{value}\n")




# converting range(corners) to df

# List to store data
corner_data = []

# Loop through each model iteration
for model, corners in corner_points_per_model.items():
    for corner_name, df_corner in corners.items():
        if not df_corner.empty:  # Ensure there's data
            for _, row in df_corner.iterrows():
                corner_data.append({
                    "Model": model,
                    "Corner": corner_name,
                    "Temp": row["Temp"],
                    "DO": row["DO"],
                    "GRP": row[model]  # Extract GRP value for this model
                })

# Convert list to DataFrame
df_corners = pd.DataFrame(corner_data)

#saving df_corners to .csv
#df_corners.to_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/1D_binaryarray_dfs/synthethic_grp_eval/grp_varianats_allcombinationsRangeTDO.csv', index = False)


# Create a scatter plot for the corner points
plt.figure(figsize=(8, 6))

    # Use different markers for each corner type
corner_markers = {"(min_T, min_DO)": "o", "(max_T, min_DO)": "s",
                      "(max_T, max_DO)": "D", "(min_T, max_DO)": "X"}

    # Plot each corner type with different markers
for corner, marker in corner_markers.items():
    subset = df_corners[df_corners["Corner"] == corner]
    plt.scatter(subset["Temp"], subset["DO"], label=corner, marker=marker, s=100, alpha=0.7)

    # Labels and title
plt.xlabel("Temperature (°C)")
plt.ylabel("Dissolved Oxygen (mg/L)")
plt.title("Corner Points of Positive GRP Regions Across Models")
#plt.legend(title="Corner Type", loc="upper right")

    # Show plot
plt.show()












" scatter all iterations in one figure "

plt.figure(figsize=(8, 6))

# Loop through GRP iterations and plot positive values
for grp_col in grp_cols:
    positive_df = df_grpfull[df_grpfull[grp_col] > 0]
    plt.scatter(positive_df["Temp"], positive_df["DO"], label=grp_col, alpha=0.3)  # Alpha for visibility

# Labels and title
plt.xlabel("Temperature (°C)")
plt.ylabel("Dissolved Oxygen (mg/L)")
plt.title("Positive GRP Regions Across 96 Model Iterations")
#plt.legend(loc="upper right", fontsize="small", ncol=2)  # Legend for models

plt.show()





"individual subplots"

num_models = len(grp_cols)
cols = 4  # Number of columns for subplots
rows = (num_models // cols) + (num_models % cols > 0)  # Dynamic rows

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))

for i, grp_col in enumerate(grp_cols):
    row, col = divmod(i, cols)
    ax = axes[row, col]

    positive_df = df_grpfull[df_grpfull[grp_col] > 0]
    ax.scatter(positive_df["Temp"], positive_df["DO"], c=positive_df[grp_col], cmap="viridis", alpha=0.7)
    
    ax.set_title(grp_col)
    ax.set_xlabel("Temp (°C)")
    ax.set_ylabel("DO (mg/L)")

plt.tight_layout()
plt.show()



#%%
#df_grp_syn = pd.read_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/1D_binaryarray_dfs/synthethic_grp_eval/grp_variants1D.csv')
#filtered_dfgrp.to_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Chapter_1/Reprocessed_Ch1_data/1D_binaryarray_dfs/synthethic_grp_eval/grp_variants_posmask1D_increasingDOandT.csv')

"""
#Chat questions - drafts and ideas

#1.how to output the (min, min), (max,min), (max, max), (min, max) for filtering columns  #(t, do)

#2.positive_condition = df[positive_cols].gt(0).all(axis=1)  # Ensures all selected columns have positive values. Instead of getting values where positive columns are all col



"""











#%%

" ---------------- Leftover maybe useful from Post_interp_process_plot in the slicing cell(s)------------"

"""
Applies to both dictionaries

location_index = -3
timescale_index = -2
year_index = -1


key_parser = {k: {'lake': parts[-3], 'timescale': parts[-2], 'year_index': parts[-1] }
              
                for k, parts in ((k, k.split('_')) for k in grp_dict.keys()) }


    

"temportary working/testing time funcs here"
years = [2021, 2022]
lakes = ['fail', 'crook']
scales = ['day', 'hr']
#genranges_og = generate_date_ranges(years, lakes)
genranges = generate_date_ranges_grp_arrs(years, lakes, scales)


# Extract start dates for each lake and year
start_dates = {
    (lake, year): genranges[f"{lake}_{year}_{scale}"][0] for lake in lakes for year in years for scale in scales
}

# Generate date arrays
date_dict = dates_for_key(start_dates, grp_dict)

for key, val in date_dict.items():
    #print('key and shape:', list(dationaryfull[key].shape))
    print(f"{key}: {val[:5]}")
    #print(f"{key}: {val[5:]}")

# Extract time slices
slices = extract_time_slices2([grp_dict, grp_binary_dict], date_dict, genranges)
grp_slices, grp_binary_slices = slices
                

check_dictlog = [(k.lower().split('_')[-3:]) for k in grp_dict.keys()] #returns list of lists
print(check_dictlog)


{print(k.lower().split('_')[-3:]) for k in date_dict.keys()} #returns list[lake, scale, year]
{print(f"{k.lower()}_preturn") for k in date_dict.keys()} #returns key_preturn
{print(k.lower().split('_')[-3:] + ['preturn']) for k in date_dict.keys()}  # returns list[lake, scale, year, preturn]


def dates_for_key(date_dict, *data_dicts):
 #########
    Works for non-preturn maybe? - used to before AI fucked it (check deepseek chat)
    Generate date arrays for each key in multiple datasets based on matching start dates.
    
    Parameters:
        date_dict (dict): Dictionary {key: (start_date, end_date)}.
        *data_dicts (dict): Multiple dictionaries {key: 2D NumPy array}.
    
    Returns:
        dict: Dictionary {key: NumPy array of dates}.
###########
    date_arrays = {}
    
    for data_dict in data_dicts:
        for key, arr in data_dict.items():
            
            num_intervals = arr.shape[1]  # Get column count for this specific key
            
            # Extract lake, timescale, var, year from key
            parts = key.lower().split('_')
            scale = parts[-2]  # Extract timescale (e.g., 'day' or 'hr')
            
            # Check if the key exists in date_dict with or without '_preturn'
            for date_key in date_dict:
            
                date_key_with_preturn = f"{key}_preturn"
                if date_key_with_preturn in date_dict:
                    start_date = date_dict[date_key_with_preturn][0]
                    print(f"preturn date {date_dict[date_key_with_preturn][0]}")
                elif date_key in date_dict:
                    start_date = date_dict[date_key][0]
                else:
                    print(f"⚠️ Warning: No start date found for {key}")
                    continue
            
                # Generate date array
                if scale == 'day':
                    dates = np.array([start_date + timedelta(days=i) for i in range(num_intervals)])
                else:  # Assume hourly if not daily
                    dates = np.array([start_date + timedelta(hours=i) for i in range(num_intervals)])
            
            date_arrays[date_key] = dates  # Store unique key-date pair
    
    return date_arrays
    



"""