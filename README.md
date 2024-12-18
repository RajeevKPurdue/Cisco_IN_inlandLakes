# Cisco_IN_inlandLakes
Processing, Summarization, Interpolation, and Visualization of inland lake oxythermal data 
1- Manually remove non-column headers from minidot .cat files and save to .csv and/or .txt
2- Remove all columns NOT "Temperature", "DO", or "Eastern Standard Time" and rename columns to "Temp C", "DO mg/l", and "ESTDateTime"
3- Convert EST column to Datetime, create hourly and daily summary files for each and save in separate nested directories
4-Split temp and DO into separate files from each file 
5- Interpolate Temp and DO for data periods and lakes 
6-Slice into full,stratified, and preturnover
6-Run GRP models, thresholds, for lakes, periods, slices
7-Visualize data

