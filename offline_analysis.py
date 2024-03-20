# Take saved raw data from csv files and process them
# Ability change filter parameters
from RealTimeProcessingClass import *

# butter bandpass filter parameters
low = 0.5
high = 40
fs = 180
order = 5

# File name
signal_csv = "sensor_data_2024-03-19 1539.csv"

# Create signal processor object
s1 = SignalProcessor(signal_csv,low,high,fs,order,filter_type='butterworth')

# optional: calculate bpm
bpm, peaks = s1.calculate_heart_rate(duration = 30)
print("bpm: ", bpm)

# Plot and save signal data
fig = s1.filter_plot(auto_close=False)
fig.savefig(signal_csv.split('.csv')[0])

