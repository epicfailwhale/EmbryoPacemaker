# To visualise the recordings in real time
# Modified on March 15: Changed port for serial communication
from RealTimeProcessingClass import *

# set up filter parameters
low = 0.5
high = 40
fs = 180
order = 7 
sig = SignalProcessor(None, low, high, fs, order)

## set up connections
arduino = SerialCommunication('COM8', 115200) 
# Update 'COM3' to your Arduino's port / Is normally COM13 for Vic, for Patricia it's 8

## online analysis
real_time_queue = RealTimeProcessor(arduino, sig, plot_window=200)
real_time_queue.start()

