# Saves recording from arduino into csv file
# To save, need to close Arduino IDE so they are not both using the port
# Modified on March 15: Port changed to COM8 instead of 13 due to usage in other computer
import serial
import csv
from datetime import datetime
import time
from RealTimeProcessingClass import*

# butter bandpass filter
low = 0.5
high = 40
fs = 180
order = 3

# Serial port and baud rate setup
ser = serial.Serial('COM8', 115200) # Update 'COM3' to your Arduino's port
sig = SignalProcessor(None, low, high, fs, order)
csv_file = "sensor_data_" + datetime.now().strftime('%Y-%m-%d %H%M') + ".csv" # CSV file name
path = "collected_data/" + csv_file
print(path)

# Duration for data recording in seconds
duration_seconds = 30

start_time = time.time() # Record the start time


# open csv file
with open(path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Recording Value", "Pulse Output"]) # Header row
    print("Writing file")
    
    # Record
    try:
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
           

            # Stop recording after the duration is exceeded, process and plot signal 
            if elapsed_time > duration_seconds:
                print("Completed data recording.")
                sig = SignalProcessor(csv_file, low, high, fs, order)
                sig.filter_plot()
                break
            
            # Acquire, save and print incoming values 
            if ser.inWaiting() > 0:
                data = ser.readline().decode('utf-8').rstrip() # Read and decode data
                writer.writerow([datetime.now(), data])  # Write data to CSV
                print(f"{datetime.now()}, {data}")  # Optional: print data to console
                
    except KeyboardInterrupt:
        print("Data collection stopped manually.")
    finally:
        ser.close() # Make sure to close the serial port


   