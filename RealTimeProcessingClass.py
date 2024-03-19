## Functions required to acquire, process and save the signal data from Arduino.
import serial
import csv
from datetime import datetime
import time
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks, lfilter, lfilter_zi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from drawnow import drawnow
import threading
import queue

class SerialCommunication():

    def __init__(self, port = 'COM8', baud_rate = 115200):
        """
        Establishes and looks after connection to Arduino
        """
        self.port = port
        self.baud_rate = baud_rate
        self.ser = serial.Serial(self.port, self.baud_rate) # Arduino settings
    
    def read_data(self):
        data = self.ser.readline().decode('utf-8').rstrip() # Read and decode data
        # print(f"{datetime.now()}, {data}")  # Optional: print data to console
        return data

    def write_data(self, data):
        """Writes data to the serial port."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(data.encode())
            except serial.SerialException as e:
                print(f"Failed to write data: {e}")
        else:
            print("Serial connection is not open.")

    def close_connection(self):
        self.ser.close()
        
        
class SignalProcessor():

    def __init__(self, signal_in, low=0.5, high=40, fs=150, order=7, filter_type = None):
        """
        Handles the filter and processing of the incoming signal, whether it is in csv or array form.
        signal_in: can handle csv input or direct list
        filter_type: can be None, butterworth, moving average, or both butter worth and moving average
        """
        self.low = low
        self.high = high
        self.fs = fs
        self.order = order
        self.path = 'collected_data/' 
        self.signal_path = None
    
        ## initialise butterworth filter state
        self.b, self.a = butter(self.order, [low/(0.5*fs), high/(0.5*fs)], btype='band')
        self.zi = lfilter_zi(self.b, self.a)

        # Assuming that the string input is a path to a CSV
        if isinstance(signal_in, str):
            self.signal_path = signal_in
            self.signal = self.read_csv() ## assume it is raw data
        elif isinstance(signal_in, (np.ndarray, list)) or signal_in is None:
            if filter_type == None:
                self.signal = signal_in
            elif filter_type == 'butterworth':
                self.signal = self.filter_butter_bandpass()
            elif filter_type == 'moving_average':
                self.signal = self.filter_moving_average()
            elif filter_type == 'butter_move_av':
                self.signal = self.filter_butter_move_av()
            else:
                raise ValueError("filter_type must be None, 'butterworth', 'moving_average' or 'butter_move_av'")
        else:
            raise ValueError("Signal must be a path to a CSV file, a list, or a NumPy array")
            
    def read_csv(self):  
        data = []
        with open(self.path + self.signal_path, 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                data.append(float(row[1])) # signal data is in second column
        return data 

    def filter_moving_average(self,window_size=15):
        window = np.ones(window_size) / window_size
        filtered_data = np.convolve(self.signal, window, mode = 'same')
        return filtered_data

    def butter_bandpass(self):
        nyq = 0.5 * self.fs
        lowcut = self.low / nyq
        highcut = self.high / nyq
        b, a = butter(self.order, [lowcut, highcut], btype='band')
        return b, a

    def filter_butter_bandpass(self):
        b, a = self.butter_bandpass()
        y = filtfilt(b, a, self.signal)
        return y   
    
    def filter_butter_move_av(self):
        moved_av = self.filter_moving_average()
        b, a = self.butter_bandpass()
        y = filtfilt(b,a,moved_av)
        return y

    def detect_peaks(self, signal):
        height, threshold, distance = self.calculate_peak_characteristics(signal)
        peaks, _ = find_peaks(signal, height,threshold=None, distance=distance)
        # peaks is the indicies of peaks in filtered_data that satisfy the conditions
        return peaks # think about just returning the number of peaks 

    def calculate_peak_characteristics(self, signal):
        """ To calculate height and distance for detect_peaks method"""
        # to find height, find histogram of signal values. then use 1 standard deviation from median as threshold
        baseline = np.median(signal) ## median or mean??
        # print("baseline: ",baseline)
        std_noise = np.std(signal - baseline)
        # print("std_noise", std_noise)
        height =  baseline + std_noise
        # print("height",height)

        # find threshold (min vert distance between peak and surround)
        threshold = 0.5*std_noise

        # print("threshold:", threshold)
        # to find min distance between peaks, calculate from expected freq of peaks
        max_hz = 150 / 60 # in embryo, highest is probably 200bpm = 3.33Hz
        min_peak_s = 1 / max_hz # convert to period 
        distance = int(min_peak_s * self.fs) # multiply by sampling rate to get the number of samples need to wait before another peak
        # print("distance: ", distance)
        return height, threshold, distance

    def calculate_heart_rate(self, duration):
        """
        Calculate the heart rate in beats per minute (BPM).

        :param peaks: The indices or the number of peaks detected in the ECG signal.
        :param duration_in_seconds: The duration of the ECG recording in seconds.
        :return: The heart rate in BPM.
        """
        peaks = self.detect_peaks(self.signal)
        # If peaks is a list of indices, the number of peaks is its length
        number_of_peaks = len(peaks) if isinstance(peaks, np.ndarray) else peaks
        # print("num peaks:", number_of_peaks)
        # Calculate the heart rate in beats per minute
        beats_per_minute = (number_of_peaks / duration) * 60
        print("bpm:", beats_per_minute)
        return beats_per_minute, peaks
    
    def plot_peaks(self):

        peaks = self.detect_peaks(self.signal)
        print(type(self.signal))
        plt.plot(self.signal)
        plt.plot(peaks, np.array(self.signal)[peaks], "x")
        plt.show()
    
    def filter_data_continuous(self, new_data_point):
        # Applies Butterworth filter to a continuous stream of data
        filtered_data, self.zi = lfilter(self.b, self.a, [new_data_point], zi=self.zi)
        return filtered_data[0]
    

    def filter_plot(self):
        """ For offline plotting"""
        data = self.signal
        print(len(data))
        butter_data = self.filter_butter_bandpass()
        print("len butter", len(butter_data))
        butter_peaks = self.detect_peaks(butter_data)
        filtered_MAV_data = self.filter_butter_move_av()
        mav_peaks = self.detect_peaks(filtered_MAV_data)
        print("Plotting data")
        # Assuming 'data' is your unfiltered data and 'filtered_data' is the result from the filter_butter_bandpass function
        # Also assuming 'fs' is your sampling frequency and is used to calculate the time axis for your plots

        # Calculate the time axis based on the sampling frequency and the length of the data
        time_axis = np.arange(len(data)) / self.fs
        print("time",time_axis)
        # Create a figure and a set of subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))  # 2 Rows, 1 Column

        # Plot unfiltered data on the first subplot
        axs[0].plot(time_axis, data, label='Unfiltered Data', color='blue')
        axs[0].set_title('Unfiltered Data')
        axs[0].set_xlabel('Time (seconds)')
        axs[0].set_ylabel('Amplitude')
        axs[0].legend()

        # Plot filtered data on the second subplot
        axs[1].plot(range(0,len(butter_data)), butter_data, label='Filtered Data', color='red')
        axs[1].plot(butter_peaks, butter_data[butter_peaks], "x")
        axs[1].set_title('Filtered Data')
        axs[1].set_xlabel('Time (seconds)')
        axs[1].set_ylabel('Amplitude')
        axs[1].legend()

        axs[2].plot(range(0,len(filtered_MAV_data)), filtered_MAV_data, label='Filtered Data', color='green')
        axs[2].plot(mav_peaks, np.array(filtered_MAV_data)[mav_peaks], "x")
        axs[2].set_title('Filtered MAV Data')
        axs[2].set_xlabel('Time (seconds)')
        axs[2].set_ylabel('Amplitude')
        axs[2].legend()

        # Display the plot
        plt.tight_layout()
        plt.show()
       

class RealTimeVisualiser():

    def __init__(self, ard, filt, window_size=1000, filter_window_size=100, sample_rate=180):
        """
        RealTimeVisualiser controls the real time graph and filtering
        ard: SerialCommunicator class
        filt: SignalProcessor class
        Window size: number of data points in dynamic graph
        filter_window_size: sliding window size
        sample_rate: frequency of sample acquisition from Arduino
        """

        self.window_size = window_size
        self.filter_window_size = filter_window_size
        self.sample_rate = sample_rate
        self.ard = ard
        self.filt = filt
        self.data_buff = deque(maxlen=window_size)
        self.filtered_data_buff = deque(maxlen=window_size)
        self.ten_sec_buff_size = filter_window_size * sample_rate
        self.ten_sec_buff = deque(maxlen = self.ten_sec_buff_size)
        self.running = False # flag to check thread is already running
        self.bpm = None
        self.peaks = None # indices of the peaks 
        self.i = 0
        self.desired_filtered_data_buff_size = window_size - 25
        self.lock = threading.Lock()

    def plot_live_data(self):
        # Clear the plot to prevent old data from persisting
        plt.cla()
        plt.plot(self.data_buff)
        plt.xlabel('Time')
        plt.ylabel('Sensor Reading')

        # Dynamically adjust Y-axis based on current data
        current_min = min(self.data_buff) if self.data_buff else 0
        current_max = max(self.data_buff) if self.data_buff else 1
        plt.ylim(current_min - 10, current_max + 10)  # Adjust the padding as needed

        plt.title('Live Sensor Data')
        plt.grid(True)
        print(f"plotting {self.i}")
        self.i += 1

    # Function to continuously read data from the serial port and update the plot
    def read_serial(self):
        self.running = True
        try:
            while self.running:
                if self.ard.ser.inWaiting() > 0:
                    data_str = self.ard.read_data()
                    try:
                        data = float(data_str)
                        self.data_buff.append(data)
                    except ValueError:
                        print("Couldn't convert data to float")

        except KeyboardInterrupt:
            print("Data collection stopped manually.")
        finally:
            self.ard.ser.close() # Make sure to close the serial port
            self.running = False

    def plot_on(self):
        if not self.running:
            # Start a thread to run the read_serial function
            serial_thread = threading.Thread(target=self.read_serial)
            serial_thread.daemon = True  # Daemonize the thread so it exits when the main program exits
            serial_thread.start()

            # Start the Tkinter main loop to update the plot
        plt.ion()  # Turn on interactive mode
        try:
            while True:
                drawnow(self.plot_live_data)  # Call drawnow to update the plot
                plt.pause(0.001)  # Pause to allow the plot to update
        except KeyboardInterrupt: 
            print("Stopped real-time plotting")
        finally:
            self.running = False

    # Function to continuously read data from the serial port and update the plot
    def read_serial_and_filter(self):
        self.running = True
        temp_buff = []
        overlap_size = 25
        try:
            while self.running:
                if self.ard.ser.inWaiting() > 0:
                    data_str = self.ard.read_data()
                    print(f"Raw data: '{data_str}'")  # Debugging line
                    try:
                        data = float(data_str)
                        temp_buff.append(data)
                        if len(temp_buff) >= self.filter_window_size: # wait until filter buffer is a full batch
                            filtered_data = self.filter_butter_bandpass(temp_buff)
                            for point in filtered_data:
                                self.filtered_data_buff.append(point) 
                            # self.filtered_data_buff.append(filtered_data[-1])
                            # self.ten_sec_buff.append(filtered_data[-1])
                            # self.filtered_data_buff.extend(filtered_data)
                            self.ten_sec_buff.extend(filtered_data)
                            temp_buff = temp_buff[-overlap_size:]
                            print("len ten sec buff",len(self.ten_sec_buff))
                            while len(self.filtered_data_buff) > self.desired_filtered_data_buff_size:
                                self.filtered_data_buff.popleft()  # Remove oldest elements
                        if len(self.ten_sec_buff) >= self.ten_sec_buff_size:
                            self.filt.signal = self.ten_sec_buff[-self.filter_window_size]
                            self.bpm, self.peaks = self.filt.calculate_heart_rate(duration = 10)
                            print("bpm:", self.bpm)
                            print("peaks:", self.peaks)
                    except ValueError:
                        print("Couldn't convert data to float")

        except KeyboardInterrupt:
            print("Data collection stopped manually.")
        finally:
            self.ard.ser.close() # Make sure to close the serial port
            self.running = False

    def plot_on_filter(self):
        if not self.running:
            # Start a thread to run the read_serial function
            serial_thread = threading.Thread(target=self.read_serial_and_filter)
            serial_thread.daemon = True  # Daemonize the thread so it exits when the main program exits
            serial_thread.start()

        plt.ion()  # Turn on interactive mode
        try:
            while True:
                drawnow(self.plot_live_data_filter)  # Call drawnow to update the plot
                plt.pause(0.01)  # Pause to allow the plot to update
        except KeyboardInterrupt: 
            print("Stopped real-time plotting")
        finally:
            self.running = False


    def butter_bandpass(self):
        nyq = 0.5 * self.filt.fs
        lowcut = self.filt.low / nyq
        highcut = self.filt.high / nyq
        b, a = butter(self.filt.order, [lowcut, highcut], btype='band')
        return b, a

    def filter_butter_bandpass(self, signal):
        b, a = self.butter_bandpass()
        y = filtfilt(b, a, signal)
        return y   
    
    def plot_live_data_filter(self):
        # Clear the plot to prevent old data from persisting
        plt.cla()
        plt.plot(self.filtered_data_buff)
        if self.peaks is not None:
            plt.plot(self.peaks, np.array(list(self.filtered_data_buff))[self.peaks], "x")
        plt.xlabel('Time')
        plt.ylabel('Sensor Reading')
        # # Dynamically adjust Y-axis based on current data
        # current_min = min(self.filtered_data_buff) if self.filtered_data_buff else 0
        # current_max = max(self.filtered_data_buff) if self.filtered_data_buff else 1
        # plt.ylim(current_min - 10, current_max + 10)  # Adjust the padding as needed

        plt.title(f'Live Sensor Data Heart rate: {self.bpm}')
        plt.grid(True)
        print(f"plotting {self.i}")
        self.i += 1


    #### try separate threads for filtering and for plotting
    def plot_filtered_data(self):
        plt.ion()  # Interactive mode on
        fig, ax = plt.subplots()
        while self.running:
            with self.lock:
                ax.clear()  # Clear previous plot
                # Ensure filtered_data_buff is converted to a list or array for plotting
                ax.plot(list(self.filtered_data_buff))  # Plot current buffer
                plt.xlabel('Time')
                plt.ylabel('Filtered Signal')
                plt.draw()
                plt.pause(0.1)  # Adjust as needed for plot update rate
                time.sleep(0.1)  # Sleep to reduce CPU load
        plt.ioff()  # Turn interactive mode off

    def start_processing(self):
        self.running = True
        # Start data processing in its own thread
        data_thread = threading.Thread(target=self.read_serial_and_filter)
        data_thread.daemon = True
        data_thread.start()
        # Start plotting in its own thread
        plot_thread = threading.Thread(target=self.plot_filtered_data)
        plot_thread.daemon = True
        plot_thread.start()

    def __init__(self, ard, filt, window_size=1000, filter_window_size=100, sample_rate=180):
        """
        Window size: number of data points in dynamic graph
        ard: SerialCommunicator class
        filt: SignalProcessor class
        """
        self.window_size = window_size
        self.filter_window_size = filter_window_size
        self.sample_rate = sample_rate
        self.smooth_window_size = sample_rate * 0.5 # get 0.5s worth of data
        self.ard = ard
        self.filt = filt
        self.read_to_filter_queue = queue.Queue()
        self.filter_to_plot_queue = queue.Queue()
        self.running = True

        self.bpm = None
        self.peaks = None # indices of the peaks 
        self.i = 0
        self.desired_filtered_data_buff_size = window_size - 25
        self.lock = threading.Lock()


    def read_serial_data(self):
        while self.running:
            if self.ard.ser.inWaiting() > 0:
                raw_data = self.ard.read_data()
                # print("read",len(raw_data))
                self.read_to_filter_queue.put(raw_data) # put data into queue

    def moving_average_filter(self,data_buff, window_size = 15):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(data_buff, window, mode='same')
    
    def filter_data(self):
        overlap_size = 25
        temp_buff = []
        
        while self.running:
            raw_data = self.read_to_filter_queue.get() # get data from queue
            # print("read_to_filter:", type(self.read_to_filter_queue.get()))
            temp_buff.append(float(raw_data))
            if len(temp_buff) >= self.smooth_window_size:
                smoothed_data = self.moving_average_filter(temp_buff,self.smooth_window_size)
                if len(smoothed_data) >= self.filter_window_size:
                    filtered_data = self.filter_butter_bandpass(smoothed_data)
                    self.filter_to_plot_queue.put(filtered_data)
                    temp_buff = temp_buff[-overlap_size:]
                    print("filt")
                self.read_to_filter_queue.task_done() # indicate queue has been processed
                    
    def plot_filtered_data(self):
        fig, ax = plt.subplots()
        line, = ax.plot([])
        def animate(frame):
            print("animate")
            if not self.filter_to_plot_queue.empty():
                data = self.filter_to_plot_queue.get()
                print("plot",data)
                # ax.plot(data)
                line.set_ydata(data)
            return line,

        ani = FuncAnimation(fig, animate, interval=1000)
        plt.show()
    
    def plot_and_filter(self):
        fig, ax = plt.subplots()
        line, = ax.plot([])
        print("plot func")
        
        def animate():
            print("len read to filt", len(self.read_to_filter_queue.get()))
            temp_buff = []
            if not self.read_to_filter_queue.empty():
                raw_data = self.read_to_filter_queue.get()
                temp_buff.append(float(raw_data))
                
                if len(temp_buff) >= self.smooth_window_size:
                    smoothed_data = self.moving_average_filter(temp_buff, self.smooth_window_size)
                    if len(smoothed_data) >= self.filter_window_size:
                        print("filter")
                        filtered_data = self.filter_butter_bandpass(smoothed_data)
                        self.filter_to_plot_queue.put(filtered_data)
                        temp_buff = temp_buff[-25:]
                        line.set_ydata(filtered_data)
            
            return line,
        
        ani = FuncAnimation(fig, animate(), interval=1000)
        plt.show()

    def start_processing(self):
        self.running = True
        # Start data processing in its own thread
        data_thread = threading.Thread(target=self.read_serial_data)
        # filter data in its own thread
        # filter_thread = threading.Thread(target = self.filter_data)
        # Start plotting in its own thread
        # plt.ion()  # Interactive mode on
        # fig, ax = plt.subplots()
        # plot_thread = threading.Thread(target=self.plot_filtered_data, args = (fig,ax),daemon = True)
    
        data_thread.start()
        # filter_thread.start()
        self.plot_and_filter()
        # self.ard.close()   # Close Serial connection when plot is closed



    def butter_bandpass(self):
        nyq = 0.5 * self.filt.fs
        lowcut = self.filt.low / nyq
        highcut = self.filt.high / nyq
        b, a = butter(self.filt.order, [lowcut, highcut], btype='band')
        return b, a

    def filter_butter_bandpass(self, signal):
        b, a = self.butter_bandpass()
        y = filtfilt(b, a, signal)
        return y   

class RealTimeProcessor:
    def __init__(self, ard, filt, plot_window=100, moving_av_window = 15, csv_save=False):
        self.ard = ard
        self.filt = filt
        self.plot_window = plot_window
        self.csv_save = csv_save
        self.data_queue = queue.Queue()
        self.running = False

        # For moving average
        self.moving_av_window = moving_av_window
        self.moving_av_buff = deque(maxlen=moving_av_window)

        # for HR
        self.bpm = 0
        self.hr_window = 30
        self.ten_sec_fs = filt.fs*self.hr_window
        # the length of ten second buffer is sammpling frequency multiplied by 10
        self.ten_sec_buff = [0] * self.ten_sec_fs
        self.overlap = int(filt.fs*self.hr_window/2) # overlap calculations every 5 seconds
        
        # parameters for heart rate monitoring
        self.heart_rates= []
        self.bpm_window_size = 4 # collects the last 4 bpm calculations
        self.stability_threshold = 2 # standard deviation
        self.increase_by = 1.1 # 1.1 increases by 10%
        self.setup_delay = 15 # after HR calculation, the time it takes to adjust stim parameters
        self.trig_time = 60 # stimulation duration 

        # Setup for real-time plotting
        self.fig, self.ax = plt.subplots()
        self.xs = list(range(plot_window))
        self.ys = [0] * plot_window
        self.line, = self.ax.plot(self.xs, self.ys)
        self.ax.set_title(f'Live Sensor Data Heart rate: {self.bpm}')
        self.ax.set_xlabel('Samples')
        self.ax.set_ylabel('Sensor Reading')
        self.ax.set_ylim(bottom=-200,top=300)
        self.ax.autoscale(enable = False, axis='y')
        
    def read_serial_data(self):
        # if self.csv_save:
        #     start_time = time.time()
        #     with open(self.path, 'w', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(["Timestamp", "Recording Value", "Pulse Output"]) # Header row
        #         print("Writing file")

        while self.running:
            data = self.ard.read_data()
            if data:
                self.data_queue.put(data)
                


    def start(self):
        self.running = True
        # Start the thread for reading serial data

        threading.Thread(target=self.read_serial_data, daemon=True).start()
        threading.Thread(target=self.trigger_pulse_generator, daemon=True).start()
        # Start the animation for plotting
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, cache_frame_data = False)
        plt.show()
    
    def stop(self):
        self.running = False
        # Optionally, stop the plotting animation
        # self.ani.event_source.stop()

    def update_plot(self, frame):
        ## this one includes moving average filter
        while not self.data_queue.empty():
            data = float(self.data_queue.get())
                # Here you could add data processing/filtering using self.signal_processor
                # For simplicity, this example assumes `data` is directly plottable
            self.moving_av_buff.append(data)
            self.ax.set_ylim(bottom=-200,top=300)
            if len(self.moving_av_buff) == self.moving_av_window:
                avg_data = sum(self.moving_av_buff) / self.moving_av_window
                filtered_data = self.filt.filter_data_continuous(avg_data)
                self.ten_sec_buff.append(filtered_data)
                # Update the plot data
                self.ys.append(filtered_data)
                self.ys = self.ys[-self.plot_window:]
                # Update the x-axis data similarly to the y-axis data
                self.xs = list(range(len(self.ys)))  # Recreate xs based on the current length of ys
                self.line.set_xdata(self.xs)  # Update the line's x-data
                self.ax.set_xlim(min(self.xs), max(self.xs))  # Optionally, set xlim to adjust dynamically with the data

                self.line.set_ydata(self.ys)
                
                self.ax.set_ylim(bottom=-200,top=300)
                self.ax.autoscale(enable = False, axis='y')
                # self.ax.relim()
                # self.ax.autoscale_view()
                if len(self.ten_sec_buff) >= self.ten_sec_fs:
                    self.filt.signal = self.ten_sec_buff
                    self.bpm, peaks = self.filt.calculate_heart_rate(duration = self.hr_window)
                    self.heart_rates.append(self.bpm)
                    self.ax.set_title(f'Live Sensor Data Heart rate: {self.bpm}')
                    self.ten_sec_buff = self.ten_sec_buff[-self.overlap:]

        return self.line,

    def is_heart_rate_stable(self):
        # while filling up
        if len(self.heart_rates) <= self.bpm_window_size: 
            print('HR is not stable')
            return False 
        

        std_dev = np.std(self.heart_rates)
        print(f"Current BPM Standard Deviation: {std_dev}")
        self.heart_rates = self.heart_rates[-self.bpm_window_size-1:] # remove the oldest
        if std_dev <=self.stability_threshold:
            print('HR is stable')
            return True
        else: 
            print("HR is not stable")
            return False
    
    def calculate_pulse_period(self,bpm):
        return 1 / (bpm/60.0) # 1/Hz for period
    
    def trigger_pulse_generator(self):
        while self.running == True:
            if self.is_heart_rate_stable == True:
                target_hr = self.increase_by * np.mean(self.heart_rates)
                new_period = self.calculate_pulse_period(target_hr)
                print(f'Current HR: {np.mean(self.heart_rates)}\nTarget HR:{target_hr}\nNew Period: {new_period}')

                time.sleep(self.setup_delay)   

                # Trigger pulse generator 
                self.ard.write_data('TRIG ON')
                time.sleep(60)
                self.ard.write_data('TRIG OFF')


