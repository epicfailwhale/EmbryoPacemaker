# EmbryoPacemaker
Designing a Pacemaker to monitor and control the cardiac activity of a chicken embryo. Input recorded using an Arduino, stimulation delivered via pulse generator with triggers controlled by Arduino.
Find out more [here](https://wiki.bme-paris.com/2024-project04/tiki-index.php).

**ecg**: the Arduino script to set up the Arduino for acquisition of signal from embryo  
**RealTimeProcessingClass**: python script with classes and methods to process, visualise signal and calculate HR in real time  
**offline_analysis**: take saved csv of raw data with ability to change filter parameters for processing  
**read_pulse_record**: saves input from Arduino into a csv  
**real_time_processing**: 'serial plotter'-like visualiser. Acquire, filter and plot signal in real time.

**example_ecg_stimulation**: Example of our results


Techniques used: signal processing, Arduino
