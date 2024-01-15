"""
Following code was written with the help of the paper https://www.mdpi.com/2227-9032/9/2/227 to try and implement their method for SQRS
Authors of the Paper: Lu Wu, Xiaoyun Xie and Yinglong Wang

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, hann, find_peaks
from scipy.stats import kurtosis
from pywt import dwt
import math

"""Runs a simple test for ecg data using the SQRS algorithm"""
"""FOR FOLLOWING CODE TO RUN, YOU MUST RUN THE MAIN.PY,SINCE THE DATA MUST BE INITIALIZED"""
def main():
    #Defining the Folder where the NumpArr and PandSeries are saved
    numpy_path = "NumpyArrays/"
    series_path = "PandaSeries/"
    #Defining the name of the files, to save the data
    x_train_unprocessed = numpy_path + "x_train_unprocessed.npy"
    x_test_unprocessed = numpy_path + "x_test_unprocessed.npy"
    y_train_path = series_path + "y_train.pkl"  
    y_test_path = series_path + "y_test.pkl"
    
    x_train = np.load(x_train_unprocessed)
    x_test = np.load(x_test_unprocessed)
    # Load Series from pickle
    y_train= pd.read_pickle(y_train_path)
    y_test = pd.read_pickle(y_test_path)

    # Original data
    #NORM
    original_data = x_train
    #MI
    #original_data = x_train[8, :, 0]
    #STTC
    #original_data = x_train[22, :, 0]
    #CD
    #original_data = x_train[32, :, 0]
    #HYP
    #original_data = x_train[30, :, 0]

    SQRS_execution(ecg_data=original_data)



def SQRS_execution(ecg_data):
    
    #Tester for different ECG Data
    first_lead = ecg_data[22, :, 0]

    #Original work had one of 5, but 3 yielded better results though does not smooth the signal as much 
    filterlengthM = 3 #5

    #Removal of noise using the a moving average filter
    def moving_average_filter(signal, window_size):

        #Create an empty list to store the denoised signal
        denoised_signal = []

        #Loop through the signal to calculate the new amplitudes according to Formula (1)
        for i in range(len(signal)):
            #Calculate the moving average of the current window
            window = signal[i:i + window_size]
            moving_average = np.mean(window) #averaging the window

            #Add the moving average to the denoised signal
            denoised_signal.append(moving_average)

        return denoised_signal
    

    #Utilizing the moving average filter from step 1 to smoothen the signal
    denoised_ecg_data = moving_average_filter(first_lead, window_size= filterlengthM)

    #Enhancing the signal for the R-Peaks to be more predominant and P and T Waves to be surpressed
    #Utilizing the QRS Enhancement functions to make the QRS-Complexes better
    squared_signal = ecg_enhancement(denoised_ecg_data)
    
    SWVT_transform = squared_signal #just changing the name

    #RR_recent and RR_all gets calculated in the code

    #As per their suggestion for 200 and 360ms
    RR_recent = 20
    RR_all = 36

    #Step 1: Generate potential R-peaks candidates
    R_peaks_candidates = generate_R_peaks_candidates(SWVT_transform)

    print(f"Peak Candidates:: {R_peaks_candidates}")
    #Step 2: Generate R-peaks with decision rules
    R_peaks = generate_R_peaks_with_decision_rules(SWVT_transform, R_peaks_candidates, RR_recent, RR_all)

    print(f"Final Peaks: {R_peaks}")
    
    
    #Plotting the signal 
    plt.figure(figsize=(12, 6))
    plt.plot(first_lead, label="Original ECG Signal")
    #plt.plot(denoised_ecg_data, label="Denoised ECG Data", color="red")
    #plt.plot(squared_signal, label="Squared Stuff", color ="green")
    plt.plot(R_peaks, squared_signal[R_peaks], "x", color="red", label="Detected R-peaks")
    plt.title("ECG Signal with only the QRS-Complexes in red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.show()

def ecg_enhancement(denoised_signal):

    #1. Non-overlapping window generation-----------------------------------------------------------
    #Taking the peak candidates using a SciPy function
    peaks, _ = find_peaks(denoised_signal, distance=50)  #Assuming a minimum distance between peaks of 50

    #Seting windo size based on the minimum width of Twave in dependence of the found peaks
    window_size = min(max(40, 2 * peaks[1] - peaks[0]), 60)
    
    #Generate windows for peaks as a list of tuples with their start and ending
    peak_windows = [(max(0, int(peak - window_size / 2)), min(len(denoised_signal), int(peak + window_size / 2))) for peak in peaks]
    
    #Generate non-peak windows between consecutive peaks as list of tuples
    non_peak_windows = []
    for i in range(len(peaks) - 1):
        #pi - w/2
        start = int((peaks[i] + peaks[i + 1]) / 2)
        end = int((peaks[i + 1] + peaks[i + 2]) / 2) if i + 2 < len(peaks) else len(denoised_signal)
        non_peak_windows.append((start, end))
    
    #Merging the windows
    all_windows = peak_windows + non_peak_windows #The non overlapping windows

    # 2. Window variance transform with Formula (2) --------------------------------------------------------------------------------------------------
    #print(all_windows)
    for window in all_windows:
        try:
            starting_window, ending_window = window[0], window[1]#Setting the window start and ending
            window_data = denoised_signal[window[starting_window]:window[ending_window]]#Setting the data for each window
            lj = len(window_data) # Setting the amount of data per window as l of j with j being the window
            window_mean = sum(window_data)/len(window_data) #Calculating the mean of the data
            #print(f"WindowMean: {window_mean}")
            for i in range(lj):
                window_data[i] = (window_data[i] - window_mean)**2 # Calculating y(t) - m(j) ^2 from the formula
            
            #Creating the WVT for each window incremently
            denoised_signal[window[starting_window]:window[ending_window]] = window_data#replacing the data
            #going to the next window
            starting_window += 1
            ending_window += 1

        except IndexError:
            pass

    # 3. Squaring/SWVT-------------------------------------------------------------------------------------------------------------
    #Squaring each value in the windows 
    denoised_signal = np.square(denoised_signal) # This is the SWVT now
    return denoised_signal


"""Generate potential candidates for RPeaks"""
def generate_R_peaks_candidates(SWVT_transform):
    #1. Generation of the R-peaks candidate set
    
    #Calculation of the 90th percentile of all data
    p90_SWVT = np.percentile(SWVT_transform, 90)
    #Mean of the data
    mean_SWVT = sum(SWVT_transform) / len(SWVT_transform)

    #print(f"Elements:: p90{p90_SWVT} , meanSWVT{mean_SWVT}")
   
    #threshold Tc using the 90th percentile and mean as in formula (4)
    Tc = 0.5 * (0.75 * p90_SWVT + 0.25 * mean_SWVT)
    #print(f"Tc {Tc}")    
    #Extracting R-peak potential candidates set Sc using threshold Tc as in formula (5)
    R_peaks_candidates = find_peaks(SWVT_transform, height=Tc)[0]
    
    return R_peaks_candidates


def generate_R_peaks_with_decision_rules(SWVT_transform, R_peaks_candidates, RR_recent, RR_all, alpha=0.5):
    #2. Generation of the R-peaks with decision rules

    #Adjusting the RR interval in accordance to formula (6)
    #Done once with initial values at the start of 20 and 36 --> 200 and 360ms each
    RR = 0.75 * RR_recent + 0.25 * RR_all

    #Initialize R-peaks list to return later
    R_peaks = []

    #Adjusting thresholds (7,8) for amplitude and kurtosis
    Ta = 0.75 * alpha * np.percentile(SWVT_transform[R_peaks_candidates], 90) #+ (1 - alpha) * np.mean(SWVT_transform[R_peaks]))
    Tk = 0.75 * alpha * np.percentile(kurtosis(SWVT_transform[R_peaks_candidates]), 90) #+ (1 - alpha) * np.mean(kurtosis(SWVT_transform[R_peaks])))
    #!! The commented out part is due to the R_peaks list still being empty, basically 0 but NumPy makes it NAN


    #Reduced thresholds for back searching Formulas (9,10)
    Ta_back_search = 0.5 * Ta
    Tk_back_search = 0.5 * Tk



    #Calculates the new updated RR Interval Formula (6)
    def RRcalc(R_peaks, RR, RR_recent = RR_recent):
        #When there are 8 or more R_peaks, we calculate the RR_recehnt for the 8 newest Peaks
        if len(R_peaks) >= 8:
            RR_recent = (R_peaks[-1] - R_peaks[-8]) / 8
            RR_all = (R_peaks[-1] - R_peaks[0]) / len(R_peaks)
            RR = 0.75 * RR_recent + 0.25 * RR_all
            return RR
        #Otherwise calculate a new RR_all if there are 2 RPeaks
        else:
            if len(R_peaks) >= 2:
                RR_all = (R_peaks[-1] - R_peaks[0]) / len(R_peaks)
                RR = 0.75 * RR_recent + 0.25 * RR_all 
                return RR
            #In the beginning RR stays the stame until we have atleast 2 RPeaks
            else:
                return RR 

    #Implements the backwards search with new thresholds if no peak was found in 1.66 of RR time
    def backwards_search(candidate, RR, newest_Rpeak, R_peaks, R_peaks_candidates=R_peaks_candidates):

        #If the last found R_peak is more time ago then 1.66 RR, we start a backsearch
        if (candidate - newest_Rpeak) > 1.66 * RR: #Last Found R_peak difference to the candidate now

            #If a search in this time did not find a R-Peak, we go this time back in the backsearch
            search_back_interval = int(candidate - newest_Rpeak)
            #Setting the start where we start our Backsearch
            search_back_position = max(0, candidate - search_back_interval)

            print(f"SeachBack Position: {search_back_position}:{candidate}")
            
            #Check decision rules for R-peaks in the search-back interval Utilization of Formulas (9,10) with NEW THRESHOLDS
            #Checking for the biggest peak in the searchback interval
            if SWVT_transform[search_back_position:candidate].max() > Ta_back_search and kurtosis(SWVT_transform[search_back_position:candidate]).max() > Tk_back_search:
                print("BackSearch")
                RR = RRcalc(R_peaks, RR)
                print("Addition in BackSearch: ", candidate)
                R_peaks.append(candidate)

                #TODO Remove if tests are as expected
                """
                R_Peak_HighestAmpl = max(SWVT_transform[search_back_position:candidate])
                print(R_Peak_HighestAmpl, " HSDF")

                index_of_max_value = np.where(SWVT_transform[search_back_position:candidate] == R_Peak_HighestAmpl)[0][0]
                print("Index: ", index_of_max_value)
                #R_peaks.append(index_of_max_value)
                if index_of_max_value in R_peaks_candidates:
                    R_peaks.append(index_of_max_value)
                    newest_Rpeak = index_of_max_value
                """


            

    
    # 3. Generation of the R-peaks with decision rules
    newest_Rpeak = 0 #If non was found, set it to the start of signal ==> 0 
    not_found_count = 0 #Amount of time since last found R_peak ==> Measurements 
    print(f"RR : {RR}")

    for candidate in R_peaks_candidates:
        #Checking if candidate is viable and if R_peaks is empty --> True
        if (0.92 * RR < candidate - newest_Rpeak < 1.16 * RR) if R_peaks else True:
            #Check decision rules for R-peaks based on formula (11,12)
            if SWVT_transform[candidate] > Ta and kurtosis(SWVT_transform[R_peaks_candidates]) > Tk:
                print("Addition in first search: ", candidate)
                R_peaks.append(candidate)#Found R_peak gets included
                newest_Rpeak = candidate
                RR = RRcalc(R_peaks, RR)

                Ta = 0.75 * (alpha * np.percentile(SWVT_transform[R_peaks_candidates], 90) + (1 - alpha) * np.mean(SWVT_transform[R_peaks]))
                Ta_back_search = 0.5 * Ta
                if len(R_peaks) >= 2:
                    Tk = 0.75 * (alpha * np.percentile(kurtosis(SWVT_transform[R_peaks_candidates]), 90) + (1 - alpha) * np.mean(kurtosis(SWVT_transform[R_peaks])))
                    Tk_back_search = 0.5 * Tk

            #Backwards Search if Threholds did not work and Peak not found in 1.66 RR time
            elif not_found_count >= 1.66 * RR:
                backwards_search(candidate=candidate, RR=RR, newest_Rpeak=newest_Rpeak, R_peaks=R_peaks)
                newest_Rpeak = candidate
                not_found_count = 0

            #Addition of candidates position as time to the notFound Count
            else:
                not_found_count += candidate
        #Backwardssearch if candidate outsite RR values and still nout found
        elif not_found_count >= 1.66 * RR:
            backwards_search(candidate=candidate, RR=RR, newest_Rpeak=newest_Rpeak, R_peaks=R_peaks)
            newest_Rpeak = candidate
            not_found_count = 0

        #Addition of candidates position as time to the notFound Count
        else:
            not_found_count += candidate
        #Adjustment of the Thresholds when there is enough R_Peaks in the R_Peak list
        try:
            if R_peaks > 1:
                #Adjusting thresholds (7,8) for amplitude and kurtosis
                Ta = 0.75 * alpha * (np.percentile(SWVT_transform[R_peaks_candidates], 90) + (1 - alpha) * np.mean(SWVT_transform[R_peaks]))
                Tk = 0.75 * alpha * (np.percentile(kurtosis(SWVT_transform[R_peaks_candidates]), 90) + (1 - alpha) * np.mean(kurtosis(SWVT_transform[R_peaks])))
                
                #Reduced thresholds for back searching Formulas (9,10)
                Ta_back_search = 0.5 * Ta
                Tk_back_search = 0.5 * Tk
        except:
            pass
    
    return R_peaks



#Runs the test
main()