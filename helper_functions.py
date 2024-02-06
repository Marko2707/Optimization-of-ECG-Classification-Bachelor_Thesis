import os
import numpy as np
import matplotlib.pyplot as plt
from peak_detection_algos.pan_tompkins_plus_plus import Pan_Tompkins_Plus_Plus

"""This module includes important functions that help implement:
    - Data Presaving for faster loading
    - Creating folders for Pre Saving
    - Plotting of ECG Signals
        - For all 5 Superclasses
        - For R-Peak Detected Plots of NORM Class

"""

"""Checks if folder is empty"""
def is_folder_empty(folder):
    #Takes in content of given folder
    content = os.listdir(folder)

    #returns boolean if folder is empty
    return len(content) == 0

import os

"""Checks if folders exist, used for NumpyArrays and PandaSeries folders"""
def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        #if folder does not exist, creates it
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        #already created
        print(f"Folder '{folder_name}' already exists.")

"""Plots all the 5 classes used in the bachelor thesis."""
def plot_all_classes(extracted_classes, extracted_ecg_ids, target_classes, x_train, y_train, number_no_superclass):
     #Iterate over y_train until each target class is extracted at least once
        for ecg_id, ecg_classes in y_train.items():

            #Check if the list of classes is not empty
            if ecg_classes:
                # Extract the first class
                ecg_class = ecg_classes[0]

                #Check if the class is in the target classes, has not been extracted yet,
                #and there are no other classes present with len(ecg_classes) == 1
                if ecg_class in target_classes and ecg_class not in extracted_classes and len(ecg_classes) == 1:
                    #Perform your desired actions with ecg_id and ecg_class
                    print(f"ecg_id: {ecg_id}, Klasse: {ecg_class}")

                    #Add the extracted class and the according ecg_id to the lists
                    extracted_classes.append(ecg_class)
                    extracted_ecg_ids.append(ecg_id)

                    #Remove the extracted class from target_classes to avoid repeated extraction
                    target_classes.remove(ecg_class)

                #heck if all target classes have been extracted
                if not target_classes:
                    break
            else:
                #Counting the amount of records with no superclass
                number_no_superclass += 1


        #Get all information of y_train of one inext

        #!To plot the different Classes remove the block comment
        #Plotting all 5 ECG Samples of each Super-Class:
        
        #Defining the 5 Superclasses I want to plot
        classes_to_plot = ["NORM", "MI", "STTC", "CD", "HYP"]

        # List of the ECG-Lead-Names
        lead_names = ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


    
        # Iterating over all 5 Classes i want to plot
        for ecg_class in classes_to_plot:
            #Finding the index with ecg class == class i want to plot
            indices = [i for i, label in enumerate(extracted_classes[:5]) if label == ecg_class]

            #Iterating over the 5 indices
            for i in indices:
                # extracting the ecg data from the x_train with the ids from before
                ecg_data = x_train[extracted_ecg_ids[i]]
                # creating the x axis as seconds with a sampling reate of 100Hz
                time_axis = np.arange(0, 10, 1/100)

                # Ploting the ecg data for each lead
                for lead in range(ecg_data.shape[1]):
                    plt.plot(time_axis, ecg_data[:, lead], label=f"{lead_names[lead]}")

                # Titel of the plot with each class
                plt.title(f"ECG Signal - Class: {ecg_class} with {extracted_ecg_ids[i]}")
                plt.xlabel("Time (s)") # Add the unit to the x-axis label
                plt.ylabel("Amplitude (mV)")  # Add the unit to the y-axis label
                plt.legend()
                plt.show()

"""Plots the PanTompkinsPlusPlus found R-peaks and the compression process aswell"""
def plot_panTompkinsPlusPlus(x_train):

    first_ekg_signal = x_train[0, :, 0]


    #Setting the frequency to 100Hz for the PanTompkinsPlusPlus
    freq = 100

    pan_tompkins = Pan_Tompkins_Plus_Plus()#init of PanTomp

    #using panTomp
    r_peaks_indices = pan_tompkins.rpeak_detection(first_ekg_signal, freq)

    #time axis for the plots
    time_axis = np.arange(0, 10, 10 / len(first_ekg_signal))

    r_peaks = r_peaks_indices.astype(int)

    print("Detected R-peaks indices:", r_peaks_indices)

    #plot of signal
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, first_ekg_signal, label="NORM Class R-Peak Detection through Pan-Tompkins++")
    plt.plot(r_peaks / freq, first_ekg_signal[r_peaks], "x", color="red")
    plt.title("NORM Class R-Peak Detection through Pan-Tompkins++")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.show()


    #------Conversion of R-peaks to the QRS complexes and further to be compressed


    #Convert R-peaks indices to integers
    r_peaks_indices = r_peaks_indices.astype(int)

    #Create a mask to set values to 0 except within a 100ms window around each R-peak
    mask = np.zeros_like(first_ekg_signal)
    window_size = int(0.1 * freq)  # 100 ms window size

    for r_peak in r_peaks_indices:
        r_peak = int(r_peak)  # Convert to integer
        mask[max(0, r_peak - window_size):min(len(mask), r_peak + window_size + 1)] = 1
    #Apply the mask to the original signal
    modified_signal = first_ekg_signal * mask

    print("Detected R-peaks indices:", r_peaks_indices)

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, first_ekg_signal, label="Original ECG Signal")
    plt.plot(time_axis, modified_signal, label="QRS-Complexes", linestyle="--", color="red")
    plt.plot(r_peaks / freq, first_ekg_signal[r_peaks], "x", color="red", label="Detected R-peaks")
    plt.title("ECG Signal with only the QRS-Complexes in red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.show()