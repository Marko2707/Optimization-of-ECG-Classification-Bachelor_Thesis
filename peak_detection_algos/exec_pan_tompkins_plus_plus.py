"""
This module used the PanTompkins++ method to extract QRS complexes of ECG signal and restructure them into a smaller data frame for faster Machine Learning.
Following module utilizes the PanTompkins++ method from Niaz et al. in their github: Found in the following Repository https://github.com/Niaz-Imtiaz/Pan-Tompkins-Plus-Plus 
"""
from peak_detection_algos.pan_tompkins_plus_plus import Pan_Tompkins_Plus_Plus
import numpy as np

""" Function which only lets the QRS Complexes in the data at the END
    This version of the data is utilized in Appendix B to showcase the effect of less complex data with the same length as the original data.
    This is unused right now in the main.py
"""
def preprocess_pantompkinsPlusPlus(ecg_data, window_size):
    # creates the same data filled with only zeros
    modified_ecg_data = np.zeros_like(ecg_data)
    print("Processing the Data with the PanTompkins++ Algorithm")
    for i in range(ecg_data.shape[0]):
        #T aking the first lead for each sample
        first_lead = ecg_data[i, :, 0]
        if i % 500 == 0:
            print(i)
        # applying the peak detection algorithm on each sampole
        freq = 100
        pan_tompkins = Pan_Tompkins_Plus_Plus()
        r_peaks_indices = pan_tompkins.rpeak_detection(first_lead, freq)
        r_peaks_indices = r_peaks_indices.astype(int)

        for peak_index in r_peaks_indices:
            #setting the first window infront of the rpeak, or start of the data
            start_index = max(0, peak_index - window_size)

            #setting the window after the rpeak or end of the data
            #len(first_lead) is 1000
            end_index = min(len(first_lead), peak_index + window_size + 1)

            # Keep the R-peaks and 100 seconds before and after unchanged
            modified_ecg_data[i, start_index:end_index, 0] = ecg_data[i, start_index:end_index, 0]

    return modified_ecg_data

"""Similiar to the preprocess_pantompkinsPlusPlus, just that it also removes the last 500 (Can be changed) 
measurements and comprimises the qrs complexes nearer to eachother.

This function utilizes the SQRS method to optimize the data and compress it accordingly as summarized in Chapter 6.2 of my bachelor thesis.
"""
def preprocess_pantompkinsPlusPlusCompression(ecg_data, window_size, data_length= 500):
    # creates the same data filled with only zeros
    modified_ecg_data = np.zeros_like(ecg_data)
    print("Processing the Data with the PanTompkins++ Algorithm with Compression")
    for i in range(ecg_data.shape[0]):
        #Taking the first lead for each sample
        first_lead = ecg_data[i, :, 0]
        if i % 500 == 0:
            print(i)
        # applying the peak detection algorithm on each sampole
        freq = 100
        pan_tompkins = Pan_Tompkins_Plus_Plus()
        r_peaks_indices = pan_tompkins.rpeak_detection(first_lead, freq)
        r_peaks_indices = r_peaks_indices.astype(int)

        starting_point = 0
        for peak_index in r_peaks_indices:
            #setting the first window infront of the rpeak, or start of the data
            start_index = max(0, peak_index - window_size)

            #setting the window after the rpeak or end of the data
            #len(first_lead) is 1000
            end_index = min(len(first_lead), peak_index + window_size + 1)
            end_point = end_index - start_index + starting_point

            # Keep the R-peaks and 100 seconds before and after unchanged
            modified_ecg_data[i, starting_point:end_point, 0] = ecg_data[i, start_index:end_index, 0]
            starting_point = end_point + 5


    #Remove the last data_length measurements along the second axis
    modified_ecg_data = modified_ecg_data[:, :-data_length, :]

    print(modified_ecg_data.shape)
    return modified_ecg_data