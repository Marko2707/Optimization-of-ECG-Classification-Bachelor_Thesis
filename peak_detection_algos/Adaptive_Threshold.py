import numpy as np

def adaptive_fixed_RPeakFinder(ecg_data):
    # List of Peaks as the actual data
    peaks = []
    #List of the according indices
    peaks_index = []
    # count for the indices, since the same data amplitude could be on multiple indices
    count = 0
    maximum = max(abs(min(ecg_data)), max(ecg_data))
    # Set threshold to be set the entire time
    fixed_threshold = 0.7
    # Set threshold for peaks to be added adaptively
    adaptive_threshold = 0.8
    # Set threshold for peaks to be removed
    removal_threshold = 0.75
    #Adding the first data
    peaks.append(abs(ecg_data[0]))
    peaks_index.append(count)
    for i in ecg_data[1:]:
        count +=1
        #Adding the maximum data, also abs values, since the lowest negative values can be the R-Peaks
        if abs(i) >= adaptive_threshold * (sum(peaks)/len(peaks)) and abs(i) >= maximum * fixed_threshold :
            peaks.append(abs(i))
            peaks_index.append(count)
            if len(peaks)>25:
                index_of_min = peaks.index(min(peaks))
                peaks.remove(min(peaks))
                peaks_index.pop(index_of_min)
    

    #Removal Function with removal-threshold
    count = 0
    for i in peaks:
        if i <= removal_threshold* (sum(peaks)/len(peaks)):
            #count += 1# to see the problem of not having this for loop, using STTC example
            peaks.remove(i)
            peaks_index.pop(count)
            count += 1
    

    #Eliminating multiple indices which point to the same peaks and are adjacent to eachother
    #i = 0
    #Going from right to left
    
    i = len(peaks_index)
    while i > 0: #len(peaks_index):
        try:
            element = peaks_index[i]
            #checking ig the element index also has further in 10 reach
            if all(abs(element - x) >= 10 for x in peaks_index if x != element):
                count = 0
            else:
                peaks_index.remove(element)
                # After deleting elements, you have to set the i one back, as it would go out of range
                #i -= 1
                i += 1
        except IndexError:
            pass
        i -= 1
    
    return peaks_index

def preprocess_adaptiveThresholdMethod(ecg_data, window_size, data_length= 500):
    # creates the same data filled with only zeros
    modified_ecg_data = np.zeros_like(ecg_data)

    print("Processing the Data with the my own adaptive threshold method Algorithm with Compression")
    for i in range(ecg_data.shape[0]):
        #T aking the first lead for each sample
        first_lead = ecg_data[i, :, 0]
        if i % 500 == 0:
            print(i)
        # applying the peak detection algorithm on each sampole
        freq = 100
        
        r_peaks_indices = adaptive_fixed_RPeakFinder(first_lead)

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


    #Remove the last 500 measurements along the second axis
    modified_ecg_data = modified_ecg_data[:, :-data_length, :]

    print(modified_ecg_data.shape)
    return modified_ecg_data

