"""
!For this code to run, you must first initialize the main.py to initialize the data
Following module tests the performance of my own method for Peak detection to find R-peaks.
It compares the R-peaks found by the renowned method of @author: Niaz in his repository https://github.com/Niaz-Imtiaz/Pan-Tompkins-Plus-Plus
with my own found R-peaks through my own methods.

The function returns the amount of identical R-peaks as hits and the missed R-peaks as misses.


"""

import numpy as np
import sys
#Import of PeakDetection Algorithms
#PanTompkins++
from peak_detection_algos.pan_tompkins_plus_plus import Pan_Tompkins_Plus_Plus
#My own adaptive Method
from peak_detection_algos.OwnMethod import adaptive_fixed_RPeakFinder, preprocess_adaptiveThresholdMethod

#-----Parameters------

#Amount of Data to be tested
test_size = 10000 #Goes up to 19601 samples, not more


def testOfOwnMethod(test_size):
    #Loading the Data from the saved NumpyArrays [the main.py must be run to initialize the dataset]
    numpy_path = "NumpyArrays/"
    #Defining the name of the files, to save the data
    x_train_unprocessed = numpy_path + "x_train_unprocessed.npy"
    ecg_data = np.load(x_train_unprocessed)


    pan_tompkins = Pan_Tompkins_Plus_Plus()

    #List for panTomkins found R-peaks and my own methods found R-peaks
    pan_r_peaks_indices_list = []
    own_r_peaks_indices_list = []
    freq=100

    #Check that the chosen test_size is correct
    if test_size > 19601 or test_size < 0:
        sys.exit()

    for i in range(test_size):
        percentage= (i + 1) / test_size*100
        if percentage % 5 == 0:
            print(f"Progress: {percentage}%")
        first_lead = ecg_data[i, :, 0]
        r_peaks_indices = pan_tompkins.rpeak_detection(first_lead, freq)
        pan_r_peaks_indices_list.append(r_peaks_indices)

        r_peaks_indices = adaptive_fixed_RPeakFinder(first_lead)
        own_r_peaks_indices_list.append(r_peaks_indices)
    
    
    #Initializing the amount of hits and misses
    hits = 0
    miss = 0
    
    hitfound = False #If no hit was found, a miss will be addes utilizing hitfound = False

    print("Testing if my found R-Peaks are also found by the PanTompkins++ method.")
    for i in range(len(own_r_peaks_indices_list)):
        for j in own_r_peaks_indices_list[i]:

            #Utilizing a range of 20ms in both directions since the absolute exact location of the R-peak is not necessary. Range(3) --> 0, 1, 2 which is added to the found R-peak
            for x in range(3):
                #if found R-peak is also found by the pantompkins algorithm(or besides by 2 measuring points) --> hit
                if j + x in pan_r_peaks_indices_list[i]:
                    hits += 1
                    hitfound = True
                if j - x in pan_r_peaks_indices_list[i]:
                    hits += 1
                    hitfound = True
            #If the found R-peak is not also in the PanTompkins Data we get a miss
            if not hitfound:
                miss +=1
            hitfound = False #gets returned to False after each tested peak

    print(f"Hits: {hits} || Misses: {miss}")
    print(f"Overlap of {hits /(hits + miss) }%\n")

    print("Testing if R-Peaks found by the PanTompkins++ method are also found by my own method.")
    for i in range(len(pan_r_peaks_indices_list)):
        for j in pan_r_peaks_indices_list[i]:

            #Utilizing a range of 20ms in both directions since the absolute exact location of the R-peak is not necessary. 
            for x in range(3):
                #if found R-peak is also found by the pantompkins algorithm(or besides by 2 measuring points) --> hit
                if j + x in own_r_peaks_indices_list[i]:
                    hits += 1
                    hitfound = True
                if j - x in own_r_peaks_indices_list[i]:
                    hits += 1
                    hitfound = True
            #If the found R-peak is not also in the PanTompkins Data we get a miss
            if not hitfound:
                miss +=1
            hitfound = False #gets returned to False after each peak
    print(f"Hits: {hits} || Misses: {miss}")
    print(f"Overlap of {hits /(hits + miss) }%")

    count = 0
    R = 0

    for i in range(len(own_r_peaks_indices_list)):
        count += len(own_r_peaks_indices_list[i])

    for x in range(len(pan_r_peaks_indices_list)):
        R += len(pan_r_peaks_indices_list[x])

    print("Found R-peaks by my novel method: ", count)
    print("Found R-peaks by the Pan-Tompkins++ method:", R)


if __name__ == "__main__":
    testOfOwnMethod(test_size=test_size)




