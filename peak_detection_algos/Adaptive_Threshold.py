
def PeakReturner(ecg_data):
    # List of Peaks as the actual data
    peaks = []
    #List of the according indices
    peaks_index = []
    # count for the indices, since the same data amplitude could be on multiple indices
    count = 0
    # Set threshold for peaks to be added
    threshold = 0.7
    #Adding the first data
    peaks.append(abs(ecg_data[0]))
    peaks_index.append(count)
    for i in ecg_data[1:]:
        count +=1
        #Adding the maximum data, also abs values, since the lowest negative values can be the R-Peaks
        if abs(i) >= threshold * (sum(peaks)/len(peaks)):
            peaks.append(abs(i))
            peaks_index.append(count)
            if len(peaks)>25:
                index_of_min = peaks.index(min(peaks))
                peaks.remove(min(peaks))
                peaks_index.pop(index_of_min)

    #Eliminating multiple indices which point to the same peaks
    i = 0
    while i < len(peaks_index):
        try:
            element = peaks_index[i]
            #checking ig the element index also has further in 10 reach
            if all(abs(element - x) >= 10 for x in peaks_index if x != element):
                count = 0
            else:
                peaks_index.remove(element)
                # After deleting elements, you have to set the i one back, as it would go out of range
                i -= 1
        except IndexError:
            pass
        i += 1
    return peaks_index



