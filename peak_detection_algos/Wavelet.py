import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pywt

def cwt_r_peak_detection_alg(ecg_data, sampling_rate=100, min_height_ratio=0.8, min_peaks=8, max_peaks=13):
    # Perform Continuous Wavelet Transform
    cwt_matrix, _ = pywt.cwt(ecg_data, np.arange(1, len(ecg_data)+1), 'cmor', sampling_rate)

    # Aggregate the CWT matrix along scales
    cwt_aggregated = np.abs(cwt_matrix).sum(axis=0)

    # Calculate the average height of the peaks in the aggregated signal
    avg_height = np.mean(cwt_aggregated)

    # Find peaks above the min_height_ratio of the average height
    cwt_peaks, _ = find_peaks(cwt_aggregated, height=min_height_ratio * avg_height)

    # Select up to max_peaks peaks, ensuring at least min_peaks are selected
    selected_peaks = cwt_peaks[:max(min_peaks, min(max_peaks, len(cwt_peaks)))]

    return selected_peaks