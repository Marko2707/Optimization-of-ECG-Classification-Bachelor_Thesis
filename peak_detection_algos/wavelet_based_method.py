import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pywt

def cwt_r_peak_detection(ecg_data, sampling_rate=100):
    # Define parameters
    wavelet = 'cmor'  # Complex Morlet wavelet
    widths = np.arange(1, 31)  # Scale (width) range for CWT

    # Perform Continuous Wavelet Transform
    
    cwt_matrix, frequencies = pywt.cwt(ecg_data, widths, wavelet, sampling_rate)

    # Extract R-peaks using peaks in the CWT matrix
    cwt_peaks, _ = find_peaks(np.abs(cwt_matrix).sum(axis=0), height=0.5)

    return cwt_peaks

def dwt_r_peak_detection(ecg_data):
    # Decompose signal using Discrete Wavelet Transform
    coeffs = pywt.wavedec(ecg_data, 'db4', level=6)

    # Extract R-peaks using peaks in the approximation coefficients
    dwt_peaks, _ = find_peaks(np.abs(coeffs[0]), height=0.2)

    return dwt_peaks

