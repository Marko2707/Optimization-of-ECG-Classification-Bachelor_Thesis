# Bachelor Thesis: Optimization of ECG Signal Preprocessing through Peak Detection

## Overview

This repository contains the code used for **optimizing** ECG signal preprocessing through **peak detection**. The main focus is on evaluating different peak detection methods and their impact on the performance of machine learning models trained on ECG data. The peak detection algorithms are used to compress the ECG data into their QRS complexes and the machine learning models are subsequenty tested on both compressed and raw data and compared. 

The work showcased significant improvements in runtime, whilst only minimally impacting performance scores such as Accuracy, Precision etc. 

## Disclaimer
The Python version used was **Python 3.11.6** for testing. 

Before executing the code, ensure that you have the following dependencies installed:

- Numpy
- PyTorch
- wfdb
- ast
- pandas
- matplotlib
- sklearn
- peakutils
- scipy
- fastai v1.0.61

To install fastai v1, use the following command:
pip install fastai==1.0.61

## Usage

To test the results of the models and generate ECG data plots, execute the `main.py` file.

Before running the `main.py` file, ensure to set the path name to the location of the PTB-XL dataset! You can download the dataset [here](https://physionet.org/content/ptb-xl/1.0.3/). It can be found in **lines 62-66** where it is elaborated how to give the path. The dataset has to be extracted in the folder you put to the pathname variable. 

Under Windows the format should be the following "C:/Users/user/ptb-xl/" --> Dont forget an "/" at the end. 

The PTB-XL dataset is utilized to create training and testing sets with corresponding labels. The data is then saved into two folders, namely `NumpyArrays` and `PandaSeries`, for future use. 

It is crucial to allow the initial data initialization to proceed without interruption to prevent data fragmentation!

## Models and Methods

The code employs three models: ResNet, LSTM, and GRU. These models are tested on both raw data and preprocessed optimized data using various Peak Detection methods, including Pan-Tompkins++, SQRS, and a novel custom approach.

Performance metrics for different data and models are displayed in the console and saved in the results folder. Additionally, the time taken for model training, evaluation runs, and preprocessing runs with different peak detection methods are provided in seconds for comparison.

## Runnable Files

- `main.py`: Executing this file guides you through the process, providing outputs at each step. Validation losses for every epoch are displayed during model runs, and performance metrics are presented with 14 decimal places using the `metrics` module from `sklearn` as well as the time the models and peak detection methods take.
  
- `ownMethodEval.py`: This file conducts an evaluation test of my custom peak detection method compared to Pan-Tompkins++. To reproduce these results, initialize the data in `main.py` and then run this file. The method is further explained in **Chapter 5.4** of my bachelor thesis.

## Modules and Files

- `classification_models`:

  - `lstm.py` contains the GRU model and the functions to execute it on the PTB-XL comressed and raw data

  - `gru.py` contains the GRU model and the functions to execute it on the PTB-XL comressed and raw data

  - `resnet_execution.py` Contains the necessary functions to execute the model on the PTB-XL compressed and raw data

  - `base_model.py` module from the PTB-XL Benchmark for the [resnet1d_wang model](https://github.com/helme/ecg_ptbxl_benchmarking)

  - `convid.py` module from the PTB-XL Benchmark for the [resnet1d_wang model](https://github.com/helme/ecg_ptbxl_benchmarking)

  - `resnet1d.py` module from the PTB-XL Benchmark containing the [resnet1d_wang model](https://github.com/helme/ecg_ptbxl_benchmarking) for experimentation

- `peak_detection_algos`:

  - `exec_pan_tompkins_plus_plus.py` module which executes the PanTomp++ to extract the R-peaks and compress the data

  - `OwnMethod.py` module contains my peak detection method + Compression of data function

  - `exec_pan_tompkins_plus_plus.py` contains the PanTompkins++ method from Md Niaz Imtiaz [PanTomp++](https://arxiv.org/abs/2211.03171)

  - `SQRS.py` module contains my interpretation of the SQRS peak detection method based on the work of Lu Wu et al. [SQRS](https://pubmed.ncbi.nlm.nih.gov/33670719/) and subsequent compression function


- `helper_functions.py` module containing helping functions such as plots, checks for folders etc.

- `main.py` contains the main experimentation run utilizing all functions above

- `ownMethodEval.py` contains the testing of my own method compared to thhe PanTompkins++ Method. For this module to run, you have to initialize the data in `main.py`.



## References

- Data Loading Function: [PTB-XL_data_load](https://physionet.org/content/ptb-xl/1.0.3/) contained with the PTB-XL dataset from Nils Strodthoff et al. 
- Peak Detection Methods:
- ResNet model: [resnet1d_wang model](https://github.com/helme/ecg_ptbxl_benchmarking) from the Work of Nils Strodthoff et al. 
- Peak Detection Methods:
  - [PanTompkins++](https://arxiv.org/abs/2211.03171) from Md Niaz Imtiaz et al. Work
  - [SQRS](https://pubmed.ncbi.nlm.nih.gov/33670719/) written with the help of Lu Wu et al. Work

The passages used from these works are referenced in the code. In addition, a lot of the work was done with the help of PyTorch, Sklearn documentations and several tutorials. 

## Notes

The code contains commented-out passages that are not currently utilized but may serve for future work or additional testing, either by reviewers or the author.

For more detailed information, please refer to the bachelor thesis or contact me via email at s7081251@stud.uni-frankfurt.de.
