# Bachelor Thesis: Optimization of ECG Signal Preprocessing through Peak Detection

## Overview

This repository contains the code used for optimizing ECG signal preprocessing through peak detection. The main focus is on evaluating different peak detection methods and their impact on the performance of machine learning models trained on ECG data.

## Disclaimer

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
- fastai v1

To install fastai v1, use the following command:
pip install fastai==1.0.61

## Usage

To test the results of the models and generate ECG data plots, execute the main file.

Before running the `main.py` file, ensure to set the path name to the location of the PTB-XL dataset. You can download the dataset [here](https://physionet.org/content/ptb-xl/1.0.3/). It can be found in lines 62-66 where it is elaborated how to give the path. The dataset has to be extracted in the folder you put to the pathname variable. 

The PTB-XL dataset is utilized to create training and testing sets with corresponding labels. The data is then saved into two folders, namely NumpyArrays and PandaSeries, for future use. It is crucial to allow the initial data initialization to proceed without interruption to prevent data fragmentation!

## Models and Methods

The code employs three models: ResNet, LSTM, and GRU. These models are tested on both raw data and preprocessed optimized data using various Peak Detection methods, including Pan-Tompkins++, SQRS, and a novel custom approach.

Performance metrics for different data and models are displayed in the console and saved in the results folder. Additionally, the time taken for model training, evaluation runs, and preprocessing runs with different peak detection methods are provided in seconds for comparison.

## Additional Files

- `main.py`: Executing this file guides you through the process, providing outputs at each step. Validation losses for every epoch are displayed during model runs, and performance metrics are presented with 14 decimal places using the `metrics` module from `sklearn` as well as the time the models and peak detection methods take.
  
- `ownMethodEval.py`: This file conducts an evaluation test of my custom peak detection method compared to Pan-Tompkins++. To reproduce these results, initialize the data in `main.py` and then run this file.

## References

- ResNet model: [resnet1d_wang model](https://github.com/helme/ecg_ptbxl_benchmarking)
- Peak Detection Methods:
  - [Method 1](https://arxiv.org/abs/2211.03171)
  - [Method 2](https://pubmed.ncbi.nlm.nih.gov/33670719/)

## Notes

The code contains commented-out passages that are not currently utilized but may serve for future work or additional testing, either by reviewers or the author.

For more detailed information, please refer to the bachelor thesis or contact me via email at s7081251@stud.uni-frankfurt.de.