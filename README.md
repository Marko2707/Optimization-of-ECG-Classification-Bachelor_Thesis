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



The main file is to be executed to test the  results of models and for the execution of several plots of the ECG Data.
You'll have to set the path name in the main.py file to the one where the PTB-XL dataset is. You can find the dataset here: https://physionet.org/content/ptb-xl/1.0.3/
The data of the PTB-XL dataset is used to create train and tests sets with according labels. The data will be saved into the two folders NumpyArrays and PandaSeries for further use later, it is important to let the first initialization of the data go trough as otherwise the data might be fragmented. 

Three models are then utilized to train and test on, including a ResNet, an LSTM and a GRU model.
These models are both tested on the raw data, as well as preprocessed optimized data utilizing different Peak Detection methods.
The peak detection methods include Pan-Tompkins++, SQRS and a novel own approach.

The performances on the different data and models are returned in the console and as outputs in the results folder, as well as the time needed for those processes to make grounds for comparison.
Running the "main.py" code will provide you with several outputs that guide you through the process of what is happening. Each run of the models will provide also validation losses of every Epoch. The performance metrics afterward will be presented in the console with 14 decimal places and are calculated using the metrics module from "sklearn". In addition to the performance metrics, the time required for both the model training and evaluation runs, as well as the preprocessing runs with the different peak detection methods will be displayed in seconds. 

In addition to the main.py file, there is an "ownMethodEval.py" file in the same location. This file runs the evaluation test of my peak detection method in comparison to the Pan-Tompkins++ method as was described in Chapter 5.4. If you are interested in also reproducing those results, you will have to initialize the data in the "main.py", after which you can run it at any time.

This code utilizes a ResNet model, namely the resnet1d_wang model sourced from the PTB-XL Benchmark that can be found here: https://github.com/helme/ecg_ptbxl_benchmarking
In addition it uses two established peak detection methods that can be found here: https://arxiv.org/abs/2211.03171 , https://pubmed.ncbi.nlm.nih.gov/33670719/

The code also includes several passages of code which is commented out and not utilized right now but incorporated some form of test or function. This was left in the code purposefully to allow for further possible work on the topic afterwards or to conduct further testing, be it by the reviewers or me. 

For further information on the general topics, please refer to the bachelor thesis. Or write me a mail under following email adress: s7081251@stud.uni-frankfurt.de