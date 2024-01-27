"""
Author: Marko Stankovic

Aspects of the code were created by other authors and will be credited in the function

"""
#Imports of Dependencies necessary for execution
import numpy as np
import pandas as pd
import wfdb
import ast
import pandas as pd
import time as tm
import matplotlib.pyplot as plt

#TODO Remove
import sys

#Machine Learning Stuff
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer


#imports of modules in this project
from helper_functions import is_folder_empty, create_folder_if_not_exists, plot_all_classes, plot_panTompkinsPlusPlus
from classification_models.resnet1d import resnet1d_wang, resnet1d18 


#Import of PeakDetection Algorithms
#PanTompkins++
from peak_detection_algos.pan_tompkins_plus_plus import Pan_Tompkins_Plus_Plus
#My own adaptive Method
from peak_detection_algos.OwnMethod import adaptive_fixed_RPeakFinder, preprocess_adaptiveThresholdMethod
#SQRS Method
from peak_detection_algos.SQRS import SQRS_PreperationWithCompression
#Wavelet Method
from peak_detection_algos.Wavelet import cwt_r_peak_detection_alg


#imports for ResNet
from classification_models.resnet_execution import model_run_resnet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix


#import for lstm
from classification_models.lstm import model_runLSTM

#import general modelrunner
from classification_models.general import model_run_GRU

#TODO ADD Several Parameters as Global Variables, to be adjusted here

#---Starting Parameters----------------------------------------
#add the folder where your ptb-xl data is (It can be found under: https://physionet.org/content/ptb-xl/1.0.3/ )
#the path structure under windows should then be like following template: "C:/Users/user/Desktop/bachelor_arbeit/code/PTB-XL", the PTB-XL dataset should be inside the path
pathname = "C:/Users/marko/Desktop/bachelor_arbeit/code/"

plot_choice = "no"

#setting the samplerate to 100Hz (choosing)
sampling_rate=100

#If you have not installed the necessary dependecies, such as fastai, you can run the code without the ResNet1d_wang model
choiceForResnet = "NO" #Set to "YES" or "NO"

def main(): 

    print("Run is initialized")
    print("Initiating ML, Plot, etc.")#TODO Let user choose what should happen, make it print what happens

    #----Data Initiation process start -------------------------------------------------------------------------------------------------------------
    #Pre-Saving the Data for faster testing, first initialization takes the longest
    #Defining the Folder where the NumpArr and PandSeries are saved
    numpy_path = "NumpyArrays/"
    series_path = "PandaSeries/"
    #Defining the name of the files, to save the data
    x_train_unprocessed = numpy_path + "x_train_unprocessed.npy"
    x_test_unprocessed = numpy_path + "x_test_unprocessed.npy"
    y_train_path = series_path + "y_train.pkl"  
    y_test_path = series_path + "y_test.pkl"
    
    #creates the folders to save the arrays
    create_folder_if_not_exists(numpy_path)
    create_folder_if_not_exists(series_path)

    #If the data is not pre saved, it initilizes it for the first time
    #Else uses the already preloaded data, making the runtime way faster
    if (is_folder_empty(numpy_path) or is_folder_empty(series_path)):
        print("Data must be loaded from the PTB-XL Folder")
        #Timetest
        data_load_time_start = tm.time()
        #Taking in the Sets
        x_train, y_train, x_test, y_test, Y = dataCreation(pathname)
        data_load_time_end = tm.time()
        #Return how long it took to get all the data and process it into our needed formats
        print("Data Load time from PTB-XL Database: ", data_load_time_end - data_load_time_start)
        
        #Saving the Numpy Arrays for x_test and x_train
        np.save(x_train_unprocessed, x_train)
        np.save(x_test_unprocessed, x_test)

        #Saving PandaSeries
        series_path = "BA/PandaSeries/"
        y_train.to_pickle(y_train_path)
        y_test.to_pickle(y_test_path)

        #Test of Equal Datasets:
        x_train_unprocessed = np.load(x_train_unprocessed)
        x_test_unprocessed = np.load(x_test_unprocessed)
        print(f"Test if the x_data is the same: {x_train_unprocessed.shape == x_train.shape and x_test_unprocessed.shape == x_test.shape}")

        # Load Series from pickle
        y_train_pickle = pd.read_pickle(y_train_path)
        y_test_pickle = pd.read_pickle(y_test_path)
        print(f"Test if the y_data is the same: {y_train_pickle.shape == y_train.shape and y_test_pickle.shape == y_test.shape}")

    else: 
        #---Loading the Data from the NumpyArrays and PandaSeries------------------------
        #Loading the NumpyArrays
        npy_load_time_start = tm.time()
        x_train = np.load(x_train_unprocessed)
        x_test = np.load(x_test_unprocessed)
        # Load Series from pickle
        y_train= pd.read_pickle(y_train_path)
        y_test = pd.read_pickle(y_test_path)
        npy_load_time_end = tm.time()
        #Return the time it takes to take the pre saved data 
        print(f"Data load time from the saved Folders: {npy_load_time_end - npy_load_time_start}")
    #----Data Initiation process ending -------------------------------------------------------------------------------------------------------------


    """
    #Counts the amount of multilabel cases
    count = 0
    for ecg_id, ecg_classes in y_train.items():
        # Check if the ecg_id has already been processed
        if len(ecg_classes) > 1:
            count += 1
    print(f"count: {count}")
    """

    #----Plotting of the 5 ECG Superclasses------------------------------------------------------------------------
    #Getting for each superclass one ECG Sample
    #All my target classes
    target_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    #Get these classes extracted
    extracted_classes = []
    #Get the according ecg_id for the class
    extracted_ecg_ids = []
    #Number of records with no superclass
    number_no_superclass = 0
    #Getting one example for each class WITHOUT any other being present
    print("Do you want to get all the Classes Ploted?")
    #plot_choice = input("[Y]es for Plots, Press Enter to skip: \n").lower()
    #Is right now hardcoded to no for faster testing TODO change later
    
    #plot_choice = "yes"

    if(plot_choice in ["yes", "y", "ye", "j", "ja", "ok", "s", "si"]):
        plot_all_classes(extracted_classes= extracted_classes, extracted_ecg_ids= extracted_ecg_ids, target_classes=target_classes, x_train=x_train, y_train=y_train, number_no_superclass=number_no_superclass)
        plot_panTompkinsPlusPlus(x_train=x_train)
        
    #-----End of Plotting----------------------------------------------------------------------------------------------
    
    #--Pan Tompkins Algorithm without 
    #x_train_panTom = preprocess_pantompkinsPlusPlus(x_train, window_size= window_size)
    #x_test_panTom = preprocess_pantompkinsPlusPlus(x_test, window_size= window_size)

    # Original data
    #NORM
    original_data = x_train[0, :, 0]
    #MI
    #original_data = x_train[8, :, 0]
    #STTC
    #original_data = x_train[22, :, 0]
    #CD
    #original_data = x_train[32, :, 0]
    #HYP
    #original_data = x_train[30, :, 0]


    #----------My Method TEST----------------------------------------------------------TODO REMOVE
    """
    adaptive_peaks = adaptive_fixed_RPeakFinder(ecg_data=original_data)
    print(adaptive_peaks)
    plt.figure(figsize=(12, 6))
    plt.title("R-Peak Detection with Adaptive and Fixed Thresholds")
    plt.plot(original_data, label="ECG Signal")
    plt.plot(adaptive_peaks , original_data[adaptive_peaks], "rx", label="Detected R-Peaks")
    plt.xlabel("Measurements with 100Hz over 10 Seconds")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.show()
    """

    #----DATA PREPROCESSING---------------------------------------------------------------------------------------------------
    length_data_compressed = 500 #TODO Außerhalb der Funktion als globale Variable
    freq = 100
    #the recording is 10 seconds --> 10 000 ms and a average qrs complex is 100ms
    window_size = int(0.1 * freq)  # 100 ms window size --> 10 measurement window


    #x_train_myData = preprocess_adaptiveThresholdMethod(ecg_data=x_train, window_size=window_size, data_length=length_data_compressed, )
    #print(f"Shape of My Data{x_train_myData.shape}")
   

    # Machine Learning Phase -------------------------------------------------------------------------------------------------------------------------------------------
    #initilizing a MultiLabelBinarizer to make the labels binary
    mlb = MultiLabelBinarizer() 
    #Transforming the Labels into binary representations [0, 0, 0, 0, 0]
    y_train_multilabel = mlb.fit_transform(y_train)     
    y_test_multilabel = mlb.transform(y_test)
    

    #(1)-------------Model Run on RAW DATA ----------------------------------------------------------------------
    #model_run_resnet(x_train=x_train, x_test=x_test, y_train_multilabel=y_train_multilabel,y_test_multilabel=y_test_multilabel, type_of_data="Raw Data on ResNet", epochs=6, length_data_compressed=0)

    #model_runLSTM(x_train, x_test, y_train_multilabel, y_test_multilabel, type_of_data="RawData on LSTM", epochs=1, length_data_compressed=0)
    model_run_GRU(x_train=x_train, x_test=x_test, y_train_multilabel=y_train_multilabel,y_test_multilabel=y_test_multilabel, type_of_data="Raw Data on Inceptionmodel", epochs=1, length_data_compressed=0)
    
    print("------Run on raw Data----------------")
    model_run_resnet(x_train=x_train, x_test=x_test, y_train_multilabel=y_train_multilabel,y_test_multilabel=y_test_multilabel, type_of_data="Raw Data on ResNet", epochs=1, length_data_compressed=0)
    model_runLSTM(x_train, x_test, y_train_multilabel, y_test_multilabel, type_of_data="RawData on LSTM", epochs=4, length_data_compressed=0)
    model_run_GRU(x_train=x_train, x_test=x_test, y_train_multilabel=y_train_multilabel,y_test_multilabel=y_test_multilabel, type_of_data="Raw Data on Inceptionmodel", epochs=9, length_data_compressed=0)
    

    


    #-----Modelllauf auf Pan Tompkins daten mit Komprimierung um 500 weniger Daten--------------------------------
    PanTompTimeStart = tm.time()
    x_train_panTom_compressed = preprocess_pantompkinsPlusPlusCompression(x_train, window_size= window_size, data_length=length_data_compressed)
    x_test_panTom_compressed = preprocess_pantompkinsPlusPlusCompression(x_test, window_size= window_size, data_length=length_data_compressed)
    PanTompTimeEnd = tm.time()
    print("Time for PanTompkins++ Compression: ", (PanTompTimeEnd-PanTompTimeStart))
    """
    more_modified_data = x_train_panTom_compressed[0, :, 0]
    # Plots the difference in the data and PanTompkins preprocessed Data
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label="Original Data", color="blue")
    #plt.plot(modified_data, label="Modified Data", color="red", linestyle="dashed")
    plt.plot(more_modified_data, label="Compressed Data", color="red", linestyle="dashed")
    plt.title("Comparison Original Data and Modified Compressed PanTompkins++ Data")
    plt.xlabel("Measurements 100Hz over 10 Seconds")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.show()
    """

    #print(f"Shape of x_train {x_train.shape}, Shape of x_test {x_test.shape}, Shape of y_train {y_train.shape} and Shape of y_test {y_test.shape}")
    #print(f"First 9 Labels of PandaSeries y_train:{y_train[:8]}")
    #print(f"First lead of the first ecg data the first 10{x_train[0][:10][0]}")
    print("--RUN ON PANTOMPKINS DATA--------------------------")
    model_run_resnet(x_train=x_train_panTom_compressed, x_test=x_test_panTom_compressed, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="PanTompkins++ Compressed Data", epochs=3, length_data_compressed=length_data_compressed)
    model_runLSTM(x_train=x_train_panTom_compressed, x_test=x_test_panTom_compressed, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="PanTompkins++ Compressed Data", epochs=4, length_data_compressed=length_data_compressed)

    
    #-----Modelllauf auf Pan Tompkins daten---------------------------------------------------------------------- 
    """
    x_train_panTom = np.transpose(x_train_panTom, (0, 2, 1))
    x_test_panTom = np.transpose(x_test_panTom, (0, 2, 1))

    #starting the countdown, to see how long the raw data model takes#
    print("Testing the model on Pan Tompkins data with full measurements")
    start = tm.time()
    print(x_train_panTom.shape)
    print(x_test_panTom.shape)

    # Assuming x_train_panTom and x_test_panTom have shapes (19601, 12, 1000) and (2198, 12, 1000), respectively
    x_train_reshaped = x_train_panTom[:, 0, :].reshape(-1, 1, 1000)
    x_test_reshaped = x_test_panTom[:, 0, :].reshape(-1, 1, 1000)

    print(f"Shapes x: {x_train_reshaped.shape} and {x_test_reshaped.shape}")
    print(f"Shapes y: {y_train_multilabel.shape} and {y_test_multilabel.shape} ")
    #initializing the model resnet1d_wang, explicitly the function train_resnet1d_wang (Returns the Metric results for each Class Entry)
    #We use all classes and all samples but only one of the 12 leads for performance reasons
    model = train_resnet1d_wang2(x_train_reshaped, y_train_multilabel,  x_test_reshaped, y_test_multilabel, epochs=2, batch_size=32, num_splits=10, type_of_data="Uncompressed PanTompkins Data")
    end = tm.time()
    time = end - start
    #Prints the time it took to train and evalutate the model:
    print(f"Training and Evaluation time on PanTompkins processed Data: {time}")
    """
    
    
    #------------Modelllauf auf meinen Algorithmus-----------------------------------
    print("Test on My Algorithm--------------")
    StartTimeOwnmethod= tm.time()
    x_train_myData = preprocess_adaptiveThresholdMethod(x_train, window_size=window_size, data_length=length_data_compressed)
    x_test_myData = preprocess_adaptiveThresholdMethod(x_test, window_size=window_size, data_length=length_data_compressed)
    EndTimeOwnmethod = tm.time()
    print("Time for my own Method: ", (EndTimeOwnmethod-StartTimeOwnmethod))
    model_run_resnet(x_train=x_train_myData, x_test=x_test_myData, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="MyOwnMethod", epochs=3, length_data_compressed=length_data_compressed)
    model_runLSTM(x_train=x_train_myData, x_test=x_test_myData, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="MyOwnMethod", epochs=4, length_data_compressed=length_data_compressed)

    #-----SQRS---------------------------
    print("TEST des SQRS--------------------")
    StartTimeSQRS = tm.time()
    x_train_SQRS = SQRS_PreperationWithCompression(x_train, window_size=window_size, data_length=length_data_compressed)
    x_test_SQRS = SQRS_PreperationWithCompression(x_test, window_size=window_size, data_length=length_data_compressed)
    EndTimeSQRS = tm.time()
    print("TIME FOR SQRS PREPROCESS: ", (EndTimeSQRS-StartTimeSQRS) )
    print("X_TRAIN_SQRS SHAPE: ", x_train_SQRS.shape, "X_TEST_SQRS SHAPE: ", x_test_SQRS.shape)
    model_run_resnet(x_train=x_train_SQRS, x_test=x_test_SQRS, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="SQRS Compressed Data", epochs=3, length_data_compressed=length_data_compressed)
    model_runLSTM(x_train=x_train_SQRS, x_test=x_test_SQRS, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="SQRS Compressed Data", epochs=4, length_data_compressed=length_data_compressed)




def train_resnet1d_wang(x_train, y_train_multilabel, x_test, y_test_multilabel, epochs=10, batch_size=32, classes=5, type_of_data="data"):
    """
    Trains the 1D ResNet model resnet1d_wang from the https://github.com/helme/ecg_ptbxl_benchmarking on ECG data.

    The different Parameters:
    - x_train: Training data.
    - y_train_multilabel: encoded training data labels.
    - x_test: Test data.
    - y_test_multilabel: encoded test data labels.
    - epochs: Number of training epochs.
    - batch_size: Batch size for training.
    - classes: Number of classes.
    - type_of_data: Type of data (e.g., "data").
    """

    # Checks for Nvidia cuda support, otherwise it uses the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Converting the data to PyTorch Tensors data structures to allow for learning
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_multilabel).float().to(device)

    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test_multilabel).float().to(device)

    # Intitilizing the model from: https://github.com/helme/ecg_ptbxl_benchmarking
    model = resnet1d_wang(input_channels=1, num_classes=classes).to(device)

    # Defining the Loss Function and the optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model across the given epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Creating tensors for training data
        x_train_fold_tensor = torch.from_numpy(x_train).float().to(device)
        y_train_fold_tensor = torch.from_numpy(y_train_multilabel).float().to(device)

        # Creation of PyTorch dataset and dataloader for training
        train_fold_dataset = TensorDataset(x_train_fold_tensor, y_train_fold_tensor)
        train_fold_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)

        # Training the Model on the entire dataset
        model.train()
        for inputs, labels in train_fold_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Testing the model on the actual final Validation Data
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()

    print(f"Test Loss: {test_loss:.4f}")
    # Preparing the Data for the Evaluation
    # Gives a Prediction as Float as to how likely the prediction data is true
    y_test_pred = torch.sigmoid(test_outputs).cpu().numpy()
    y_test_true = y_test_tensor.cpu().numpy()

    threshold = 0.4  # TODO Move to the Start, 0.4 resulted in the best results

    print(f"Threshold is set at --{threshold}--")
    # Rounding the values of the prediction into 1 and 0 based on the threshold above
    y_test_pred_rounded = np.where(y_test_pred >= threshold, 1, 0)

    y_test_entry = y_test_true.flatten().tolist()
    y_pred_entry = y_test_pred_rounded.flatten().tolist()

    print(f"--------Results for {type_of_data}------------------------------------")

    print("Metrics tested for each Instance of all classes:")

    accuracy_eachEntry = accuracy_score(y_test_entry, y_pred_entry)
    precision_eachEntry = precision_score(y_test_entry, y_pred_entry)
    recall_eachEntry = recall_score(y_test_entry, y_pred_entry)
    print(f"Accuracy Each Entry {accuracy_eachEntry:.4f}")
    print(f"Precision Each Entry {precision_eachEntry:.4f}")
    print(f"Recall Each Entry {recall_eachEntry:.4f}")

    roc_auc_eachEntry = roc_auc_score(y_test_entry, y_pred_entry)
    print(f"ROC AUC: {roc_auc_eachEntry:.4f}")

    fmax_score_eachEntry = f1_score(y_test_entry, y_pred_entry)
    print(f"F-1 Score Each Entry: {fmax_score_eachEntry:.4f}")
    print("------------------------------------------------------------------------")

    conf_matrix_eachEntry = confusion_matrix(y_test_entry, y_pred_entry)
    print(conf_matrix_eachEntry)

    print(y_test_entry[0:100:5])
    print(y_pred_entry[0:100:5])
    print("-----------------")

    print(y_test_entry[1:100:5])
    print(y_pred_entry[1:100:5])
    print("-----------------")

    print(y_test_entry[2:100:5])
    print(y_pred_entry[2:100:5])
    print("-----------------")

    print(y_test_entry[3:100:5])
    print(y_pred_entry[3:100:5])
    print("-----------------")

    print(y_test_entry[4:100:5])
    print(y_pred_entry[4:100:5])

    return model


""" Function which only lets the QRS Complexes in the data be"""
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

"""Similiar to the preprocess_pantompkinsPlusPlus, just that it also removes the last 500 (Can be changed and is changed in the main) measurements and comprimises the qrs complexes nearer to eachother """
def preprocess_pantompkinsPlusPlusCompression(ecg_data, window_size, data_length= 500):
    # creates the same data filled with only zeros
    modified_ecg_data = np.zeros_like(ecg_data)
    print("Processing the Data with the PanTompkins++ Algorithm with Compression")
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

 
"""
This Function implements the general way the data is loaded from the PTB-XL dataset.
It utilizes most functions given directly by the authors of the PTB-XL dataset and their python file: https://physionet.org/content/ptb-xl/1.0.3/ 

It takes in the pathname under windows as input and returns the train and test data.
"""
def dataCreation(pathname):

    def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    path =  "C:/Users/marko/Desktop/bachelor_arbeit/code/"
    sampling_rate=100
    # load and convert annotation data
    Y = pd.read_csv(path+"ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+"scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

    # Split data into train and test
    test_fold = 10

    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    return (X_train, y_train, X_test, y_test, Y)


if __name__ == "__main__":
    main()
