"""
@author: Marko Stankovic

This function "main.py" implements my experimentation setup of my Bachelor Thesis for Optimization of ECG classification through peak detection.
It implements several things and utilizes the PTB-XL dataset which can be found here: https://physionet.org/content/ptb-xl/1.0.3/

The data of the PTB-XL dataset is used to create train and tests sets with according labels. 

Three models are then utilized to train and test on, including a ResNet, an LSTM and a GRU model.
These models are both tested on the raw data, as well as preprocessed optimized data utilizing different Peak Detection method.
The peak detection methods include Pan-Tompkins++, SQRS and a novel own approach.

The performances on the different data and models are returned in the console, as well as the time needed for those processes to make grounds for comparison.

For further information, please refer to the ReadMe and the Bachelor Thesis.
Aspects of the code created by other authors and will be credited in the function

When you run the "main.py", the results will be displayed in the following sequence:
    Raw Data performances on the models
    Pan-Tompkins++ compression runtime
    Pan-Tompkins++ Model performances
    My own Methods of compression runtime
    My own Methods Model performances
    SQRS compression runtime
    SQRS Model Performances
"""
#Imports of Dependencies necessary for execution
import numpy as np
import pandas as pd
import wfdb
import ast
import pandas as pd
import time as tm

#Machine Learning Stuff
from sklearn.preprocessing import MultiLabelBinarizer


#imports of modules in this project to help execution
from helper_functions import is_folder_empty, create_folder_if_not_exists, plot_all_classes, plot_panTompkinsPlusPlus


#Import of PeakDetection Algorithms
#PanTompkins++
from peak_detection_algos.pan_tompkins_plus_plus import Pan_Tompkins_Plus_Plus
from peak_detection_algos.exec_pan_tompkins_plus_plus import preprocess_pantompkinsPlusPlusCompression
#My own adaptive Method
from peak_detection_algos.OwnMethod import preprocess_adaptiveThresholdMethod
#SQRS Method
from peak_detection_algos.SQRS import SQRS_PreperationWithCompression

#imports for ResNet
from classification_models.resnet_execution import model_run_resnet
#import for lstm
from classification_models.lstm import model_runLSTM
#import general modelrunner
from classification_models.gru import model_run_GRU




#---Starting Parameters----------------------------------------
#add the folder where your ptb-xl data is (It can be found under: https://physionet.org/content/ptb-xl/1.0.3/ )
#the path structure under windows should then be like following template:  "C:/Users/marko/Desktop/ptb-xl" as string, the PTB-XL dataset should be inside the folder
pathname = "C:/Users/marko/Desktop/ptb-xl" #Make sure not to have a "/" at the end

#decision if certain plots of the bachelor thesis should be run
plot_choice = "no" # "yes" or "no"
#If you decide to create them, they will have to be closed for the rest of the code to continue

#Decision by which amount the data should get compressed (0-500) (500 is suggested) (It should not be less than 300, as information might get loss)
length_data_compressed = 500 #By how much the data gets reduced in measuring points (1000 is the original data length)

def main(): 
    print("Run is initialized")
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


    if(plot_choice in ["yes", "y", "ye", "j", "ja", "ok", "s", "si"]):
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
    
        print("Plotting all 5 classes, please close the plot for the program to continue.\n")
        plot_all_classes(extracted_classes= extracted_classes, extracted_ecg_ids= extracted_ecg_ids, target_classes=target_classes, x_train=x_train, y_train=y_train, number_no_superclass=number_no_superclass)
        print("Plotting the extracted R-peaks by the PanTompkins++ algorithm, as well as the compression process.\n")
        plot_panTompkinsPlusPlus(x_train=x_train)
        
    #-----End of Plotting----------------------------------------------------------------------------------------------
    

    #Found Samples for different Classes 
    #NORM
    #original_data = x_train[0, :, 0]
    #MI
    #original_data = x_train[8, :, 0]
    #STTC
    #original_data = x_train[22, :, 0]
    #CD
    #original_data = x_train[32, :, 0]
    #HYP
    #original_data = x_train[30, :, 0]


    #----DATA PREPROCESSING---------------------------------------------------------------------------------------------------
    freq = 100
    #the recording is 10 seconds --> 10 000 ms and a average qrs complex is 100ms - 120ms
    window_size = int(0.1 * freq)  # 100 ms window size --> 10 measurement window
    #The window size is used both before and after the found R-peaks to extract a 200ms window that contains the QRS-complex in its entirety
   

    # Machine Learning Phase -------------------------------------------------------------------------------------------------------------------------------------------
    #initilizing a MultiLabelBinarizer to make the labels binary
    mlb = MultiLabelBinarizer() 
    #Transforming the Labels into binary representations [0, 0, 0, 0, 0]
    y_train_multilabel = mlb.fit_transform(y_train)     
    y_test_multilabel = mlb.transform(y_test)

    print("The results of the different methods will be written into the .txts in the results folder\n\n")
    #(1)-------------Model Run on RAW DATA ----------------------------------------------------------------------
 
    
    print("--Run on the original Raw Data is initialized--")
    model_run_resnet(x_train=x_train, x_test=x_test, y_train_multilabel=y_train_multilabel,y_test_multilabel=y_test_multilabel, type_of_data="Raw Data on ResNet Model", epochs=3, length_data_compressed=0)
    model_runLSTM(x_train, x_test, y_train_multilabel, y_test_multilabel, type_of_data="RawData on LSTM Model", epochs=4, length_data_compressed=0)
    model_run_GRU(x_train=x_train, x_test=x_test, y_train_multilabel=y_train_multilabel,y_test_multilabel=y_test_multilabel, type_of_data="Raw Data on GRU Model", epochs=9, length_data_compressed=0)
    print("--End of Raw Data Run --\n\n")

    #-----Modelllauf auf Pan Tompkins daten mit Komprimierung um 500 weniger Daten--------------------------------
    print("_______________________________________________________________________")
    print("Compression Process is initialized with the PanTompkins++ Algorithm")
    PanTompTimeStart = tm.time()
    x_train_panTom_compressed = preprocess_pantompkinsPlusPlusCompression(x_train, window_size= window_size, data_length=length_data_compressed)
    x_test_panTom_compressed = preprocess_pantompkinsPlusPlusCompression(x_test, window_size= window_size, data_length=length_data_compressed)
    PanTompTimeEnd = tm.time()
    print("Time for PanTompkins++ Compression: ", (PanTompTimeEnd-PanTompTimeStart))
    print("_______________________________________________________________________")


    print("--Model run on PanTompkins++ compressed Data--")
    model_run_resnet(x_train=x_train_panTom_compressed, x_test=x_test_panTom_compressed, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="PanTompkins++ Compressed Data on ResNet", epochs=3, length_data_compressed=length_data_compressed)
    model_runLSTM(x_train=x_train_panTom_compressed, x_test=x_test_panTom_compressed, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="PanTompkins++ Compressed Data on LSTM", epochs=4, length_data_compressed=length_data_compressed)
    model_run_GRU(x_train=x_train_panTom_compressed, x_test=x_test_panTom_compressed, y_train_multilabel=y_train_multilabel,y_test_multilabel=y_test_multilabel, type_of_data="PanTompkins++ Compressed Data on GRU", epochs=9, length_data_compressed=length_data_compressed)
    print("--End of Model run on PanTompkins++ compressed Data--\n\n")

    #------------Modelllauf auf meinen Algorithmus-----------------------------------
    print("_______________________________________________________________________")
    print("Compression Process with my own Algorithm is initialized")
    StartTimeOwnmethod= tm.time()
    x_train_myData = preprocess_adaptiveThresholdMethod(x_train, window_size=window_size, data_length=length_data_compressed)
    x_test_myData = preprocess_adaptiveThresholdMethod(x_test, window_size=window_size, data_length=length_data_compressed)
    EndTimeOwnmethod = tm.time()
    print("Time for my own Method: ", (EndTimeOwnmethod-StartTimeOwnmethod))
    print("_______________________________________________________________________")

    print("--Model run on my own Peak Detection method's compressed Data--")
    model_run_resnet(x_train=x_train_myData, x_test=x_test_myData, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="Own Method Compressed Data on ResNet", epochs=3, length_data_compressed=length_data_compressed)
    model_runLSTM(x_train=x_train_myData, x_test=x_test_myData, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="Own Method Compressed Data on LSTM", epochs=4, length_data_compressed=length_data_compressed)
    model_run_GRU(x_train=x_train_myData, x_test=x_test_myData, y_train_multilabel=y_train_multilabel,y_test_multilabel=y_test_multilabel, type_of_data="Own Method Compressed Data on GRU", epochs=9, length_data_compressed=length_data_compressed)
    print("--End of Model run on my own Peak Detection method's compressed Data--\n\n")


    #-----SQRS---------------------------
    print("_______________________________________________________________________")
    print("Compression Process with the SQRS algorithm")
    StartTimeSQRS = tm.time()
    x_train_SQRS = SQRS_PreperationWithCompression(x_train, window_size=window_size, data_length=length_data_compressed)
    x_test_SQRS = SQRS_PreperationWithCompression(x_test, window_size=window_size, data_length=length_data_compressed)
    EndTimeSQRS = tm.time()
    print("TIME FOR SQRS PREPROCESS: ", (EndTimeSQRS-StartTimeSQRS) )
    print("_______________________________________________________________________")
   
    print("--Model run on SQRS Peak Detection method's compressed Data--")
    model_run_resnet(x_train=x_train_SQRS, x_test=x_test_SQRS, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="SQRS Compressed Data on ResNet", epochs=3, length_data_compressed=length_data_compressed)
    model_runLSTM(x_train=x_train_SQRS, x_test=x_test_SQRS, y_train_multilabel=y_train_multilabel,y_test_multilabel= y_test_multilabel, type_of_data="SQRS Compressed Data on LSTM", epochs=4, length_data_compressed=length_data_compressed)
    model_run_GRU(x_train=x_train_SQRS, x_test=x_test_SQRS, y_train_multilabel=y_train_multilabel,y_test_multilabel=y_test_multilabel, type_of_data="SQRS Compressed Data on GRU", epochs=9, length_data_compressed=length_data_compressed)
    print("--End of Model run on SQRS Peak Detection method's compressed Data--\n\n")


 
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

    # Training data creation
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Testing data creation
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    return (X_train, y_train, X_test, y_test, Y)


if __name__ == "__main__":
    main()
