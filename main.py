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


#Machine Learning Stuff
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer


#imports of modules in this project
from helper_functions import is_folder_empty, create_folder_if_not_exists
from classification_models.resnet1d import resnet1d_wang, resnet1d18 



from peak_detection_algos.pan_tompkins_plus_plus import Pan_Tompkins_Plus_Plus

#imports for ResNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit



#add the folder where your ptb-xl data is 
# the path structure under windows should than be: "C:/Users/marko/Desktop/bachelor_arbeit/code/PTB-XL", the PTB-XL dataset should be inside the path
pathname = "C:/Users/marko/Desktop/bachelor_arbeit/code/"

#setting the samplerate to 100Hz (choosing)
sampling_rate=100

#Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(pathname+"scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def main(): 

    print("Code initialized")

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

    """
    #Counts the amount of multilabel cases
    count = 0
    for ecg_id, ecg_classes in y_train.items():
        # Check if the ecg_id has already been processed
        if len(ecg_classes) > 1:
            count += 1
    print(f"count: {count}")
    """

    #----Plotting of the 5 ECG Superclasses----------
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
    plot_choice = "no"

    if(plot_choice in ["yes", "y", "ye", "j", "ja", "ok", "s", "si"]):
        
        # Iterate over y_train until each target class is extracted at least once
        for ecg_id, ecg_classes in y_train.items():
            # Check if the ecg_id has already been processed
            #if ecg_id in extracted_ecg_ids:
            #    continue

            # Check if the list of classes is not empty
            if ecg_classes:
                # Extract the first class
                ecg_class = ecg_classes[0]

                # Check if the class is in the target classes, has not been extracted yet,
                # and there are no other classes present with len(ecg_classes) == 1
                if ecg_class in target_classes and ecg_class not in extracted_classes and len(ecg_classes) == 1:
                    # Perform your desired actions with ecg_id and ecg_class
                    print(f"ecg_id: {ecg_id}, Klasse: {ecg_class}")

                    # Add the extracted class and the according ecg_id to the lists
                    extracted_classes.append(ecg_class)
                    extracted_ecg_ids.append(ecg_id)

                    # Remove the extracted class from target_classes to avoid repeated extraction
                    target_classes.remove(ecg_class)

                # Check if all target classes have been extracted
                if not target_classes:
                    break
            else:
                #Counting the amount of records with no superclass
                number_no_superclass += 1


        #Get all information of y_train of one inext
        #print(Y.loc[y_train.index[0]])

        #!To plot the different Classes remove the block comment
        #Plotting all 5 ECG Samples of each Super-Class:
        
        #Defining the 5 Superclasses I want to plot
        classes_to_plot = ["NORM", "MI", "STTC", "CD", "HYP"]

        # List of the ECG-Lead-Names
        lead_names = ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


    
        # Iterating over all 5 Classes i want to plot
        for ecg_class in classes_to_plot:
            #Finding the index with ecg class == class i want to plot
            indices = [i for i, label in enumerate(extracted_classes[:5]) if label == ecg_class]

            #Iterating over the 5 indices
            for i in indices:
                # extracting the ecg data from the x_train with the ids from before
                ecg_data = x_train[extracted_ecg_ids[i]]
                # creating the x axis as seconds with a sampling reate of 100Hz
                time_axis = np.arange(0, 10, 1/100)

                # Ploting the ecg data for each lead
                for lead in range(ecg_data.shape[1]):
                    plt.plot(time_axis, ecg_data[:, lead], label=f"{lead_names[lead]}")

                # Titel of the plot with each class
                plt.title(f"ECG Signal - Class: {ecg_class}")
                plt.xlabel("Time (s)") # Add the unit to the x-axis label
                plt.ylabel("Amplitude (mV)")  # Add the unit to the y-axis label
                plt.legend()
                plt.show()
        
    #-----End of Plotting--------------------------

    """
    #Erster Umgang mit PanTompkinsPlusPlus
    
    # Extrahieren Sie das erste EKG-Signal (1. Leitung) "NORM"
    first_ekg_signal = x_train[0, :, 0]
    # CD Klasse
    #first_ekg_signal = x_train[31,:,0]

    #Setting the frequency to 100Hz for the PanTompkinsPlusPlus
    freq = 100

    # Initialisieren Sie eine Instanz des Pan_Tompkins_Plus_Plus-Algorithmus
    pan_tompkins = Pan_Tompkins_Plus_Plus()

    # Wenden Sie den Algorithmus auf das erste EKG-Signal an
    r_peaks_indices = pan_tompkins.rpeak_detection(first_ekg_signal, freq)

    # Zeitachse für das Plot
    time_axis = np.arange(0, 10, 1/ freq )

    r_peaks = r_peaks_indices.astype(int)


    # Ausgabe der detektierten R-Peaks-Indizes
    print("Detected R-peaks indices:", r_peaks_indices)

    

    # Plot des EKG-Signals
    plt.figure(figsize=(12, 6))
    plt.plot( first_ekg_signal, label='ECG Signal')
    plt.plot(r_peaks, first_ekg_signal[r_peaks], "x", color="red")
    plt.title('ECG Signal with Detected R-peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

   # Convert R-peaks indices to integers
    r_peaks_indices = r_peaks_indices.astype(int)

    # Create a mask to set values to 0 except within a 100ms window around each R-peak
    mask = np.zeros_like(first_ekg_signal)
    window_size = int(0.1 * freq)  # 100 ms window size

    for r_peak in r_peaks_indices:
        r_peak = int(r_peak)  # Convert to integer
        mask[max(0, r_peak - window_size):min(len(mask), r_peak + window_size + 1)] = 1
    # Apply the mask to the original signal
    modified_signal = first_ekg_signal * mask

    # Ausgabe der detektierten R-Peaks-Indizes
    print("Detected R-peaks indices:", r_peaks_indices)
    # Plot des EKG-Signals mit modifizierten Werten
    plt.figure(figsize=(12, 6))
    plt.plot(first_ekg_signal, label='Original ECG Signal')
    plt.plot(modified_signal, label='Modified Signal', linestyle='--', color='red')
    plt.plot(r_peaks, first_ekg_signal[r_peaks], "x", color="red", label='Detected R-peaks')
    plt.title('ECG Signal with Modified Values around R-peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    """

    freq = 100
    #the recording is 10 seconds --> 10 000 ms and a average qrs complex is 100ms
    window_size = int(0.1 * freq)  # 100 ms window size
    x_train_panTom = preprocess_pantompkinsPlusPlus(x_train, window_size= window_size)
    x_test_panTom = preprocess_pantompkinsPlusPlus(x_test, window_size= window_size)

    # Original data
    original_data = x_train[0, :, 0]

    # Modified data
    modified_data = x_train_panTom[0, :, 0]
    
    # Plots the difference in the data and PanTompkins preprocessed Data
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Original Data', color='blue')
    plt.plot(modified_data, label='Modified Data', color='red', linestyle='dashed')
    plt.title('Comparison Original Data and Modified PanTompkins++ Data')
    plt.xlabel('Measurements 100Hz over 10 Seconds')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.show()
    


    print(f"Shape of x_train {x_train.shape}, Shape of x_test {x_test.shape}, Shape of y_train {y_train.shape} and Shape of y_test {y_test.shape}")
    print(f"First 9 Labels of PandaSeries y_train:{y_train[:8]}")
    print(f"First lead of the first ecg data the first 10{x_train[0][:10][0]}")

    # Used to Change Data to fit model but not anymore TODO Remove or TODO Utilize
    """
    #print(f"y_train Before Update: Test für Multilabel: {y_train[35:45]}" )
    #Changing the Labels to Integers for the Machine Learning Models
    classes = ["NORM", "MI", "STTC", "CD", "HYP"] 
    translation = [1, 2, 3, 4, 5]
    y_train = label_changer_stringToInt(classes, translation, y_train)
    y_test = label_changer_stringToInt(classes, translation, y_test)
    #print(f"y_train Updated Test für Multilabel: {y_train[35:45]}" )

    #resnet1d_model = resnet1d_wang(kernel_size=[7, 5], kernel_size_stem=8, stride_stem=2, pooling_stem=True, inplanes=256)
    #print(resnet1d_model)
    # Annahme: Sie verwenden resnet1d_wang, Sie können auch ein anderes Modell wählen
    model = resnet1d_wang(num_classes=5)

    #print(model)
    # Daten in DataLoader umwandeln
    print(x_train.shape)
    print(y_train.shape)
    """

    # Wandeln Sie die Listen von Labels in binäre Vektoren um
    #initilizing a MultiLabelBinarizer to make the labels binary
    mlb = MultiLabelBinarizer()
    #Transforming the Labels into binary representations
    y_train_multilabel = mlb.fit_transform(y_train)
    y_test_multilabel = mlb.transform(y_test)

        
    #Expanding the dimension by one, necessary for the use with the model TODO Remove 
    y_train_onehot = np.expand_dims(y_train_multilabel, axis=1)
    y_test_onehot = np.expand_dims(y_test_multilabel, axis=1)


    #-----Modelllauf auf Pan Tompkins daten----------------------------------------------------------------------
    x_train_panTom = np.transpose(x_train_panTom, (0, 2, 1))
    x_test_panTom = np.transpose(x_test_panTom, (0, 2, 1))

    #starting the countdown, to see how long the raw data model takes
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
    model = train_resnet1d_wang2(x_train_reshaped, y_train_multilabel,  x_test_reshaped, y_test_multilabel, epochs=2, batch_size=32, num_splits=10)
    end = tm.time()
    time = end - start
    #Prints the time it took to train and evalutate the model:
    print(f"Training and Evaluation time on PanTompkins processed Data: {time}")

    
    #--------------Modellauf auf rohdaten----------------------------------------------------------------------



    #Sorting the Data, so that it fits the expected way of the models
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))

    # Assuming x_train and x_test have shapes (19601, 12, 1000) and (2198, 12, 1000), respectively
    x_train_reshaped = x_train[:, 0, :].reshape(-1, 1, 1000)
    x_test_reshaped = x_test[:, 0, :].reshape(-1, 1, 1000)

    print(f"Shapes: {x_train.shape} und {x_test.shape}")
    print(f"Shapes y: {y_train_multilabel.shape} and {y_test_multilabel.shape} ")

    #starting the countdown, to see how long the raw data model takes
    start = tm.time()
    #initializing the model resnet1d_wang, explicitly the function train_resnet1d_wang (Returns the Metric results for each Class Entry)
    #We use all classes and all samples but only one of the 12 leads for performance reasons
    model = train_resnet1d_wang2(x_train_reshaped, y_train_multilabel,  x_test_reshaped, y_test_multilabel, epochs=10, batch_size=32, num_splits=10)
    end = tm.time()
    time = end - start
    #Prints the time it took to train and evalutate the model:
    print(f"Training and Evaluation time on unprocessed Data: {time}")
    

   
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
"""
author = Marko Stankovic
Following code was written with the help of https://github.com/helme/ecg_ptbxl_benchmarking and ChatGPT
"""
def train_resnet1d_wang2(x_train, y_train_multilabel, x_test, y_test_multilabel, epochs=10, batch_size=32, num_splits=10, classes = 5):
    """
    Trains the 1D ResNet model resnet1d_wang from the https://github.com/helme/ecg_ptbxl_benchmarking on ECG data.

    Parameters:
    - x_train: Training data .
    - y_train_multilabel:encoded training data labels.
    - x_test: Test data .
    - y_test_multilabel:encoded test data labels.
    - epochs: Number of training epochs.
    - batch_size: Batch size for training.
    - num_splits: Number of splits for stratified k-fold cross-validation.
    """

    #Checks for Nvidia cuda support, otherwise it uses the CPU
    #On my Hardware it was the RTX 2070 Super, which should use CUDA 12.1 Pytorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Converting the data to Pytorch Tensors to allow for learning
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_multilabel).float().to(device)
    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test_multilabel).float().to(device)

    #Creating a Pytorch Dataset and a pytorch DataLoader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #Intitilizing the model from: https://github.com/helme/ecg_ptbxl_benchmarking
    model = resnet1d_wang(input_channels=1, num_classes= classes).to(device)
    #Defining the Loss Function and the optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Splits the data for Training, num_splits is 10 as wanted by the PTB-XL Benchmark
    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=42)

    #Training the model across the given epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        #
        for train_index, valid_index in sss.split(x_train, y_train_multilabel.argmax(axis=1)):
            # Dividing the data in training and validation
            # This part is important, so that under each training iteration the model can get tested to improve
            x_train_fold, x_valid_fold = x_train[train_index], x_train[valid_index]
            y_train_fold, y_valid_fold = y_train_multilabel[train_index], y_train_multilabel[valid_index]


            #Creating further tensors the training data
            #Dividing it into training and validation tensors
            x_train_fold_tensor = torch.from_numpy(x_train_fold).float().to(device)
            y_train_fold_tensor = torch.from_numpy(y_train_fold).float().to(device)
            x_valid_fold_tensor = torch.from_numpy(x_valid_fold).float().to(device)
            y_valid_fold_tensor = torch.from_numpy(y_valid_fold).float().to(device)

            # creation of pytorch datasets and dataloaders for the training 
            train_fold_dataset = TensorDataset(x_train_fold_tensor, y_train_fold_tensor)
            train_fold_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)
            # creation of pytorch datasets and dataloaders but for the validation 
            valid_fold_dataset = TensorDataset(x_valid_fold_tensor, y_valid_fold_tensor)
            valid_fold_loader = DataLoader(valid_fold_dataset, batch_size=batch_size, shuffle=False)

            #training the model
            model.train()
            for inputs, labels in train_fold_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs#.unsqueeze(1)  
                loss = criterion(outputs, labels)#bringing it into the right shape for the training
                loss.backward()
                optimizer.step()

            # Validating the model
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_fold_loader:
                    outputs = model(inputs)
                    total_loss += criterion(outputs, labels).item()

            avg_loss = total_loss / len(valid_fold_loader)
            print(f"  Validation Loss: {avg_loss:.4f}")

    # testing the model on how it runs
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()

    print(f"Test Loss: {test_loss:.4f}")

    #Gives a Prediction as Float as to how likely the prediction data is true
    y_test_pred = torch.sigmoid(test_outputs).cpu().numpy()
    y_test_true = y_test_tensor.cpu().numpy()

    print(f"Shapes of data {y_test_pred.shape} and {y_test_true.shape}")

    #threshold value on which the prediction percentage is supposed to be done into True Value
    threshold = 0.75 

    # Rounding the values of the prediction into 1 and 0 based on the threshold above
    y_test_pred_rounded = np.where(y_test_pred >= threshold, 1, 0)
   
    """
    #Calculation of the Metrics across all values of each sample
    accuracy = accuracy_score(y_test_true[:, 0], y_test_pred_rounded)
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(y_test_true[:, 0], y_test_pred_rounded, average='weighted')
    print(f"Precision: {precision:.4f}")

    recall = recall_score(y_test_true[:, 0], y_test_pred_rounded, average='weighted')
    print(f"Recall: {recall:.4f}")
    """

    #Actual Values
    #Calculating the Value for each instance of each class instead of samplewise
    y_test_entry = y_test_true.reshape(-1).astype(int)
    y_test_entry = y_test_entry.tolist()

    #Prediction Values
    #Reshaping each individual array and concatenate them
    y_pred_entry = np.concatenate([arr.reshape(-1) for arr in y_test_pred_rounded])
    #Converting to a  list, each value being in the right orderin the prediction as in the actual values
    y_pred_entry = y_pred_entry.tolist()

    print("Metrics tested for each Instance of all classes:")

    accuracy_eachEntry = accuracy_score(y_test_entry, y_pred_entry)
    precision_eachEntry = precision_score(y_test_entry, y_pred_entry, average="weighted")
    recall_eachEntry = recall_score(y_test_entry, y_pred_entry, average="weighted")
    print(f"Accuracy Each Entry {accuracy_eachEntry}")
    print(f"Precision Each Entry {precision_eachEntry}")
    print(f"Recall Each Entry {recall_eachEntry}")

    roc_auc_eachEntry = roc_auc_score(y_test_entry, y_pred_entry, average="weighted")
    print(f"ROC AUC: {roc_auc_eachEntry}")

    # Annahme: y_test_entry und y_pred_entry sind Numpy-Arrays oder Listen
    fmax_score_eachEntry = f1_score(y_test_entry, y_pred_entry, average="weighted")

    print(f"F-Max Score Each Entry: {fmax_score_eachEntry}")

    """
    print(y_test_entry[:10])
    print(y_pred_entry[:10])
    # Berechne F1-Score
    #f1 = f1_score(y_test_true[:, 0, :], y_test_pred_rounded[:, 0], average='weighted')
    #print(f"F1-Score: {f1:.4f}")
    print(y_test_entry[0:200:5])
    print(y_pred_entry[0:200:5])
    print("-----------------")

    print(y_test_entry[1:200:5])
    print(y_pred_entry[1:200:5])
    print("-----------------")


    print(y_test_entry[2:200:5])
    print(y_pred_entry[2:200:5])
    print("-----------------")

    print(y_test_entry[3:200:5])
    print(y_pred_entry[3:200:5])
    print("-----------------")

    print(y_test_entry[4:200:5])
    print(y_pred_entry[4:200:5])
    """
    return model

 
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
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # Split data into train and test
    test_fold = 10

    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    return (X_train, y_train, X_test, y_test, Y)

"""Unused Function right now"""
def label_changer_stringToInt(classes, translation, y_train):
    for ecg_id, ecg_classes in y_train.items():
        if len(ecg_classes) < 1:
            y_train[ecg_id] = 0
        elif len(ecg_classes) == 1:
            y_train[ecg_id] = translation[classes.index(ecg_classes[0])]
        elif len(ecg_classes) == 2:
            y_train[ecg_id][0] = translation[classes.index(ecg_classes[0])] 
            y_train[ecg_id][1] = translation[classes.index(ecg_classes[1])]
        elif len(ecg_classes) == 3:
            y_train[ecg_id][0] = translation[classes.index(ecg_classes[0])] 
            y_train[ecg_id][1] = translation[classes.index(ecg_classes[1])]
            y_train[ecg_id][2] = translation[classes.index(ecg_classes[2])] 
        elif len(ecg_classes) == 4:
            y_train[ecg_id][0] = translation[classes.index(ecg_classes[0])] 
            y_train[ecg_id][1] = translation[classes.index(ecg_classes[1])]
            y_train[ecg_id][2] = translation[classes.index(ecg_classes[2])] 
            y_train[ecg_id][3] = translation[classes.index(ecg_classes[3])]
        else:
            y_train[ecg_id][0] = translation[classes.index(ecg_classes[0])] 
            y_train[ecg_id][1] = translation[classes.index(ecg_classes[1])]
            y_train[ecg_id][2] = translation[classes.index(ecg_classes[2])] 
            y_train[ecg_id][3] = translation[classes.index(ecg_classes[3])]
            y_train[ecg_id][4] = translation[classes.index(ecg_classes[4])]
    return y_train


if __name__ == "__main__":
    main()
