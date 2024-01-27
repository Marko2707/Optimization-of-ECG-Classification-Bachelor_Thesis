import numpy as np
import pandas as pd
import time as tm

#Machine Learning Stuff
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score


#imports for ResNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

from classification_models.resnet1d import resnet1d_wang



def model_run_resnet(x_train, x_test, y_train_multilabel, y_test_multilabel, type_of_data, epochs=1, length_data_compressed=0 ):
    #Sorting the Data, so that it fits the expected way of the models
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))

    # Assuming x_train and x_test have shapes (19601, 12, 1000) and (2198, 12, 1000), respectively
    x_train_reshaped = x_train[:, 0, :].reshape(-1, 1, (1000 - length_data_compressed))
    x_test_reshaped = x_test[:, 0, :].reshape(-1, 1, (1000 - length_data_compressed))

    print(f"Shapes x: {x_train.shape} und {x_test.shape}")
    print(f"Shapes y: {y_train_multilabel.shape} and {y_test_multilabel.shape} ")

    #starting the countdown, to see how long the raw data model takes
    start = tm.time()
    #initializing the model resnet1d_wang, explicitly the function train_resnet1d_wang (Returns the Metric results for each Class Entry)
    #We use all classes and all samples but only one of the 12 leads for performance reasons
    #model = train_lstm(x_train, y_train_multilabel, x_test, y_test_multilabel, epochs=10, batch_size=32, num_splits=10, classes=5, type_of_data="data")
    model = train_resnet1d_wang(x_train_reshaped, y_train_multilabel,  x_test_reshaped, y_test_multilabel, epochs=epochs, batch_size=32, type_of_data=type_of_data)
    #model = train_resnet1d_wang2(x_train_reshaped, y_train_multilabel,  x_test_reshaped, y_test_multilabel, epochs=epochs, batch_size=32, num_splits=10, type_of_data=type_of_data)
    end = tm.time()
    time = end - start
    print(f"Training and Evaluation time on {type_of_data}: {time}")

"""
author = Marko Stankovic
Following code was written with the help of https://github.com/helme/ecg_ptbxl_benchmarking and ChatGPT
"""
def train_resnet1d_wang(x_train, y_train_multilabel, x_test, y_test_multilabel, epochs=10, batch_size=32, num_splits=10, classes = 5, type_of_data="data"):
    """
    Trains the 1D ResNet model resnet1d_wang from the https://github.com/helme/ecg_ptbxl_benchmarking on ECG data.

    The different Parameters:
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

    #Converting the data to Pytorch Tensors datastruchtures to allow for learning
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_multilabel).float().to(device)
    
    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test_multilabel).float().to(device)

    #---NOT USED right now, due to the StratifiedShuffleSplit----
    #Creating a Pytorch Dataset and a pytorch DataLoader
    #train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    #NOT USED due to the StratifiedShuffleSplit, since we train with the splits
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #Intitilizing the model from: https://github.com/helme/ecg_ptbxl_benchmarking
    model = resnet1d_wang(input_channels=1, num_classes= classes).to(device)

    #Defining the Loss Function and the optimizer
    criterion = nn.BCEWithLogitsLoss() #Binary Cross Entropy Loss (Either True or False)
    optimizer = optim.Adam(model.parameters(), lr=0.001) #adjusts the models parameters based on the loss function

    #Splits the data for TRAINING into Train/Test Split
    #NumSplits is 10 in accordance to PTB-XL --> Split into 10 different train and validation sets
    #80% of all 10 is training and 20% is for the validation process (test_size=0.2)
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    #Training the model across the given epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Gives the 10 random splits for training for each epochy
        for train_index, valid_index in sss.split(x_train, y_train_multilabel.argmax(axis=1)):
            # Dividing the data into training and validation
            x_train_fold, x_valid_fold = x_train[train_index], x_train[valid_index]
            y_train_fold, y_valid_fold = y_train_multilabel[train_index], y_train_multilabel[valid_index]

            # Creating further tensors of the newly formed training data
            x_train_fold_tensor = torch.from_numpy(x_train_fold).float().to(device)
            y_train_fold_tensor = torch.from_numpy(y_train_fold).float().to(device)
            x_valid_fold_tensor = torch.from_numpy(x_valid_fold).float().to(device)
            y_valid_fold_tensor = torch.from_numpy(y_valid_fold).float().to(device)

            # Creation of PyTorch datasets and dataloaders for training
            train_fold_dataset = TensorDataset(x_train_fold_tensor, y_train_fold_tensor)
            train_fold_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)  #Shuffle the training data

            # Creation of PyTorch datasets and dataloaders for validation
            valid_fold_dataset = TensorDataset(x_valid_fold_tensor, y_valid_fold_tensor)
            valid_fold_loader = DataLoader(valid_fold_dataset, batch_size=batch_size, shuffle=True)

            # Training the Model on 80%
            model.train()
            for inputs, labels in train_fold_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validating the model during the Training process according to the SSS split 20%
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_fold_loader:
                    outputs = model(inputs)
                    total_loss += criterion(outputs, labels).item()

            avg_loss = total_loss / len(valid_fold_loader)
            print(f"Validation Set Loss: {avg_loss:.4f}")


    # Testing the model on the actual final Validation Data
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()

    print(f"Test Set Loss: {test_loss:.4f}")
    #Preparing the Data for the Evaluatiin 
    #Gives a Prediction as Float as to how likely the prediction data is true
    y_test_pred = torch.sigmoid(test_outputs).cpu().numpy()
    y_test_true = y_test_tensor.cpu().numpy()
    print(f"Shapes of data {y_test_pred.shape} and {y_test_true.shape}")
    #threshold value on which the prediction percentage is supposed to be done into True Value
    threshold = 0.4 #TODO Move to the Start, 0.4 resulted in the best results
    
    print(f"Threshold is set at --{threshold}--")
    #Rounding the values of the prediction into 1 and 0 based on the threshold above
    y_test_pred_rounded = np.where(y_test_pred >= threshold, 1, 0)
   

    y_test_entry = y_test_true.flatten().tolist()
    y_pred_entry = y_test_pred_rounded.flatten().tolist()



    print(f"--------Results for {type_of_data}------------------------------------")

    print("Metrics tested for each Instance of all classes:")

    accuracy_eachEntry = accuracy_score(y_test_entry, y_pred_entry)
    precision_eachEntry = precision_score(y_test_entry, y_pred_entry)
    recall_eachEntry = recall_score(y_test_entry, y_pred_entry)
    print(f"Accuracy Each Entry {accuracy_eachEntry}")
    print(f"Precision Each Entry {precision_eachEntry}")
    print(f"Recall Each Entry {recall_eachEntry}")

    roc_auc_eachEntry = roc_auc_score(y_test_entry, y_pred_entry)
    print(f"ROC AUC: {roc_auc_eachEntry}")

    # Annahme: y_test_entry und y_pred_entry sind Numpy-Arrays oder Listen
    f1_score_eachEntry = f1_score(y_test_entry, y_pred_entry)

    print(f"F-1 Score Each Entry: {f1_score_eachEntry}")
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