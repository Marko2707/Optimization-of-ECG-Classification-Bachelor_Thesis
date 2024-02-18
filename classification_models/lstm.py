"""This module contains the initialization of a basic LSTM model and the functions to train and test said model on the PTB-XL in the main.py"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import numpy as np
import time as tm
import torch.nn.functional as F

"""This model is a basic implementation of a RNN, namely an LSTM. It was written with the help of the PyTorch Documentation
It includes dropout regularization and layer normalization for stabilizing training. 
Additionally, it utilizes a multihead self-attention mechanism to capture dependencies in the input data before passing it through a fully connected layer for final classification.
"""
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(LSTM, self).__init__()
        #The LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        #Dropout layer used for regularization to avoid overfitting
        self.dropout = nn.Dropout(dropout)
        #Normalization to stabilize training
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        #Multi head layer to capture different dependencies
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=2)
        #Fully connected layer for the final classification process
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    #To Apply the different Layers
    def forward(self, x):
        #Pass input through the LSTM
        out, _ = self.lstm(x)
        #Applying the Layer Normalization
        out = self.layer_norm(out)
        #Application of multi head self attention
        out, _ = self.attention(out.permute(1, 0, 2), out.permute(1, 0, 2), out.permute(1, 0, 2))
        #The dropout
        out = self.dropout(out[-1, :, :])
        #Finall Pass through
        out = self.fc(out)
        return out

""" 
This function preprocessed the data and gives it to the model for training
The data here is already preprocessed to be shorter in the instances where compression is applied.
"""
def model_runLSTM(x_train, x_test, y_train_multilabel, y_test_multilabel, type_of_data, epochs=5, length_data_compressed=0 ):
    #Selecting the 0th lead for each sample of general and optimized data
    x_train_selected_lead = x_train[:, :, 0]
    x_test_selected_lead = x_test[:, :, 0]

    #Adaptation of the Shapes to allow the models to work on the data
    x_train_reshaped = x_train_selected_lead[:, :, np.newaxis]
    x_test_reshaped = x_test_selected_lead[:, :, np.newaxis]
    start = tm.time()
    
    #Init of the basic LSTM model
    #We use all classes and all samples but only one of the 12 leads for performance reasons
    model = train_lstm(x_train_reshaped, y_train_multilabel, x_test_reshaped, y_test_multilabel, epochs=epochs, batch_size=16, num_splits=10, classes=5, type_of_data=type_of_data)
    end = tm.time()
    time = end - start
    print(f"Training and Evaluation time on {type_of_data}: {time}\n") 
    
def train_lstm(x_train, y_train_multilabel, x_test, y_test_multilabel, epochs=5, batch_size=32, num_splits=10, classes=5, type_of_data="data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Shapes of Data: {x_train.shape}, {x_test.shape}")
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_multilabel).float().to(device)

    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test_multilabel).float().to(device)

    model = LSTM(input_size=1, hidden_size=64, num_layers=2, num_classes=classes).to(device)
    
    #Defining the Loss Function and the optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    #Splits the data for TRAINING into Train/Test Split
    #NumSplits is 10 in accordance to PTB-XL --> Split into 10 different train and validation sets
    #80% of all 10 is training and 20% is for the validation process (test_size=0.2)
    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=42)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        for train_index, valid_index in sss.split(x_train, y_train_multilabel.argmax(axis=1)):
            x_train_fold, x_valid_fold = x_train[train_index], x_train[valid_index]
            y_train_fold, y_valid_fold = y_train_multilabel[train_index], y_train_multilabel[valid_index]

            x_train_fold_tensor = torch.from_numpy(x_train_fold).float().to(device)
            y_train_fold_tensor = torch.from_numpy(y_train_fold).float().to(device)
            x_valid_fold_tensor = torch.from_numpy(x_valid_fold).float().to(device)
            y_valid_fold_tensor = torch.from_numpy(y_valid_fold).float().to(device)

            train_fold_dataset = TensorDataset(x_train_fold_tensor, y_train_fold_tensor)
            train_fold_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)

            valid_fold_dataset = TensorDataset(x_valid_fold_tensor, y_valid_fold_tensor)
            valid_fold_loader = DataLoader(valid_fold_dataset, batch_size=batch_size, shuffle=False)

            model.train()
            for inputs, labels in train_fold_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_fold_loader:
                    outputs = model(inputs)
                    total_loss += criterion(outputs, labels).item()

            avg_loss = total_loss / len(valid_fold_loader)
            print(f"  Validation Loss: {avg_loss:.4f}")

    
    model.eval()
    with torch.no_grad():
        batch_size = 2  #Adjusting the Batch size to two, to use less vram at once
        num_batches = len(x_test_tensor) // batch_size
        all_outputs = []

        for i in range(num_batches):
            start_idx = i * batch_size # start index (x * batchsize = 0, batchsize, 2batchsize)
            end_idx = (i + 1) * batch_size # end index

            x_test_batch = x_test_tensor[start_idx:end_idx]# Creating the test tensor for the batch

            outputs = model(x_test_batch)
            all_outputs.append(outputs)

        # Concatenate the outputs from all batches along the batch dimension
        test_outputs = torch.cat(all_outputs, dim=0)
        test_loss = criterion(test_outputs, y_test_tensor).item()

    print(f"Test Loss: {test_loss:.4f}")

    #preprocesses the outputs for the metrics
    y_test_pred = torch.sigmoid(test_outputs).cpu().numpy()
    y_test_true = y_test_tensor.cpu().numpy()

    threshold = 0.4
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

    f1_score_eachEntry = f1_score(y_test_entry, y_pred_entry)
    print(f"F-1 Score Each Entry: {f1_score_eachEntry:.4f}")
    print("------------------------------------------------------------------------")

    #returning the results as .txt in the results folder
    path_of_results = "results/"
    file_name = path_of_results + type_of_data + "_PerformanceMetrics.txt"

    with open(file_name, "w") as file:
        file.write(f"{type_of_data} Performance Metrics: \n")
        file.write(f"Accuracy: {accuracy_eachEntry}\n")
        file.write(f"Precision:{precision_eachEntry}\n")
        file.write(f"Recall:{recall_eachEntry}\n")
        file.write(f"ROC AUC: {roc_auc_eachEntry}\n")
        file.write(f"F-1 Score: {f1_score_eachEntry}\n")
    return model

