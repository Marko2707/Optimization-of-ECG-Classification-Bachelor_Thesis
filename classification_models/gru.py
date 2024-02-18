"""Following module includes the structure of a basic GRU model and functions to train and evaluate the model on the PTB-XL data"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit #was utilized to test differences, is now left out to showcase the effect of the SSS in my bachelor thesis
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import numpy as np
import time as tm


"""
This model is a basic implementation of a Gated Recurrent Unit (GRU) neural network with bidirectional architecture. 
It includes dropout regularization and layer normalization for stabilizing training. 
Additionally, it utilizes a multihead self-attention mechanism to capture dependencies in the input data before passing it through a fully connected layer for final classification.
"""
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(GRU, self).__init__()
        #GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        #Dropout layer used for regularization of overfitting
        self.dropout = nn.Dropout(dropout)
        #Normalization to stabilize training
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        #Multihead layer to capture dependencies better with 2 heads
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=2)
        #Fully connected layer for the final classification process
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    #To Apply the different Layers
    def forward(self, x):
        #Pass input through the GRU
        out, _ = self.gru(x)
        # pplying the Layer Normalization
        out = self.layer_norm(out)
        # Application of multi-head self-attention
        out, _ = self.attention(out.permute(1, 0, 2), out.permute(1, 0, 2), out.permute(1, 0, 2))
        #The dropout process
        out = self.dropout(out[-1, :, :])
        #Finally pass through
        out = self.fc(out)
        return out

""" 
This function preprocessed the data and gives it to the model for training
The data here is already preprocessed to be shorter in the instances where compression is applied.
"""
def model_run_GRU(x_train, x_test, y_train_multilabel, y_test_multilabel, type_of_data, epochs=5, length_data_compressed=0):

    #Selecting the first lead that was possibly updated lead for each sample
    x_train_selected_lead = x_train[:, :, 0]
    x_test_selected_lead = x_test[:, :, 0]

    #Adaptation of the Shapes to allow the models to work on the data
    x_train_reshaped = x_train_selected_lead[:, :, np.newaxis]
    x_test_reshaped = x_test_selected_lead[:, :, np.newaxis]
    
    start = tm.time()
    #Initialization of model training
    model = train_model(x_train_reshaped, y_train_multilabel, x_test_reshaped, y_test_multilabel, epochs=epochs, batch_size=16, num_splits=10, classes=5, type_of_data=type_of_data)
    end = tm.time()
    time = end - start
    print(f"Training and Evaluation time on {type_of_data}: {time}\n") 

"""Following function trains the GRU model and returns several performance metrics for it"""
def train_model(x_train, y_train_multilabel, x_test, y_test_multilabel, epochs=5, batch_size=32, num_splits=10, classes=5, type_of_data="data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #adaptive setting for functioning

    #creation of tensors for training and testing
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_multilabel).float().to(device)

    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test_multilabel).float().to(device)

    #init of GRU model defined above in this module
    model = GRU(input_size=1, hidden_size=64, num_layers=2, num_classes=5).to(device)

    #Loss Function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    #Tensor Training set and data loader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    #Training the model across the given epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        #Evaluation during training
        model.eval()
        with torch.no_grad():
            batch_size = 2  # Adjust the batch size according to your available memory
            num_batches = len(x_test_tensor) // batch_size

            all_outputs = []

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size

                x_test_batch = x_test_tensor[start_idx:end_idx]
                y_test_batch = y_test_tensor[start_idx:end_idx]

                outputs = model(x_test_batch)
                all_outputs.append(outputs)

            # Concatenate the outputs from all batches along the batch dimension
            test_outputs = torch.cat(all_outputs, dim=0)

            test_loss = criterion(test_outputs, y_test_tensor).item()
        print(f"Test Loss: {test_loss:.4f}")
    #Evaluation after training
    model.eval()
    with torch.no_grad():
        batch_size = 2  # Adjust the batch size according to your available memory
        num_batches = len(x_test_tensor) // batch_size

        all_outputs = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            x_test_batch = x_test_tensor[start_idx:end_idx]
            y_test_batch = y_test_tensor[start_idx:end_idx]

            outputs = model(x_test_batch)
            all_outputs.append(outputs)

        # Concatenate the outputs from all batches along the batch dimension
        test_outputs = torch.cat(all_outputs, dim=0)

        test_loss = criterion(test_outputs, y_test_tensor).item()

    print(f"Final Test Loss: {test_loss:.4f}")

    #Creating the labels accordingly
    y_test_pred = torch.sigmoid(test_outputs).cpu().numpy()

    y_test_true = y_test_tensor.cpu().numpy()

    threshold = 0.4 #threshold by which to round the predictions
    y_test_pred_rounded = np.where(y_test_pred >= threshold, 1, 0)

    #Making each sample into the correct predictions
    y_test_entry = y_test_true.flatten().tolist()
    y_pred_entry = y_test_pred_rounded.flatten().tolist()

    #Results printing
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

    f1_score_eachEntry = f1_score(y_test_entry, y_pred_entry)
    print(f"F-1 Score Each Entry: {f1_score_eachEntry}")
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