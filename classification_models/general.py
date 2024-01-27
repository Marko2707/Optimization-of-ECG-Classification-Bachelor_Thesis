import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import numpy as np
import time as tm
import torch.nn.functional as F

from torch.optim import lr_scheduler

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(GRU, self).__init__()
        # The GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # Dropout layer used for regularization
        self.dropout = nn.Dropout(dropout)
        # Normalization to stabilize training
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        # Multihead layer to capture dependencies
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=1)
        # Fully connected layer for the final classification process
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    # To Apply the different Layers
    def forward(self, x):
        # Pass input through the GRU
        out, _ = self.gru(x)
        # Applying the Layer Normalization
        out = self.layer_norm(out)
        # Application of multi-head self-attention
        out, _ = self.attention(out.permute(1, 0, 2), out.permute(1, 0, 2), out.permute(1, 0, 2))
        # The dropout
        out = self.dropout(out[-1, :, :])
        # Finally pass through
        out = self.fc(out)
        return out
class InceptionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.5):
        super(InceptionModel, self).__init__()

        # Linear layer to match the input size to the hidden size
        self.linear = nn.Linear(input_size, hidden_size)

        # Inception module with Multihead Attention
        self.attention1 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)
        self.attention2 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)
        self.attention3 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Linear layer with activation function
        out = F.relu(self.linear(x))

        # Inception module with Multihead Attention
        out1, _ = self.attention1(out.transpose(0, 1), out.transpose(0, 1), out.transpose(0, 1))
        out2, _ = self.attention2(out.transpose(0, 1), out1, out1)
        out3, _ = self.attention3(out.transpose(0, 1), out2, out2)

        # Concatenate the outputs of the three attention modules along the feature dimension
        out = torch.cat((out1, out2, out3), dim=0)

        # Layer normalization
        out = self.layer_norm(out)

        # Apply dropout
        out = self.dropout(out[-1, :, :])

        # Fully connected layer
        out = self.fc(out)

        return out

def model_run_GRU(x_train, x_test, y_train_multilabel, y_test_multilabel, type_of_data, epochs=5, length_data_compressed=0 ):
    # Assuming your original data shape is (number of samples, timeseries data of sample, lead of ECG)
    # Selecting the 0th lead for each sample
    x_train_selected_lead = x_train[:, :, 0]
    x_test_selected_lead = x_test[:, :, 0]

    # Assuming x_train_selected_lead and x_test_selected_lead have shape (number of samples, timeseries data of sample)
    # You can then add a third dimension to represent the single lead for each sample
    x_train_reshaped = x_train_selected_lead[:, :, np.newaxis]
    x_test_reshaped = x_test_selected_lead[:, :, np.newaxis]
    start = tm.time()
    
    #initializing the model resnet1d_wang, explicitly the function train_resnet1d_wang (Returns the Metric results for each Class Entry)
    #We use all classes and all samples but only one of the 12 leads for performance reasons
    model = train_model(x_train_reshaped, y_train_multilabel, x_test_reshaped, y_test_multilabel, epochs=epochs, batch_size=16, num_splits=10, classes=5, type_of_data=type_of_data)
    end = tm.time()
    time = end - start
    print(f"Training and Evaluation time on {type_of_data}: {time}") 

def train_model(x_train, y_train_multilabel, x_test, y_test_multilabel, epochs=5, batch_size=32, num_splits=10, classes=5, type_of_data="data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_multilabel).float().to(device)

    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test_multilabel).float().to(device)

    #model = InceptionModel(input_size=1, hidden_size=16, num_classes=5).to(device)
    model = GRU(input_size=1, hidden_size=64, num_layers=2, num_classes=5).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=None)




    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
        """
        
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
        """
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
    print(f"Accuracy Each Entry {accuracy_eachEntry}")
    print(f"Precision Each Entry {precision_eachEntry}")
    print(f"Recall Each Entry {recall_eachEntry}")

    roc_auc_eachEntry = roc_auc_score(y_test_entry, y_pred_entry)
    print(f"ROC AUC: {roc_auc_eachEntry}")

    f1_score_eachEntry = f1_score(y_test_entry, y_pred_entry)
    print(f"F-1 Score Each Entry: {f1_score_eachEntry}")
    print("------------------------------------------------------------------------")

    return model