import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import numpy as np
import time as tm
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
    
def model_runLSTM(x_train, x_test, y_train_multilabel, y_test_multilabel, type_of_data, epochs=1, length_data_compressed=0 ):
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
    model = train_lstm(x_train_reshaped, y_train_multilabel, x_test_reshaped, y_test_multilabel, epochs=epochs, batch_size=32, num_splits=10, classes=5, type_of_data=type_of_data)
    end = tm.time()
    time = end - start
    print(f"Training and Evaluation time on {type_of_data}: {time}") 
    
def train_lstm(x_train, y_train_multilabel, x_test, y_test_multilabel, epochs=5, batch_size=32, num_splits=10, classes=5, type_of_data="data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_multilabel).float().to(device)

    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test_multilabel).float().to(device)

    model = LSTMModel(input_size=1, hidden_size=64, num_layers=3, num_classes=5).to(device)


    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=None)

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
        test_outputs = model(x_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()

    print(f"Test Loss: {test_loss:.4f}")

    y_test_pred = torch.sigmoid(test_outputs).cpu().numpy()
    y_test_true = y_test_tensor.cpu().numpy()

    threshold = 0.4
    y_test_pred_rounded = np.where(y_test_pred >= threshold, 1, 0)

    y_test_entry = y_test_true.reshape(-1).astype(int)
    y_test_entry = y_test_entry.tolist()

    y_pred_entry = np.concatenate([arr.reshape(-1) for arr in y_test_pred_rounded])
    y_pred_entry = y_pred_entry.tolist()

    print(f"--------Results for {type_of_data}------------------------------------")

    print("Metrics tested for each Instance of all classes:")

    accuracy_eachEntry = accuracy_score(y_test_entry, y_pred_entry)
    precision_eachEntry = precision_score(y_test_entry, y_pred_entry, average="weighted")
    recall_eachEntry = recall_score(y_test_entry, y_pred_entry, average="weighted")
    print(f"Accuracy Each Entry {accuracy_eachEntry}")
    print(f"Precision Each Entry {precision_eachEntry}")
    print(f"Recall Each Entry {recall_eachEntry}")

    roc_auc_eachEntry = roc_auc_score(y_test_entry, y_pred_entry, average="weighted")
    print(f"ROC AUC: {roc_auc_eachEntry}")

    fmax_score_eachEntry = f1_score(y_test_entry, y_pred_entry, average="weighted")
    print(f"F-Max Score Each Entry: {fmax_score_eachEntry}")
    print("------------------------------------------------------------------------")

    return model

# Example usage:
# model = train_lstm(x_train, y_train_multilabel, x_test, y_test_multilabel, epochs=10, batch_size=32, num_splits=10, classes=5, type_of_data="data")
