#!/usr/bin/env python
# coding: utf-8

# # Imports and functions



import pandas as pd
import numpy as np
import os
import csv
import argparse
import time
import optuna
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score


import torchvision
import engine
import helper_functions 
from helper_functions import set_seeds
from engine import train_with_early_stopping

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, f1_score
from collections import deque  # Import deque for early stopping
import warnings

# Suppress torch UserWarning related to the initialization of in_features in 2nd layer conv
warnings.filterwarnings("ignore", category=UserWarning, message="Initializing zero-element tensors is a no-op")

# Reset the warning filters if necessary
# warnings.resetwarnings()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
torch.set_num_threads(10)

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# Function to perform one-hot encoding for DNA sequences
def one_hot_encode(sequence):
    encoding = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    one_hot_sequence = [encoding.get(base, [0, 0, 0, 0]) for base in sequence]
    return np.array(one_hot_sequence)


# # Iterate throuht TFs of target_names_6.txt
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Script for training CNN models on TF data.')
parser.add_argument('tf_name', type=str, help='Name of the TF (TensorFlow)')

args = parser.parse_args()

# Extract TF name from command-line arguments
tf = args.tf_name
print(tf)

file_path = f'/mnt/raid1/thalassini/home/filtered_fimo_1kb/{tf.upper()}/datasets/data.csv'
df = pd.read_csv(file_path)
df


# # Sequence length inspection and normalization



lengths = [len(item) for item in df['data']] 
print(f'Max length: {max(lengths)}, Min length: {min(lengths)}')

if max(lengths) == min(lengths):  
    cut_sequences = df['data']
else:
    # Pad or cut sequences to ensure they all have the same length
    limit = 200
    cut_sequences = [seq[:limit] for seq in df['data']]




# # Define X,y variables




X=[one_hot_encode(sequence) for sequence in cut_sequences]
X = torch.from_numpy(np.array(X)).type(torch.float)

y=np.array(df['class'])
y = torch.from_numpy(y).type(torch.float)


# # Split the data in train, val and test set

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


# Split the data into a stratified hold-out validation set and the rest
stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_val_index, test_index in stratified_splitter.split(X, y):
    X_train_val, X_test = X[train_val_index], X[test_index]
    y_train_val, y_test = y[train_val_index], y[test_index]

# Split the remaining data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42)



# # Create datasets and Dataloaders




from torch.utils.data import DataLoader, TensorDataset

set_seeds()

# Expand the dimensions of y to make it 2-dimensional
y_train = y_train.view(-1, 1)
y_test = y_test.view(-1,1)
y_val = y_val.view(-1,1)
# Concatenate the complement of y to create a 2-dimensional tensor
y_train_one_hot = torch.cat([1 - y_train, y_train], dim=1).float()
y_test_one_hot = torch.cat([1 - y_test, y_test], dim=1).float()
y_val_one_hot = torch.cat([1 - y_val, y_val], dim=1).float()


# Move tensors to the same device
X_train, y_train_one_hot, y_train = X_train.to(device), y_train_one_hot.to(device), y_train.to(device)
X_val, y_val_one_hot, y_val = X_val.to(device), y_val_one_hot.to(device), y_val.to(device)
X_test, y_test_one_hot, y_test = X_test.to(device), y_test_one_hot.to(device), y_test.to(device)



# Create datasets
train_dataset = TensorDataset(X_train, y_train_one_hot, y_train)
val_dataset = TensorDataset(X_val, y_val_one_hot, y_val)
test_dataset = TensorDataset(X_test, y_test_one_hot, y_test)


set_seeds()
batch_size=64


# Convert to PyTorch DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# # Computing class weights

from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

# Assuming we have defined dataset
# train_dataset = TensorDataset(X_train, y_train_one_hot, y_train)

# Extract the labels from the dataset
_, _, y_train = zip(*train_dataset)

# Count occurrences of each class
class_0_count = sum(1 for label in y_train if label == 0)
class_1_count = sum(1 for label in y_train if label == 1)

# Total length of y_train
total_samples = len(y_train)

# Calculate class weights
weights = [total_samples / class_0_count, total_samples / class_1_count]


# Print the list of class weights
print(f"{tf.upper()} Class Weights List:", weights)

weights = torch.tensor(weights).to(device)



"""
# # Optimising  a 2 layer CNN with Maxpooling and Dropout



class Conv2(nn.Module):
    def __init__(self, out_channels1, kernel_size1, out_channels2, kernel_size2, dropout_rate):
        super(Conv2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=out_channels1, kernel_size=kernel_size1)

        #self.conv2 = nn.Conv1d(in_channels = out_channels1, out_channels = out_channels2, kernel_size=kernel_size2)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc = torch.nn.Linear(in_features=0, out_features=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(torch.tanh(self.conv1(x)), 5)
        #x = F.max_pool1d(torch.tanh(self.conv2(x)), 5)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        # Dynamically set the in_features for the fc layer
        if self.fc.in_features == 0:
            self.fc.in_features = x.size(1)
            self.fc = nn.Linear(in_features=x.size(1), out_features=2)  

        out = self.fc(x)
        return out
"""

class Conv_v0(torch.nn.Module):

    def __init__(self, out_channels1, kernel_size1, dropout_rate):
        super(Conv_v0, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=4, out_channels= out_channels1, kernel_size = kernel_size1)
        self.activation = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(kernel_size=5)
        
        self.dropout = torch.nn.Dropout(p=dropout_rate) 
        self.fc = torch.nn.Linear(in_features=0,out_features=2)
        #elf.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x= x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        
        # Reshape the output of the max pooling layer before passing it to the fully connected layer
        x = x.view(x.size(0), -1)
              
        x=self.dropout(x)
        
        # Dynamically set the in_features for the fc layer
        if self.fc.in_features == 0:
            self.fc.in_features = x.size(1)
            self.fc = nn.Linear(in_features=x.size(1), out_features=2)  
      
        x = self.fc(x)
       #x = self.sigmoid(x)
        return x



# Define the objective function
def objective_conv0(trial):
    # Define hyperparameters to be optimized
    out_channels1 = trial.suggest_int("out_channels1", 3, 5)
    kernel_size1 = trial.suggest_int("kernel_size1", 10, 25 )
    #out_channels2 = trial.suggest_int("out_channels2", 3, 5)
    #kernel_size2 = trial.suggest_int("kernel_size2", 10, 20)
    dropout_rate = trial.suggest_float("dropout_rate", 0.5, 0.9)

  
    model = Conv_v0(out_channels1, kernel_size1, dropout_rate)

    loss_fn = nn.BCEWithLogitsLoss(weight=weights)

    # Define oprimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Define exponetntial lr with lr_scheduler
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.01)


    results, _ = engine.train_with_early_stopping(model=model,
                                                  train_dataloader=train_loader,
                                                  valid_dataloader=test_loader,
                                                  optimizer=optimizer,
                                                  loss_fn=loss_fn,
                                                  epochs= 600,
                                                  device=device)


    val_mcc = results["valid_mcc"][-1]
    # Return the MCC for optimization
    return val_mcc





#### Run optimization ####


start = time.time()
# Create a study
study = optuna.create_study(direction="maximize")
study.optimize(objective_conv0, n_trials=50)

end = time.time()
execution_time = (end-start)/60 

#### Save study Results ####

# Get the home directory
base_directory = os.path.expanduser("~")
sub_directory = f"CNNs/{tf}"
save_directory = os.path.join(base_directory, sub_directory)
os.makedirs(save_directory, exist_ok=True)
study_file_name =  "CNN1_trial_results.txt"

study_path = os.path.join(save_directory, study_file_name)

# Open the file in write mode
with open(study_path, "w") as file:
    file.write("Number of finished trials: {}\n".format(len(study.trials)))
    file.write("Params and MCC for each trial:\n")

    for trial in study.trials:
        file.write(f"Trial number {trial.number}:\n")
        file.write(f"    Value (MCC): {trial.value}\n")
        file.write("    Params:\n")
        for key, value in trial.params.items():
            file.write(f"        {key}: {value}\n")

        file.write(f"Best trial: {study.best_trial}\n")

    # Write execution time to the file
    file.write(f"Total execution time: {execution_time} minutes\n")

print(f"CNN1 Optimization results saved to: {file_path}")
print(f"Total execution time: {execution_time:.2f} minutes\n")


#### Extract and Save the best parameters ####
best_trial = study.best_trial
best_trial_params = best_trial.params

params_file_name = "CNN1_best_params.csv"

params_path=os.path.join(save_directory, params_file_name)


import csv
with open(params_path, 'w', newline='') as csvfile:
    # Define the CSV writer
    csv_writer = csv.writer(csvfile)

    # Write header
    header = ['Parameter', 'Value']
    csv_writer.writerow(header)

    # Write best trial parameters
    for key, value in best_trial_params.items():
        csv_writer.writerow([key, value])

print(f"Best study parameters saved to: {params_path}")


model = Conv_v0(**best_trial_params)
loss_fn = nn.BCEWithLogitsLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)   
exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.01)


##### Train the model again with best hyperparameters #####


results, model = engine.train_with_early_stopping(model=model,
                                                  train_dataloader=train_loader,
                                                  valid_dataloader=test_loader,
                                                  optimizer=optimizer,
                                                  loss_fn=loss_fn,
                                                  epochs=600,
                                                  device=device)



##### Save model and results ######
results_file_path = os.path.join(save_directory, "training_results.csv")
model_file_path = os.path.join(save_directory, "trained_model.pth")

os.makedirs(results_file_path, exist_ok=True)
os.makedirs(model_file_path, exist_ok=True)

# Save the PyTorch model
torch.save(model.state_dict(), model_file_path)
print(f"Model saved to: {model_file_path}")

# Save the last metrics of training
last_values = {key: values[-1] for key, values in results.items()}



with open(results_file_path , 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header
    csv_writer.writerow(['Metric', 'Last_Value'])

    # Write data
    for key, value in last_values.items():
        csv_writer.writerow([key, value])

print(f"Best model and results saved to: {save_directory}")






##### Plot and Save curves #####

helper_functions.plot_loss_curves(results)
plt.savefig(os.path.join(save_directory, f'{tf}_CNN1_ADAM_BCEWithLogits_seq=200b.png'));plt.title(f"Loss Curves 1{tf}_CNN1_ADAM_BCEWithLogits_seq=200b.png")


    

    














