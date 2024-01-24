#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import argparse
import pandas as pd
import numpy as np
import os
import csv 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, f1_score

import torchvision
import torch
from torch import nn
import torch.nn.functional as F

import engine
import helper_functions
from helper_functions import set_seeds

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, f1_score
from collections import deque  # Import deque for early stopping
import warnings
import csv



# Function to perform one-hot encoding for DNA sequences
def one_hot_encode(sequence):
    encoding = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    one_hot_sequence = [encoding.get(base, [0, 0, 0, 0]) for base in sequence]
    return np.array(one_hot_sequence)


# # Set up device agnostic code

device = "cuda" if torch.cuda.is_available() else "cpu"
device


torch.set_num_threads(10)

# Set dictionary containing length of each tf motif that will be used as a kernel 
kernel = {
    'ATF2': [10] ,
    'ATF3' : [10] ,
    'CEBPB' : [10] ,
    'CREB1' : [8] ,
    'CTCF' : [15] ,
    'E2F4' : [13] ,
    'EGR1' : [10] ,
    'EFL1' : [9] ,
    'ELK1' : [9] ,
    'FOS' : [8] ,
    'FOXA1' : [8] ,
    'GABPA' : [10] ,
    'JUN' : [14] ,
    'JUND' : [11] ,
    'MAFK' : [10] ,
    'MAX' : [6] ,
    'MAZ' : [8] ,
    'MXI1' : [6] ,
    'MYC' : [8] ,
    'NRF1' : [11] ,
    'RELA' : [10] ,
    'REST' : [20] ,
    'RFX5' : [14] ,
    'SP1' : [9] ,
    'SRF' : [16] ,
    'TCF7L2' : [9] ,
    'TCF12' : [7] ,
    'TEAD4' : [8] ,
    'USF1' : [10] ,
    'USF2' : [10] ,
    'YY1' : [12] ,
    'ZBTB33' : [10] ,
    'ZNF274' : [12]
}

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Script for training CNN models on TF data.')
parser.add_argument('tf_name', type=str, help='Name of the TF (TensorFlow)')

args = parser.parse_args()

# Extract TF name from command-line arguments
tf = args.tf_name
print(tf)


file_path = f'/mnt/raid1/thalassini/home/filtered_fimo_200b_balanced/{tf.upper()}/datasets/data.csv'

df = pd.read_csv(file_path)



### Sequence length inspection and normalization ###
lengths= [len(item) for item in df['data']] 
print(f'Max length:{max(lengths)}, Min length: {min(lengths)}')



# # Define X,y variables


cut_sequences=df["data"]

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
torch.manual_seed(42)

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
set_seeds()
train_dataset = TensorDataset(X_train, y_train_one_hot, y_train)
val_dataset = TensorDataset(X_val, y_val_one_hot, y_val)
test_dataset = TensorDataset(X_test, y_test_one_hot, y_test)





# Convert to PyTorch DataLoader
set_seeds()
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

"""

  ### Construct a 1 and 2 layer CNN with Maxpooling and Dropout ###


class Conv1(nn.Module):
    def __init__(self, out_channels1, kernel_size1, dropout_rate):
        super(Conv1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=out_channels1, kernel_size=kernel_size1)

        #self.conv2 = nn.Conv1d(in_channels = out_channels1, out_channels = out_channels2, kernel_size=kernel_size2)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc = torch.nn.Linear(in_features=276, out_features=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(torch.tanh(self.conv1(x)), 2)
        #x = F.max_pool1d(torch.tanh(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

      
          

        out = self.fc(x)
        return out




"""



class Conv_v0(torch.nn.Module):

    def __init__(self,tf_kernel):
        super(Conv_v0, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=4, out_channels=3, kernel_size=17)
        self.activation = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(kernel_size=tf_kernel)
        
        self.dropout = torch.nn.Dropout(p=0.25) 
        #in_features = 108 for 200kb, 588 for 1kb, 1188 for 2kb
        self.fc =  torch.nn.LazyLinear(out_features=2)
        #self.sigmoid = torch.nn.Sigmoid() will not be used since its intergraded in BCEWithLogitsLoss()

    def forward(self, x):
        x= x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        
        # Reshape the output of the max pooling layer before passing it to the fully connected layer
        x = x.view(x.size(0), -1)
        
        #print("Size after reshaping:", x.size())
        x=self.dropout(x)
        x = self.fc(x)
        #x = self.sigmoid(x)
        return x



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
print("Class Weights List:", weights)




# # Set up loss function and optimizer



from torch.optim import lr_scheduler
weights = torch.tensor(weights).to(device)

set_seeds()

# Initialize model instance
model = Conv_v0(kernel[tf])
print(f'Initializing model with kernel: {kernel[tf]}')

# define the CrossEntropyLoss with weights
#loss_fn = nn.BCEWithLogitsLoss(weight=weights)
loss_fn = nn.BCEWithLogitsLoss()

# Define oprimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Define exponetntial lr with lr_scheduler
exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.01)



# View the frist 5 outputs of the forward pass on the test data
y_logits = model(X_test.to(device))[:5]
y_logits


# # Results of 1 layer CNN


results, model = engine.train_with_early_stopping(model=model,
                                                  train_dataloader=train_loader,
                                                  valid_dataloader=val_loader,
                                                  optimizer=optimizer,
                                                  loss_fn=loss_fn,
                                                  epochs=300,
                                                  device=device)





##### Save model and results ######

base_directory = os.path.expanduser("~")
sub_directory = f"CNN_results_balanced_kernels/{tf}/CNN1"
save_directory = os.path.join(base_directory, sub_directory)
os.makedirs(save_directory, exist_ok=True)

results_file_path = os.path.join(save_directory, "training_results.csv")
model_file_path = os.path.join(save_directory, "trained_model.pth")

# Save the PyTorch model
torch.save(model.state_dict(), model_file_path)
print(f"Model saved to: {model_file_path}")

# Save the last metrics of training
last_values = {key: values[-1] for key, values in results.items()}

   # Save the last values to CSV
with open(results_file_path , 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header
    csv_writer.writerow(['Metric', 'Last_Value'])

    # Write data
    for key, value in last_values.items():
        csv_writer.writerow([key, value])

print(f"Best model and results saved to: {save_directory}")


helper_functions.plot_loss_curves(results)

plt.savefig(f"{save_directory}/{tf}_CNN1_ADAM_BCEWithLogits_seq=200b.png")
plt.title(f"Loss Curves {tf}_CNN1_ADAM_BCEWithLogits_seq=200b.png")





# In[ ]:


"""
CTCF	15, 19 (TF has several motif variants), 54
EP300	
JUN	14
NR3C1	15
TEAD4	8
ZNF274	12
USF2	10
TCF12	7
SRF	16
NRF1	11
MAZ	8
HES2	9
FOS	8
Cebpa	10
FOXA1	8
USF1	10
RELA/p65	10
GABPA	10
ELF1	9
EGR1	10
ATF3	10
ZBTB33	10
SP1	9
CREB1	8
BCL3	
YY1	12
JUND	11
FOSL2	10
MAX	6
MYC	8
CEBPB	10
REST	20
	
"""


# In[1]:





# In[ ]:




