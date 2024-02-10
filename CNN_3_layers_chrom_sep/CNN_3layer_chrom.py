


#!/usr/bin/env python
# coding: utf-8



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
from torch.optim import lr_scheduler

import engine
import engine_boosted
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
kernel2 = {
    'ATF2': [9] ,
    'ATF3' : [18] ,
    'CEBPB' : [10] ,
    'CREB1' : [14] ,
    'CTCF' : [18] ,
    'E2F4' : [10] ,
    'EGR1' : [20] ,
    'ELF1' : [9] ,
    'ELK1' : [15] ,
    'FOS' : [20] ,
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

kernel1 = {
    'ATF2': [10] ,
    'ATF3' : [10] ,
    'CEBPB' : [10] ,
    'CREB1' : [8] ,
    'CTCF' : [15] ,
    'E2F4' : [13] ,
    'EGR1' : [10] ,
    'ELF1' : [9] ,
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


file_path = f'/mnt/raid1/thalassini/home/filtered_fimo_regs_200/{tf.upper()}/datasets/data.csv'

df = pd.read_csv(file_path)



### Sequence length inspection and normalization ###
lengths= [len(item) for item in df['data']] 
print(f'Max length:{max(lengths)}, Min length: {min(lengths)}')


# Seperate train an test df
# Select chromosomes for test set (e.g., 'chr1' and 'chr2')
test_chromosomes = ['chr1', 'chr2']
test_df = df[df['chrom'].isin(test_chromosomes)]

# Select chromosomes for training set (excluding 'chr1' and 'chr2')
train_df = df[~df['chrom'].isin(test_chromosomes)]

# Print the shapes of train and test sets
print("Train set shape:", train_df.shape)
print("Test set shape:", test_df.shape)




# Create train and valid loader

X=[one_hot_encode(sequence) for sequence in train_df['data']]
X = torch.from_numpy(np.array(X)).type(torch.float)
y=np.array(train_df['class'])
y = torch.from_numpy(y).type(torch.float)
train_loader, val_loader = helper_functions.make_2loaders(X,y)

# Create test loader

X=[one_hot_encode(sequence) for sequence in test_df['data']]
X = torch.from_numpy(np.array(X)).type(torch.float)
y=np.array(test_df['class'])
y = torch.from_numpy(y).type(torch.float)
test_loader = helper_functions.make_loader(X,y)







  ### ==== Construct a 1 and 2 layer CNN with Maxpooling and Dropout ==== ###


class Conv_v1(torch.nn.Module):

    def __init__(self):
        super(Conv_v1, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels= 4, out_channels= 30, kernel_size= 3)
        self.conv2 = torch.nn.Conv1d(in_channels= 30, out_channels= 60, kernel_size= 5)
        self.conv3 = torch.nn.Conv1d(in_channels= 60, out_channels= 100, kernel_size= 5)
        self.activation = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(kernel_size= 2)

        self.dropout = torch.nn.Dropout(p=0.25)
        #in_features = 108 for 200kb, 588 for 1kb, 1188 for 2kb
        self.fc =  torch.nn.LazyLinear(out_features=2)
        #self.sigmoid = torch.nn.Sigmoid() will not be used since its intergraded in BCEWithLogitsLoss()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.dropout(x)


        # Reshape the output of the max pooling layer before passing it to the fully connected layer
        x = x.view(x.size(0), -1)

        #print("Size after reshaping:", x.size())
        #x=self.dropout(x)
        x = self.fc(x)
        #x = self.softmax(x)
        #x = self.sigmoid(x)
        return x






# # Set up loss function and optimizer

set_seeds()

# Initialize model instance
model = Conv_v1()


# define the CrossEntropyLoss with weights
#loss_fn = nn.BCEWithLogitsLoss(weight=weights)
loss_fn = nn.BCEWithLogitsLoss()

# Define oprimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Define exponetntial lr with lr_scheduler
exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.01)




# # Results of 1 layer CNN


results, model = engine_boosted.train_with_early_stopping(model=model,
                                                  train_dataloader=train_loader,
                                                  valid_dataloader=val_loader,
                                                  optimizer=optimizer,
                                                  loss_fn=loss_fn,
                                                  epochs=1000,
                                                  device=device)





##### Save model and results ######

base_directory = os.path.expanduser("~")
sub_directory = f"CNN_results_200_batch_128_chrom/{tf}/CNN3"
save_directory = os.path.join(base_directory, sub_directory)
os.makedirs(save_directory, exist_ok=True)

results_file_path = os.path.join(save_directory, "training_results.csv")
model_file_path = os.path.join(save_directory, "trained_model.pth")

# Save the PyTorch model
torch.save(model.state_dict(), model_file_path)
print(f"Model saved to: {model_file_path}")



test_results = engine_boosted.test_step2(model=model,
                                                  dataloader=test_loader,
                                                  loss_fn=loss_fn,
                                                  device=device)




helper_functions.save_results_to_csv(test_results, f'{save_directory}/test_results.csv')

print(f"Best model and results saved to: {save_directory}")


helper_functions.plot_loss_curves(results)
plt.savefig(f"{save_directory}/LOSS_AUC{tf}_CNN3_ADAM_BCEWithLogits.png")
plt.title(f"Loss Curves {tf}_CNN3_ADAM_BCEWithLogits.png")


helper_functions.plot_auroc(results)
plt.savefig(f"{save_directory}/AUROC{tf}_CNN3_ADAM_BCEWithLogits.png")
plt.title(f"AUROC {tf}_CNN3_ADAM_BCEWithLogits.png")

helper_functions.plot_f_score(results)
plt.savefig(f"{save_directory}/F1_SCORE{tf}_CNN3_ADAM_BCEWithLogits.png");
plt.title(f"F1-score{tf}_CNN3_ADAM_BCEWithLogits.png")






