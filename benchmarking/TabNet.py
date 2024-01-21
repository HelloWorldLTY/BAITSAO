import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L


import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip

import matplotlib.pyplot as plt

# pip install pytorch-tabnet



df_deepsy = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/synergy_prediction/predictions_result_synergy.csv")

df_deepsy

df_grountruth_score = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/synergy_prediction/labels_synergy_value.csv")



hyperparameter_file = 'hyperparameters' # textfile which contains the hyperparameters of the model
data_file = '../data_test_fold3_tanh.p.gz' # pickle file which contains the data (produced with normalize.ipynb)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



# file = gzip.open(data_file, 'rb')
# X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)
# file.close()

import pickle

with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_deepsynergycellline.pickle", 'rb') as f:
    cellline_name_getembedding = pickle.load(f)
with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_deepsynergydrug.pickle", 'rb') as f:
    drug_name_getembedding = pickle.load(f)

df_grountruth_score.head()



import sklearn.model_selection
test_fold = 0

train_index = df_grountruth_score[df_grountruth_score['fold'] != test_fold].index

train_list = {}
for item in train_index:
    d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
    value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
#     value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
    train_list[item] = value_list

X_train = np.array(list(train_list.values()))
y_train = df_grountruth_score.loc[train_index]['synergy'].values

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, random_state=2023)


df_test = df_grountruth_score[df_grountruth_score['fold'] == test_fold]

test_list = {}
for item in df_test.index.values:
    d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
    value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
#     value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
    test_list[item] = value_list

X_test = np.array(list(test_list.values()))
y_test = df_grountruth_score.loc[df_test.index.values]['synergy'].values



# X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test =torch.FloatTensor(X_train),torch.FloatTensor(X_val),torch.FloatTensor(X_train),torch.FloatTensor(X_test),torch.FloatTensor(y_train), torch.FloatTensor(y_val), torch.FloatTensor(y_train), torch.FloatTensor(y_test)

# train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
# valid_dataset = torch.utils.data.TensorDataset(X_val, y_val)
# test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
clf = TabNetRegressor(n_a=8, n_d=8, gamma=1.1, n_steps=3)  #TabNetRegressor()
clf.fit(
  X_train, y_train.reshape(-1, 1),
  eval_set=[(X_val, y_val.reshape(-1, 1))]
)

y_pred = clf.predict(X_test)

import scipy.stats
import sklearn.metrics

cor, pval = scipy.stats.pearsonr(y_pred.transpose()[0], y_test)

r2score = sklearn.metrics.r2_score(y_test, y_pred.transpose()[0])

print(cor, " ", r2score)