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

import matplotlib.pyplot as plt
    
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import scipy
import sklearn.metrics

def run_model_baseon_regression_fold(df_grountruth_score, fold): #fold means test fold
    import pickle

    with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_deepsynergycellline.pickle", 'rb') as f:
        cellline_name_getembedding = pickle.load(f)
    with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_deepsynergydrug.pickle", 'rb') as f:
        drug_name_getembedding = pickle.load(f)
        
    df_train = df_grountruth_score[df_grountruth_score['fold'] != fold]


    train_list = {}
    for item in df_train.index.values:
        d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
        value_list = np.hstack([drug_name_getembedding[d1],drug_name_getembedding[d2],cellline_name_getembedding[cl]])
        train_list[item] = value_list

    X_train = np.array(list(train_list.values()))
    y_train = df_grountruth_score.loc[df_train.index.values]['synergy'].values
    
    
    df_test = df_grountruth_score[df_grountruth_score['fold'] == fold]

    test_list = {}
    for item in df_test.index.values:
        d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
        value_list = np.hstack([drug_name_getembedding[d1],drug_name_getembedding[d2],cellline_name_getembedding[cl]])
        test_list[item] = value_list

    X_test = np.array(list(test_list.values()))
    y_test = df_grountruth_score.loc[df_test.index.values]['synergy'].values
    
    
    clf = XGBRegressor(max_depth=10, min_child_weight=3, n_estimators=10)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    cor, pvalue = scipy.stats.pearsonr(y_pred_test, y_test)
    r2score = sklearn.metrics.r2_score(y_test, y_pred_test)
    
    print("We have the test results for fold for regression:", fold)
    print(cor)
    print(r2score)

df_grountruth_score = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/synergy_prediction/labels_synergy_value.csv")    
run_model_baseon_regression_fold(df_grountruth_score, 0)