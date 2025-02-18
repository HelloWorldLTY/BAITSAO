# # Pretrain-finetuning



import keras
import keras.backend as K
import tensorflow as tf
from __future__ import print_function, division
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Sequential, Model
from keras import regularizers
import np_utils
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip

import matplotlib.pyplot as plt


df_grountruth_score = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/synergy_prediction/labels_synergy_value.csv")

fold = 0

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


import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
# import lightning as L

import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip
import sklearn.model_selection
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics

df_grountruth_score = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/summary_v_1_5.csv")

df_grountruth_score = df_grountruth_score.drop(df_grountruth_score[df_grountruth_score['synergy_loewe'] == '\\N'].index)

df_grountruth_score = df_grountruth_score.dropna(subset=['drug_col', 'drug_row'])


df_grountruth_score = df_grountruth_score.sample(frac=0.5, random_state=2023)
df_grountruth_score = df_grountruth_score.reset_index(drop=True)

import pickle

with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_pretraincelllineupdate.pickle", 'rb') as f:
    cellline_name_getembedding = pickle.load(f)
with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_pretraindrugupdate.pickle", 'rb') as f:
    drug_name_getembedding = pickle.load(f)

test_fold = 0


train_index = df_grountruth_score.index
n_dim = 1536
drug_num_dim = 16
train_list = []
for item in train_index:
    d1, d2, cl = df_grountruth_score.loc[item]['drug_row'], df_grountruth_score.loc[item]['drug_col'], df_grountruth_score.loc[item]['cell_line_name']
    if str(d2) == 'nan':
        #value_list = drug_name_getembedding[d1] + cellline_name_getembedding[cl]
        value_list = np.hstack([drug_name_getembedding[d1], cellline_name_getembedding[cl]])
        value_list = np.hstack([value_list, 1])
    else:
        #value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
        value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
        value_list = np.hstack([value_list, 2])    
    train_list.append(value_list)

# X_train = np.array(train_list)
# y_train = df_grountruth_score.loc[train_index]['synergy_loewe'].values


# In[12]:


X_train = np.array(train_list)
y_train = df_grountruth_score[['synergy_zip']].values

y_train = np.array(y_train).astype('float').T[0]

# y_train_class = (y_train[:,1] > 30)*1

# y_train = np.hstack([y_train, np.array([y_train_class]).T])

X_tr, X_test, y_tr, y_test = sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state = 2023)

X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(X_tr, y_tr, test_size=0.1, random_state = 2023)

X_tr


# In[ ]:





# In[13]:


print(X_train.shape)
print(y_train.shape)


# In[14]:


### Data formatting to fit MARSY's input requirements ###
def data_preparation(X_tr, X_tst, pair_range):
    X_tr = []
    X_tst = []
    
    #Extract Pairs and Triples from the input vector of each sample
    #Pairs refers to features of both drugs (3912 features)
    #Triple refers to the features of both drugs and the cancer cell line (8551 features)
    pair = []
    for i in X_train:
        temp_pair = i[:pair_range]
        pair.append(temp_pair)
    pair = np.asarray(pair)
    x1 = np.asarray(X_train)
    
    X_tr.append(x1)
    X_tr.append(pair)
    
    pair = []
    for i in X_test:
        temp_pair = i[:pair_range]
        pair.append(temp_pair)
    pair = np.asarray(pair)
    x2 = np.asarray(X_test)
    
    X_tst.append(x2)
    X_tst.append(pair)
    
    return X_tr, X_tst    


### Implementation of the MARSY model ###
def MARSY(X_tr, Y_tr, param):
    #Encoder for Triple
    tuple_vec = Input(shape=(param[0],))
    tpl = Dense(2048, activation='linear', kernel_initializer='he_normal')(tuple_vec)
    tpl = Dropout(param[2])(tpl)
    out_tpl1 = Dense(4096, activation='relu')(tpl)
    model_tpl = Model(tuple_vec, out_tpl1)

    tpl_inp = Input(shape=(param[0],))
    out_tpl = model_tpl(tpl_inp)
    
    #Encoder for Pair
    pair_vec = Input(shape=(param[1],))
    pair1 = Dense(1024, activation='linear', kernel_initializer='he_normal')(pair_vec)
    pair1 = Dropout(param[2])(pair1)
    out_p1 = Dense(2048, activation = 'relu')(pair1)
    model_pair = Model(pair_vec, out_p1)

    pair_inp = Input(shape=(param[1],))
    out_pair = model_pair(pair_inp)

    #Decoder to predict the synergy score and the single drug response of each drug
    concatenated_tpl = keras.layers.concatenate([out_pair, out_tpl])
    out_c1 = Dense(4096, activation='relu')(concatenated_tpl)
    out_c1 = Dropout(param[3])(out_c1)
    out_c1 = Dense(1024, activation='relu')(out_c1)
    out_c1 = Dense(3, activation='linear', name="Predictor_Drug_Combination")(out_c1)

    multitask_model = Model(inputs= [tpl_inp, pair_inp], outputs =[out_c1])

    multitask_model.compile(optimizer= tf.keras.optimizers.legacy.Adamax(learning_rate=float(0.001), 
                                                    beta_1=0.9, beta_2=0.999, epsilon=1e-07), 
                                                    loss={'Predictor_Drug_Combination': 'mse'}, 
                                                    metrics={'Predictor_Drug_Combination': 'mse'})

    es = EarlyStopping(monitor='val_mse', mode='min', verbose=0, patience=param[6])

    multi_conc = multitask_model.fit(X_tr, Y_tr, batch_size=param[5], epochs=param[4], verbose=0, 
                                     validation_split=0.2, callbacks=es)
    
    multitask_model.save("pretrained_marsy.keras")
    
    return multitask_model 


# In[15]:


# !nvidia-smi


# In[16]:


### Parameters ###
triple_length = 4609
pair_length = 3912
dropout_encoders = 0.2
dropout_decoder = 0.5
epochs = 10
batch_size = 64
tol_stopping = 10

param = [triple_length, pair_length, dropout_encoders, dropout_decoder, epochs, batch_size, tol_stopping]


### Training and Prediction Example ###

training_set, testing_set = data_preparation(X_train, X_val, pair_length)
trained_MARSY = MARSY(training_set, y_train, param)
pred = trained_MARSY.predict(testing_set)
