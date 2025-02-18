#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import sklearn.model_selection
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics

df_grountruth_score = pd.read_csv("./deepsynergy_data/summary_v_1_5.csv")

df_grountruth_score = df_grountruth_score.drop(df_grountruth_score[df_grountruth_score['synergy_loewe'] == '\\N'].index)

df_grountruth_score = df_grountruth_score.dropna(subset=['drug_col', 'drug_row'])

# pip install torchmetrics

L.seed_everything(0)

df_grountruth_score = df_grountruth_score.sample(frac=0.5, random_state=2023)
df_grountruth_score = df_grountruth_score.reset_index(drop=True)

# df_grountruth_score['ri_row']

import pickle

# with open("/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/ensem_emb_marsydrugall.pickle", 'rb') as f:
#     drug_name_getembedding = pickle.load(f)
# with open("/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/ensem_emb_marsycelllinesall.pickle", 'rb') as f:
#     cellline_name_getembedding = pickle.load(f)

with open("./deepsynergy_data/ensem_emb_pretraincelllineupdate.pickle", 'rb') as f:
    cellline_name_getembedding = pickle.load(f)
with open("./deepsynergy_data/ensem_emb_pretraindrugupdate.pickle", 'rb') as f:
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


# In[2]:


# X_train = np.array(train_list)
# y_train = df_grountruth_score.loc[train_index]['synergy_loewe'].values

X_train = np.array(train_list)
y_train = df_grountruth_score[['synergy_zip']].values


# In[3]:


# y_train[:,0]


# In[4]:


y_train = np.array(y_train).astype('float')

y_train_class = (y_train[:,0] > 30)*1

# y_train = np.hstack([y_train, np.array([y_train_class]).T])

X_tr, X_test, y_tr, y_test = sklearn.model_selection.train_test_split(X_train, y_train_class, test_size=0.1, random_state = 2023)

X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(X_tr, y_tr, test_size=0.1, random_state = 2023)

X_tr


# In[5]:


y_tr


# In[6]:


# df_grountruth_score.head()

# import sklearn.model_selection
# test_fold = 4

# train_index = df_grountruth_score[df_grountruth_score['fold'] != test_fold].index

# train_list = {}
# for item in train_index:
#     d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
#     value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
#     #value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
#     train_list[item] = value_list

# X_train = np.array(list(train_list.values()))
# y_train = (df_grountruth_score.loc[train_index]['synergy'].values > 30)*1

# X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(X_train, y_train, random_state=2023)

# # layers = [8182,4096,1] 
# # epochs = 1000 
# # act_func = 'GELU'
# # dropout = 0.5 
# # input_dropout = 0.2
# # eta = 0.00001 
# # norm = 'tanh' 

layers = [X_tr.shape[1],4096,1] 
epochs = 10 
act_func = 'relu'
dropout = 0.5 
input_dropout = 0.2
eta = 0.00001 
norm = 'tanh' 

# df_test = df_grountruth_score[df_grountruth_score['fold'] == test_fold]

# test_list = {}
# for item in df_test.index.values:
#     d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
#     value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
#     #value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
#     test_list[item] = value_list

# X_test = np.array(list(test_list.values()))
# y_test = (df_grountruth_score.loc[df_test.index.values]['synergy'].values > 30)*1

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(X_tr.shape[1], layers[0]), 
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[0], layers[1]),
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[1], layers[2]),
                                nn.Sigmoid()
                               )

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.apply(init_weights)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self.encoder(x)
        loss = F.binary_cross_entropy(z,y.view(x.size(0), -1).float())
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.encoder(x)
        val_loss = F.binary_cross_entropy(z,y.view(x.size(0), -1).float())
        return val_loss

        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.encoder(x)
        test_loss = F.binary_cross_entropy(z,y.view(x.size(0), -1).float())
        self.log("test_loss", test_loss)
        return test_loss
        
    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=eta, momentum=0.5)
        return optimizer

X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test =torch.FloatTensor(X_tr),torch.FloatTensor(X_val),torch.FloatTensor(X_train),torch.FloatTensor(X_test),torch.FloatTensor(y_tr), torch.FloatTensor(y_val), torch.FloatTensor(y_train), torch.FloatTensor(y_test)


# In[ ]:





# In[9]:


train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
valid_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

model = LitAutoEncoder(Encoder())


# In[ ]:


from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
train_loader = DataLoader(train_dataset, batch_size=64)
valid_loader = DataLoader(valid_dataset, batch_size=64)

# train with both splits
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_loader, valid_loader)

trainer.test(model, dataloaders=DataLoader(test_dataset))

# with torch.no_grad():
#     y_pred = model.encoder(X_test).detach()

# import scipy.stats
# import sklearn.metrics

# print("We use the fold with number:", test_fold)
# print(sklearn.metrics.roc_auc_score(y_test, y_pred.t()[0]), sklearn.metrics.accuracy_score(y_test, (y_pred.t()[0]>0.5)*1))


# In[15]:


trainer.checkpoint_callback.best_model_path


# In[14]:


with torch.no_grad():
    y_pred = model.encoder(X_test).detach()

import scipy.stats
import sklearn.metrics

print("We use the fold with number:", test_fold)
print(sklearn.metrics.roc_auc_score(y_test, y_pred.t()[0]), sklearn.metrics.accuracy_score(y_test, (y_pred.t()[0]>0.5)*1))


# In[ ]:





# In[33]:


a = 1


# In[34]:


print(sklearn.metrics.roc_auc_score(y_test, y_pred.t()[0]), sklearn.metrics.accuracy_score(y_test, (y_pred.t()[0]>0.5)*1))


# In[20]:


print(sklearn.metrics.roc_auc_score(y_test, y_pred.t()[0]), sklearn.metrics.accuracy_score(y_test, (y_pred.t()[0]>0.5)*1))


# # finetune training classification

# In[1]:


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
import sklearn.model_selection
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics

# df_grountruth_score = pd.read_csv("./deepsynergy_data/summary_v_1_5.csv")

# df_grountruth_score = df_grountruth_score.drop(df_grountruth_score[df_grountruth_score['synergy_loewe'] == '\\N'].index)

# df_grountruth_score = df_grountruth_score.dropna(subset=['drug_col', 'drug_row'])

# # pip install torchmetrics

# L.seed_everything(0)

# df_grountruth_score = df_grountruth_score.sample(frac=0.5, random_state=2023)
# df_grountruth_score = df_grountruth_score.reset_index(drop=True)

# # df_grountruth_score['ri_row']

# import pickle

# # with open("/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/ensem_emb_marsydrugall.pickle", 'rb') as f:
# #     drug_name_getembedding = pickle.load(f)
# # with open("/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/ensem_emb_marsycelllinesall.pickle", 'rb') as f:
# #     cellline_name_getembedding = pickle.load(f)

# with open("./deepsynergy_data/ensem_emb_pretraincelllineupdate.pickle", 'rb') as f:
#     cellline_name_getembedding = pickle.load(f)
# with open("./deepsynergy_data/ensem_emb_pretraindrugupdate.pickle", 'rb') as f:
#     drug_name_getembedding = pickle.load(f)

# test_fold = 0


# train_index = df_grountruth_score.index
# n_dim = 1536
# drug_num_dim = 16
# train_list = []
# for item in train_index:
#     d1, d2, cl = df_grountruth_score.loc[item]['drug_row'], df_grountruth_score.loc[item]['drug_col'], df_grountruth_score.loc[item]['cell_line_name']
#     if str(d2) == 'nan':
#         #value_list = drug_name_getembedding[d1] + cellline_name_getembedding[cl]
#         value_list = np.hstack([drug_name_getembedding[d1], cellline_name_getembedding[cl]])
#         value_list = np.hstack([value_list, 1])
#     else:
#         #value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
#         value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
#         value_list = np.hstack([value_list, 2])    
#     train_list.append(value_list)


# In[ ]:





# In[2]:


df_grountruth_score = pd.read_csv("./deepsynergy_data/labels_synergy_value.csv")


# In[ ]:





# In[3]:


import pickle


# In[4]:


with open("./deepsynergy_data/ensem_emb_deepsynergycellline.pickle", 'rb') as f:
    cellline_name_getembedding = pickle.load(f)
with open("./deepsynergy_data/ensem_emb_deepsynergydrug.pickle", 'rb') as f:
    drug_name_getembedding = pickle.load(f)


# In[5]:


df_grountruth_score.head()


# In[ ]:





# In[85]:


import sklearn.model_selection
test_fold = 4


# In[86]:


train_index = df_grountruth_score[df_grountruth_score['fold'] != test_fold].index


# In[87]:


train_list = {}
for item in train_index:
    d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
    value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
    value_list = np.hstack([value_list, 2])
    train_list[item] = value_list


# In[88]:


X_train = np.array(list(train_list.values()))
y_train = (df_grountruth_score.loc[train_index]['synergy'].values > 30)*1


# In[89]:


X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, random_state=2023)
X_tr = X_train 
y_tr = y_train


# In[90]:


# layers = [8182,4096,1] 
# epochs = 1000 
# act_func = 'GELU'
# dropout = 0.5 
# input_dropout = 0.2
# eta = 0.00001 
# norm = 'tanh' 

# layers = [8182,4096,1] 
# epochs = 1000 
# act_func = 'GELU'
# dropout = 0.5 
# input_dropout = 0.2
# eta = 1e-4 
# norm = 'tanh' 

layers = [10240,4096,1] 
epochs = 1000 
act_func = 'GELU'
dropout = 0.5 
input_dropout = 0.2
eta = 1e-5
norm = 'tanh'
drug_num_dim = 16


# In[91]:


df_test = df_grountruth_score[df_grountruth_score['fold'] == test_fold]


# In[92]:


test_list = {}
for item in df_test.index.values:
    d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
    value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
    value_list = np.hstack([value_list, 2])
    test_list[item] = value_list


# In[93]:


X_test = np.array(list(test_list.values()))
y_test = (df_grountruth_score.loc[df_test.index.values]['synergy'].values> 30)*1


# In[94]:


import torchvision


# In[95]:


X_train.shape


# In[96]:


y_train


# In[ ]:





# In[97]:


# df_grountruth_score.head()

# import sklearn.model_selection
# test_fold = 4

# train_index = df_grountruth_score[df_grountruth_score['fold'] != test_fold].index

# train_list = {}
# for item in train_index:
#     d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
#     value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
#     #value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
#     train_list[item] = value_list

# X_train = np.array(list(train_list.values()))
# y_train = (df_grountruth_score.loc[train_index]['synergy'].values > 30)*1

# X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(X_train, y_train, random_state=2023)

# # layers = [8182,4096,1] 
# # epochs = 1000 
# # act_func = 'GELU'
# # dropout = 0.5 
# # input_dropout = 0.2
# # eta = 0.00001 
# # norm = 'tanh' 

layers = [X_tr.shape[1],4096,1] 
epochs = 10 
act_func = 'relu'
dropout = 0.5 
input_dropout = 0.2
eta = 0.00001 
norm = 'tanh' 

# df_test = df_grountruth_score[df_grountruth_score['fold'] == test_fold]

# test_list = {}
# for item in df_test.index.values:
#     d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
#     value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
#     #value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
#     test_list[item] = value_list

# X_test = np.array(list(test_list.values()))
# y_test = (df_grountruth_score.loc[df_test.index.values]['synergy'].values > 30)*1

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(X_tr.shape[1], layers[0]), 
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[0], layers[1]),
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[1], layers[2]),
                                nn.Sigmoid()
                               )

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.apply(init_weights)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self.encoder(x)
        loss = F.binary_cross_entropy(z,y.view(x.size(0), -1).float())
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.encoder(x)
        val_loss = F.binary_cross_entropy(z,y.view(x.size(0), -1).float())
        return val_loss

        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.encoder(x)
        test_loss = F.binary_cross_entropy(z,y.view(x.size(0), -1).float())
        self.log("test_loss", test_loss)
        return test_loss
        
    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=eta, momentum=0.5)
        return optimizer

X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test =torch.FloatTensor(X_tr),torch.FloatTensor(X_val),torch.FloatTensor(X_train),torch.FloatTensor(X_test),torch.FloatTensor(y_tr), torch.FloatTensor(y_val), torch.FloatTensor(y_train), torch.FloatTensor(y_test)


# In[98]:


model = LitAutoEncoder(Encoder())

checkpoint = torch.load('/gpfs/radev/project/ying_rex/tl688/synerg_predict/lightning_logs/version_187382/checkpoints/epoch=9-step=46810.ckpt')
model.load_state_dict(checkpoint["state_dict"])


# In[99]:


model


# In[100]:


# model.encoder.l1[7] = nn.Identity()


# In[101]:


train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)

valid_dataset = torch.utils.data.TensorDataset(X_val, y_val)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)


# In[102]:


from lightning.pytorch.callbacks import LearningRateMonitor

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

lr_monitor = LearningRateMonitor(logging_interval='step')
from lightning.pytorch.callbacks.early_stopping import EarlyStopping



train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=1, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=2048, num_workers=1)



# train with both splits

trainer = L.Trainer(max_epochs=100)

trainer.fit(model, train_loader, valid_loader)

trainer.test(model, dataloaders=DataLoader(test_dataset, batch_size=1024, num_workers=5))



for m in model.encoder.modules():

    for child in m.children():

        if type(child) == nn.BatchNorm1d:

            child.track_running_stats = False

            child.running_mean = None

            child.running_var = None

model.encoder.eval()


# In[103]:


with torch.no_grad():
    y_pred = model.encoder(X_test).detach()

import scipy.stats
import sklearn.metrics

print("We use the fold with number:", test_fold)

print(sklearn.metrics.roc_auc_score(y_test, y_pred.t()[0]), sklearn.metrics.accuracy_score(y_test, (y_pred.t()[0]>0.5)*1))


# In[ ]:





# In[ ]:





# # finetune training regression

# In[5]:


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
import sklearn.model_selection
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics

# df_grountruth_score = pd.read_csv("./deepsynergy_data/summary_v_1_5.csv")

# df_grountruth_score = df_grountruth_score.drop(df_grountruth_score[df_grountruth_score['synergy_loewe'] == '\\N'].index)

# df_grountruth_score = df_grountruth_score.dropna(subset=['drug_col', 'drug_row'])

# # pip install torchmetrics

# L.seed_everything(0)

# df_grountruth_score = df_grountruth_score.sample(frac=0.5, random_state=2023)
# df_grountruth_score = df_grountruth_score.reset_index(drop=True)

# # df_grountruth_score['ri_row']

# import pickle

# # with open("/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/ensem_emb_marsydrugall.pickle", 'rb') as f:
# #     drug_name_getembedding = pickle.load(f)
# # with open("/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/ensem_emb_marsycelllinesall.pickle", 'rb') as f:
# #     cellline_name_getembedding = pickle.load(f)

# with open("./deepsynergy_data/ensem_emb_pretraincelllineupdate.pickle", 'rb') as f:
#     cellline_name_getembedding = pickle.load(f)
# with open("./deepsynergy_data/ensem_emb_pretraindrugupdate.pickle", 'rb') as f:
#     drug_name_getembedding = pickle.load(f)

# test_fold = 0


# train_index = df_grountruth_score.index
# n_dim = 1536
# drug_num_dim = 16
# train_list = []
# for item in train_index:
#     d1, d2, cl = df_grountruth_score.loc[item]['drug_row'], df_grountruth_score.loc[item]['drug_col'], df_grountruth_score.loc[item]['cell_line_name']
#     if str(d2) == 'nan':
#         #value_list = drug_name_getembedding[d1] + cellline_name_getembedding[cl]
#         value_list = np.hstack([drug_name_getembedding[d1], cellline_name_getembedding[cl]])
#         value_list = np.hstack([value_list, 1])
#     else:
#         #value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
#         value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
#         value_list = np.hstack([value_list, 2])    
#     train_list.append(value_list)


# In[ ]:





# In[30]:


df_grountruth_score = pd.read_csv("./deepsynergy_data/labels_synergy_value.csv")


# In[ ]:





# In[31]:


import pickle


# In[32]:


with open("./deepsynergy_data/ensem_emb_deepsynergycellline.pickle", 'rb') as f:
    cellline_name_getembedding = pickle.load(f)
with open("./deepsynergy_data/ensem_emb_deepsynergydrug.pickle", 'rb') as f:
    drug_name_getembedding = pickle.load(f)


# In[33]:


df_grountruth_score.head()


# In[ ]:





# In[34]:


import sklearn.model_selection
test_fold = 4


# In[35]:


train_index = df_grountruth_score[df_grountruth_score['fold'] != test_fold].index


# In[36]:


train_list = {}
for item in train_index:
    d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
    value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
    value_list = np.hstack([value_list, 2])
    train_list[item] = value_list


# In[37]:


X_train = np.array(list(train_list.values()))
y_train = df_grountruth_score.loc[train_index]['synergy'].values 


# In[38]:


X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, random_state=2023)
X_tr = X_train 
y_tr = y_train


# In[39]:


# layers = [8182,4096,1] 
# epochs = 1000 
# act_func = 'GELU'
# dropout = 0.5 
# input_dropout = 0.2
# eta = 0.00001 
# norm = 'tanh' 

# layers = [8182,4096,1] 
# epochs = 1000 
# act_func = 'GELU'
# dropout = 0.5 
# input_dropout = 0.2
# eta = 1e-4 
# norm = 'tanh' 

layers = [10240,4096,1] 
epochs = 1000 
act_func = 'GELU'
dropout = 0.5 
input_dropout = 0.2
eta = 1e-5
norm = 'tanh'
drug_num_dim = 16


# In[40]:


df_test = df_grountruth_score[df_grountruth_score['fold'] == test_fold]


# In[41]:


test_list = {}
for item in df_test.index.values:
    d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
    value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
    value_list = np.hstack([value_list, 2])
    test_list[item] = value_list


# In[42]:


X_test = np.array(list(test_list.values()))
y_test = df_grountruth_score.loc[df_test.index.values]['synergy'].values


# In[43]:


import torchvision


# In[44]:


X_train.shape


# In[45]:


y_train


# In[46]:


y_test


# In[47]:


# df_grountruth_score.head()

# import sklearn.model_selection
# test_fold = 4

# train_index = df_grountruth_score[df_grountruth_score['fold'] != test_fold].index

# train_list = {}
# for item in train_index:
#     d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
#     value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
#     #value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
#     train_list[item] = value_list

# X_train = np.array(list(train_list.values()))
# y_train = (df_grountruth_score.loc[train_index]['synergy'].values > 30)*1

# X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(X_train, y_train, random_state=2023)

# # layers = [8182,4096,1] 
# # epochs = 1000 
# # act_func = 'GELU'
# # dropout = 0.5 
# # input_dropout = 0.2
# # eta = 0.00001 
# # norm = 'tanh' 

layers = [X_tr.shape[1],4096,1] 
epochs = 10 
act_func = 'relu'
dropout = 0.5 
input_dropout = 0.2
eta = 0.00001 
norm = 'tanh' 

# df_test = df_grountruth_score[df_grountruth_score['fold'] == test_fold]

# test_list = {}
# for item in df_test.index.values:
#     d1, d2, cl = df_grountruth_score.loc[item]['Unnamed: 0'].split('_')
#     value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
#     #value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
#     test_list[item] = value_list

# X_test = np.array(list(test_list.values()))
# y_test = (df_grountruth_score.loc[df_test.index.values]['synergy'].values > 30)*1

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(X_tr.shape[1], layers[0]), 
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[0], layers[1]),
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[1], layers[2]),
                                nn.Sigmoid()
                               )

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.apply(init_weights)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self.encoder(x)
        loss = F.mse_loss(z,y.view(x.size(0), -1).float())
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.encoder(x)
        val_loss = F.mse_loss(z,y.view(x.size(0), -1).float())
        return val_loss

        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.encoder(x)
        test_loss = F.mse_loss(z,y.view(x.size(0), -1).float())
        self.log("test_loss", test_loss)
        return test_loss
        
    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=eta, momentum=0.5)
        return optimizer

X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test =torch.FloatTensor(X_tr),torch.FloatTensor(X_val),torch.FloatTensor(X_train),torch.FloatTensor(X_test),torch.FloatTensor(y_tr), torch.FloatTensor(y_val), torch.FloatTensor(y_train), torch.FloatTensor(y_test)


# In[48]:


model = LitAutoEncoder(Encoder())

checkpoint = torch.load('/gpfs/radev/project/ying_rex/tl688/synerg_predict/lightning_logs/version_187382/checkpoints/epoch=9-step=46810.ckpt')
model.load_state_dict(checkpoint["state_dict"])


# In[49]:


model


# In[50]:


model.encoder.l1[7] = nn.Identity()


# In[51]:


train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)

valid_dataset = torch.utils.data.TensorDataset(X_val, y_val)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)


# In[52]:


from lightning.pytorch.callbacks import LearningRateMonitor

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

lr_monitor = LearningRateMonitor(logging_interval='step')
from lightning.pytorch.callbacks.early_stopping import EarlyStopping



train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=1, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=2048, num_workers=1)



# train with both splits

trainer = L.Trainer(max_epochs=100)

trainer.fit(model, train_loader, valid_loader)

trainer.test(model, dataloaders=DataLoader(test_dataset, batch_size=1024, num_workers=5))



for m in model.encoder.modules():

    for child in m.children():

        if type(child) == nn.BatchNorm1d:

            child.track_running_stats = False

            child.running_mean = None

            child.running_var = None

model.encoder.eval()


# In[ ]:





# In[53]:


with torch.no_grad():
    y_pred = model.encoder(X_test).detach()

import scipy.stats
import sklearn.metrics

print("We use the fold with number:", test_fold)
cor, pval = scipy.stats.pearsonr(y_test, y_pred.t()[0])
r2score = sklearn.metrics.mean_squared_error(y_test,y_pred.t()[0])

print(cor, " ", r2score)





