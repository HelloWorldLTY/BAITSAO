from ray import train, tune

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
import sklearn.model_selection

df_grountruth_score = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/summary_v_1_5.csv")
df_grountruth_score = df_grountruth_score.drop(df_grountruth_score[df_grountruth_score['synergy_loewe'] == '\\N'].index)
df_grountruth_score = df_grountruth_score.dropna(subset=['drug_col', 'drug_row'])
df_grountruth_score_update = df_grountruth_score.reset_index(drop=True)

df_grountruth_score_update['combine'] = [i+'_'+j+'_'+k for i,j,k in zip(df_grountruth_score_update['drug_row'].values, df_grountruth_score_update['drug_col'].values, df_grountruth_score_update['cell_line_name'].values) ]

df_grountruth_score_update

df_grountruth_score = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/MARSY/data/split_data.csv", index_col=0)

df_grountruth_score['combine'] = [i+'_'+j+'_'+k for i,j,k in zip(df_grountruth_score['Drug1_PC3'].values, 
                                                            df_grountruth_score['Drug2_PC3'].values, 
                                                            df_grountruth_score['Cell_line'].values) ]

df_grountruth_score_update_clean_index = []
for i in df_grountruth_score_update.index:
    if df_grountruth_score_update.loc[i]['combine'] not in df_grountruth_score['combine'].values:
        df_grountruth_score_update_clean_index.append(i)

df_final_marsy = df_grountruth_score_update.loc[df_grountruth_score_update_clean_index]

df_final_marsy.to_csv("df_final_marsy.csv")



a = 1

from ray import train, tune

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
import sklearn.model_selection

df_grountruth_score = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/summary_v_1_5.csv")
df_grountruth_score = df_grountruth_score.drop(df_grountruth_score[df_grountruth_score['synergy_loewe'] == '\\N'].index)
df_grountruth_score = df_grountruth_score.dropna(subset=['drug_col', 'drug_row'])
df_grountruth_score_update = df_grountruth_score.reset_index(drop=True)

df_grountruth_score_update.index= [i+'_'+j+'_'+k for i,j,k in zip(df_grountruth_score_update['drug_row'].values, df_grountruth_score_update['drug_col'].values, df_grountruth_score_update['cell_line_name'].values) ]

df_grountruth_score = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/MARSY/data/split_data.csv", index_col=0)

fold = 0

import pickle

with open("/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/ensem_emb_marsydrugall.pickle", 'rb') as f:
    drug_name_getembedding = pickle.load(f)
with open("/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/ensem_emb_marsycelllinesall.pickle", 'rb') as f:
    cellline_name_getembedding = pickle.load(f)

df_train = df_grountruth_score[df_grountruth_score['kfold'] != fold]


train_list = []
for item in df_train.index.values:
    d1, d2, cl = df_grountruth_score.loc[item]['Drug1_PC3'], df_grountruth_score.loc[item]['Drug2_PC3'], df_grountruth_score.loc[item]["Cell_line"]
    value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2], cellline_name_getembedding[cl]])
    value_list = np.hstack([value_list,2])
    train_list.append(value_list)

X_train = np.array(train_list)
y_train = df_grountruth_score.loc[df_train.index.values]['Synergy_Zip'].values
y_train = y_train.astype('float')


df_test = df_grountruth_score[df_grountruth_score['kfold'] == fold]

test_list = []
final_index = []
for item in df_test.index.values:
    d1, d2, cl = df_grountruth_score.loc[item]['Drug1_PC3'], df_grountruth_score.loc[item]['Drug2_PC3'], df_grountruth_score.loc[item]["Cell_line"]
    test_index = d1 + '_' + d2 +'_' + cl
    if test_index not in df_grountruth_score_update.index:
        value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2], cellline_name_getembedding[cl]])
        value_list = np.hstack([value_list,2])
        test_list.append(value_list)
        final_index.append(item)

X_test = np.array(test_list)
y_test = df_grountruth_score.loc[final_index]['Synergy_Zip'].values
y_test = y_test.astype('float')


len(X_test)

len(y_test)



X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(X_train, y_train, random_state = 2023)

# X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(X_tr, y_tr, test_size=0.1, random_state = 2023)

# X_tr

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
input_dropout = 0.0
eta = 1e-4
norm = 'tanh' 
n_dim_drug = 16

n_dim = 1536

import torch
n_dim = 1536


class MultiTaskLoss(torch.nn.Module):



    def __init__(self, is_regression, reduction='none'):

        super(MultiTaskLoss, self).__init__()

        self.is_regression = is_regression

        self.n_tasks = len(is_regression)

        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))

        self.reduction = reduction



    def forward(self, losses):

        dtype = losses.dtype

        device = losses.device

        stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)

        self.is_regression = self.is_regression.to(device).to(dtype)

        coeffs = 1 / ( (self.is_regression+1)*(stds**2) )

        multi_task_losses = coeffs*losses + torch.log(stds)



        if self.reduction == 'sum':

            multi_task_losses = multi_task_losses.sum()

        if self.reduction == 'mean':

            multi_task_losses = multi_task_losses.mean()

    

        return multi_task_losses

def init_weights(m):

    if isinstance(m, nn.Linear):

        torch.nn.init.kaiming_normal_(m.weight)

class Encoder(nn.Module):

    def __init__(self, k=3):

        super().__init__()

        self.l1 = nn.Sequential(nn.Linear(layers[1], layers[0]), 

                                nn.BatchNorm1d(layers[0]),

                                nn.ReLU(), 

                                nn.Dropout(input_dropout),

                                nn.Linear(layers[0], layers[1]),

                                nn.BatchNorm1d(layers[1]),

                                nn.ReLU(), 

                                nn.Dropout(input_dropout),

                               )

        



        self.loewe = nn.Linear(layers[1], layers[2])

        self.hsa = nn.Linear(layers[1], layers[2])

        self.bliss = nn.Linear(layers[1], layers[2])

        self.sigm = nn.Sequential(nn.Linear(layers[1], layers[2]),

                                  nn.Sigmoid()

                               )

        



        self.loewe_input = nn.Linear(n_dim*2 + drug_num_dim, layers[1])

        self.hsa_input = nn.Linear(n_dim*2 + drug_num_dim, layers[1])

        self.bliss_input = nn.Linear(n_dim*2 + drug_num_dim, layers[1])

        self.sigm_input = nn.Linear(n_dim*2 + drug_num_dim, layers[1])

        

        self.drug_num_emb = nn.Embedding(10, drug_num_dim)



    def forward(self, x):

        

        x_new = torch.cat([(x[:,0:n_dim] + x[:,n_dim:n_dim*2])/2, x[:,n_dim*2:n_dim*3], self.drug_num_emb(x[:,-1].long())], axis=1)

        x_row = torch.cat([x[:,0:n_dim], x[:,n_dim*2:n_dim*3], self.drug_num_emb(x[:,-1].long() - 1)], axis=1)

        #x_col = torch.cat([x[:,n_dim:n_dim*2], x[:,n_dim*2:n_dim*3], self.drug_num_emb(x[:,-1].long())], axis=1)

        

        return {

            'loewe':self.loewe(self.l1(self.loewe_input(x_new))),

            'ic50_row':self.hsa(self.l1(self.hsa_input(x_row))),

            #'ic50_col':self.bliss(self.l1(self.bliss_input(x_col))),

            'classify':self.sigm(self.l1(self.sigm_input(x_new)))

        }

    

    def inference_task(self,x,label):

        if label == 'ic50_row':

            x_new = x

            return self.forward(x_new)[label]



        if label == 'ic50_col':

            x_new = x

            return self.forward(x_new)[label]

        x_new = x

        output = self.forward(x_new)[label]

        return output
    
    def getzip(self, x):
        x_new = torch.cat([(x[:,0:n_dim] + x[:,n_dim:n_dim*2])/2, x[:,n_dim*2:n_dim*3], self.drug_num_emb(x[:,-1].long())], axis=1)
        return self.loewe(self.l1(self.loewe_input(x_new)))





class LitAutoEncoder(L.LightningModule):

    def __init__(self, encoder):

        super().__init__()

        self.encoder = encoder
        self.is_regression = torch.Tensor([True, True, False])
        self.loss_mode = MultiTaskLoss(self.is_regression, reduction = 'sum')


    def training_step(self, batch, batch_idx):

        # training_step defines the train loop.

        x, y = batch

        z = self.encoder(x)

        loss = F.mse_loss(z['loewe'],y.view(x.size(0), -1))
    
        multitaskloss = loss

        return multitaskloss

    

    def validation_step(self, batch, batch_idx):

        # this is the validation loop

        x, y = batch

        z = self.encoder(x)

    
        val_loss = F.mse_loss(z['loewe'],y.view(x.size(0), -1))

        self.log("val_loss", val_loss)

        return val_loss



        

    def test_step(self, batch, batch_idx):

        # this is the test loop

        x, y = batch

        z = self.encoder(x)

        

        test_loss = F.mse_loss(z['loewe'],y.view(x.size(0), -1))


        self.log("test_loss", test_loss)

        return test_loss

        

    def forward(self, x):

        return self.encoder(x)



    def configure_optimizers(self):

            optimizer = torch.optim.Adam(

                params=self.parameters(), 

                lr=eta

            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

                optimizer,

                patience=10,

                verbose=True

            )

            return {

               'optimizer': optimizer,

               'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler

               'monitor': 'val_loss'

           }


X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test =torch.FloatTensor(X_tr),torch.FloatTensor(X_val),torch.FloatTensor(X_train),torch.FloatTensor(X_test),torch.FloatTensor(y_tr), torch.FloatTensor(y_val), torch.FloatTensor(y_train), torch.FloatTensor(y_test)
train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
valid_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

layers = [10240,4096,1] 

epochs = 1000 

act_func = 'ReLU'

dropout = 0.5 

input_dropout = 0.0

eta = 1e-4

norm = 'tanh' 

drug_num_dim = 16

model = LitAutoEncoder(Encoder())

checkpoint = torch.load("/gpfs/gibbs/pi/zhao/tl688/MARSY/lightning_logs/version_36482933/checkpoints/epoch=101-step=14790.ckpt")
model.load_state_dict(checkpoint["state_dict"])


train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=5, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=2048, num_workers=5)


for m in model.encoder.modules():

    for child in m.children():

        if type(child) == nn.BatchNorm1d:

            child.track_running_stats = False

            child.running_mean = None

            child.running_var = None

model.encoder.eval()

import scipy.stats
import sklearn.metrics

with torch.no_grad():
    y_pred = model.encoder.getzip(X_test).detach()

cor, pval = scipy.stats.pearsonr(y_test, y_pred.t()[0])
r2score = sklearn.metrics.r2_score(y_test,y_pred.t()[0])

print(cor, " ", r2score)

