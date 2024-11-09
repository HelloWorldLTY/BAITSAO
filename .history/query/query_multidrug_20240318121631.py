import torch
import torch.nn.functional as F
import lightning as L
import pickle
import os
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn

# Please modify these path with your own path.

with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_pretraincelllineupdate.pickle", 'rb') as f:

    cellline_name_getembedding = pickle.load(f)

with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_pretraindrugupdate.pickle", 'rb') as f:

    drug_name_getembedding = pickle.load(f)

df = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/ThreeDrugCombs.csv")

df['Drug1'] = df['Drug1 ']

data_train = df

n_dim = 1536
train_list = {}
comb_list = []
for item in data_train.index:
    d1, d2, d3, cl = data_train.loc[item]['Drug1'], data_train.loc[item]['Drug2'], data_train.loc[item]['Drug3'], data_train.loc[item]['Cell Name']
    if d1+'_'+d2+'_'+d3+'_'+cl not in comb_list:
    #     value_list = drug_name_getembedding[d1] + drug_name_getembedding[d2] + cellline_name_getembedding[cl]
    #     value_list = np.hstack([value_list, 2])
        value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , drug_name_getembedding[d3] , cellline_name_getembedding[cl]])
        value_list = np.hstack([value_list, 3])
        train_list[d1+'_'+d2+'_'+d3+'_'+cl] = value_list
        comb_list.append(d1+'_'+d2+'_'+d3+'_'+cl)
        print(item)

X_train = np.array(list(train_list.values()))


layers = [10240,4096,1] 
epochs = 1000 
act_func = 'GELU'
dropout = 0.5 
input_dropout = 0.2
eta = 1e-4
norm = 'tanh' 
drug_num_dim = 16
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


        x_new = torch.cat([(x[:,0:n_dim] + x[:,n_dim:n_dim*2] + x[:,n_dim*2:n_dim*3])/3, x[:,n_dim*3:n_dim*4], self.drug_num_emb(x[:,-1].long())], axis=1)
#         x_row = torch.cat([x[:,0:n_dim], x[:,n_dim*2:n_dim*3], self.drug_num_emb(x[:,-1].long() - 1)], axis=1)

        #x_col = torch.cat([x[:,n_dim:n_dim*2], x[:,n_dim*2:n_dim*3], self.drug_num_emb(x[:,-1].long())], axis=1)

        

        return {

            'loewe':self.loewe(self.l1(self.loewe_input(x_new))),

#             'ic50_row':self.hsa(self.l1(self.hsa_input(x_row))),

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





class LitAutoEncoder(L.LightningModule):

    def __init__(self, encoder):

        super().__init__()

        self.encoder = encoder

#         self.encoder.apply(init_weights)

        self.is_regression = torch.Tensor([True, True, False])

        self.loss_mode = MultiTaskLoss(self.is_regression, reduction = 'mean')



    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        x, y = batch

        

        z = self.encoder(x)

        

        loss = F.binary_cross_entropy(z['classify'],y.view(x.size(0), -1))
    
        multitaskloss = loss

        return multitaskloss

    

    def validation_step(self, batch, batch_idx):

        # this is the validation loop

        x, y = batch

        z = self.encoder(x)

        
        
        val_loss = F.binary_cross_entropy(z['classify'],y.view(x.size(0), -1))

        self.log("val_loss", val_loss)

        return val_loss



        

    def test_step(self, batch, batch_idx):

        # this is the test loop

        x, y = batch

        z = self.encoder(x)

        

        test_loss = F.binary_cross_entropy(z['classify'],y.view(x.size(0), -1))


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

X_tr = X_train

X_tr = torch.FloatTensor(X_tr)
train_dataset = torch.utils.data.TensorDataset(X_tr)

model = LitAutoEncoder(Encoder())

checkpoint = torch.load("/gpfs/gibbs/pi/zhao/tl688/DeepSynergy/cv_example/epoch=798-step=117453.ckpt")
model.load_state_dict(checkpoint["state_dict"])


confid_int = []
for _ in range(0,100):
    with torch.no_grad():
        y_pred = model.encoder(X_tr)['loewe'].detach()
        confid_int.append(y_pred.t()[0].numpy())

confid_int = np.array(confid_int)

with open('multidrug_confiinterval.pickle', 'wb') as handle:
    pickle.dump(confid_int, handle, protocol=pickle.HIGHEST_PROTOCOL)


