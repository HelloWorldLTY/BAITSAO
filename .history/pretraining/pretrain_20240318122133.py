import torch
import torch.nn.functional as F
import lightning as L
import os
import pandas as pd
import numpy as np
import pickle
import sklearn.model_selection
import scipy.stats
import sklearn.metrics

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn

# Please modify the fold path for your own dataset.

df_grountruth_score = pd.read_csv("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/summary_v_1_5.csv")

df_grountruth_score = df_grountruth_score.drop(df_grountruth_score[df_grountruth_score['synergy_loewe'] == '\\N'].index)

df_grountruth_score = df_grountruth_score.dropna(subset=['drug_col', 'drug_row'])

with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_pretraincelllineupdate.pickle", 'rb') as f:
    cellline_name_getembedding = pickle.load(f)
with open("/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/GenePT/ensem_emb_pretraindrugupdate.pickle", 'rb') as f:
    drug_name_getembedding = pickle.load(f)


train_index = df_grountruth_score.index
n_dim = 1536
drug_num_dim = 16
train_list = []
for item in train_index:
    d1, d2, cl = df_grountruth_score.loc[item]['drug_row'], df_grountruth_score.loc[item]['drug_col'], df_grountruth_score.loc[item]['cell_line_name']
    if str(d2) == 'nan':
        value_list = np.hstack([drug_name_getembedding[d1], cellline_name_getembedding[cl]])
        value_list = np.hstack([value_list, 1])
    else:
        value_list = np.hstack([drug_name_getembedding[d1] , drug_name_getembedding[d2] , cellline_name_getembedding[cl]])
        value_list = np.hstack([value_list, 2])    
    train_list.append(value_list)

X_train = np.array(train_list)
y_train = df_grountruth_score[['synergy_zip', 'synergy_loewe', 'ri_row', 'ri_col']].values

y_train = np.array(y_train).astype('float')

y_train_class = (y_train[:,1] > 30)*1

y_train = np.hstack([y_train, np.array([y_train_class]).T])

X_tr, X_test, y_tr, y_test = sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state = 2023)

X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(X_tr, y_tr, test_size=0.1, random_state = 2023)


# Implementation for UW.
class MultiTaskLoss(torch.nn.Module):

    def __init__(self, is_regression, reduction='none'):
        super(MultiTaskLoss, self).__init__()
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction
        self.eps = 1e-8

    def forward(self, losses):
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)
        coeffs = 1 / ( (self.is_regression+1)*(stds**2 + self.eps) )
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
        self.shared_layers = nn.Sequential(nn.Linear(layers[1], layers[0]), 
                                nn.BatchNorm1d(layers[0]),
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[0], layers[1]),
                                nn.BatchNorm1d(layers[1]),
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                               )
        
        self.output_layers = nn.ModuleDict({'loewe':nn.Linear(layers[1], layers[2]),
                                   'ri_row':nn.Linear(layers[1], layers[2]),
                                   'ri_col':nn.Linear(layers[1], layers[2]),
                                   'classify':nn.Sequential(nn.Linear(layers[1], layers[2]),
                                  nn.Sigmoid()
                               )})

        self.input_layers = nn.ModuleDict({'loewe':nn.Linear(n_dim*2 + drug_num_dim, layers[1]),
                                   'ri_row':nn.Linear(n_dim*2 + drug_num_dim, layers[1]),
                                   'ri_col':nn.Linear(n_dim*2 + drug_num_dim, layers[1]),
                                   'classify':nn.Linear(n_dim*2 + drug_num_dim, layers[1])})

        self.drug_num_emb = nn.Embedding(10, drug_num_dim)

    def forward(self, x, key):
        
        x_new = torch.cat([(x[:,0:n_dim] + x[:,n_dim:n_dim*2]) / 2, x[:,n_dim*2:n_dim*3], self.drug_num_emb(x[:,-1].long())], axis=1)
        x_row = torch.cat([x[:,0:n_dim], x[:,n_dim*2:n_dim*3], self.drug_num_emb(x[:,-1].long() - 1)], axis=1)
        x_col = torch.cat([x[:,n_dim:n_dim*2], x[:,n_dim*2:n_dim*3], self.drug_num_emb(x[:,-1].long() - 1)], axis=1)
        
        x_input = x_new
        if key =='ri_row':
            x_input = x_row
        if key =='ri_col':
            x_input = x_col
        
        output_data = self.output_layers[key](self.shared_layers(self.input_layers[key](x_input)))
        return output_data
        
    
    def inference_task(self,x,label):
        x_new = x
        output = self.forward(x_new, label)
        return output


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        #self.encoder.apply(init_weights)
        self.is_regression = torch.Tensor([True, True, False])
        self.loss_mode = MultiTaskLoss(self.is_regression, reduction = 'sum')

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        
        loss = torch.stack([F.mse_loss(self.encoder(x, 'loewe') ,y[:,1].view(x.size(0), -1))
        ,F.mse_loss(self.encoder(x, 'ri_row'),y[:,2].view(x.size(0), -1))
        ,F.binary_cross_entropy(self.encoder(x, 'classify'), y[:,4].view(x.size(0), -1))])
        
        multitaskloss = self.loss_mode(loss)
        return multitaskloss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
#         z = self.encoder(x)
        
        val_loss = torch.stack([F.mse_loss(self.encoder(x, 'loewe') ,y[:,1].view(x.size(0), -1))
        ,F.mse_loss(self.encoder(x, 'ri_row'),y[:,2].view(x.size(0), -1))
        ,F.binary_cross_entropy(self.encoder(x, 'classify'), y[:,4].view(x.size(0), -1))])
        
        # val_loss = self.loss_mode(val_loss)
        val_loss = max(val_loss)

        self.log("val_loss", val_loss.item())
        return val_loss

        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
#         z = self.encoder(x)
        
        test_loss = torch.stack([F.mse_loss(self.encoder(x, 'loewe') ,y[:,1].view(x.size(0), -1))
        ,F.mse_loss(self.encoder(x, 'ri_row'),y[:,2].view(x.size(0), -1))
        ,F.binary_cross_entropy(self.encoder(x, 'classify'), y[:,4].view(x.size(0), -1))])
        
        test_loss = self.loss_mode(test_loss)
        self.log("test_loss", test_loss.item())
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
                patience=100,
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
act_func = 'GELU'
dropout = 0.5 
input_dropout = 0.2
eta = 1e-4
norm = 'tanh' 
drug_num_dim = 16

model = LitAutoEncoder(Encoder())

print(model.log) #see log and log dir for model place



from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
lr_monitor = LearningRateMonitor(logging_interval='step')

train_loader = DataLoader(train_dataset, batch_size=4096, num_workers=5, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2048, num_workers=5)

# train with both splits
trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=500)], max_epochs=1000)

trainer.log_dir

trainer.fit(model, train_loader, valid_loader)

trainer.test(model, dataloaders=DataLoader(test_dataset, batch_size=1024, num_workers=5))



for m in model.encoder.modules():

    for child in m.children():

        if type(child) == nn.BatchNorm1d:

            child.track_running_stats = False

            child.running_mean = None

            child.running_var = None
model.encoder.eval()


"df_grountruth_score[['synergy_zip', 'synergy_loewe', 'synergy_hsa', 'synergy_bliss']].values"

with torch.no_grad():
    y_pred = model.encoder.inference_task(X_test, 'loewe').detach()

cor, pval = scipy.stats.pearsonr(y_pred.t()[0], y_test[:,1])

r2score = sklearn.metrics.r2_score(y_test[:,1],y_pred.t()[0])

print(cor, " ", r2score)



with torch.no_grad():
    y_pred = model.encoder.inference_task(X_test, 'ri_row').detach()



cor, pval = scipy.stats.pearsonr(y_pred.t()[0], y_test[:,2])

r2score = sklearn.metrics.r2_score(y_test[:,2],y_pred.t()[0])

print(cor, " ", r2score)


with torch.no_grad():
    y_pred = model.encoder.inference_task(X_test, 'ri_col').detach()


cor, pval = scipy.stats.pearsonr(y_pred.t()[0], y_test[:,3])

r2score = sklearn.metrics.r2_score(y_test[:,3],y_pred.t()[0])

print(cor, " ", r2score)

with torch.no_grad():
    y_pred = model.encoder.inference_task(X_test, 'classify').detach()
#     y_pred = torch.sigmoid(y_pred)
import scipy.stats
import sklearn.metrics

print(sklearn.metrics.roc_auc_score(y_test[:,4], y_pred.t()[0]), sklearn.metrics.accuracy_score(y_test[:,4], (y_pred.t()[0]>0.5)*1))


