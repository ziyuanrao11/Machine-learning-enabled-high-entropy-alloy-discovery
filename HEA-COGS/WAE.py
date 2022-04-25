# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:23:08 2022

@author: Po-Yen Tung; Ziyuan Rao
"""


import cv2
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# root = '/content/'

sns.set(color_codes=True)

class WAE(nn.Module):
    def __init__(self, input_size):
        super(WAE, self).__init__()
        self.input_size = input_size

        # encoder
        self.encoder = nn.Sequential(
                        nn.Linear(self.input_size, 80),
                        nn.LayerNorm(80),
                        nn.ReLU(),
                        nn.Linear(80, 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Linear(64, 48),
                        nn.LayerNorm(48),
                        nn.ReLU(),
                        nn.Linear(48, 2),
                        )

        # decoder
        self.decoder = nn.Sequential(
                        nn.Linear(2, 48),
                        nn.LayerNorm(48),
                        nn.ReLU(),
                        nn.Linear(48, 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Linear(64, 80),
                        nn.LayerNorm(80),
                        nn.ReLU(),
                        nn.Linear(80, self.input_size),
                        nn.Softmax(dim=1) #(softmad along dimension 1)
                        )
        self.apply(weights_init)
    
    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon, z
    
    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
#%% Data loading, params - here you can play around with all different combinations of paramters to reduce the total loss. You can visualize your training history just to see how good your chosen set of hyperparameters performs 
same_seeds(1) #seed equals to 1

params = {
    'num_epoch' : 200,
    'batch_size' : 20,
    'lr' : 5e-4,
    'weight_decay' : 0.0,
    'sigma' : 8.0,
    'MMD_lambda' : 1e-4,
    'model_name' : 'WAE_v1',
} # for WAE training
all = pd.read_csv('data_base.csv', header=0).iloc[:,1:19].to_numpy()
raw_x = all[:696,:6]
raw_y = all[:696,17].reshape(-1,1)
dataset = FeatureDataset(raw_x[:], raw_y[:]) #numpy to tensor
dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True) # tensor to dataloader
print(raw_x[50:55])
#%%train the WAE
model = WAE(raw_x.shape[1]).to(device) # initialize the model 
optimizer = Adam(model.parameters(), lr = params['lr'], weight_decay = params['weight_decay']) # optimizer
def train_WAE(model, optimizer, dataloader, params):
    model_name = params['model_name']
    num_epoch = params['num_epoch']
    sigma = params['sigma'] # assuming the latent space follows Gaussian
    MMD_lambda = params['MMD_lambda'] #WAE distance (maximum mean discrepancy)

    folder_dir = os.path.join(root, model_name) # a folder to save models
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)
    loss_=[]   
    for epoch in range(num_epoch):
        start_time = time.time()
        total_loss = [] #save for plot, recon loss+MMD
        total_recon = [] # binary cross entropy
        total_MMD = [] #maximum mean discrepancy
        
        for i, data in enumerate(dataloader):
            x = data[0].to(device)
            y = data[1].to(device)
            model.train() # model goes to train mode
            recon_x, z_tilde = model(x) # latent space is Z_tilde
            z = sigma*torch.randn(z_tilde.size()).to(device) # z is sampled from a Gaussian that has the same dimension (but no relation to z_tilde).

            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean') #lowest reconstruction loss 
            #recon_loss = F.mse_loss(recon_x, x, reduction='mean') 
            #recon_loss = F.l1_loss(recon_x, x, reduction='mean')
            
            MMD_loss = imq_kernel(z_tilde, z, h_dim=2).to(device) #W-distance between z_tilde and z
            MMD_loss = MMD_loss / x.size(0) #averaging, because recon loss is mean.
            loss = recon_loss + MMD_loss * MMD_lambda #MM_lambda: learning-rate alike, hyperparamer

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())# from tensor to values
            total_recon.append(recon_loss.item())
            total_MMD.append(MMD_loss.item())

        avg_loss = sum(total_loss)/len(total_loss)
        avg_recon = sum(total_recon)/len(total_recon)
        avg_MMD = sum(total_MMD)/len(total_MMD)
        loss_.append(avg_loss)

        #scheduler.step(avg_loss)

        print('[{:03}/{:03}] loss: {:.6f} Recon_loss: {:.6f}, MMD_loss:{:.6f}, time: {:.3f} sec'.format(\
                                        epoch+1, num_epoch, \
                                        avg_loss, \
                                        avg_recon, avg_MMD, time.time() - start_time))
        # save the model every 5 epoches
        if (epoch+1) % 5 == 0:
            save_model_dir = str(model_name + "_{}.pth".format(epoch+1))
            torch.save(model.state_dict(), os.path.join(folder_dir, save_model_dir))
    return loss_

loss_=train_WAE(model, optimizer, dataloader, params)
plt.figure()
sns.set_style('ticks')
plt.plot(range(len(loss_)),loss_)
#%%Double check on the reconstructed compositions
#one way to find out whether WAE (or any other VAE) has learned the repsentation is 
# to compare the reconstructed and original compositions.if you are not happy with the 
# reconstruction. go back to the previous step and change the params.
#double check on the recontructed compositions
#t = time.localtime()
model_dir = os.path.join(root,'{}/{}_{}.pth'.format(params['model_name'], params['model_name'],params['num_epoch']))#load your model
model = WAE(raw_x.shape[1]).to(device)
model.load_state_dict(torch.load(model_dir))
model.eval()
with torch.no_grad():
    test = torch.FloatTensor(raw_x).to(device)
    recon_x, z = model(test)
    recon_x = model.decoder(z)
    recon_x = recon_x.cpu().detach().numpy()

column_name = ['Fe','Ni','Co','Cr','V','Cu']#,'VEC','AR1','AR2','PE','Density','TC','MP','FI','SI','TI','M']
#recon_x = (recon_x * (max-min)) + min
pd.DataFrame(recon_x.round(3), columns=column_name).loc[690:695]
csv_data = pd.read_csv('data_base.csv', header=0).iloc[:,1:19]
csv_data.iloc[690:702,:6].round(3)
#%%Visualize the WAE latent space
# Here we assign different colors to alloy with and without Copper, as we expected them to differ significantly in the latent space.
sns.set_style('ticks')
model = WAE(raw_x.shape[1]).to(device)
model.load_state_dict(torch.load(model_dir))
dataset = FeatureDataset(raw_x[:], raw_y[:])
latents = get_latents(model, dataset)

low_cu = raw_x[:,5] < 0.05
low_cu_latent = latents[low_cu]
low_cu_color = raw_y[:][low_cu]

high_cu = raw_x[:,5] >= 0.05
high_cu_latent = latents[high_cu]
high_cu_color = raw_y[:][high_cu]


# figure settings
fig, axs = plt.subplots(figsize = (3, 3),dpi=200)

#axs.set_aspect(1.)
#axs.set_ylim(-7,7)
#axs.set_xlim(-11,5)

axs.set_yticks(np.arange(-6, 8, step=2))
axs.set_xticks(np.arange(-10, 5, step=2))

axs.set_yticklabels(np.arange(-6, 8, step=2), fontsize=7)
axs.set_xticklabels(np.arange(-10, 5, step=2), fontsize=7)


for axis in ['top','bottom','left','right']:
  axs.spines[axis].set_linewidth(1.)


axs.tick_params(axis='both', which='major', top=False, labeltop=False, direction='out', width=1., length=4)
axs.tick_params(axis='both', which='major', right=False, labelright=False, direction='out', width=1., length=4)

#scatter1 = axs.scatter(low_cu_latent[:,0], low_cu_latent[:,1], c=low_cu_color, alpha=.75, s=10, linewidths=0, cmap='viridis')
#scatter2 = axs.scatter(high_cu_latent[:,0], high_cu_latent[:,1], c=high_cu_color, alpha=.75, s=9, linewidths=0, cmap='Reds')

scatter1 = axs.scatter(low_cu_latent[:,0], low_cu_latent[:,1], c='steelblue', alpha=.55, s=8, linewidths=0, label='Alloys w/o Cu')
scatter2 = axs.scatter(high_cu_latent[:,0], high_cu_latent[:,1], c='firebrick', alpha=.65, s=14, linewidths=0, marker='^', label='Alloys w/ Cu')
#scatter3 = axs.scatter(latents_exp_4[:,0], latents_exp_4[:,1], alpha=1., s=10, linewidths=.75, edgecolors='darkslategray', facecolors='w')#, label='New FeCoNiCr HEAs')
#scatter4 = axs.scatter(latents_exp_5[:,0], latents_exp_5[:,1], alpha=1., s=16, linewidths=.75, edgecolors='darkred', facecolors='w',marker='^')#, label='New FeCoNiCrCu HEAs')

handles,labels = axs.get_legend_handles_labels()
handles = handles[::1]
labels = labels[::1]

legend_properties = {'size':7.5}
axs.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.015,1.017), handletextpad=-0.3, frameon=False, prop=legend_properties)
#axs.legend(handles, labels, loc='upper left', bbox_to_anchor=(-0.045,1.017), handletextpad=-0.3, frameon=False, prop=legend_properties)

#rect = patches.Rectangle((-19.4,15.0), 18, 4.5, linewidth=0,edgecolor=None,facecolor='k', alpha=0.03,linestyle=None,zorder=-10) #(0.2,15.4), 14, 4.1
#axs.add_patch(rect)

fig.savefig('Figure3_a.tif', bbox_inches = 'tight', pad_inches=0.01)
