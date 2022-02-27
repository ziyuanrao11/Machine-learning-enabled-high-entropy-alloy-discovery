# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:36:04 2022

@author: z.rao
"""


import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random

class MAPELoss(nn.Module):
     def __init__(self):
        super(MAPELoss, self).__init__() 
        
     def forward (self, output, target):
         loss = torch.mean(torch.abs((target - output) / target))
         # loss = (-1)*loss
         return loss
     
def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)    
    return (data - min)/(max-min)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class FeatureDataset(Dataset):
    '''
    Args: x is a 2D numpy array [x_size, x_features]
    '''
    def __init__(self, x):
        self.x = x
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx])

    def getBatch(self, idxs = []):
        if idxs == None:
            return idxs
        else:
            x_features = []
            for i in idxs:
                x_features.append(self.__getitem__(i))
            return torch.FloatTensor(x_features)
        
def normalizing_data(data, seed=42):  
  df_all = data.drop(columns=['alloy'])
  #create a min max processing object
  composition = df_all [['Fe','Ni','Co','Cr','V','Cu']]
  min_max_scaler = preprocessing.MinMaxScaler()
  normalized_atomic_properties = min_max_scaler.fit_transform(df_all[['VEC','AR1','AR2','PE','Density',
                                              'TC','MP','FI','SI','TI','M']])
  x = pd.concat([composition,pd.DataFrame(normalized_atomic_properties)],axis=1)
  x=x.iloc[:714]
  y = df_all[['TEC']][:714]
  # bins     = [18,35,48,109,202,234,525,687,695]
  bins     = [18,35,48,109,202,234,525,687]
  y_binned = np.digitize(y.index, bins, right=True) #stratified 7-fold: each folder contains a specific type of alloys (7 types in total, each takes 85% and 15% as training and testing)

  x = torch.FloatTensor(x.values) #numpy to tensor
  y = torch.FloatTensor(y.values) #numpy to tensor

  if torch.cuda.is_available():
      x = x.cuda()
      y = y.cuda() 
  
  train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=0.15, random_state=seed, stratify=y_binned)
  return train_features, test_features, train_labels, test_labels
