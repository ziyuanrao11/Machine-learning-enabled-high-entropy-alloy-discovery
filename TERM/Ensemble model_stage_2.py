# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:45:03 2022

@author: z.rao
"""


import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import datetime
import torch.utils.data as Data
import pandas as pd
import torch
import torch.nn.functional as F     # 激励函数都在这
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from My_Modules import MAPELoss
from bayes_opt import BayesianOptimization
import time
import os
import pickle 
import seaborn as sns

#提取数据
plt.close('all')
data = pd.read_excel('Data_base_DFT_Thermal.xlsx')# the input with DFT and Calphad 
df_all = data.drop(columns=['alloy'])
Comp_total = pd.DataFrame()
x1 = df_all [['Fe','Ni','Co','Cr','V','Cu','VEC','AR1','AR2','PE','Density','TermalC','MP','FI','SI','TI',
              'M','TC','MS','MagS_O']]
X = x1[0:696] # these composition are for training
Comp = x1[717:727]# these compoisitions are the high-ranking composition from the stage 1
Comp=Comp.rename({717:0,718:1,719:2,720:3,721:4,722:5,723:6,724:7,725:8,726:9},axis=0)
y1 = df_all [['TEC']]
# y1 = y1[0:705]
Y = y1[0:696]
# bins     = [18,35,48,109,202,234,525,687,695]
bins     = [18,35,48,109,202,234,525,687]
y_binned = np.digitize(Y.index, bins, right=True)

def Tree(n,seed):
    #提取你需要的参数
    target = pd.read_excel('BO/06-01-18-04-Invar_GBDT_with_DFT_(50+500).xlsx')
    colsample_bytree = target.at[n,'colsample_bytree']
    learning_rate = target.at[n,'learning_rate']
    max_bin = target.at[n,'max_bin']
    max_depth = target.at[n,'max_depth']
    max_bin = target.at[n,'max_bin']
    min_child_samples = target.at[n,'min_child_samples']
    min_child_weight = target.at[n,'min_child_weight']
    min_split_gain= target.at[n,'min_split_gain']
    n_estimators = target.at[n,'n_estimators']
    num_leaves = target.at[n,'num_leaves']
    reg_alpha = target.at[n,'reg_alpha']
    reg_lambda = target.at[n,'reg_lambda']
    subsample = target.at[n,'subsample']
    params = {
        "num_leaves": int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        'learning_rate': learning_rate,
        'n_estimators': int(round(n_estimators)),
        'max_bin': int(round(max_bin)),
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'subsample': max(min(subsample, 1), 0),
        'max_depth': int(round(max_depth)),
        'reg_lambda':  max(reg_lambda, 0),
        'reg_alpha': max(reg_alpha, 0),
        'min_split_gain': min_split_gain,
        'min_child_weight': min_child_weight,
        'objective': 'regression',
        'verbose': -1
                 }
    #将数据分为training和testing data
    X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.15, random_state=seed, stratify=y_binned)
    #进行训练（核心代码只有一句话：））
    model = LGBMRegressor(**params)
    model.fit(X_train, Y_train)
    preds = model.predict(Comp)
    return preds

class Net(nn.Module):  
        def __init__(self, n_feature, n_hidden, n_output, w):
            super(Net, self).__init__()   
            # self.BN=torch.nn.BatchNorm1d(n_hidden)
            self.hidden1 = torch.nn.Linear(n_feature, n_hidden) 
            nn.init.kaiming_normal_(self.hidden1.weight)
            
            self.hiddens = nn.ModuleList ([nn.Linear(n_hidden, n_hidden) for i in range(w)])                            
            for m in self.hiddens:
                nn.init.kaiming_normal_(m.weight)   
            
            self.predict = torch.nn.Linear(n_hidden, n_output) 
            nn.init.kaiming_normal_(self.predict.weight)
    
        def forward(self, x): 
            x = self.hidden1(x)
            # x = self.BN(x)
            # x = self.Dropout (x)
            x = F.relu(x)   
            
            for m in self.hiddens:
                x = m(x)
                # x = self.BN(x)
                x = F.relu(x) 
                          
            x = self.predict(x)
            # x = self.BN_3(x)
            # x = self.Dropout (x)
            return x
        
def NN(n,seed):
    #read your data
    data = pd.read_excel('Data_base_DFT_Thermal.xlsx')
    df_all = data.drop(columns=['alloy'])
    x1 = df_all [['Fe','Ni','Co','Cr','V','Cu']]
    # x2 = df_all [['VEC','AR1','AR2','PE','Density',
    #                 'TermalC','MP','FI','SI','TI','M']]
    x2 = df_all [['VEC','AR1','AR2','PE','Density',
                    'TermalC','MP','FI','SI','TI','M','TC','MS','MagS_O']]
    # min_r = [8,135,124,1.8235,7874,75.905,1181,737.14,1562.98,2957.4,0.6]
    # max_r = [10,140,125.679,1.91,8908,116.559,1850.4,762.47,1753.03,3395,2.22] 
    min_r = [8,135,124,1.8235,7874,75.905,1181,737.14,1562.98,2957.4,0.6,2.2439,0.0009,0.0037]
    max_r = [10,140,125.679,1.91,8908,116.559,1850.4,762.47,1753.03,3395,2.22,1456.5497,2.4483,0.0649] 
    min_r = np.array(min_r)
    max_r = np.array(max_r)
    x2_normalization = (x2 - min_r)/(max_r-min_r)
    x_combined = pd.concat ((x1, x2_normalization), axis=1)
    # x_c_u = x_combined[0:705]
    Comp_N = x_combined[717:727]

    
    y1 = df_all [['TEC']]
    # y1 = y1[0:705]
    y1 = y1[0:696]
    #read your hyperparameter
    target = pd.read_excel('BO/05-28-14-50-Invar_NN_DFT_BO(10+100)_only_with_TC_MS_MagS_O.xlsx')
    batch_size = target.at[n,'batch_size']
    lr = target.at[n,'lr']
    module__n_hidden = target.at[n,'module__n_hidden']
    module__w = target.at[n,'module__w']
    module__n_hidden = int(module__n_hidden)
    module__w = int(module__w)
    batch_size = int(batch_size)
    net = Net(n_feature=20, n_hidden=module__n_hidden, n_output=1, w = module__w)
    print(net)
    #load模型
    net.load_state_dict(torch.load('Results/05-28-14-50-Invar_NN_DFT_BO(10+100)_only_with_TC_MS_MagS_O_ensemble/{}-{}.pt'.format(n,seed-10)))
    net.eval()
    Comp_NN = torch.FloatTensor(Comp_N.values)
    preds = net(Comp_NN)
    preds=preds.data.numpy()
    return preds

#训练    
r=0
for i in range(1,11):
    for j in range(50,55):
        #训练Tree
        print ('prediction_Tree_{}'.format(r))
        prediction = Tree(i,j)
        Comp_total['pred_Z_Tree_{}'.format(r)] = prediction
        #训练NN
        print ('prediction_NN_{}'.format(r))
        prediction = NN(i,j)
        Comp_total['pred_Z_NN_{}'.format(r)] = prediction    
        r += 1
               
prediciton_mean = Comp_total.mean(axis=1)
prediciton_std = Comp_total.std(axis=1)
Comp.insert(Comp.shape[1],'pred_Z_mean',prediciton_mean)
Comp.insert(Comp.shape[1],'pred_Z_std',prediciton_std)


Comp.to_csv('Results/05-28-14-50-Invar_NN_DFT_BO(10+100)_only_with_TC_MS_MagS_O+_ensemble_First_round_seed_50_55.csv',index =False)
Comp_total.to_csv('Results/05-28-14-50-Invar_NN_DFT_BO(10+100)_only_with_TC_MS_MagS_ensemble_Comp_total_First_round_seed_50_55.csv',index =False)


