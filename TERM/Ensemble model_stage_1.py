# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:41:22 2022

@author: z.rao
"""


import os
import time
from bayes_opt import BayesianOptimization
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import numpy as np
#import seaborn as sns
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime
import pandas as pd
#for GBDT training
def Tree(n,j, WAE_x):
    target = pd.read_excel('BO/05-18-12-09-Invar_GBDT(50+500).xlsx')
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
    data=pd.read_csv('data_base.csv')             
    train_features, test_features, train_labels, test_labels = normalizing_data(data,seed=j)
    train_features, test_features = train_features.cpu().data.numpy(),test_features.cpu().data.numpy()
    train_labels, test_labels = train_labels.cpu().data.numpy(), test_labels.cpu().data.numpy()
    train_labels, test_labels = train_labels.reshape(-1), test_labels.reshape(-1)   
    model = LGBMRegressor(**params)
    model.fit(train_features, train_labels)
    preds = model.predict(WAE_x)
    return preds
# build the NN architechture
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
#define the NN traning        
def NN(n,seed, WAE_x):
    target = pd.read_excel('BO/05-18-19-09-Invar_NN_BO(10+100).xlsx')
    batch_size = target.at[n,'batch_size']
    lr = target.at[n,'lr']
    module__n_hidden = target.at[n,'module__n_hidden']
    module__w = target.at[n,'module__w']
    module__n_hidden = int(module__n_hidden)
    module__w = int(module__w)
    batch_size = int(batch_size)
    net = Net(n_feature=17, n_hidden=module__n_hidden, n_output=1, w = module__w)
    print(net)
    #load模型
    net.load_state_dict(torch.load('Results/Invar_NN_BO(10+100)_6/{}-seed_{}.pt'.format(n,seed)))
    net.eval()
    Comp_NN = torch.FloatTensor(WAE_x.values)
    preds = net(Comp_NN)
    preds=preds.data.numpy()
    return preds

#start the emsemble training 
r=0
Comp_total = pd.DataFrame()
for i in range(0,10):
    for j in range(40,50):
        #Tree
        print ('prediction_Tree_{}'.format(r))
        prediction = Tree(i,j,WAE_x)
        Comp_total['pred_Z_Tree_{}'.format(r)] = prediction
        #NN
        print ('prediction_NN_{}'.format(r))
        prediction = NN(i,j,WAE_x)
        Comp_total['pred_Z_NN_{}'.format(r)] = prediction    
        r += 1
