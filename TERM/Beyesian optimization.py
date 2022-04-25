# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:37:38 2022

@author: z.rao
"""


import datetime
import torch.utils.data as Data
import pandas as pd
import torch
import torch.nn.functional as F    
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from bayes_opt import BayesianOptimization
import time
import os
from sklearn import preprocessing

bounds = {'lr': (0.0005,0.001), 'batch_size': (32,64), 'module__n_hidden': (16,526),
          'module__w': (2,10)} #here we define the bounnds to optimize
optimizer = BayesianOptimization(
    f=train_model,
    pbounds=bounds,
    random_state=1,
)

optimizer.maximize(init_points=10, n_iter=1) #set the parameter for BO
print(optimizer.max)
#save the results 
table = pd.DataFrame(columns=['target','batch_size','lr','module__n_hidden','module__w'])
for res in optimizer.res:
    table=table.append(pd.DataFrame({'target':[res['target']],'batch_size':[res['params']['batch_size']],
                                     'lr':[res['params']['lr']], 'module__n_hidden':[res['params']['module__n_hidden']],
                                     'module__w':[res['params']['module__w']]}),ignore_index=True)
table=table.append(pd.DataFrame({'target':[optimizer.max['target']],'batch_size':[optimizer.max['params']['batch_size']],
                                    'lr':[optimizer.max['params']['lr']], 'module__n_hidden':[optimizer.max['params']['module__n_hidden']],
                                    'module__w':[optimizer.max['params']['module__w']]}),ignore_index=True)
model_name = 'Invar_inference_NN'
file_name = '{}.xlsx'.format(model_name)
endtime = datetime.datetime.now()
Rtime = endtime - starttime
print(Rtime)
table.to_excel(file_name)
