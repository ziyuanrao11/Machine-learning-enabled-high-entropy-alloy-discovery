# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:40:00 2022

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

t = time.localtime()
model_name = 'Invar_inference_GBDT'
file_name = '{}.xlsx'.format(model_name)
data = pd.read_csv('data_base.csv')
train_features, test_features, train_labels, test_labels = normalizing_data(data,seed=42)
train_features, test_features = train_features.cpu().data.numpy(),test_features.cpu().data.numpy()
train_labels, test_labels = train_labels.cpu().data.numpy(), test_labels.cpu().data.numpy()
train_labels, test_labels = train_labels.reshape(-1), test_labels.reshape(-1) 
#define the model
def train_model(num_leaves,
                min_child_samples,
            learning_rate,
            n_estimators, 
            max_bin,
            colsample_bytree, 
            subsample, 
            max_depth, 
            reg_alpha,
            reg_lambda,
            min_split_gain,
            min_child_weight
            ):
    params = {
        "num_leaves": int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        'learning_rate': learning_rate,
        'n_estimators': int(round(n_estimators)),
        'max_bin': int(round(max_bin)),
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'subsample': max(min(subsample, 1), 0),
        'max_depth': int(round(max_depth)),
        'reg_alpha':  max(reg_alpha, 0),
        'reg_lambda': max(reg_lambda, 0),
        'min_split_gain': min_split_gain,
        'min_child_weight': min_child_weight,
        'verbose': -1
                  }
    model = LGBMRegressor(**params)
    model.fit(train_features, train_labels)
    y_pred = model.predict(test_features)
    error = -np.mean(np.abs((test_labels - y_pred) / test_labels))       # print(error)     
    return error
#define the parameters for optimize
bounds = {'num_leaves': (5, 60),#50
          'min_child_samples':(1, 50),
          'learning_rate': (0.001, 1),
          'n_estimators': (5, 200),#100
            'max_bin': (5, 100),#10
          'colsample_bytree': (0.5, 1),
          'subsample': (0.1, 2),
          'max_depth': (1, 60),#10
          'reg_alpha': (0.01, 1), #5
          'reg_lambda': (0.01, 1),#5
          'min_split_gain': (0.001, 0.1),
          'min_child_weight': (0.0001, 30)}
optimizer = BayesianOptimization(
    f=train_model,
    pbounds=bounds,
    random_state=1,
)
#optimize the parameter with BO
optimizer.maximize(init_points = 10, n_iter=1)
#save the results
table = pd.DataFrame(columns=['target', 'colsample_bytree', 'learning_rate', 'max_bin',
                      'max_depth','min_child_samples','min_child_weight','min_split_gain',
                      'n_estimators','num_leaves','reg_alpha','reg_lambda','subsample'])
for res in optimizer.res:
    table=table.append(pd.DataFrame({'target':[res['target']],'colsample_bytree':[res['params']['colsample_bytree']],
                                     'colsample_bytree':[res['params']['colsample_bytree']],
                                     'learning_rate':[res['params']['learning_rate']],
                                     'max_bin':[res['params']['max_bin']],
                                     'max_depth':[res['params']['max_depth']],
                                     'min_child_samples':[res['params']['min_child_samples']],
                                     'min_child_weight':[res['params']['min_child_weight']],
                                     'min_split_gain':[res['params']['min_split_gain']],
                                     'n_estimators':[res['params']['n_estimators']],
                                     'num_leaves':[res['params']['num_leaves']],
                                     'reg_alpha':[res['params']['reg_alpha']],
                                     'reg_lambda':[res['params']['reg_lambda']],
                                     'subsample':[res['params']['subsample']]}),
                                     ignore_index=True)
table=table.append(pd.DataFrame({'target':[optimizer.max['target']],'colsample_bytree':[optimizer.max['params']['colsample_bytree']],
                                 'colsample_bytree':[optimizer.max['params']['colsample_bytree']],
                                 'learning_rate':[optimizer.max['params']['learning_rate']],
                                 'max_bin':[optimizer.max['params']['max_bin']],
                                 'max_depth':[optimizer.max['params']['max_depth']],
                                 'min_child_samples':[optimizer.max['params']['min_child_samples']],
                                 'min_child_weight':[optimizer.max['params']['min_child_weight']],
                                 'min_split_gain':[optimizer.max['params']['min_split_gain']],
                                 'n_estimators':[optimizer.max['params']['n_estimators']],
                                 'num_leaves':[optimizer.max['params']['num_leaves']],
                                 'reg_alpha':[optimizer.max['params']['reg_alpha']],
                                 'reg_lambda':[optimizer.max['params']['reg_lambda']],
                                 'subsample':[optimizer.max['params']['subsample']]}),
                                 ignore_index=True)
table.to_excel(file_name)
endtime = datetime.datetime.now()
print ('running time {}'.format(endtime - starttime))
