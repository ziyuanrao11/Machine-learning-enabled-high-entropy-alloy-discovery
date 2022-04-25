# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:33:41 2022

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

# A simple neural network classifier that predicts INVAR based on composition.
class Classifier(nn.Module): #a very simple classifer with large dropout. intuition here: as simple as possible, given that we only have 2d input
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2,8),
            nn.Dropout(0.5),
            nn.Linear(8,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

#%%Classifier training, Again you can play around with params just to see how it affects the model accuracy (training and tesing)

from matplotlib.pyplot import MultipleLocator
same_seeds(1)

params['cls_bs'] = 16
params['cls_lr'] = 1e-4
params['cls_epoch'] = 100
params['num_fold'] = 5


params['label_y'] = np.where(raw_y<5, 1, 0)
params['latents'] = latents

cls = Classifier().to(device)
opt = Adam(cls.parameters(), lr=params['cls_lr'], weight_decay=0.)


def training_Cls(model, optimizer, params):
    label_y = params['label_y']
    latents = params['latents']
    cls_epoch = params['cls_epoch']

    kf = KFold(n_splits=params['num_fold'])
    train_acc = []
    test_acc = []

    k=1
    for train, test in kf.split(latents):
        x_train, x_test, y_train, y_test = latents[train], latents[test], label_y[train], label_y[test]
        cls_dataset = AttributeDataset(x_train, y_train)
        cls_dataloader = DataLoader(cls_dataset, batch_size=params['cls_bs'], shuffle=True)
        cls_testDataset = AttributeDataset(x_test, y_test)
        cls_testDataloader = DataLoader(cls_testDataset, batch_size=cls_testDataset.__len__(), shuffle=False)


        for epoch in range(cls_epoch):
            t = time.time()
            total_loss = []
            total_acc = []
            cls.train()
            
            for i, data in enumerate(cls_dataloader):
                x = data[0].to(device)
                y = data[1].to(device)
                y_pred = cls(x)
                loss = F.binary_cross_entropy(y_pred, y)
                total_acc.append(torch.sum(torch.where(y_pred>=0.5,1,0) == y).detach().cpu().numpy())
                total_loss.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()
            
            #eval
            cls.eval()
            for test in cls_testDataloader:
                x = test[0].to(device)
                y = test[1].to(device)
                y_pred = cls(x)
                accuracy = torch.sum(torch.where(y_pred>=0.5,1,0) == y) / y_pred.size(0)
                test_loss = F.binary_cross_entropy(y_pred, y)

            #print(f'[{epoch+1:03}/{cls_epoch}] loss:{sum(total_loss)/len(total_loss):.3f} test_loss:{test_loss.item():.3f} acc:{sum(total_acc)/cls_dataset.__len__():.3f} test_acc:{accuracy:.3f} time:{time.time()-t:.3f}')
        
        print('[{}/{}] train_acc: {:.04f} || test_acc: {:.04f}'.format(k, params['num_fold'], sum(total_acc)/cls_dataset.__len__(), accuracy.item()))
        train_acc.append(sum(total_acc)/cls_dataset.__len__())
        test_acc.append(accuracy.item())
        k+=1
    print('train_acc: {:.04f} || test_acc: {:.04f}'.format(sum(train_acc)/len(train_acc), sum(test_acc)/len(test_acc)))
    plt.figure()
    sns.set_style()
    plt.xlabel('number of folds')
    plt.ylabel('loss')
    x=range(1,params['num_fold']+1)
    sns.set_style("darkgrid")
    x_major_locator=MultipleLocator(1)
    ax=plt.gca()
    plt.plot(x, train_acc)
    plt.plot(x, test_acc, linestyle=':', c='steelblue')
    plt.legend(["train_accuracy", "test_accuracy"])
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig('figure/binary_classifier.png',dpi=300)
    return train_acc, test_acc

train_acc, test_acc = training_Cls(cls, opt, params)
