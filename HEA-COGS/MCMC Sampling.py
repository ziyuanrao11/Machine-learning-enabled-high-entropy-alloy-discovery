# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:29:52 2022

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

def MCMC(gm, classifier, n_samples, sigma=0.1): #MCMC
    sample_z = []

    z = gm.sample(1)[0]
    for i in range(n_samples):
        uniform_rand = np.random.uniform(size=1)
        z_next = np.random.multivariate_normal(z.squeeze(),sigma*np.eye(2)).reshape(1,-1)

        z_combined = np.concatenate((z, z_next),axis=0)
        scores = cls(torch.Tensor(z_combined).to(device)).detach().cpu().numpy().squeeze() 
        z_score, z_next_score = np.log(scores[0]), np.log(scores[1]) #z score needes to be converted to log, coz gm score is log.
        z_prob, z_next_prob = (gm.score(z)+z_score), (gm.score(z_next)+z_next_score) # two log addition, output: log probability
        accepence = min(0, (z_next_prob - z_prob))

        if i == 0:
            sample_z.append(z.squeeze())

        if np.log(uniform_rand) < accepence:
            sample_z.append(z_next.squeeze())
            z = z_next
        else:
            pass

    return np.stack(sample_z)

#%%Sample 5000 times with sigma=0.5
sample_z = MCMC(gm=gm, classifier=cls, n_samples=5000, sigma=0.5)
WAE_comps = model._decode(torch.Tensor(sample_z).to(device)).detach().cpu().numpy()  # new_comps save as csv and goes to TERM
print('Sample size:', sample_z.shape)   
WAE_comps=pd.DataFrame(WAE_comps)
WAE_comps.columns=column_name
WAE_comps.to_csv('comps_WAE.csv',index=False)
WAE_comps.head() 
