# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:31:05 2022

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

# Here the GMM is applied, you might wonder why 4 is chosen, the answer can be found below
gm = GaussianMixture(n_components=4, random_state=0, init_params='kmeans').fit(latents) #plot a n_components v.s. Average negative log likelihood
print('Average negative log likelihood:', -1*gm.score(latents))
plot_gmm(gm, latents)

# Using elbow method to find out the best # of components, the lower the negative log likehood the better the model is, but too many cluster is trivial. just imagine you fit each individual data points with a Gaussian, in this case, you would have a very good model. but the such fitting is not very useful.

# In this case, the best number of cluster is either 4 or 5.

scores=[] #using elbow method to find out the best # of components
for i in range(1,8):
  gm = GaussianMixture(n_components=i, random_state=0, init_params='kmeans').fit(latents)
  print('Average negative log likelihood:', -1*gm.score(latents))
  scores.append(-1*gm.score(latents))
  
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
plt.figure()
plt.scatter(range(1,8), scores,color='green')
plt.plot(range(1,8),scores)
plt.savefig('elbow_plot.png', format='png', dpi=300)
plt.show()
