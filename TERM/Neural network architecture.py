# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:37:02 2022

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
#build the model
class Net(nn.Module):  
    def __init__(self, n_feature=17, n_hidden=218, n_output=1, w = 6):
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
          # 输出值
        return x
#set the parameters for training
def train(net, num_epochs, batch_size, train_features, test_features, train_labels, test_labels,
          train_loader,
          optimizer):
    print ("\n=== train begin ===")
    print(net)
    train_ls, test_ls = [], []
    loss = MAPELoss() # MAPE means Mean Absolute percentile error 
    for epoch in range(num_epochs):
        for x, y in train_loader:
            ls = loss(net(x).view(-1, 1), y.view(-1, 1))
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        if epoch % 100 == 0:
            train_ls.append(loss(net(train_features).view(-1, 1), train_labels.view(-1, 1)).item())
            test_ls.append(loss(net(test_features).view(-1, 1), test_labels.view(-1, 1)).item())
            print ("epoch %d: train loss %f, test loss %f" % (epoch, train_ls[-1], test_ls[-1]))
        
    print ("=== train end ===")
#set the parameters for testing    
def test(model, test_loader):
    model.eval()
    test_loss = 0
    n = 0
    loss = MAPELoss() 
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss(output.view(-1, 1), target.view(-1, 1)).item()  # sum up batch loss
            n += 1

    test_loss /= n
    
    print('Test set: Average loss: {:.4f}'.format(
        test_loss))
    
    return test_loss   

#train the data
def train_model(batch_size,lr, module__n_hidden,module__w):
    module__n_hidden = int(module__n_hidden) # number of neurons per layer
    module__w = int(module__w) # number of hidden layers
    batch_size = int(batch_size)
    train_dataset = Data.TensorDataset(train_features, train_labels)
    test_dataset = Data.TensorDataset(test_features, test_labels)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=True) 
    net = Net(n_feature=17, n_hidden=module__n_hidden, n_output=1, w = module__w)
    if torch.cuda.is_available():
      net = net.cuda()
    n_epochs = 20 
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
    train(net, n_epochs, batch_size,train_features, test_features, 
          train_labels, test_labels,train_loader, optimizer)
    train_loss= test(net,train_loader)
    test_loss = test(net, test_loader)

    
    r = -np.abs(train_loss-test_loss)
    
    return -test_loss
#build the ensemble traning for NN
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
import time
import os
import pickle 
import seaborn as sns
t = time.localtime() 
plt.close('all')
target = pd.read_excel('BO/05-18-19-09-Invar_NN_BO(10+100).xlsx')
starttime = datetime.datetime.now()
for i in range(7,8): # This is to choose best 10 points 
    for j in range(42,44): # 10 different seeds
        train_features, test_features, train_labels, test_labels = normalizing_data(data, seed=j)
        lr = target.at[i,'lr'] # the same
        module__n_hidden = target.at[i,'module__n_hidden']
        module__w = target.at[i,'module__w']
        batch_size = target.at[i,'batch_size']
        
        module__n_hidden = int(module__n_hidden)
        module__w = int(module__w)
        batch_size = int(batch_size)
        print (module__w)
        
        batch_size = target.at[i,'batch_size'] # choose 'batch_size' paramter at ith row
        lr = target.at[i,'lr'] # the same
        module__n_hidden = target.at[i,'module__n_hidden']
        module__w = target.at[i,'module__w']
        
        module__n_hidden = int(module__n_hidden)
        module__w = int(module__w)
        batch_size = int(batch_size)
        print (module__w)
        train_dataset = Data.TensorDataset(train_features, train_labels)
        test_dataset = Data.TensorDataset(test_features, test_labels)
        train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=True) 
            

        class Net(nn.Module):  
            def __init__(self, n_feature=17, n_hidden=module__n_hidden, n_output=1, w = module__w):
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
   
        def plotCurve(x_vals, y_vals, 
                        x_label, y_label, 
                        x2_vals=None, y2_vals=None, 
                        legend=None,
                        figsize=(3.5, 2.5)):
            # set figsize
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.plot(x_vals, y_vals)
            if x2_vals and y2_vals:
                plt.plot(x2_vals, y2_vals, linestyle=':')
            
            if legend:
                plt.legend(legend)
        #training 
        print ("\n=== train begin ===")
        
        net = Net()
        print(net)
        if torch.cuda.is_available():
            net = net.cuda()    
        train_ls, test_ls = [], []
        loss = MAPELoss() 
        n_epochs = 3000
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
        for epoch in range(n_epochs):
            for x, y in train_loader:
                ls = loss(net(x).view(-1, 1), y.view(-1, 1))
                optimizer.zero_grad()
                ls.backward()
                optimizer.step()
            train_ls.append(loss(net(train_features).view(-1, 1), train_labels.view(-1, 1)).item())
            test_ls.append(loss(net(test_features).view(-1, 1), test_labels.view(-1, 1)).item())
            if epoch % 100 == 0:
                print ("epoch %d: train loss %f, test loss %f" % (epoch, train_ls[-1], test_ls[-1]))
        print ("plot curves")
        plotCurve(range(1, n_epochs + 1), train_ls,"epoch", "loss",range(1, n_epochs + 1), test_ls,["train", "test"])
        plt.text(60, 0.7, 'Loss=%.4f' % test_ls[-1], fontdict={'size': 20, 'color':  'red'})
        folder_dir = 'Results/Invar_NN_BO(10+100)_6'
        if not os.path.isdir(folder_dir):
          os.mkdir(folder_dir)
        folder_dir = 'Results/Invar_NN_BO(10+100)_6/Figures'
        if not os.path.isdir(folder_dir):
          os.mkdir(folder_dir)
        fig_name_1 = 'Results/Invar_NN_BO(10+100)_6/Figures/{}-seed_{}_1.png'.format(i,j)
        plt.savefig(fig_name_1, format='png', dpi=300)            
                   
        #plotting
        net.eval()
        predict=net(test_features)
        predict=predict.cpu()
        predict=predict.data.numpy()  
        plt.figure()
        sns.regplot(x=predict, y=test_labels.cpu().data.numpy(), color='g') 
        fig_name_2 = 'Results/Invar_NN_BO(10+100)_6/Figures/{}-seed_{}.png'.format(i,j)
        plt.savefig(fig_name_2, format='png', dpi=300)
         
        #save the models
        net_name = 'Results/Invar_NN_BO(10+100)_6/{}-seed_{}.pt'.format(i,j)
        torch.save(net.state_dict(), net_name)
        
endtime = datetime.datetime.now()
Rtime = endtime - starttime
print(Rtime)
