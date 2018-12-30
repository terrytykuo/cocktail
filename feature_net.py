import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset

import matplotlib.pyplot as plt
import pickle
import os
import json
import numpy as np



#=============================================
#        Hyperparameters
#=============================================

epoch = 2
lr = 0.001
mom = 0.9
bs = 16

#======================================
clean_dir = '/home/tk/Documents/clean/' 
clean_label_dir = '/home/tk/Documents/clean_labels/' 
#========================================

cleanfolder = os.listdir(clean_dir)
cleanfolder.sort()

cleanlabelfolder = os.listdir(clean_label_dir)
cleanlabelfolder.sort()

clean_list = []
clean_label_list = []

#========================================

class featureDataSet(Dataset):
    
    def __init__(self, clean_dir, clean_label_dir):
                

        for i in cleanfolder:
            with open(clean_dir + '{}'.format(i)) as f:
                clean_list.append(torch.Tensor(json.load(f)))
                
        for i in cleanlabelfolder:
            with open(clean_label_dir + '{}'.format(i)) as f:
                clean_label_list.append(torch.Tensor(json.load(f)))
        
        cleanblock = torch.cat(clean_list, 0)
        self.spec = torch.cat([cleanblock], 0)
                
        cleanlabel = torch.cat(clean_label_list, 0)
        self.label = torch.cat([cleanlabel], 0)

        
    def __len__(self):
        return self.spec.shape[0]

                
    def __getitem__(self, index): 

        spec = self.spec[index]
        label = self.label[index]
        return spec, label

    
#=================================================    
#           Dataloader 
#=================================================
featureset = featureDataSet(clean_dir, clean_label_dir)
trainloader = torch.utils.data.DataLoader(dataset = featureset,
                                                batch_size = bs,
                                                shuffle = True)

#=================================================    
#           model 
#=================================================
class featureNet(nn.Module):
    def __init__(self):
        super(featureNet, self).__init__()
        self.fc1 = nn.Linear(256 * 128, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

        
    def forward(self, x):
        x = x.view(-1, 256*128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x
    
model = featureNet()
print (model)

#============================================
#              optimizer
#============================================
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = mom)

#============================================
#              training
#============================================

loss_record = []
every_loss = []
epoch_loss = []


model.train()
for epoch in range(70):
    
    
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
        every_loss.append(loss.item())
        print ('[%d, %5d] loss: %.3f' % (epoch, i, loss.item()))
        
    epoch_loss.append(np.mean(every_loss))
    every_loss = []

            
            
torch.save(model, '/home/tk/Documents/FeatureNet.pkl')


plt.figure(figsize = (20, 10))
plt.plot(loss_record)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.show()

plt.figure(figsize = (20, 10))
plt.plot(epoch_loss)
plt.xlabel('iterations')
plt.ylabel('epoch_loss')
plt.savefig('epoch_loss')
plt.show()