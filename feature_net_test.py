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
import random 
random.seed(0)


#=============================================
#        Hyperparameters
#=============================================

epoch = 2
lr = 0.001
mom = 0.9
bs = 10

#======================================
clean_dir = '/home/tk/Documents/clean_test/' 
clean_label_dir = '/home/tk/Documents/clean_labels_test/' 
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
testloader = torch.utils.data.DataLoader(dataset = featureset,
                                                batch_size = bs,
                                                shuffle = True)

#=================================================    
#           model 
#=================================================
class featureNet(nn.Module):
    def __init__(self):
        super(featureNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(2,2), stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size = (2,2))
        self.batchnorm = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(16*8*8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        
    def forward(self, x):
        x = x.view(bs, 1 ,256, 128)
        x = F.relu(self.maxpool(self.conv1(x)))
        x = F.relu(self.maxpool(self.conv2(x)))
        x = self.batchnorm(x)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim = 1)
    
model = featureNet()
model.load_state_dict(torch.load('/home/tk/Documents/FeatureNet.pkl'))
print (model)

#============================================
#              optimizer
#============================================
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = mom)

#============================================
#              training
#============================================

loss_record = []
every_loss = []
epoch_loss = []
correct = 0
total = 0
blank = []

model.eval()

with torch.no_grad():       
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = model(inputs)
        labels = labels.to(dtype=torch.long)

        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        
        correct += (predicted == labels).sum()
        print ("correct = ", correct, 'total =', total)
        
        accuracy = 100 * correct/ total
        loss_record.append(loss.item())
        every_loss.append(loss.item())
        print ('[%d, %5d] loss: %.3f, acc: %.3f' % (epoch, i, loss.item(), accuracy))
        
    epoch_loss.append(np.mean(every_loss))
    every_loss = []

            
           
plt.figure(figsize = (20, 10))
plt.plot(loss_record)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.show()