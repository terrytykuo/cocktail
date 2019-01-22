import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torch.autograd import Variable


import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset

import matplotlib.pyplot as plt
import pickle
import os
import json
import numpy as np
import random
random.seed(7)


#=============================================
#        Hyperparameters
#=============================================

epoch = 5
lr = 0.001
mom = 0.8
bs = 10

SAMPLES_PER_JSON = 200

#======================================
clean_dir = '/home/tk/Documents/clean/' 
clean_label_dir = '/home/tk/Documents/clean_labels/' 
#========================================

cleanfolder = os.listdir(clean_dir)
cleanfolder.sort()

cleanlabelfolder = os.listdir(clean_label_dir)
cleanlabelfolder.sort()

# clean_list = []
# clean_label_list = []

#========================================

class featureDataSet(Dataset):
    def __init__(self):
        self.curr_json_index = 0

        f = open(clean_dir + '{}'.format(cleanfolder[self.curr_json_index]))
        self.spec = torch.Tensor(json.load(f)) 
        f = open(clean_label_dir + '{}'.format(cleanlabelfolder[self.curr_json_index]))
        self.label = torch.Tensor(json.load(f))

    def __len__(self):
        return SAMPLES_PER_JSON * len(cleanfolder)

    def __getitem__(self, index):
        # print("__getitem__: " + str(index))
        
        newest_json_index = index // SAMPLES_PER_JSON
        offset_in_json = index % SAMPLES_PER_JSON
        
        if not (self.curr_json_index == newest_json_index):
            self.curr_json_index = newest_json_index

            f = open(clean_dir + '{}'.format(cleanfolder[newest_json_index]))
            self.spec = torch.Tensor(json.load(f)) 
            f = open(clean_label_dir + '{}'.format(cleanlabelfolder[newest_json_index]))
            self.label = torch.Tensor(json.load(f))

        spec = self.spec[offset_in_json]
        label = self.label[offset_in_json]
        return spec, label

    
#=================================================    
#           Dataloader 
#=================================================
featureset = featureDataSet()
trainloader = torch.utils.data.DataLoader(dataset = featureset,
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
import feature_net_test

loss_record = []
every_loss = []
epoch_loss = []
epoch_accu = []

model.train()
for epo in range(epoch):
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        inputs = Variable(inputs)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.to(dtype=torch.long)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
        every_loss.append(loss.item())
        print ('[%d, %5d] loss: %.3f' % (epo, i, loss.item()))
        
    epoch_loss.append(np.mean(every_loss))
    every_loss = []
    accuracy = test(model)
    epoch_accu.append(accuracy)
            
            
torch.save(model.state_dict(), '/home/tk/Documents/FeatureNet.pkl')


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
plt.savefig('epoch_loss.png')
plt.show()

plt.figure(figsize = (20, 10))
plt.plot(epoch_accu)
plt.xlabel('iterations')
plt.ylabel('accu')
plt.savefig('accuracy.png')
plt.show()