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

CLASSES = 10
ENTRIES_PER_JSON = 100
SAMPLES_PER_JSON = 1000

#======================================
clean_dir = '/home/tk/Documents/clean_test/' 

cleanfolder = os.listdir(clean_dir)
cleanfolder.sort()
#========================================

#========================================
class featureDataSet(Dataset):
    def __init__(self):
        self.curr_json_index = -1

        self.spec = None
        self.labels = None

    def __len__(self):
        return SAMPLES_PER_JSON * len(cleanfolder)

    def __getitem__(self, index):

        newest_json_index = index // SAMPLES_PER_JSON
        offset_in_json = index % SAMPLES_PER_JSON
        
        if not (self.curr_json_index == newest_json_index):
            self.curr_json_index = newest_json_index

            f = open(clean_dir + '{}'.format(cleanfolder[newest_json_index]))
            self.spec  = torch.Tensor(
                np.concatenate(np.array(json.load(f)).transpose(1,0,2,3), axis=0)
            )

            self.labels = np.array([np.arange(CLASSES) for _ in range(ENTRIES_PER_JSON)])
            self.labels = torch.Tensor( np.concatenate(self.labels, axis=0) )

        spec = self.spec[offset_in_json]
        label = self.labels[offset_in_json]
        return spec, label

    
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


#============================================
#              optimizer
#============================================

#=================================================    
#           Dataloader 
#=================================================
featureset = featureDataSet()
testloader = torch.utils.data.DataLoader(dataset = featureset,
                                                batch_size = bs,
                                                shuffle = False)

#============================================
#              testing
#============================================

def test(model):
    criterion = torch.nn.NLLLoss()

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

            loss_record.append(loss.item())
            every_loss.append(loss.item())
            
        epoch_loss.append(np.mean(every_loss))
        every_loss = []

        plt.figure(figsize = (20, 10))
        plt.plot(loss_record)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.savefig('loss.png')
        plt.close()

    return correct, total

if __name__ == '__main__':
    model_for_test = featureNet()
    model_for_test.load_state_dict(torch.load('/home/tk/Documents/FeatureNet.pkl'))
    print(model_for_test)
    test(model)
