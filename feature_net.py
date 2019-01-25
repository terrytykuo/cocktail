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
lr = 0.01
mom = 0.9
bs = 10

ENTRIES_PER_JSON = 100
CLASSES = 10
SAMPLES_PER_JSON = 1000

#======================================
clean_dir = '/home/tk/Documents/clean/' 
clean_label_dir = '/home/tk/Documents/clean_labels/' 
#========================================

cleanfolder = os.listdir(clean_dir)
cleanfolder.sort()

cleanlabelfolder = os.listdir(clean_label_dir)
cleanlabelfolder.sort()

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
            self.spec = np.array(json.load(f)).transpose(1,0,2,3)
            self.spec = np.concatenate(self.spec, axis=0)

            self.labels = np.array([np.arange(CLASSES) for _ in range(ENTRIES_PER_JSON)])
            self.labels = np.concatenate(self.labels, axis=0)

            indexes = np.arange(ENTRIES_PER_JSON * CLASSES)
            random.shuffle(indexes)
            print(self.labels[:50])
            print(indexes[:50])

            self.spec = torch.Tensor(self.spec[indexes]).squeeze()
            self.labels = torch.Tensor(self.labels[indexes]).squeeze()

            del indexes

        spec = self.spec[offset_in_json]
        label = self.labels[offset_in_json]
        return spec, label

#=================================================    
#           Dataloader 
#=================================================
featureset  = featureDataSet()
trainloader = torch.utils.data.DataLoader(dataset = featureset,
                                                batch_size = bs,
                                                shuffle = False) # must be False for efficiency

#=================================================    
#           model 
#=================================================
''' ResBlock '''
class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ResBlock, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, kernel_size=(3,3), padding=1)

    def forward(self, x):
        if self.channels_out > self.channels_in:
            x1 = F.relu(self.conv1(x))
            x1 =        self.conv2(x1)
            x  = self.sizematch(self.channels_in, self.channels_out, x)
            return F.relu(x + x1)
        elif self.channels_out < self.channels_in:
            x = F.relu(self.conv1(x))
            x1 =       self.conv2(x)
            x = x + x1
            return F.relu(x)
        else:
            x1 = F.relu(self.conv1(x))
            x1 =        self.conv2(x1)
            x = x + x1
            return F.relu(x)

    def sizematch(self, channels_in, channels_out, x):
        zeros = torch.zeros( (x.size()[0], channels_out - channels_in, x.shape[2], x.shape[3]), dtype=torch.float )
        return torch.cat((x, zeros), dim=1)


class featureNet(nn.Module):
    def __init__(self):
        super(featureNet, self).__init__()

        self.conv1 = nn.Sequential(
            ResBlock(1, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            nn.BatchNorm2d(8)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(
            ResBlock(8, 8),
            ResBlock(8, 16),
            ResBlock(16, 16),
            nn.BatchNorm2d(16)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(
            ResBlock(16, 16),
            ResBlock(16, 32),
            ResBlock(32, 32),
            nn.BatchNorm2d(32),
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 64),
            ResBlock(64, 64),
            nn.BatchNorm2d(64),
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 128),
            ResBlock(128, 128),
            nn.BatchNorm2d(128),
        )
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.conv6 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 256),
            ResBlock(256, 256),
            nn.BatchNorm2d(256),
        )
        self.maxpool6 = nn.MaxPool2d(kernel_size=2)

        self.conv7 = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 512),
            ResBlock(512, 512),
            nn.BatchNorm2d(512),
        )
        self.maxpool7 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(bs, 1 ,256, 128)
        
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = self.conv6(x)
        x = self.maxpool6(x)
        x = self.conv7(x)
        x = self.maxpool7(x)

        x = x.reshape(bs, 1024)
        x = F.relu(self.fc(x))

        return F.log_softmax(x, dim = 1)
    
model = featureNet()
try:
    model.load_state_dict(torch.load('/home/tk/Documents/FeatureNet.pkl'))
except:
    print("model not available")
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
from feature_net_test import test as test

loss_record = []
every_loss = []
epoch_loss = []
epoch_accu = []

model.train()
for epo in range(epoch):
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.to(dtype=torch.long)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
        every_loss.append(loss.item())
        print ('[%d, %5d] loss: %.3f hits: %d/%d' % (
                epo, i, loss.item(), 
                np.sum( np.argmax(outputs.detach().numpy(), axis=1) == labels.detach().numpy()),
                bs
            )
        )

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

#plt.figure(figsize = (20, 10))
#plt.plot(epoch_accu)
#plt.xlabel('iterations')
#plt.ylabel('accu')
#plt.savefig('accuracy.png')
#plt.show()
