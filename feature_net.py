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

#======================================
clean_dir = '/home/tk/Documents/clean/' 
mix_dir = '/home/tk/Documents/mix/' 
clean_label_dir = '/home/tk/Documents/clean_labels/' 
mix_label_dir = '/home/tk/Documents/mix_labels/' 
#========================================

cleanfolder = os.listdir(clean_dir)
cleanfolder.sort()

mixfolder = os.listdir(mix_dir)
mixfolder.sort()

cleanlabelfolder = os.listdir(clean_label_dir)
cleanlabelfolder.sort()

mixlabelfolder = os.listdir(mix_label_dir)
mixlabelfolder.sort()

clean_list = []
mix_list = []
clean_label_list = []
mix_label_list = []

#========================================

class MSourceDataSet(Dataset):
    
    def __init__(self, clean_dir, mix_dir, clean_label_dir, mix_label_dir):
                

        for i in cleanfolder:
            with open(clean_dir + '{}'.format(i)) as f:
                clean_list.append(torch.Tensor(json.load(f)))

        for i in mixfolder:
            with open(mix_dir + '{}'.format(i)) as f:
                mix_list.append(torch.Tensor(json.load(f)))
                
        for i in cleanlabelfolder:
            with open(clean_label_dir + '{}'.format(i)) as f:
                clean_label_list.append(torch.Tensor(json.load(f)))

        for i in mixlabelfolder:
            with open(mix_label_dir + '{}'.format(i)) as f:
                mix_label_list.append(torch.Tensor(json.load(f)))
        
        cleanblock = torch.cat(clean_list, 0)
        mixblock = torch.cat(mix_list, 0)
        self.spec = torch.cat([cleanblock, mixblock], 0)
                
        cleanlabel = torch.cat(clean_label_list, 0)
        mixlabel = torch.cat(mix_label_list, 0)
        self.label = torch.cat([cleanlabel, mixlabel], 0)

        
    def __len__(self):
        return self.spec.shape[0]

                
    def __getitem__(self, index): 

        spec = self.spec[index]
        label = self.label[index]
        return spec, label
    
trainset = MSourceDataSet(clean_dir, mix_dir, clean_label_dir, mix_label_dir)
trainloader = torch.utils.data.DataLoader(dataset = trainset,
                                                batch_size = 16,
                                                shuffle = True)

# testloader = torch.utils.data.DataLoader(dataset = testset,
#                                                batch_size = 4,
#                                                shuffle = True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1025*16, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

        
    def forward(self, x):
        x = x.view(-1, 1025*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x
    
model = Net()
print (model)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.00003)

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

            
            
torch.save(model, '/home/tk/Documents/FeatureNet_ada_0.0001.pkl')
with open ('/home/tk/Documents/FeatureNet_0.0001.json', 'w') as f:
    json.dump(epoch_loss, f)


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
