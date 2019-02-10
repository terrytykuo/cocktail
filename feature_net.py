import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data


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
bs = 1

train_dir = '/home/tk/cocktail/cleanblock/' 


#=================================================    
#               Dataloader 
#=================================================
from Spec_Label_Dataset import Spec_Label_Dataset as Spec_Label_Dataset
featureset  = Spec_Label_Dataset(train_dir)
trainloader = torch.utils.data.DataLoader(dataset = featureset,
                                                batch_size = bs,
                                                shuffle = False) # must be False for efficiency

#=================================================    
#               load
#=================================================
from featureNet.featureNet import featureNet as featureNet

model = featureNet()
try:
    if sys.argv[1]=="reuse":
        model.load_state_dict(torch.load('/home/tk/cocktail/FeatureNet.pkl'))
except:
    print("reused model not available")
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
        print ('[%d, %5d] loss: %.3f hits: %d/%d' % 
            (
                epo, i, loss.item(), 
                np.sum( np.argmax(outputs.detach().numpy(), axis=1) == labels.detach().numpy()),
                bs
            )
        )

    epoch_loss.append(np.mean(every_loss))
    every_loss = []
    corr, total = test(model)
    accuracy = (float)(corr) / total
    epoch_accu.append(accuracy)
    print('test: [%d] accuracy: %.4f' % (epo, accuracy))

            
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
