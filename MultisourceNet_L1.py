import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.init as init
import pytorch_ssim

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import pickle
import os
import json
import numpy as np
import gc
import cv2

#=============================================
#        Hyperparameters
#=============================================

epoch = 200
lr = 0.01
mom = 0.9
bs = 1

#=============================================
#        Define Functions
#=============================================

def odd(w):
    return list(np.arange(1, w, step=2, dtype='long'))

def even(w):
    return list(np.arange(0, w, step=2, dtype='long'))

def white(x):
    fw, tw = x.shape[1], x.shape[2]

    first = F.relu(torch.normal(mean=torch.zeros(fw, tw), std=torch.ones(fw, tw)) ) * 0.05
    second_seed = F.relu(torch.normal(mean=torch.zeros(fw//2, tw//2), std=torch.ones(fw//2, tw//2))) * 0.03
    second = torch.zeros(fw, tw)

    row_x  = torch.zeros(int(fw//2), tw)
    # row_x = torch.zeros(int(fw/2), tw)

    row_x[:, odd(tw)]  = second_seed
    row_x[:, even(tw)] = second_seed

    second[odd(fw), :]  = row_x
    second[even(fw), :] = row_x

    return second + first


#=============================================
#        path
#=============================================

server = False

root_dir = '/home/tk/Documents/'
if server == True:
    root_dir = '/home/guotingyou/cocktail_phase2/'


clean_dir = root_dir + 'mix_pool/feature/' 
mix_dir = root_dir + 'mix_pool/mix_spec/' # 10-people mix
target_spec_dir = root_dir + 'mix_pool/target_spec/' 
target_label_dir = root_dir + 'mix_pool/target_label/'

full_audio = ['birdstudybook', 'captaincook', 'cloudstudies_02_clayden_12', 
      'constructivebeekeeping',
      'discoursesbiologicalgeological_16_huxley_12', 
      'natureguide', 'pioneersoftheoldsouth', 
      'pioneerworkalps_02_harper_12', 
      'romancecommonplace', 'travelstoriesretold']


#=============================================
#       Define Datasets
#=============================================


class mixDataSet(Dataset):
    
    def __init__(self, mix_dir, target_spec_dir, target_label_dir):           
        
        mix_list = []
        target_spec_list = []
        target_label_list = []

        for na in os.listdir(mix_dir):
            with open(mix_dir + '{}'.format(na)) as f:
                mix_list.append(torch.Tensor(json.load(f)))

        for na in os.listdir(target_spec_dir):
            with open(target_spec_dir + '{}'.format(na)) as f:
                target_spec_list.append(torch.Tensor(json.load(f)))

        for na in os.listdir(target_label_dir):
            with open(target_label_dir + '{}'.format(na)) as f:
                target_label_list.append(torch.Tensor(json.load(f)))

        
        mixblock = torch.cat(mix_list, 0)
        targetblock = torch.cat(target_spec_list, 0)
        targetlabel = torch.cat(target_label_list, 0)

        

        self.mix_spec = mixblock
        self.target_spec = targetblock
        self.target_label = targetlabel

                
        
    def __len__(self):
        return self.mix_spec.shape[0]


    def __getitem__(self, index): 

        mix_spec = self.mix_spec[index]
        target_spec = self.target_spec[index]
        target_label = self.target_label[index]
        return mix_spec, target_spec, target_label


class featureDataSet(Dataset):
    
    def __init__(self, clean_dir, label):
        

        feature_list = []

        with open(clean_dir + full_audio[label] + '/0.json') as f:
            feature_list.append(torch.Tensor(json.load(f)))      
        
        featureblock = torch.cat(feature_list, 0)
        
        self.featurespec = featureblock
        self.label = label
                
        
    def __len__(self):
        return self.featurespec.shape[0]

                
    def __getitem__(self): 
            
        index = int(self.label)
        featurespec = self.featurespec[index]
        return featurespec, index
     
#=============================================
#        Define Dataloader
#=============================================


mixset = mixDataSet( mix_dir, target_spec_dir, target_label_dir)

mixloader = torch.utils.data.DataLoader(dataset = mixset,
    batch_size = bs,
    shuffle = False)

#=============================================
#        Model
#=============================================

'''featureNet'''
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
        feat = F.relu(self.fc2(x))
        x = self.fc3(feat)
        
        return feat, F.log_softmax(x, dim = 1)
    
featurenet = featureNet()
featurenet.load_state_dict(torch.load(root_dir + 'FeatureNet.pkl'))
    
'''ANet'''
class ANet(nn.Module):
    
    def __init__(self):
        super(ANet, self).__init__()

        self.linear7 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        self.linear6 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.linear5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.linear4 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(256, 16),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(bs, 1, 256)

        a7 = self.linear7(x).view(bs, 512, 1, 1)
        a6 = self.linear6(x).view(bs, 256, 1, 1)
        a5 = self.linear5(x).view(bs, 128, 1, 1)
        a4 = self.linear4(x).view(bs, 64, 1, 1)
        a3 = self.linear3(x).view(bs, 32, 1, 1)
        a2 = self.linear2(x).view(bs, 16, 1, 1)

        return a7, a6, a5, a4, a3, a2

A_model = ANet()
# A_model.load_state_dict(torch.load(root_dir + 'cocktail/combinemodel_fullconv/A.pkl'))


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
            x1 = F.relu(self.conv1(x), inplace = True)
            x1 =        self.conv2(x1)
            x  = self.sizematch(self.channels_in, self.channels_out, x)
            return F.relu(x + x1, inplace = True)
        elif self.channels_out < self.channels_in:
            x = F.relu(self.conv1(x))
            x1 =       self.conv2(x)
            x = x + x1
            return F.relu(x, inplace = True)
        else:
            x1 = F.relu(self.conv1(x), inplace = True)
            x1 =        self.conv2(x1)
            x = x + x1
            return F.relu(x, inplace = True)

    def sizematch(self, channels_in, channels_out, x):
        zeros = torch.zeros( (x.size()[0], channels_out - channels_in, x.shape[2], x.shape[3]), dtype = torch.float32)
        return torch.cat((x, zeros), dim=1)

class ResTranspose(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ResTranspose, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out

        self.deconv1 = nn.ConvTranspose2d(in_channels=channels_in, out_channels=channels_out, kernel_size=(2,2), stride=2)
        self.deconv2 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, kernel_size=(3,3), padding=1)

    def forward(self, x):
        # cin = cout
        x1 = F.relu(self.deconv1(x), inplace = True)
        x1 =        self.deconv2(x1)
        x = self.sizematch(x)
        return F.relu(x + x1, inplace = True)

    def sizematch(self, x):
        # expand
        x2 = torch.zeros(x.shape[0], self.channels_in, x.shape[2]*2, x.shape[3]*2)

        row_x  = torch.zeros(x.shape[0], self.channels_in, x.shape[2], 2*x.shape[3])
        row_x[:,:,:,odd(x.shape[3]*2)]   = x
        row_x[:,:,:,even(x.shape[3]*2)]  = x
        x2[:,:, odd(x.shape[2]*2),:] = row_x
        x2[:,:,even(x.shape[2]*2),:] = row_x

        return x2


def initialize(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)
    if isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight)



class ResDAE(nn.Module):
    def __init__(self):
        super(ResDAE, self).__init__()

        # 128x128x1

        self.upward_net1 = nn.Sequential(
            ResBlock(1, 1),
            ResBlock(1, 8),
            ResBlock(8, 8),
            nn.BatchNorm2d(8),
        )

        # 64x64x8

        self.upward_net2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(8, 8),
            ResBlock(8, 16),
            ResBlock(16, 16),
            nn.BatchNorm2d(16),
        )

        # 32x32x16

        self.upward_net3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(16, 16),
            ResBlock(16, 32),
            ResBlock(32, 32),
            nn.BatchNorm2d(32),
        )

        # 16x16x32

        self.upward_net4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(32, 32),
            ResBlock(32, 64),
            ResBlock(64, 64),
            nn.BatchNorm2d(64),
        )

        # 8x8x64

        self.upward_net5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 128),
            ResBlock(128, 128),
            nn.BatchNorm2d(128),
        )

        # 4x4x128

        self.upward_net6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(128, 128),
            ResBlock(128, 256),
            ResBlock(256, 256),
            nn.BatchNorm2d(256),
        )

        # 2x2x256

        self.upward_net7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(256, 256),
            ResBlock(256, 512),
            ResBlock(512, 512),
            nn.BatchNorm2d(512),
        )

        # 1x1x512
        self.downward_net7 = nn.Sequential(
            ResBlock(512, 512),
            ResBlock(512, 256),
            ResBlock(256, 256),
            ResTranspose(256, 256),
#            nn.ConvTranspose2d(256, 256, kernel_size = (2,2), stride = 2),
            nn.BatchNorm2d(256),
        )

        # 2x2x256

        self.downward_net6 = nn.Sequential(
            # 8x8x64
            ResBlock(256, 256),
            ResBlock(256, 128),
            ResBlock(128, 128),
            ResTranspose(128, 128),
#            nn.ConvTranspose2d(128, 128, kernel_size = (2,2), stride = 2),
            nn.BatchNorm2d(128),
        )

        # 4x4x128
        # cat -> 4x4x256
        self.uconv5 = nn.Conv2d(256, 128, kernel_size=(3,3), padding=(1,1))
        # 4x4x128
        self.downward_net5 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 64),
            ResBlock(64, 64),
            ResTranspose(64, 64),
#            nn.ConvTranspose2d(64, 64, kernel_size = (2,2), stride = 2),
            nn.BatchNorm2d(64),
        )

        # 8x8x64
        # cat -> 8x8x128
        self.uconv4 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=(1,1))
        # 8x8x64
        self.downward_net4 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResTranspose(32, 32),
#            nn.ConvTranspose2d(32, 32, kernel_size = (2,2), stride = 2),
            nn.BatchNorm2d(32),
        )

        # 16x16x32
        # cat -> 16x16x64
        self.uconv3 = nn.Conv2d(64, 32, kernel_size=(3,3), padding=(1,1))
        # 16x16x32
        self.downward_net3 = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResTranspose(16, 16),
#            nn.ConvTranspose2d(16, 16, kernel_size = (2,2), stride = 2),
            nn.BatchNorm2d(16),
        )

        # 32x32x16
        # cat -> 32x32x32
        self.uconv2 = nn.Conv2d(32, 16, kernel_size=(3,3), padding=(1,1))
        # 32x32x16
        self.downward_net2 = nn.Sequential(
            ResBlock(16, 16),
            ResBlock(16, 8),
            ResBlock(8, 8),
            ResTranspose(8, 8),
#            nn.ConvTranspose2d(8, 8, kernel_size = (2,2), stride = 2),
            nn.BatchNorm2d(8),
        )

        # 64x64x8
        self.downward_net1 = nn.Sequential(
            ResBlock(8, 8),
            ResBlock(8, 1),
            ResBlock(1, 1),
            ResBlock(1, 1),
            nn.BatchNorm2d(1),
        )

        # 128x128x1
        
        self.apply(initialize)


    def upward(self, x, a7= True, a6= True, a5= True, a4= True, a3= True, a2= True):
        x = x.view(bs, 1, 256, 128)

        # 1x256x128
#        print ("initial", x.shape)

        x = self.upward_net1(x)
#        print ("after conv1", x.shape)


        # 8x128x64
        x = self.upward_net2(x)
        if a2 is not None: x = x * a2
        self.x2 = x
#        print ("after conv2", x.shape)

        # 16x64x32
        x = self.upward_net3(x)
        if a3 is not None: x = x * a3
        self.x3 = x
#        print ("after conv3", x.shape)

        # 32x32x16
        x = self.upward_net4(x)
        if a4 is not None: x = x * a4
        self.x4 = x
#        print ("after conv4", x.shape)

        # 64x16x8
        x = self.upward_net5(x)
        if a5 is not None: x = x * a5
        self.x5 = x
#        print ("after conv5", x.shape)

        
        # 128x8x4
        x = self.upward_net6(x)
        if a6 is not None: x = x * a6
#        print ("after conv6", x.shape)

        # 256x4x2
        x = self.upward_net7(x)
        if a7 is not None: x = x * a7
#        print ("after conv7", x.shape)

        # 512x2x1
        return x


    def downward(self, y, shortcut= True):
#        print ("begin to downward, y.shape = ", y.shape)
        # 512x2x1
        y = self.downward_net7(y)
#        print ("after down7", y.shape)

        # 256x4x2
        y = self.downward_net6(y)
#        print ("after down6", y.shape)

        # 128x8x4
        if shortcut:
            y = torch.cat((y, self.x5), 1)
            y = F.relu(self.uconv5(y))
        y = self.downward_net5(y)
#        print ("after down5", y.shape)

        # 64x16x8
        if shortcut:
            y = torch.cat((y, self.x4), 1)
            y = F.relu(self.uconv4(y))
        y = self.downward_net4(y)
#        print ("after down4", y.shape)

        # 32x32x16
        if shortcut:
            y = torch.cat((y, self.x3), 1)
            y = F.relu(self.uconv3(y))
        y = self.downward_net3(y)
#        print ("after down3", y.shape)

        # 16x64x32
        if shortcut:
            y = torch.cat((y, self.x2), 1)
            y = F.relu(self.uconv2(y))
        y = self.downward_net2(y)
#        print ("after down2", y.shape)

        # 8x128x64
        y = self.downward_net1(y)
#        print ("after down1", y.shape)
 
        # 1x256x128
        return y

Res_model = ResDAE()
Res_model = torch.load(root_dir + 'DAE_SSIM.pkl')
# Res_model.load_state_dict(torch.load(root_dir + 'cocktail/combinemodel_fullconv/res.pkl'))


# print (model)

#=============================================
#        Optimizer
#=============================================

#import pytorch_ssim
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(Res_model.parameters(), lr = lr, momentum = mom)
optimizer_A = torch.optim.SGD(A_model.parameters(), lr = lr, momentum = mom)


#=============================================
#        Loss Record
#=============================================

loss_record = []
# every_loss = []
# epoch_loss = []


#=============================================
#        Train
#=============================================

Res_model.train()
for epo in range(epoch):
    for i, data in enumerate(mixloader, 0):

    
        # get mix spec & label
        mix_spec, target_spec, target_label = data
        inputs = Variable(mix_spec)
        targets = target_spec
        
        optimizer.zero_grad()
        optimizer_A.zero_grad()

        
        # get feature
        featureset = featureDataSet(clean_dir, int(target_label))
        feat_data, index = featureset.__getitem__()  
        feat, _ = featurenet(feat_data) 

        # feed in feature to ANet
        att = A_model(feat)
        
        # Res_model
        top = Res_model.upward(inputs) #+ white(inputs))
        outputs = Res_model.downward(top, shortcut = True)
        outputs = outputs.view(bs, 1, 256, 128)
        
        
        target = targets.view(bs, 1, 256, 128)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()
        optimizer_A.step()

        
        loss_record.append(loss.item())
        print ('[%d, %2d] loss: %.3f' % (epo, i, loss.item()))
    
    
    

        if i % 100 == 0:

            inn = inputs.view(256, 128).detach().numpy() * 255
            np.clip(inn, np.min(inn), 1)
            cv2.imwrite(root_dir + 'cocktail/combinemodel_fullconv/L1/' + '_' + str(epo) + str(i)  + "_mix.png", inn)

            tarr = target.view(256, 128).detach().numpy() * 255
            np.clip(tarr, np.min(tarr), 1)
            cv2.imwrite(root_dir + 'cocktail/combinemodel_fullconv/L1/' + '_' + str(epo) + str(i)  + "_tar.png", tarr)

            outt = outputs.view(256, 128).detach().numpy() * 255
            np.clip(outt, np.min(outt), 1)
            cv2.imwrite(root_dir + 'cocktail/combinemodel_fullconv/L1/' + '_' + str(epo) + str(i)  + "_sep.png", outt)

        if i % 200 == 0:
            plt.figure(figsize = (20, 10))
            plt.plot(loss_record)
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.savefig(root_dir + 'cocktail/combinemodel_fullconv/L2')
 
   
    gc.collect()
    plt.close("all")



    
#=============================================
#        Save Model & Loss
#=============================================

torch.save(Res_model.state_dict(), root_dir + 'cocktail/combinemodel_fullconv/L1/res.pkl')
torch.save(A_model.state_dict(), root_dir + 'cocktail/combinemodel_fullconv/L1/A.pkl')
torch.save(featurenet.state_dict(), root_dir + 'cocktail/combinemodel_fullconv/L1/feat.pkl')




