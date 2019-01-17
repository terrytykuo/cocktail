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

epoch = 10
lr = 0.001
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


clean_dir = '/home/tk/Documents/mix_pool/feature/' 
mix_dir = '/home/tk/Documents/mix_pool/mix_spec/' 
# clean_label_dir = '/home/tk/Documents/clean_labels/' 
mix_label_dir = '/home/tk/Documents/mix_pool/mix_labels/' 
target_spec_dir = '/home/tk/Documents/mix_pool/target_spec/'
target_label_dir = '/home/tk/Documents/mix_pool/target_label/'


# cleanfolder = os.listdir(clean_dir)
# cleanfolder.sort()

# mixfolder = os.listdir(mix_dir)
# mixfolder.sort()

#=============================================
#       Define Datasets
#=============================================

class mixDataSet(Dataset):
    
    def __init__(self, mix_dir, target_spec_dir, target_label_dir):           
        
        mix_list = []
        target_spec_list = []
        target_label_list = []

        with open(mix_dir + 'mix_spec2.json') as f:
            mix_list.append(torch.Tensor(json.load(f)))
        
        with open(target_spec_dir + 'target_spec2.json') as f:
            target_spec_list.append(torch.Tensor(json.load(f)))
            
        with open(target_label_dir + 'target_label2.json') as f:
            target_label_list.append(torch.Tensor(json.load(f)))
        
        

        mixblock = torch.cat(mix_list, 0)
        targetspec = torch.cat(target_spec_list, 0)
        targetlabel = torch.cat(target_label_list, 0)
        
        self.mix_spec = mixblock
        self.target_spec = targetspec
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
        
        full_audio = ['birdstudybook', 'captaincook', 'cloudstudies_02_clayden_12', 
              'constructivebeekeeping',
              'discoursesbiologicalgeological_16_huxley_12', 
              'natureguide', 'pioneersoftheoldsouth', 
              'pioneerworkalps_02_harper_12', 
              'romancecommonplace', 'travelstoriesretold']

        feature_list = []

        with open(clean_dir + full_audio[label] + '/0.json') as f:
            feature_list.append(torch.Tensor(json.load(f)))      
        
        featureblock = torch.cat(feature_list, 0)
        
        self.featurespec = featureblock
                
        
    def __len__(self):
        return self.featurespec.shape[0]

                
    def __getitem__(self): 
        
        featurespec = self.featurespec
        return featurespec
    
#=============================================
#        Dataloader
#=============================================


mixset = mixDataSet(mix_dir, target_spec_dir, target_label_dir)
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
        self.fc1 = nn.Linear(256*128, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

        
    def forward(self, x):
        x = x.view(-1, 256*128)
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(feat))
        
        return feat, x

featurenet = featureNet()
# model.load_state_dict(torch.load('/home/tk/Documents/FeatureNet.pkl'))
    
'''ANet'''
class ANet(nn.Module):
    
    def __init__(self):
        super(ANet, self).__init__()

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

        a5 = self.linear5(x).view(bs, 128, 1, 1)
        a4 = self.linear4(x).view(bs, 64, 1, 1)
        a3 = self.linear3(x).view(bs, 32, 1, 1)
        a2 = self.linear2(x).view(bs, 16, 1, 1)

        return a5, a4, a3, a2

A_model = ANet()
# A_model = torch.load(root_dir + 'cocktail/anet_2.pickle')


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

        # 1x256x128
        self.upward_net1 = nn.Sequential(
            ResBlock(1, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            nn.BatchNorm2d(8),
        )

        # 8x256x128
        self.upward_net2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            nn.BatchNorm2d(16),
        )
        # 16x128x64
        self.upward_net3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            nn.BatchNorm2d(32),
        )
        # 32x64x32
        self.upward_net4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.BatchNorm2d(64),
        )
        # 64x32x16
        self.upward_net5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.BatchNorm2d(64),
        )
        # 64x16x8

        self.linear1 = nn.Linear(8192, 512)

        self.linear2 = nn.Linear(512, 8192)

        # 64x16x8
        self.uconv5 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1))
        self.downward_net5 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResTranspose(64, 64),
            nn.BatchNorm2d(64),
        )

        # 64x32x16
        self.uconv4 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1))
        self.downward_net4 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResTranspose(32, 32),
            nn.BatchNorm2d(32),
        )

        # 32x64x32
        self.uconv3 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.downward_net3 = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResTranspose(16, 16),
            nn.BatchNorm2d(16),
        )

        # 16x128x64
        self.uconv2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.downward_net2 = nn.Sequential(
            ResBlock(16, 16),
            ResBlock(16, 8),
            ResBlock(8, 8),
            ResTranspose(8, 8),
            nn.BatchNorm2d(8),
        )

        # 8x256x128
        self.uconv1 = nn.Conv2d(8, 8, kernel_size=(3,3), padding=(1,1))
        self.downward_net1 = nn.Sequential(
            ResBlock(8, 8),
            ResBlock(8, 4),
            ResBlock(4, 1),
            ResBlock(1, 1),
            ResTranspose(1, 1),
            nn.BatchNorm2d(1),
        )
        # 1x256x128

    def upward(self, x, a5=None, a4=None, a3=None, a2=None, a1=None):
        x = x.view(bs, 1, 256, 128)
        
        print (x.shape)
        x = self.upward_net1(x)
        self.x1 = x

        print (x.shape)
        x = self.upward_net2(x)         # 8x64x64
        if a1 is not None: x = x * a1   
        self.x2 = x
 
        print (x.shape)
        x = self.upward_net3(x)         # 16x32x32
        if a2 is not None: x = x * a2
        self.x3 = x

        print (x.shape)
        x = self.upward_net4(x)         # 32x16x16
        if a3 is not None: x = x * a3
        self.x4 = x

        print (x.shape)
        x = self.upward_net5(x)         # 64x8x8
        if a4 is not None: x = x * a4
        self.x5 = x

        print (x.shape)
        x = x.view(bs, 1, 8192)
        x = F.relu(self.linear1(x))
        if a5 is not None: x = x * a5

        self.top = x

        return x


    def downward(self, y, shortcut=True):


        y = F.relu(self.linear2(y))
        y = y.view(bs, 64, 16, 8)
        print (y.shape)
        if shortcut:
            y = torch.cat((y, self.x5), 1)
            y = F.relu(self.uconv5(y))
        y = self.downward_net5(y)
        
        if shortcut:
            y = torch.cat((y, self.x4), 1)
            y = F.relu(self.uconv4(y))
        y = self.downward_net4(y)
        
        if shortcut:
            y = torch.cat((y, self.x3), 1)
            y = F.relu(self.uconv3(y))
        y = self.downward_net3(y)
        
        if shortcut:
            y = torch.cat((y, self.x2), 1)
            y = F.relu(self.uconv2(y))
        y = self.downward_net2(y)
        
        # if shortcut:
        #     y = torch.cat((y, self.x1), 1)
        #     y = F.relu(self.uconv1(y))
        y = self.downward_net1(y)

        return y



Res_model = ResDAE()
# Res_model = torch.load(root_dir + 'cocktail/dae_2.pickle')
# print (model)

#=============================================
#        Optimizer
#=============================================

#import pytorch_ssim
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(Res_model.parameters(), lr = lr, momentum = 0.9)

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
        
        # get feature
        featureset = featureDataSet(clean_dir, int(target_label))
        feat_data = featureset.__getitem__()[2]  
        feat, _ = featurenet(feat_data) 

        # feed in feature to ANet
        att = A_model(feat)
        
        # Res_model
        top = Res_model.upward(inputs) #+ white(inputs))
        outputs = Res_model.downward(top, shortcut = True)
        outputs = outputs.view(bs, 1, 256, 128)
        
        
        target = target.view(bs, 1, 256, 128)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        loss_record.append(loss.item())
    
    plt.figure(figsize = (20, 10))
    plt.plot(loss_record)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.savefig(root_dir + 'recover/combine/DAE_loss.png')
    
    inn = inputs.view(256, 128).detach().numpy() * 255
    np.clip(inn, np.min(inn), 1)
    cv2.imwrite("/home/tk/Documents/recover/combine/" + str(epo)  + "_mix.png", inn)

    tarr = target.view(256, 128).detach().numpy() * 255
    np.clip(tarr, np.min(tarr), 1)
    cv2.imwrite("/home/tk/Documents/recover/combine/" + str(epo)  + "_tar.png", tarr)

    outt = outputs.view(256, 128).detach().numpy() * 255
    np.clip(outt, np.min(outt), 1)
    cv2.imwrite("/home/tk/Documents/recover/combine/" + str(epo)  + "_sep.png", outt)

    
    print ('[%d] loss: %.3f' % (epo, loss.item()))
#            print ('[%d, %5d] ssim: %.3f' % (epo, i, ssim_value))
   
    gc.collect()
    plt.close("all")



    
#=============================================
#        Save Model & Loss
#=============================================

# torch.save(model, root_dir + 'recover/combine/combine_SSIM.pkl')