import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.init as init

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

root_dir = '/home/tk/cocktail/'


train_dir = root_dir + 'cleanblock/'
test_dir  = root_dir + 'clean_test/'

# 22 in train dir, 4 in test dir
# 22 = 3+19, 4 = 1+3

def list_json_in_dir(dir):
    temp = os.listdir(dir)
    temp.sort()
    i = 0
    for t in temp:
        if '.json' in t:
            i += 1
        else:
            del temp[i]
    return temp

all_json_in_train_dir = list_json_in_dir(train_dir)
spec_train_blocks = all_json_in_train_dir[:21]
feat_train_block = all_json_in_train_dir[21:]

all_json_in_test_dir = list_json_in_dir(test_dir)
spec_test_blocks = all_json_in_test_dir[:3]
feat_test_block = all_json_in_test_dir[3:]

#=============================================
#       Define Datasets
#=============================================

CLASSES = 10
RANDOM_SAMPLES_PER_ENTRY = 20
ALL_SAMPLES_PER_ENTRY = CLASSES * (CLASSES - 1) // 2
ENTRIES_PER_JSON = 100
SPEC_TRAIN_JSONS = len(spec_train_blocks)
SPEC_TEST_JSONS = len(spec_test_blocks)

#=============================================
#        Hyperparameters
#=============================================

epoch = 10
lr = 0.005
mom = 0.9
BS = 10
BS_TEST = ALL_SAMPLES_PER_ENTRY

def gen_all_pairs():
    all_pairs = []
    for i in range(CLASSES):
        for j in range(CLASSES):
            if(i==j): continue
            all_pairs.append([i, j])
    return np.array(all_pairs)

all_combinations = gen_all_pairs()
all_combination_indices = np.arange(CLASSES * (CLASSES-1) // 2)

def gen_rand_pairs(num_pairs):
    ''' 至多C(10,2)对组合 '''
    assert(2 * num_pairs <= CLASSES * (CLASSES - 1))
    ''' 长为 num_pairs 的 list ，为 [0,CLASSES-1]x[0,CLASSES-1] 中的序偶 '''
    chosen = all_combinations[ 
        np.array( np.random.choice(all_combination_indices, num_pairs, replace=False) ) 
    ]
    return chosen

def gen_f_a_b(spec_block, entry_index, feat_block, random_mode=True):
    if random_mode: 
        samples_selected = RANDOM_SAMPLES_PER_ENTRY
    else:
        samples_selected = ALL_SAMPLES_PER_ENTRY
    a_b_indexes = gen_rand_pairs(samples_selected).transpose()
    a_index_list, b_index_list = a_b_indexes[0], a_b_indexes[1]

    a_b = np.array([
        spec_block[entry_index, a_index_list], 
        spec_block[entry_index, b_index_list]
    ])
    feats = feat_block[
                np.random.randint(feat_block.shape[0]),
                a_index_list
            ].reshape(1, samples_selected, 256, 128)
    return np.concatenate((feats, a_b), axis=0)

    
class BlockBasedDataSet(Dataset):
    def __init__(self, block_dir, feat_block_list, spec_block_list, gen_fab_random_mode):
        self.feat_block = []
        for block in feat_block_list:
            self.feat_block.append( json.load(open(block_dir + block, "r")) )
        self.feat_block = np.concatenate( np.array(self.feat_block), axis=1 ).transpose(1,0,2,3)

        self.curr_json_index = 0
        self.curr_entry_index = 0

        self.spec_block = np.array(json.load(open(block_dir + spec_block_list[0], "r"))).transpose(1,0,2,3)
        self.f_a_b = gen_f_a_b(self.spec_block, self.curr_entry_index, self.feat_block, random_mode=gen_fab_random_mode)

        self.curr_fab_index = 0

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return None

class trainDataSet(BlockBasedDataSet):

    # 不变性：
    # 总保有一份 spec_block ，一份 feat_block
    # 每次访问时，有长为bs的f-a-b列表，每次取下标从列表中取得
    # f ：随机一个下标，取目标编号的spectrogram

    def __init__(self):
        print("trainDataSet: feature blocks: ", len(feat_train_block))
        super(trainDataSet, self).__init__(train_dir, feat_train_block, spec_train_blocks, gen_fab_random_mode=True)

    def __len__(self):
        return ENTRIES_PER_JSON * RANDOM_SAMPLES_PER_ENTRY * SPEC_TRAIN_JSONS // BS

    def __getitem__(self, dummy_index): # index is dummy, cuz doing ordered traverse
        '''
        数据规格协议：
        - block块，标号为self.curr_json_index；需支持 get_next_entry ，内部方法： get_next_block
            - entry流，标号为self.entry_index；需支持 gen_fab
        - fab块：通过 self.curr_fab_index 取下标；需支持 get_next_batch ，内部方法： get_next_entry 与拼接
            - batch流：顺序遍历无标号
        '''
        # to next batch


        fab = None
        if self.curr_fab_index + BS <= self.f_a_b.shape[1]:
            fab = self.f_a_b[:, self.curr_fab_index : self.curr_fab_index + BS]
            self.curr_fab_index += BS
        else: # load next entry
            self.curr_entry_index += 1

            if self.curr_entry_index == ENTRIES_PER_JSON: # load next block
                self.curr_json_index += 1
                self.spec_block = np.array(
                        json.load(open(train_dir + spec_train_blocks[self.curr_json_index], "r"))
                    ).transpose(1,0,2,3)
                self.curr_entry_index = 0

            if self.curr_fab_index < self.f_a_b.shape[1]:
                # print("shape1", self.f_a_b[:, self.curr_fab_index:self.f_a_b.shape[0]].shape)
                # print("shape2", gen_f_a_b(self.spec_block, self.curr_entry_index, self.feat_block).shape)
                self.f_a_b = np.concatenate(
                    (   self.f_a_b[:, self.curr_fab_index:self.f_a_b.shape[1]], 
                        gen_f_a_b(self.spec_block, self.curr_entry_index, self.feat_block)  ),
                    axis=1
                )
            else:
                self.f_a_b = gen_f_a_b(self.spec_block, self.curr_entry_index, self.feat_block)

            self.curr_fab_index = 0
            fab = self.f_a_b[:, self.curr_fab_index : self.curr_fab_index + BS]

            self.curr_fab_index += BS

        # print("dummy_index = {} | block_index = {}, entry_index = {}, fab_index = {}-{}".format(
        #        dummy_index, self.curr_json_index, self.curr_entry_index, self.curr_fab_index - BS, self.curr_fab_index))

        assert(fab.shape == (3, BS, 256, 128))
        return (torch.Tensor(fab[0]).view(BS, 256, 128),
                torch.Tensor(fab[1]).view(BS, 256, 128),
                torch.Tensor(fab[2]).view(BS, 256, 128))

class testDataSet(BlockBasedDataSet):
    # 不用考虑batch了，直接一个一个读取
    # 从block中，取出entry
    # 从entry中，取出一系列f-a-b
    def __init__(self):
        print("testDataSet: feature blocks: ", len(feat_test_block))
        super(testDataSet, self).__init__(test_dir, feat_test_block, spec_test_blocks, gen_fab_random_mode=False)

    def __len__(self):
        return ENTRIES_PER_JSON * SPEC_TEST_JSONS * ALL_SAMPLES_PER_ENTRY

    def __getitem__(self, index):

        # block号
        newest_json_index = index // (ENTRIES_PER_JSON * ALL_SAMPLES_PER_ENTRY)
        entry_offset = index % (ENTRIES_PER_JSON * ALL_SAMPLES_PER_ENTRY)
        newest_entry_index = entry_offset // ALL_SAMPLES_PER_ENTRY
        newest_fab_index = entry_offset % ALL_SAMPLES_PER_ENTRY

        print("json index = {}, newest_entry_index = {}, fab_offset = {}".format(
            newest_json_index, newest_entry_index, newest_fab_index)
        )

        if not (self.curr_json_index == newest_json_index):
            self.curr_json_index = newest_json_index
            f = open(clean_dir + '{}'.format(cleanfolder[newest_json_index]))
            self.spec_block = np.array(json.load(f)).transpose(1,0,2,3)

        if not (self.curr_entry_index == newest_entry_index) or not (self.curr_json_index == newest_json_index):
            self.curr_entry_index = newest_entry_index
            self.f_a_b = gen_f_a_b(self.spec_block, self.curr_entry_index, self.feat_block, random_mode=False)

        print(self.f_a_b.shape)

        return torch.Tensor(self.f_a_b[0, newest_fab_index]), \
               torch.Tensor(self.f_a_b[1, newest_fab_index]), \
               torch.Tensor(self.f_a_b[2, newest_fab_index])


#=============================================
#        Define Dataloader
#=============================================

mixset = trainDataSet()
mixloader = torch.utils.data.DataLoader(dataset = mixset,
    batch_size = 1,
    shuffle = False) # batch size is controlled by BS in hard-code, here batch_size is set to 1

testset = testDataSet()
testloader = torch.utils.data.DataLoader(dataset = testset,
    batch_size = 1,
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
        x = x.view(-1, 1 ,256, 128)
        x = F.relu(self.maxpool(self.conv1(x)))
        x = F.relu(self.maxpool(self.conv2(x)))
        x = self.batchnorm(x)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        x = self.fc3(feat)
        
        return feat #, F.log_softmax(x, dim = 1)


featurenet = featureNet()
try:
    featurenet.load_state_dict(torch.load(root_dir + 'cocktail/combinemodel_fullconv/feat.pkl'))
except:
    print("F-model not available")



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
        x = x.view(-1, 1, 256)

        a7 = self.linear7(x).view(-1, 512, 1, 1)
        a6 = self.linear6(x).view(-1, 256, 1, 1)
        a5 = self.linear5(x).view(-1, 128, 1, 1)
        a4 = self.linear4(x).view(-1, 64, 1, 1)
        a3 = self.linear3(x).view(-1, 32, 1, 1)
        a2 = self.linear2(x).view(-1, 16, 1, 1)

        return a7, a6, a5, a4, a3, a2

A_model = ANet()
try:
    A_model.load_state_dict(torch.load(root_dir + 'cocktail/combinemodel_fullconv/A.pkl'))
except:
    print("A-model not available")
# print(A_model)



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
            # nn.ConvTranspose2d(256, 256, kernel_size = (2,2), stride = 2),
            nn.BatchNorm2d(256),
        )

        # 2x2x256

        self.downward_net6 = nn.Sequential(
            # 8x8x64
            ResBlock(256, 256),
            ResBlock(256, 128),
            ResBlock(128, 128),
            ResTranspose(128, 128),
            # nn.ConvTranspose2d(128, 128, kernel_size = (2,2), stride = 2),
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
            # nn.ConvTranspose2d(64, 64, kernel_size = (2,2), stride = 2),
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
            # nn.ConvTranspose2d(32, 32, kernel_size = (2,2), stride = 2),
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
            # nn.ConvTranspose2d(16, 16, kernel_size = (2,2), stride = 2),
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
            # nn.ConvTranspose2d(8, 8, kernel_size = (2,2), stride = 2),
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
        

    def upward(self, x, a7=None, a6=None, a5=None, a4=None, a3=None, a2=None):
        x = x.view(-1, 1, 256, 128)
        # 1x128x128
        # print ("initial", x.shape)
        x = self.upward_net1(x)
        # print ("after conv1", x.shape)


        # 8x64x64
        x = self.upward_net2(x)
        if a2 is not None: x = x * a2
        self.x2 = x
        # print ("after conv2", x.shape)

        # 16x32x32
        x = self.upward_net3(x)
        if a3 is not None: x = x * a3
        self.x3 = x
        # print ("after conv3", x.shape)

        # 32x16x16

        x = self.upward_net4(x)
        if a4 is not None: x = x * a4
        self.x4 = x
        # print ("after conv4", x.shape)

        # 64x8x8

        x = self.upward_net5(x)
        if a5 is not None: x = x * a5
        self.x5 = x
        # print ("after conv5", x.shape)

        
        # 128x4x4
        x = self.upward_net6(x)
        if a6 is not None: x = x * a6
        # print ("after conv6", x.shape)

        # 256x2x2

        x = self.upward_net7(x)
        if a7 is not None: x = x * a7
        # print ("after conv7", x.shape)

        # 512x1x1

        return x


    def downward(self, y, shortcut= True):
        # print ("begin to downward, y.shape = ", y.shape)
        # 512x1x1
        y = self.downward_net7(y)
        # print ("after down7", y.shape)


        # 256x2x2
        y = self.downward_net6(y)
        # print ("after down6", y.shape)

        # 128x4x4
        if shortcut:
            y = torch.cat((y, self.x5), 1)
            y = F.relu(self.uconv5(y))
        y = self.downward_net5(y)
        # print ("after down5", y.shape)

        # 64x8x8
        if shortcut:
            y = torch.cat((y, self.x4), 1)
            y = F.relu(self.uconv4(y))
        y = self.downward_net4(y)
        # print ("after down4", y.shape)

        # 32x16x16
        if shortcut:
            y = torch.cat((y, self.x3), 1)
            y = F.relu(self.uconv3(y))
        y = self.downward_net3(y)
        # print ("after down3", y.shape)

        # 16x32x32
        if shortcut:
            y = torch.cat((y, self.x2), 1)
            y = F.relu(self.uconv2(y))
        y = self.downward_net2(y)
        # print ("after down2", y.shape)

        # 8x64x64
        y = self.downward_net1(y)
        # print ("after down1", y.shape)
 
        # 1x128x128

        return y


Res_model = ResDAE()
try:
    Res_model.load_state_dict(torch.load(root_dir + 'cocktail/combinemodel_fullconv/res.pkl'))
except:
    print("Res-model not available")
# print(Res_model)



#=============================================
#        Optimizer
#=============================================

criterion = nn.MSELoss()
feat_optimizer = torch.optim.SGD(featurenet.parameters(), lr = lr, momentum=mom)
anet_optimizer = torch.optim.SGD(A_model.parameters(), lr = lr, momentum=mom)
res_optimizer = torch.optim.SGD(Res_model.parameters(), lr = lr, momentum=mom)



#=============================================
#        Loss Record
#=============================================

loss_record = []
test_record = []
epoch_train = []
epoch_test = []

#=============================================
#        Train
#=============================================

Res_model.train()
for epo in range(epoch):
    # train
    for i, data in enumerate(mixloader, 0):

        # get mix spec & label
        feat_data, a_specs, b_specs = data

        feat_data = feat_data.squeeze()
        a_specs = a_specs.squeeze()
        b_specs = b_specs.squeeze()

        mix_specs = a_specs + b_specs
        target_specs = a_specs

        feat_optimizer.zero_grad()
        anet_optimizer.zero_grad()
        res_optimizer.zero_grad()

        # get feature
        print(feat_data.shape)
        feats = featurenet(feat_data)

        # feed in feature to ANet
        a7, a6, a5, a4, a3, a2 = A_model(feats)

        # Res_model
        tops = Res_model.upward(mix_specs, a7, a6, a5, a4, a3, a2) #+ white(inputs))
        outputs = Res_model.downward(tops, shortcut = True)

        loss_train = criterion(outputs, target_specs)

        loss_train.backward()
        res_optimizer.step()
        anet_optimizer.step()
        feat_optimizer.step()

        loss_record.append(loss_train.item())
        print ("training batch #{}".format(i))

        if i % 20 == 0:

            print ('[%d, %2d] loss_train: %.3f' % (epo, i, loss_train.item()))

            inn = mix_specs[0].view(256, 128).detach().numpy() * 255
            np.clip(inn, np.min(inn), 1)
            cv2.imwrite(root_dir + 'cocktail/combinemodel_fullconv/' + str(epo)  + "_mix.png", inn)

            tarr = target_specs[0].view(256, 128).detach().numpy() * 255
            np.clip(tarr, np.min(tarr), 1)
            cv2.imwrite(root_dir + 'cocktail/combinemodel_fullconv/' + str(epo)  + "_tar.png", tarr)

            outt = outputs[0].view(256, 128).detach().numpy() * 255
            np.clip(outt, np.min(outt), 1)
            cv2.imwrite(root_dir + 'cocktail/combinemodel_fullconv/' + str(epo)  + "_sep.png", outt)

    # test
    Res_model.eval()
    for i, data in enumerate(testloader, 0):
        feat_data, a_specs, b_specs = data

        feat_data = feat_data.squeeze()
        a_specs = a_specs.squeeze()
        b_specs = b_specs.squeeze()

        mix_specs = a_specs + b_specs
        target_specs = a_specs

        feat = featurenet(feat_data)

        a7, a6, a5, a4, a3, a2 = A_model(feat)

        top = Res_model.upward(mix_spec, a7, a6, a5, a4, a3, a2) #+ white(inputs))
        output = Res_model.downward(top, shortcut = True)

        loss_test = criterion(output, target_spec)

        test_record.append(loss_test.item())

    plt.figure(figsize = (20, 10))
    plt.plot(loss_record)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.savefig(root_dir + 'cocktail/training.png')
    gc.collect()
    plt.close("all")

    plt.figure(figsize = (20, 10))
    plt.plot(test_record)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.savefig(root_dir + 'cocktail/testing.png')
    gc.collect()
    plt.close("all")

    train_average_loss = np.average(loss_record)
    test_average_loss = np.average(test_record)

    epoch_train.append(train_average_loss)
    epoch_test.append(test_average_loss)

    print ("train finish epoch #{}, loss average {}".format(epo, train_average_loss))
    print ("test finish epoch #{}, loss average {}".format(epo, test_average_loss))

    loss_record = []
    test_record = []


#=============================================
#        Save Model & Loss
#=============================================

torch.save(Res_model.state_dict(), root_dir + 'cocktail/combinemodel_fullconv/res.pkl')
torch.save(A_model.state_dict(), root_dir + 'cocktail/combinemodel_fullconv/A.pkl')
torch.save(featurenet.state_dict(), root_dir + 'cocktail/combinemodel_fullconv/feat.pkl')

