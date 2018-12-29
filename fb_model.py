'''imports'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np


def odd(w):
    return list(np.arange(1, w, step=2, dtype='long'))

def even(w):
    return list(np.arange(0, w, step=2, dtype='long'))


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
            return x + x1
        elif self.channels_out < self.channels_in:
            x = F.relu(self.conv1(x))
            x1 =       self.conv2(x)
            x = x + x1
            return x
        else:
            x1 = F.relu(self.conv1(x))
            x1 =        self.conv2(x1)
            x = x + x1
            return x

    def sizematch(self, channels_in, channels_out, x):
        zeros = torch.zeros( (x.size()[0], channels_out - channels_in, x.shape[2], x.shape[3]), dtype=torch.float )
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
        x1 = F.relu(self.deconv1(x))
        x1 =        self.deconv2(x1)
        x = self.sizematch(x)
        return x + x1

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
            ResBlock(1, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            nn.BatchNorm2d(8),
        )

        # 64x64x8

        self.upward_net2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(8, 8),
            ResBlock(8, 16),
            ResBlock(16, 16),
            nn.BatchNorm2d(16),
        )

        # 32x32x16

        self.upward_net3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=2),
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
            nn.BatchNorm2d(256),
        )

        # 2x2x256

        self.downward_net6 = nn.Sequential(
            # 8x8x64
            ResBlock(256, 256),
            ResBlock(256, 128),
            ResBlock(128, 128),
            ResTranspose(128, 128),
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
            nn.BatchNorm2d(8),
        )

        # 64x64x8
        self.downward_net1 = nn.Sequential(
            ResBlock(8, 8),
            ResBlock(8, 4),
            ResBlock(4, 1),
            ResBlock(1, 1),
            ResTranspose(1, 1),
            nn.BatchNorm2d(1),
        )

        # 128x128x1


    def upward(self, x, a7=None, a6=None, a5=None, a4=None, a3=None, a2=None):
        bs = x.shape[0]
        x = x.view(bs, 1, 128, 128)
        # 1x128x128

        x = self.upward_net1(x)
        # 8x64x64

        x = self.upward_net2(x)
        if a2 is not None: x = x * a2
        self.x2 = x
        # 16x32x32

        x = self.upward_net3(x)
        if a3 is not None: x = x * a3
        self.x3 = x
        # 32x16x16

        x = self.upward_net4(x)
        if a4 is not None: x = x * a4
        self.x4 = x
        # 64x8x8

        x = self.upward_net5(x)
        if a5 is not None: x = x * a5
        self.x5 = x
        # 128x4x4

        x = self.upward_net6(x)
        if a6 is not None: x = x * a6
        # 256x2x2

        x = self.upward_net7(x)
        if a7 is not None: x = x * a7
        # 512x1x1

        return x


    def downward(self, y, shortcut=True):

        # 512x1x1
        y = self.downward_net7(y)

        # 256x2x2
        y = self.downward_net6(y)

        # 128x4x4
        if shortcut:
            y = torch.cat((y, self.x5), 1)
            y = F.relu(self.uconv5(y))
        y = self.downward_net5(y)

        # 64x8x8
        if shortcut:
            y = torch.cat((y, self.x4), 1)
            y = F.relu(self.uconv4(y))
        y = self.downward_net4(y)
        
        # 32x16x16
        if shortcut:
            y = torch.cat((y, self.x3), 1)
            y = F.relu(self.uconv3(y))
        y = self.downward_net3(y)

        # 16x32x32
        if shortcut:
            y = torch.cat((y, self.x2), 1)
            y = F.relu(self.uconv2(y))
        y = self.downward_net2(y)

        # 8x64x64
        y = self.downward_net1(y)
        
        # 1x128x128

        return y



class ANet(nn.Module):
    def __init__(self):
        super(ANet, self).__init__()

        self.linear7 = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
        )
        self.linear6 = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
        )
        self.linear5 = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
        )
        self.linear4 = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
        )

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, 2)

        a7 = self.linear7(x).view(bs, 512, 1, 1)
        a6 = self.linear6(x).view(bs, 256, 1, 1)
        a5 = self.linear5(x).view(bs, 128, 1, 1)
        a4 = self.linear4(x).view(bs, 64, 1, 1)
        a3 = self.linear3(x).view(bs, 32, 1, 1)
        a2 = self.linear2(x).view(bs, 16, 1, 1)

        return a7, a6, a5, a4, a3, a2