# training.py
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
## import cv2


import model.fb_model as model
from model.fb_model import ResDAE, ANet

import pickle
## import logger
import dataset_utils
from meta import *

import os
import pdb


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


if __name__=="__main__":

    clear_dir(logroot, kw="events")
    clear_dir(training_result, kw="png")

    ''' clean task '''
    # iteration / observation meta params
    import sys
    ITER1 = 5200
    ITER2 = 5200
    BS = 5
    OBS = 1
    LOWEST = int(sys.argv[1])


    # directory & index control
    block_names = dataset_utils.get_block_names("train")
    block_iter = 0
    temp_specs, temp_attens = dataset_utils.load_trivial('train', block_names[0])
    block_pos = 0


    # training prep
    model = ResDAE(LOWEST)
    anet = ANet()
    lr = 0.005
    params = []
    params.extend(list(model.parameters()))
    params.extend(list(anet.parameters()))
    optimizer = torch.optim.SGD(params, lr = lr, momentum=0.9); optimizer.zero_grad()
    lossF = nn.MSELoss(size_average=True) # must be True


    # training
    # logger = logger.Logger(logroot)
    for t in range(ITER1):

        ''' indexing '''
        end = min(block_pos+BS, len(temp_specs))


        ''' load batch '''
        source = temp_specs[block_pos:end]
        source = torch.tensor(source, dtype=torch.float)

        target = temp_attens[block_pos:end]
        target = torch.tensor(target, dtype=torch.float)


        ''' forward pass'''
        # a5, a4, a3, a2, a1 = anet(target)
        top = model.upward(source + white(source))
        recover = model.downward(top).view(end-block_pos, 128, 128)


        ''' loss-bp '''
        loss = lossF(recover, source)
        loss.backward()


        ''' observe '''
        # logger.scalar_summary("loss", loss, t)
        print("t={} loss={}".format(t, loss) )

        # if t%OBS==0:
        #     if OBS < 16: OBS *= 2

        #     cv2.imwrite(training_result+"top{}.png".format(t), model.top.detach().numpy().reshape(end-block_pos, 512) * 255)

        #     spec = source[0].view(128, 128).detach().numpy()
        #     cv2.imwrite(training_result+"source_{}.png".format(t), spec * 255)

        #     y1 = recover[0].view(128, 128).detach().numpy()
        #     cv2.imwrite(training_result+"y1_{}.png".format(t), y1 * 255)


        ''' stepping '''
        optimizer.step()
        optimizer.zero_grad()


        ''' next batch '''
        block_pos += BS
        if block_pos >= len(temp_attens):
            block_iter += 1
            if block_iter >= len(block_names): block_iter = 0
            temp_specs, temp_attens = dataset_utils.load_trivial('train', block_names[block_iter])

            block_pos = 0

    ''' dump '''
    

    ''' mixed-task '''
    block_names = dataset_utils.get_mix_names("train")
    block_iter = 0
    temp_mix, temp_attens, temp_clean = dataset_utils.load_mix('train', block_names[0])
    block_pos = 0

    # training prep
    OBS = 1

    # training
    for t in range(ITER2):
        ''' [s:e] '''
        end = min(block_pos+BS, temp_attens.shape[0])
        

        ''' load batch '''
        source = torch.tensor(temp_mix[block_pos:end], dtype=torch.float)
        attens = torch.tensor(temp_attens[block_pos:end], dtype=torch.float)
        clean  = torch.tensor(temp_clean[block_pos:end], dtype=torch.float)


        ''' indexing '''
        # a5, a4, a3, a2, a1 = anet(attens)
        a7, a6, a5, a4, a3, a2 = anet(attens)
        top = model.upward(source, a7, a6, a5, a4, a3, a2)
        recover = model.downward(top, shortcut=True).view(end-block_pos, 128, 128)

        
        ''' loss-bp '''
        loss = lossF(recover, clean.view(end-block_pos, 128, 128))
        loss.backward()


        ''' observe '''
        # logger.scalar_summary("loss2", loss, t)
        print("t={}, loss={}".format(t, loss))


        ''' stepping '''
        optimizer.step()
        optimizer.zero_grad()


        ''' next batch '''
        if end==temp_attens.shape[0]:
            # next & initial pos
            block_iter += 1
            if block_iter >= len(block_names): block_iter = 0
            temp_mix, temp_attens, temp_clean = dataset_utils.load_mix('train', block_names[block_iter])
            block_pos = 0
        else:
            block_pos += BS


    ''' dump '''
    ph = open("anet_{}.pickle".format(LOWEST), "wb")
    pickle.dump(anet, ph)
    ph = open("dae_{}.pickle".format(LOWEST), "wb")
    pickle.dump(model, ph)

#end