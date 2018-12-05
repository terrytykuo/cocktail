# test.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import cv2

import model.fb_model as model
from model.fb_model import ResDAE, ANet

import json
import pickle
import logger
import dataset_utils
from meta import *

import os

import pdb


''' utils '''
def one_hot(zo):
    ans = np.zeros(2, dtype=float)
    ans[zo] = 1
    return ans

def atten(zo):
    try:
        if(zo[0]==1):
            return 0
        elif(zo[1]==1):
            return 1
    except:
        print("atten error")

def AMI(recover_a, recover_b, clean_a, clean_b):
    def cov(u, v):
        u = u.reshape(128*128)
        v = v.reshape(128*128)
        return u.dot(v)

    ami = cov(recover_a, clean_a) + cov(recover_b, clean_b) - cov(recover_b, clean_a) - cov(recover_a, clean_b)
    return ami


''' main '''
if __name__=="__main__":

    import sys
    LOWEST = sys.argv[1]

    clear_dir(logroot, kw="events")
    clear_dir(result, kw=".png")

    ''' dataset & indexing: ( mixed , atten , pure ) '''
    # block_names = dataset_utils.get_block_names("test")
    # block_names.sort(key=lambda b:int(b[5:-5]))
    mixed_names = dataset_utils.get_mix_names("test")
    mixed_names.sort(key=lambda m:int(m[6:-5]))
    test_num = len(mixed_names)

    ''' training prep '''
    dae = pickle.load(open("dae_{}.pickle".format(LOWEST), "rb"))
    anet = pickle.load(open("anet_{}.pickle".format(LOWEST), "rb"))
    if(LOWEST=='2'): dae.lowest = 2

    print("dae.lowest = {}".format(dae.lowest))

    lossF = nn.MSELoss()



    ''' testing '''

    # values for evaluation
    loss_a_list = []
    loss_b_list = []
    ami_list = []


    for i in range(test_num):
        print("block/mixed #i={}".format(i))

        # mixed
        mixed_specs, a_clean, b_clean = dataset_utils.load_mix('test', mixed_names[i])
        mixed_specs_tensor = torch.tensor(mixed_specs, dtype=torch.float)
        bs = mixed_specs.shape[0]

        mixed_attens_a = np.zeros((bs, 2)); mixed_attens_a[:,0] = 1; mixed_attens_a = torch.tensor(mixed_attens_a, dtype=torch.float)
        mixed_attens_b = np.zeros((bs, 2)); mixed_attens_b[:,1] = 1; mixed_attens_b = torch.tensor(mixed_attens_b, dtype=torch.float)

        _a7, _a6, _a5, _a4, _a3, _a2 = anet(mixed_attens_a)
        _top = dae.upward(mixed_specs_tensor,  _a7, _a6, _a5, _a4, _a3, _a2)
        recover_a = dae.downward(_top).view(bs, 128, 128).detach().numpy()

         _a7, _a6, _a5, _a4, _a3, _a2 = anet(mixed_attens_b)
        _top = dae.upward(mixed_specs_tensor,  _a7, _a6, _a5, _a4, _a3, _a2)
        recover_b = dae.downward(_top).view(bs, 128, 128).detach().numpy()

        _top = dae.upward(torch.tensor(a_clean, dtype=torch.float))
        single_a = dae.downward(_top).view(bs, 128, 128).detach().numpy()

        _top = dae.upward(torch.tensor(b_clean, dtype=torch.float))
        single_b = dae.downward(_top).view(bs, 128, 128).detach().numpy()



        # [0..bs-1]
        for j in range(bs):
            dirname = "testid={}_{}/".format(i,j)
            try:
                os.mkdir(result+dirname)
            except:
                remove_dir(result+dirname)
                os.mkdir(result+dirname)
            

            # values for this batch
            ami = AMI(recover_a[j], recover_b[j], a_clean[j], b_clean[j])
            lossa = float( lossF(torch.tensor(recover_a[j], dtype=torch.float), torch.tensor(a_clean[j], dtype=torch.float)).detach().numpy() )
            lossb = float( lossF(torch.tensor(recover_b[j], dtype=torch.float), torch.tensor(b_clean[j], dtype=torch.float)).detach().numpy() )
            

            # print
            print("\tAMI = {}".format(ami))
            print("\tloss = a:{}\tb:{}".format(lossa, lossb))


            # store
            ami_list.append(ami)
            loss_a_list.append(lossa)
            loss_b_list.append(lossb)


            # # demo pics
            cv2.imwrite(result+dirname+"testid={}_{}_mixed.png".format(i, j), mixed_specs[j].reshape(128, 128)*255)
            cv2.imwrite(result+dirname+"testid={}_{}_recover_atten=0.png".format(i, j), recover_a[j]*255)
            cv2.imwrite(result+dirname+"testid={}_{}_recover_atten=1.png".format(i, j), recover_b[j]*255)
            cv2.imwrite(result+dirname+"testid={}_{}_clean_0.png".format(i, j), a_clean[j].reshape(128, 128)*255)
            cv2.imwrite(result+dirname+"testid={}_{}_clean_1.png".format(i, j), b_clean[j].reshape(128, 128)*255)
            cv2.imwrite(result+dirname+"testid={}_{}_single_0.png".format(i, j), single_a[j].reshape(128, 128)*255)
            cv2.imwrite(result+dirname+"testid={}_{}_single_1.png".format(i, j), single_b[j].reshape(128, 128)*255)


    loss_a_fhand = open("losses_a_lowest_{}.json".format(LOWEST), "w")
    loss_b_fhand = open("losses_b_lowest_{}.json".format(LOWEST), "w")
    ami_fhand = open("AMIs_lowest_{}.json".format(LOWEST), "w")
    loss_a_fhand.write(json.dumps(loss_a_list))
    loss_b_fhand.write(json.dumps(loss_b_list))
    ami_fhand.write(json.dumps(ami_list))

# end