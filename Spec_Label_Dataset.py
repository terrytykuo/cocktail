import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import random
random.seed(7)

import os


ENTRIES_PER_JSON = 100
CLASSES = 10
SAMPLES_PER_JSON = 1000


class Spec_Label_Dataset(Dataset):
    def __init__(self, block_dir):
        super(Spec_Label_Dataset, self).__init__()
        
        self.curr_json_index = -1
        self.block_dir = block_dir
        self.cleanfolder = os.listdir(clean_dir)
        self.cleanfolder.sort()
        self.spec = None
        self.labels = None

    def __len__(self):
        return SAMPLES_PER_JSON * len()

    def __getitem__(self, index):

        newest_json_index = index // SAMPLES_PER_JSON
        offset_in_json = index % SAMPLES_PER_JSON

        if not (self.curr_json_index == newest_json_index):
            self.curr_json_index = newest_json_index

            f = open(self.block_dir + '{}'.format(self.cleanfolder[newest_json_index]))
            self.spec = np.array(json.load(f)).transpose(1,0,2,3)
            self.spec = np.concatenate(self.spec, axis=0)

            self.labels = np.array([np.arange(CLASSES) for _ in range(ENTRIES_PER_JSON)])
            self.labels = np.concatenate(self.labels, axis=0)

            indexes = np.arange(ENTRIES_PER_JSON * CLASSES)
            random.shuffle(indexes)

            # indexes: randomly arranged 0:999
            # self.labels: 0-9, 0-9, ..., 0-9

            self.spec = torch.Tensor(self.spec[indexes]).squeeze()
            self.labels = torch.Tensor(self.labels[indexes]).squeeze()

            del indexes

        spec = self.spec[offset_in_json]
        label = self.labels[offset_in_json]
        return spec, label