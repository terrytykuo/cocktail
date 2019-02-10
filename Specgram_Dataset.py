import torch
from torch.utils.data import Dataset, DataLoader



class MSourceDataSet(Dataset):
    
    def __init__(self, clean_dir):
        self.curr_json_index = 0
        self.spec_block = None
        
    def __len__(self):
        return self.spec.shape[0]
                
    def __getitem__(self, index): 
        spec = self.spec[index]
        return spec