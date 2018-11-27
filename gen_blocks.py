# Package
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import gc
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile
import numpy as np
import json
import data_process

# parameters:

## 10 sec slicing:

# full audio will be stored here
full_audio_path = '/home/tk/Documents/full_audio/' 

# 10 sec sliced will be stored here
sec10_sliced_path = '/home/tk/Documents/slice_10sec/' 

# 0.1 sec slices will be stored here
point_sec_sliced_path = '/home/tk/Documents/slice_pointsec/' 

# block will be stored here
block_path = '/home/tk/Documents/blocks/'  

# controls datapoint in single column
multiplication = 1

# blocks 
blocks_volume = 50

#======================================================================


audio_list = os.listdir(full_audio_path)
if ".DS_Store" in audio_list:
    audio_list.remove(".DS_Store")
print ("There are", len(audio_list), "fully concatenated files")


for p in range(blocks_volume):

    # remove current files 
    to_clear = os.listdir(point_sec_sliced_path)
    for c in to_clear:
        os.remove(point_sec_sliced_path + c)

    pieces = p # controls which 10 sec segments will be processed 0.1 sec slicing 
               # pieces = 0 means the 1st 10-sec file, 
               # pieces = 1 means the 2nd 10-sec file, etc.

    ## 0.1 sec slicing
    print ('slicing', p, 'segment now')
    file_name = []
    for i in audio_list:
        slice_name = i[:-4]
        file_name.append(slice_name) # generate 0.1 sec file list

    for file in file_name:
        for i in range(pieces * multiplication , (pieces + 1) * multiplication):
            name = file + "_{0}.wav".format(i)
            data_process.slice_it(name, sec10_sliced_path, point_sec_sliced_path, length = 100)

    ## checkpoint: spec_name should have 1,000 files
    spec_name = os.listdir(point_sec_sliced_path)
    if len(spec_name) == 1000 * multiplication: 
        print ("checked")
    spec_name.sort()
    
    
    
    ## Generate Spectrogram and concatenate 
    print ('generating spectrogram now')

    big_pcs = []
    all_index = []

    for i in range(100):
        small_pcs = []
        index_record = []

        for j in range(0 + i, 1000 + i, 100):
            spec = data_process.gen_spectrogram(point_sec_sliced_path + spec_name[i])
            small_pcs.append(spec)

        # generate mixed column
        mixed = []
        index_record = []
        pc1, pc2 = np.random.choice([0,1,2,3,4,5,6,7,8,9], 2)

        mixed_spec = (small_pcs[pc1] * small_pcs[pc2])/ (small_pcs[pc1] + small_pcs[pc2])

        # append mixed spec into small_pcs
        small_pcs.append(mixed_spec)

        # record index
        index_record.append([pc1, pc2])
        single_row = np.vstack([small_pcs])    

        big_pcs.append(single_row)
        all_index.append(index_record)
        
        if i % 10 == 0:
            print (i, "row done")

    big_matrix = np.vstack([big_pcs])
    print ("big_matrix shape = ", big_matrix.shape)
    index_matrix = np.vstack(all_index)
    print ("index_matrix shape = ", index_matrix.shape)

    print ("The", p, "th block done. Start writing .json file")
    ## x_train = big_matrix --> json
    ## y_train = index_record --> json
    with open(block_path + "block" + str(p) + '.json', 'w') as jh:
        json.dump(big_matrix.tolist(), jh)
    
    with open(block_path + "index" + str(p) + '.json', 'w') as f:
        json.dump(index_matrix.tolist(), f)