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

    big_pieces = []
    for i in range(10):
        big_pieces.append(data_process.s_matrix(i, point_sec_sliced_path, multiplication))
        print ('column',i, 'done')
        
    ## concatenate each column into a big_matrix 
    big_matrix = np.vstack((big_pieces))
    
    ## generate the mixed column, and concatenate it with original big_matrix
    mixed_column = []
    index_record = []
    for i in range(len(big_matrix[1])):
        pc1, pc2 = np.random.choice([0,1,2,3,4,5,6,7,8,9], 2)
        mixed_spec = (big_matrix[pc1][i] * big_matrix[pc2][i])/ (big_matrix[pc1][i] + big_matrix[pc2][i])
        mixed_column.append(mixed_spec)
        index_record.append([pc1, pc2])

    mixed_column = np.stack([mixed_column])

    print (big_matrix.shape)
    print ('mixed_column shape =', mixed_column.shape)
    
    big_matrix = np.vstack([big_matrix, mixed_column])
    index_record = np.vstack([index_record])

    print ("big_matrix shape =", big_matrix.shape)
    print ('index_record shape =', index_record.shape)

    ## x_train = big_matrix --> json
    ## y_train = index_record --> json
    with open(block_path + "block" + str(p) + '.json', 'w') as jh:
        json.dump(big_matrix.tolist(), jh)
    
    with open(block_path + "index" + str(p) + '.json', 'w') as f:
        json.dump(index_record.tolist(), f)