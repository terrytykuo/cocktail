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
blocks_volume = 70

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
               # pieces = 0 means 0~10, 
               # pieces = 1 means 10~20, etc.

    ## 0.1 sec slicing
    print ('slicing', p, 'segment now')
    file_name = []
    for i in audio_list:
        slice_name = i[:-4]
        file_name.append(slice_name)

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
        print (i, 'column done')
        big_pieces.append(data_process.s_matrix(i, point_sec_sliced_path, multiplication))

    big_matrix = np.vstack((big_pieces))
    print (big_matrix.shape)

    ## big_matrix --> json
    jh = open(block_path + "block" + str(p) + '.json', 'w')
    json.dump(big_matrix.tolist(), jh)
