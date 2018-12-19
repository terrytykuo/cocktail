# Package
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import gc
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scipy
import scipy.io.wavfile
import numpy as np
import json
import data_process



server = False

root_dir = '/home/tk/Documents/'
if server == True:
    root_dir = '/home/guotingyou/cocktail_phase2/'


# parameters:

## 10 sec slicing:

# full audio will be stored here
full_audio_path = root_dir + 'full_audio/' 

# 10 sec sliced will be stored here
sec10_sliced_path = root_dir + 'slice_10sec/' 

# 0.1 sec slices will be stored here
point_sec_sliced_path = root_dir + 'slice_pointsec/' 

# clean audios will be stored here
block_path = root_dir + 'clean/'  

# clean labels will be stored here
labels_path = root_dir + 'clean_labels/' 

# controls datapoint in single column
multiplication = 1

# blocks 
blocks_volume = 20

#minimum audio length
length = 0.5
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

    ## 0.1 sec slicing
    print ('slicing segment', p, 'now')
    file_name = []
    for i in audio_list:
        slice_name = i[:-4]
        file_name.append(slice_name) # generate 0.1 sec file list

    for file in file_name:
        for i in range(pieces * multiplication , (pieces + 1) * multiplication):
            name = file + "_{0}.wav".format(i)
            data_process.slice_it(name, sec10_sliced_path, point_sec_sliced_path, length = length *1000)

    ## checkpoint: spec_name should have 1,000 files
    spec_name = os.listdir(point_sec_sliced_path)
    if len(spec_name) == (10/length)* 10 * multiplication: 
        print ((10/length)* 10 * multiplication)
        print ("checked")
    spec_name.sort()
    
        
    small_pcs = []
    pc = []
    for i in range(int(10/length) * 10 * multiplication):
        spec = data_process.gen_spectrogram(point_sec_sliced_path + spec_name[i])
        small_pcs.append(spec) # record spectrograms
        
        # one-hot encoding
        index = int(i/((10/length) * multiplication))
        z = np.zeros((10))
        z[index] = 1
        pc.append(z) # record indexs
        print ("i =", i, "; index =", index ,"\n", "z =",z)
    
    big_matrix = np.stack(small_pcs)
    index_matrix = np.stack(pc)

    print ("big_matrix shape = ", big_matrix.shape)
    print ("index_matrix shape = ", index_matrix.shape)

    print ("The", p, "th block done. Start writing .json file")
    
    ## x_train = big_matrix --> json
    ## y_train = index_record --> json
    with open(block_path + "clean" + str(p) + '.json', 'w') as jh:
        json.dump(big_matrix.tolist(), jh)
    
    with open(labels_path + "clean_label" + str(p) + '.json', 'w') as f:
        json.dump(index_matrix.tolist(), f)
        
    print (".json done")