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

# parameters:

## 10 sec slicing:

#=============================================
#        path
#=============================================

server = False

root_dir = '/home/tk/Documents/'
if server == True:
    root_dir = '/home/guotingyou/cocktail_phase2/'


clean_dir = root_dir + 'clean/' 
mix_dir = root_dir + 'mix/' 
clean_label_dir = root_dir + 'clean_labels/' 
mix_label_dir = root_dir + 'mix_labels/' 

cleanfolder = os.listdir(clean_dir)
cleanfolder.sort()

# mixfolder = os.listdir(mix_dir)
# mixfolder.sort()


clean_list = []
# mix_list = []

# controls datapoint in single column
multiplication = 1

# blocks 
blocks_volume = 10

#minimum audio length
length = 0.5
#======================================================================


audio_list = os.listdir(root_dir + "full_audio/")
if ".DS_Store" in audio_list:
    audio_list.remove(".DS_Store")
print ("There are", len(audio_list), "fully concatenated files")


for p in range(blocks_volume):

    # remove current files 
    to_clear = os.listdir(root_dir + "slice_pointsec/")
    for c in to_clear:
        os.remove(root_dir + "slice_pointsec/" + c)

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
            data_process.slice_it(name, root_dir + "slice_10sec/", root_dir + "slice_pointsec/", length *1000)

    ## checkpoint: spec_name should have 1,000 files
    spec_name = os.listdir(root_dir + "slice_pointsec/")
    if len(spec_name) == (10/length)* 10 * multiplication: 
        print ((10/length)* 10 * multiplication)
        print ("checked")
    spec_name.sort()
    
    
    
    ## Generate Spectrogram and concatenate 
    print ('generating spectrogram now')

    big_pcs = []
    all_index = []

    for i in range(int(10/length) * multiplication):
        small_pcs = []
        index_record = []
        mix_pcs = []
        
        
        # generate clean column
        for j in range(0 + i, int(10/length) * 10 * multiplication + i, int(10/length) * multiplication):
            spec = data_process.gen_spectrogram(root_dir + 'slice_pointsec/' + spec_name[j])
            small_pcs.append(spec)

        # generate mixed column
        mixed = []
        index_record = []
        pc1, pc2 = np.random.choice([0,1,2,3,4,5,6,7,8,9], 2, replace = False)
        
        mixed_spec = (small_pcs[pc1] + small_pcs[pc2])
        print ("mixed_spec shape =", mixed_spec.shape)

        # append mixed spec into small_pcs
        small_pcs.append(mixed_spec)

        print ("pc1 = ", pc1, ", pc2 = ", pc2)
        index_record.append([pc1, pc2])

       
        
#        # record index
#        ## turn [pc1, pc2] into one-hot encoding
#        z = np.zeros((10,))
#        z[pc1] = 1
#        z[pc2] = 1
#        print ("pc1 = ", pc1, "pc2 = ", pc2, "\n", "z = " ,z)
#        index_record.append(z)
        
#        single_row = np.stack(mix_pcs)    

        big_pcs.append(small_pcs)
        all_index.append(index_record)
        
        print (i, "row done")

    big_matrix = np.vstack(big_pcs)
    print ("big_matrix shape = ", big_matrix.shape)
    index_matrix = np.vstack(all_index)
    print ("index_matrix shape = ", index_matrix.shape)

    print ("The", p, "th block done. Start writing .json file")
    ## x_train = big_matrix --> json
    ## y_train = index_record --> json
    
    with open(root_dir + "mix/mix" + str(p) + '.json', 'w') as jh:
        print(root_dir + "mix/mix" + str(p) + '.json')
        json.dump(big_matrix.tolist(), jh)
    
    with open(root_dir + "mix_labels/mix_label" + str(p) + '.json', 'w') as f:
        print(root_dir + "mix_labels/mix_label" + str(p) + '.json')
        json.dump(index_matrix.tolist(), f)