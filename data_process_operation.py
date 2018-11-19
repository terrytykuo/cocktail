# data_process
# Input:
# 	long audio files

# Expected output:
# 	data matrix with below structure 
# 	 |[mixed], [a0], [a1], [a2],...|
# 	(|		   	                   |)
# 	 |                             |

# Operations:
# 1. concat several short audio files spoken by same person into single long audio files
#	  --> retrieve raw file from multiple folder
#	  --> save in concated_audio_path
# 2. slice down long audio file into small ones (3 sec/ each)
#	  --> retrieve from concat_audio_path
#	  --> save in fragment_audio_path
# 3. turn fragmented audio files into spectrograms
# 4. store spectrograms into above structure 


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


# parameters:

## 10 sec slicing:
full_audio_path = '/home/tk/Documents/full_audio/' # full audio will be stored here
sec10_sliced_path = '/home/tk/Documents/slice_10sec/' # 10 sec sliced will be stored here

## 0.1 sec slicing
pieces = 0 # controls which 10 sec segments will be processed 0.1 sec slicing 
		   # pieces = 1 means 0~10, 
		   # pieces = 2 means 10~20, etc.

point_sec_sliced_path = '/home/tk/Documents/slice_pointsec/' # 0.1 sec slices will be stored here

## generate block
block_path = '/home/tk/Documents/blocks/'   


#-----------------------------------------------------------------
# Operations
## 10 sec slicing
audio_list = os.listdir(full_audio_path)

if ".DS_Store" in audio_list:
	audio_list.remove(".DS_Store")
print ("There are", len(audio_list), "fully concatenated files")

for audio in audio_list:
     data_process.slice_it(audio, full_audio_path, sec10_sliced_path, length = 10000)
    print ('done slicing')


## 0.1 sec slicing
file_name = []
for i in audio_list:
    slice_name = i[:-4]
    file_name.append(slice_name)

# controls sliced segments, 
# pieces = 1 means 0~10, 
# pieces = 2 means 10~20, etc.

for file in file_name:
    for i in range(pieces * 10, (pieces + 1) * 10):
        name = file + "_{0}.wav".format(i)
         data_process.slice_it(name, sec10_sliced_path, point_sec_sliced_path, length = 100)

## checkpoint: spec_name should have 10,000 files
spec_name = os.listdir(point_sec_sliced_path)
spec_name.sort()
if len(spec_name) == 10000:
	print ("10000 checked!")


## Generate Spectrogram and concatenate
from data_process import gen_spectrogram
big_matrix = data_process.big_matrix()

## big_matrix --> json
jh = open("~/Documents/blocks/" + "block0" + '.json', 'w')
json.dump(big_matrix.tolist(), jh)