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
import data_process

# parameters:

## 10 sec slicing:
full_audio_path = '/home/tk/Documents/full_audio/' # full audio will be stored here
sec10_sliced_path = '/home/tk/Documents/slice_10sec/' # 10 sec sliced will be stored here

#=======================================================================
# Operations
## 10 sec slicing
audio_list = os.listdir(full_audio_path)

if ".DS_Store" in audio_list:
	audio_list.remove(".DS_Store")
print ("There are", len(audio_list), "fully concatenated files")

for audio in audio_list:
    os.mkdir(sec10_sliced_path + audio[:-4])
    data_process.slice_it(audio, full_audio_path, sec10_sliced_path + audio + "/", length = 10000)
    print ('done slicing')
