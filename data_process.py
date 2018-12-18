# data_process
# Input:
# 	long audio files

# Expected output:
# 	data matrix with below structure (tuple?)
# 	 |mixed, a0, a1, a2,...|
# 	(|			           |)
# 	 |                     |

# Operations:
# 1. concat several short audio files spoken by same person into single long audio files
# 2. slice down long audio file into small ones (3 sec/ each)
# 3. turn small audio files into spectrograms
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


## concat
def concat_audio(raw_path, concated_audio_path):
    from pydub import AudioSegment
    import os

    # generate a filename list according to given path
    name_list = os.listdir(raw_path)
    
    # delete '.DS_Store' in the list
    if '.DS_Store' in name_list:
        name_list.remove('.DS_Store')
    else:
        pass

    # take the 1st element in name_list as final output's file name 
    output_name = name_list[0][:-7]
    
    # concat
    sound = AudioSegment.from_mp3(raw_path + name_list[0]) # format alert, modify .from_mp3 method if file format is not mp3
    
    for i in range(2, len(name_list)):
        sound1 = AudioSegment.from_mp3(raw_path + name_list[i])
        sound = sound + sound1
    
    # export
    output_path = concated_audio_path + str(output_name) + '.wav' # format alert
    sound.export(output_path, format="wav") # format alert, output .wav file
    print ('done concat')



## slice
def slice_it(filename, input_path, output_path, length):
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    

    # slice the file
    myaudio = AudioSegment.from_file(input_path + filename, 'wav') 
    chunks = make_chunks(myaudio, length) # Make chunks

    #Export all individual chunks as wav files
    for i, chunk in enumerate(chunks):
        chunk_name = filename[:-4] + "_{0}.wav".format(i) # select first 6 characters as file name
        print ("exporting", chunk_name)
        chunk.export(output_path + chunk_name, format="wav")

    # dump the last slice (might be an incomplete slice)
    dump_file = [] 
    chunk_list = os.listdir(output_path)
    if '.DS_Store' in chunk_list:
        chunk_list.remove('.DS_Store')


## spectrogram
def gen_spectrogram(wav):
    
    import matplotlib.pyplot as plt
    import gc    
    
    fs, x = scipy.io.wavfile.read(wav) # read audio file as np array
    spec, _, _, _= plt.specgram(x, Fs=fs, NFFT=2048, noverlap=1900)
    spec = spec[:500,:]
    plt.close('all')
    gc.collect()

    return spec


def s_matrix(segment, point_sec_sliced_path, multiplication):
    spec_name = os.listdir(point_sec_sliced_path)
    spec_name.sort()

    cnt = 0
    s_pieces = []
    # every 100 datapoint belongs to the same category
    # i = 0 means it will transfer filename_1_0 ~ filename_1_99 into spectrograms and concatenate them 
    for filename in spec_name[(segment)* 100 * multiplication : (segment+1) * 100 * multiplication]: 
        spec = gen_spectrogram(point_sec_sliced_path + filename)
        s_pieces.append(spec)
        
        cnt = cnt+1
        if cnt % 50 == 0:
            print (cnt)
    # concatenate filename_1_0 ~ filename_9_99 into a matrix
    s_matrix = np.stack([s_pieces])
    
    return s_matrix


# concatenate the final block
def big_matrix():
    big_pieces = []
    for segment in range(len(big_pieces)):
        print (segment)
        big_pieces.append(s_matrix(segment))
    big_matrix = np.vstack((big_pieces))
    return big_matrix

# seems useless now :ppppp
def mix(audio_source):
    # randomly select 2 audio source with the list
    pieces1, pieces2 = np.random.choice(audio_source, 2)
    
    # normalize the 2 mixed audios
    m = (pieces1 * pieces2) / (pieces1 + pieces2)
    
    # return the index
    pc1 = audio_source.index(pieces1)
    pc2 = audio_source.index(pieces2)
    
    return m, pc1, pc2