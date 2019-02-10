import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import gc
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile
import json
import acoustics
import cv2

root_dir = '/home/tk/Documents/'
sliced_pool_path = '/home/tk/Documents/sliced_pool/'
mixed_pool_path =  '/home/tk/Documents/mix_pool/'

full_audio = ['birdstudybook', 'captaincook', 'cloudstudies_02_clayden_12', 
              'constructivebeekeeping',
              'discoursesbiologicalgeological_16_huxley_12', 
              'natureguide', 'pioneersoftheoldsouth', 
              'pioneerworkalps_02_harper_12', 
              'romancecommonplace', 'travelstoriesretold']
              



              
blocks = 1

def gen_spectrogram(wav):
    
    fs, x = scipy.io.wavfile.read(wav) # read audio file as np array
    spec, _, _, _= plt.specgram(x, Fs=fs, NFFT=2048, noverlap=1900)
    plt.close('all')
    gc.collect()

    freq_wid = 342
    spec = spec[:freq_wid, :]

    spec_ = spec[:256, :128]
    mean = np.mean(spec_)
    spec_ = spec_/ mean
    spec_[spec_ >= 1] = 1

    white = acoustics.generator.white(2048)
    white = white.reshape(256, 128)
    spec_ = white + spec_

    return spec_


for i in range(blocks):
    for name in full_audio:
        
        all_clean_spec = []
        if (mixed_pool_path + 'feature/' + name) == False:
            os.mkdir(mixed_pool_path + 'feature/' + name)
        
        file_name_list = os.listdir(sliced_pool_path + name + '/clean/')
        file_name = np.random.choice(file_name_list, 10)
        
        for k in file_name:
            spec = gen_spectrogram(sliced_pool_path + name + '/clean/' + k)
            print (k)
            all_clean_spec.append(spec)

            # print white noise
            cv2.imwrite(mixed_pool_path +  '/white/' + name + '/' + str(i), spec)

            
        all_clean_spec = np.array(all_clean_spec)
        all_clean_spec = np.stack(all_clean_spec)
            
        print ("name = ", name , ", shape = ", all_clean_spec.shape)
    
        with open(mixed_pool_path +  '/white/' + name + '/' + str(i) + '.json', 'w') as jh:
            json.dump(all_clean_spec.tolist(), jh)


