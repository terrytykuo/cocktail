import numpy as np
import os
from data_process import gen_spectrogram
import json
import imageio
import acoustics


root_dir = '/home/tk/Documents/'
sliced_pool_path = '/home/tk/Documents/sliced_pool/'
mixed_pool_path =  '/home/tk/Documents/mix_pool/'

full_audio = ['birdstudybook'],
             # 'captaincook', 'cloudstudies_02_clayden_12', 
             #  'constructivebeekeeping',
             #  'discoursesbiologicalgeological_16_huxley_12', 
             #  'natureguide', 'pioneersoftheoldsouth', 
             #  'pioneerworkalps_02_harper_12', 
             #  'romancecommonplace', 'travelstoriesretold']
              
              
blocks = 1
target_snr = 1
noise_type = 'white'

# gen_spectrogram
def gen_noise_spectrogram(wav, target_snr, noise_type):
    
    import matplotlib.pyplot as plt
    import gc    
    
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

    if noise_type == 'white':
        noise = acoustics.generator.white(256*128).reshape(256, 128)
    
    if noise_type == 'pink':
        noise = acoustics.generator.pink(256*128).reshape(256, 128)

    current_snr = (np.mean(spec_))/ np.std(noise)
    noise = noise * (current_snr/ target_snr)
    return spec_ + noise


for i in range(blocks):
    for ind, name in enumerate(full_audio):
        
        all_clean_spec = []
        all_clean_label = []

        if (mixed_pool_path + 'feature/' + name) == False:
            os.mkdir(mixed_pool_path + 'feature/' + name)
        
        if (mixed_pool_path + 'feature_label/' + name) == False:
            os.mkdir(mixed_pool_path + 'feature_label/' + name)


        file_name_list = os.listdir(sliced_pool_path + name + '/clean/')
        file_name = np.random.choice(file_name_list, 1000)
        

        for k in file_name:
            spec = gen_spectrogram(sliced_pool_path + name + '/clean/' + k, target_snr, noise_type)
            print (k)
            all_clean_spec.append(spec)
            all_clean_label.append(ind)
            print (ind)
            
            
        all_clean_spec = np.array(all_clean_spec)
        all_clean_spec = np.stack(all_clean_spec)

        all_clean_label = np.array(all_clean_label)
        all_clean_label = np.stack(all_clean_label)

            
        print ("name = ", name , ", shape = ", all_clean_spec.shape)
        print ("label = ", name , ", shape = ", all_clean_label.shape)

    
        with open(mixed_pool_path +  'feature/' + name + '/' + name + str(i) + '.json', 'w') as jh:
            json.dump(all_clean_spec.tolist(), jh)

        with open(mixed_pool_path +  'feature_label/' + name + '/' + name + str(i) + '.json', 'w') as jh:
            json.dump(all_clean_label.tolist(), jh)


