import numpy as np
import os
from data_process import gen_spectrogram
import json


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
            
            
        all_clean_spec = np.array(all_clean_spec)
        all_clean_spec = np.stack(all_clean_spec)
            
        print ("name = ", name , ", shape = ", all_clean_spec.shape)
    
        with open(mixed_pool_path +  '/feature/' + name + '/' + str(i) + '.json', 'w') as jh:
            json.dump(all_clean_spec.tolist(), jh)


