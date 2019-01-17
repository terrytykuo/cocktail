import numpy as np
import os
from data_process import gen_spectrogram
import json

##======================================
##               path
##======================================
root_dir = '/home/tk/Documents/'
sliced_pool_path = '/home/tk/Documents/sliced_pool/'

full_audio = ['birdstudybook', 'captaincook', 'cloudstudies_02_clayden_12', 
              'constructivebeekeeping',
              'discoursesbiologicalgeological_16_huxley_12', 
              'natureguide', 'pioneersoftheoldsouth', 
              'pioneerworkalps_02_harper_12', 
              'romancecommonplace', 'travelstoriesretold']

##======================================
##               control
##======================================

# number of mix blocks
blocks = 10

for i in range(blocks):
    all_target_label = []
    all_target_spec = []
    all_mix_spec = []
    
    for j in range(100):
        mix_spec = np.zeros((256, 128))
        cnt = 0
        
        all_spec = []
        all_selected_file = []

        for name in full_audio:
            
            
            file_name_list = os.listdir(sliced_pool_path + name + '/')
            file_name = np.random.choice(file_name_list)
            spec = gen_spectrogram(sliced_pool_path + name + '/' + file_name)
            
            print (file_name)
            
            all_selected_file.append(file_name)
            all_spec.append(spec)
            
            mix_spec = mix_spec + spec

            cnt+=1
                
        target_ind = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        target_spec = all_spec[target_ind]
        
        all_mix_spec.append(mix_spec)
        all_target_label.append(target_ind)
        all_target_spec.append(target_spec)

    all_target_label = np.array(all_target_label)
    all_target_label = np.stack(all_target_label)
    
    all_target_apec = np.array(all_target_spec)
    all_target_spec = np.stack(all_target_spec)
    
    all_mix_spec = np.array(all_mix_spec)
    all_mix_spec = np.stack(all_mix_spec)
    
    print("target label shape = ", all_target_label.shape)
    print("target spec shape = ", all_target_spec.shape)
    print("mix spec shape = ", all_mix_spec.shape)
        


    with open(root_dir + "mix_pool/mix_spec/" + 'mix_spec' + str(i) + '.json', 'w') as jh:
        json.dump(all_mix_spec.tolist(), jh)

    with open(root_dir + "mix_pool/target_spec/" + 'target_spec' + str(i) + '.json', 'w') as jh:
        json.dump(all_target_spec.tolist(), jh)

    with open(root_dir + "mix_pool/target_label/" 'target_label' + str(i) + '.json', 'w') as jh:
        json.dump(all_target_label.tolist(), jh)