import numpy as np
import os
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
              
              

for audio_name in full_audio:

    single_audio = []
    file_list = os.listdir('/home/tk/Documents/mix_pool/feature/' + audio_name + '/')

    for i in file_list:
        with open('/home/tk/Documents/mix_pool/feature/' + audio_name + '/' + i) as f:
            single_audio.append(json.load(f))


    single_audio = np.array(single_audio)
    single_audio = np.hstack(single_audio)

    print ("shape = ", single_audio.shape)


    with open(mixed_pool_path +  'feature/' + audio_name + '.json', 'w') as jh:
        json.dump(single_audio.tolist(), jh)

    # with open(mixed_pool_path +  'feature_label/' + name + '/' + str(i) + '.json', 'w') as jh:
    #     json.dump(all_clean_label.tolist(), jh)



