from data_process import gen_spectrogram
import numpy as np
import os


root_dir = '/home/tk/Documents/'
sliced_pool_path = '/home/tk/Documents/sliced_pool/'

female_audio = ['birdstudybook', 'captaincook', 'cloudstudies_02_clayden_12', 
              'discoursesbiologicalgeological_16_huxley_12', 
              'natureguide', 'pioneerworkalps_02_harper_12']

male_audio = ['constructivebeekeeping', 'pioneersoftheoldsouth',
                'romancecommonplace','travelstoriesretold']

full_audio = male_audio + female_audio

selection_space = male_audio

mix_spec = []
spec0_cluster = []
spec1_cluster = []
label =[]

for i in range(10):
    for j in range(100):

        # randomly select 2 numbers from 0~9 as index of full_audio
        ind = np.random.choice(len(selection_space), size = 2, replace = False)

        # generate spectrograms 
        spec_file0 = np.random.choice(os.listdir(sliced_pool_path + selection_space[int(ind[0])] + '/for_mix/'))
        spec0 = gen_spectrogram(sliced_pool_path + selection_space[int(ind[0])] + '/for_mix/' + spec_file0)

        spec_file1 = np.random.choice(os.listdir(sliced_pool_path + selection_space[int(ind[1])] + '/for_mix/'))
        spec1 = gen_spectrogram(sliced_pool_path + selection_space[int(ind[1])] + '/for_mix/' + spec_file1)
        print ('spec0 = ', spec_file0, 'spec1 = ', spec_file1)
        # generate mix_spectrograms
        mixed_spec = spec0 + spec1

        # records
        mix_spec.append(mixed_spec)
        spec0_cluster.append(spec0) # target_spec
        label.append(ind[0]) # assign the first number as target_label




    mix_spec = np.array(mix_spec)
    mix_spec = np.stack(mix_spec)
    print ('mixed_spec shape = ', mix_spec.shape)

    spec0_cluster = np.array(spec0_cluster)
    spec0_cluster = np.stack(spec0_cluster)
    print ('target_spec shape = ', spec0_cluster.shape)

    label = np.array(label)
    label = np.stack(label)
    print ('label shape = ', label.shape)

    with open(root_dir + "mix_pool/two_mix_spec/mix_spec/mix_spec" + str(i) + '.json', 'w') as jh:
        json.dump(mix_spec.tolist(), jh)

    with open(root_dir + "mix_pool/two_mix_spec/tatget_spec/tatget_spec" + str(i) + '.json', 'w') as jh:
        json.dump(spec0_cluster.tolist(), jh)

    with open(root_dir + "mix_pool/two_mix_spec/label/label" + str(i) + '.json', 'w') as jh:
        json.dump(label.tolist(), jh) 