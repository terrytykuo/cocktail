from data_process import gen_spectrogram
import numpy as np
import os


root_dir = '/home/tk/Documents/'
sliced_pool_path = '/home/tk/Documents/sliced_pool/'

full_audio = ['birdstudybook', 'captaincook', 'cloudstudies_02_clayden_12', 
              'constructivebeekeeping',
              'discoursesbiologicalgeological_16_huxley_12', 
              'natureguide', 'pioneersoftheoldsouth', 
              'pioneerworkalps_02_harper_12', 
              'romancecommonplace', 'travelstoriesretold']

mix_spec = []
spec0_cluster = []
spec1_cluster = []
label =[]



for i in range(10):
    for j in range(100):

        # randomly select 2 numbers from 0~9 as index of full_audio
        ind = np.random.choice([0,1,2,3,4,5,6,7,8,9], 
            size = 2, replace = False)

        # generate spectrograms 
        spec_file = np.random.choice(os.listdir(sliced_pool_path + full_audio[int(ind[0])] + '/'))
        spec0 = gen_spectrogram(sliced_pool_path + full_audio[int(ind[0])] + '/' + spec_file)
        print (spec_file)

        spec_file = np.random.choice(os.listdir(sliced_pool_path + full_audio[int(ind[1])] + '/'))
        spec1 = gen_spectrogram(sliced_pool_path + full_audio[int(ind[1])] + '/' + spec_file)
        print (spec_file)

        # generate mix_spectrograms
        mixed_spec = spec0 + spec1

        # records
        mix_spec.append(mixed_spec)
        spec0_cluster.append(spec0) # target_spec
        label.append(ind[0]) # assign the first number as target_label




    mix_spec = np.array(mix_spec)
    mix_spec = np.atack(mix_spec)
    print ('mixed_spec shape = ', mix_spec.shape)

    spec0_cluster = np.array(spec0_cluster)
    spec0_cluster = np.atack(spec0_cluster)
    print ('target_spec shape = ', spec0_cluster.shape)

    label = np.array(label)
    label = np.atack(label)
    print ('label shape = ', label.shape)

    with open(root_dir + "mix_pool/two_mix_spec/mix_spec/mix_spec" + str(i) + '.json', 'w') as jh:
        json.dump(mix_spec.tolist(), jh)

    with open(root_dir + "mix_pool/two_mix_spec/tatget_spec/tatget_spec" + str(i) + '.json', 'w') as jh:
        json.dump(spec0_cluster.tolist(), jh)

    with open(root_dir + "mix_pool/two_mix_spec/label/label" + str(i) + '.json', 'w') as jh:
        json.dump(label.tolist(), jh) 