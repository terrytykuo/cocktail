import numpy as np 
import json
import os

full_audio = ['birdstudybook', 'captaincook', 'cloudstudies_02_clayden_12', 
              'constructivebeekeeping',
              'discoursesbiologicalgeological_16_huxley_12', 
              'natureguide', 'pioneersoftheoldsouth', 
              'pioneerworkalps_02_harper_12', 
              'romancecommonplace', 'travelstoriesretold']



clean_dir = '/home/tk/Documents/clean/'
cleanfolder = os.listdir(clean_dir)
cleanfolder.sort()

clean_list = []

for count in [1,2,3,4,5,6,7,8,9]:
	for i in full_audio:
		file_name = i + count + '.json'
		print (file_name)
		file_list.append((json.load(f)))

	for j in file_list:
		with open(clean_dir + '{}'.format(j)) as f:
			clean_list.append((json.load(f)))


	cleanblock = np.stack(clean_list)
	print (cleanblock.shape)

	with open('/home/tk/Documents/clean/' + 'datablock' + count + '.json', 'w') as jh:
		json.dump(cleanblock.tolist(), jh)

