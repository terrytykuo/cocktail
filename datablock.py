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

for count in [1,2,3,4,5,6,7,8,9]:

	file_list = []
	clean_list = []

	for i in full_audio:
		file_name = str(i) + str(count) + '.json'
		print (file_name)
		file_list.append(file_name)

	for j in file_list:
		print (j)
		with open(clean_dir + '{}'.format(j)) as jh:
			clean_list.append((json.load(jh)))


	cleanblock = np.stack(clean_list)
	print (cleanblock.shape)

	with open('/home/tk/Documents/ok/' + 'datablock' + str(count) + '.json', 'w') as jh:
		json.dump(cleanblock.tolist(), jh)

	for k in file_list:
	 	os.remove('/home/tk/Documents/clean/' + k)


