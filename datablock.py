import numpy as np 
import json
import os

full_audio = ['birdstudybook', 'captaincook', 'cloudstudies_02_clayden_12', 
              'constructivebeekeeping',
              'discoursesbiologicalgeological_16_huxley_12', 
              'natureguide', 'pioneersoftheoldsouth', 
              'pioneerworkalps_02_harper_12', 
              'romancecommonplace', 'travelstoriesretold']



clean_dir = '/home/tk/Documents/clean/temp/'
cleanfolder = os.listdir(clean_dir)
cleanfolder.sort()

clean_list = []


for j in cleanfolder:
	print (j)
	with open(clean_dir + '{}'.format(j)) as f:
		clean_list.append((json.load(f)))


	cleanblock = np.stack(clean_list)
	print (cleanblock.shape)

	with open(clean_dir + 'temp/datablock' + '0.json', 'w') as jh:
	    json.dump(cleanblock.tolist(), jh)

