# generate 0.5 sec sliced audio from ~/Documents/slice_10sec/
# take _60~_160 for each file
# save the sliced file at ~/Documents/sliced_pool/ 
import os
##=============================
##    define slicing function
##=============================

def slice_it(filename, input_path, output_path, length):
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    

    # slice the file
    myaudio = AudioSegment.from_file(input_path + filename, 'wav') 
    chunks = make_chunks(myaudio, length) # Make chunks

    #Export all individual chunks as wav files
    for i, chunk in enumerate(chunks):
        chunk_name = filename[:-4] + "_{0}.wav".format(i) # select first 6 characters as file name
        print ("exporting", chunk_name)
        chunk.export(output_path + chunk_name, format="wav")

    # dump the last slice (might be an incomplete slice)
    dump_file = [] 
    chunk_list = os.listdir(output_path)
    if '.DS_Store' in chunk_list:
        chunk_list.remove('.DS_Store')

##=============================
##      full audio name
##=============================
full_audio = ['birdstudybook', 'captaincook', 'cloudstudies_02_clayden_12', 
              'constructivebeekeeping',
              'discoursesbiologicalgeological_16_huxley_12', 
              'natureguide', 'pioneersoftheoldsouth', 
              'pioneerworkalps_02_harper_12', 
              'romancecommonplace', 'travelstoriesretold']

full_audio_path = '/home/tk/Documents/full_audio/' # full audio will be stored here
sec10_sliced_path = '/home/tk/Documents/slice_10sec/' # 10 sec sliced will be stored here
sliced_pool_path =  '/home/tk/Documents/sliced_pool/'


##=============================
##       slicing
##=============================
for name in full_audio:
    sliced_list = []
    if (sliced_pool_path + name) == False :
        os.mkdir(sliced_pool_path + name)
    
    
    for i in range(225, 250): # numbers in range() controls which segements to take
        m = name + '_' + str(i) + '.wav'
        sliced_list.append(m)
        del m
        
    for j in sliced_list:
        slice_it(j, sec10_sliced_path + name + '/', sliced_pool_path + name + '/for_mix/', 500)

        
#for j in sliced_list:
#    slice_it(j, '/home/tk/Documents/slice_10sec/' + name, '/home/tk/Documents/sliced_pool/a/', 500)
    
##=============================
# This file will create 1000*20 = 20,000 audio files, 
# where 1000 = 100 (_60~_160) * 10 (numbers of full audio files) * (10 sec/0.5 sec)
##=============================
