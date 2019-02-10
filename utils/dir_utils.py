
import os

def list_json_in_dir(dir):
    temp = os.listdir(dir)
    temp.sort()
    i = 0
    for t in temp:
        if '.json' in t:
            i += 1
        else:
            del temp[i]
    return temp

ROOT_DIR = '/home/tk/cocktail/'

TRAIN_DIR = root_dir + 'cleanblock/'
TEST_DIR  = root_dir + 'clean_test/'