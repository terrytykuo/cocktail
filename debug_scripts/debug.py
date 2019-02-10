
import numpy as np
import json

spec_block = np.array(
    json.load(open(train_dir + spec_train_blocks[self.curr_json_index], "r"))
).transpose(1,0,2,3)
