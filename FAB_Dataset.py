import torch
from torch.utils.data import Dataset, DataLoader

from util.dataset_meta import *
from utils.dir_utils import TRAIN_DIR as TRAIN_DIR



class BlockBasedDataSet(Dataset):
    '''
    基于 __CLASSES__ENTRIES_PER_JSON__256__128__ 格式数据块
    的 dataloader
    load的结果为一个f(feat)-a(speaker-A)-b(speaker-B)

    构造方法：
        - 提供 block_dir ，feat_list 和 spec_list ，并指示是否 generate_fab_randomly

    类变量：
        - self.feat_block：读取到进程内存中，用于作为feat分量
        - 读取标号：
         - self.curr_json_index：trainset中的block号
         - self.curr_entry_index：block中的entry号
         - self.curr_fab_index：entry对应的所有(ALL_SAMPLES_PER_ENTRY) fab的编号
    '''
    def __init__(self, block_dir, feat_block_list, spec_block_list, gen_fab_random_mode):
        self.feat_block = []
        for block in feat_block_list: self.feat_block.append( json.load(open(os.path.join(block_dir, block), "r")) )
        self.feat_block = np.concatenate( np.array(self.feat_block), axis=1 ).transpose(1,0,2,3)

        self.curr_json_index = 0
        self.curr_entry_index = 0

        self.spec_block = np.array(json.load(open(os.path.join(block_dir, spec_block_list[0]), "r"))).transpose(1,0,2,3)
        self.f_a_b = gen_f_a_b(self.spec_block, self.curr_entry_index, self.feat_block, random_mode=gen_fab_random_mode)

        self.curr_fab_index = 0

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return None

class trainDataSet(BlockBasedDataSet):

    # 不变性：
    # 总保有一份 spec_block ，一份 feat_block
    # 每次访问时，有长为bs的f-a-b列表，每次取下标从列表中取得
    # f ：随机一个下标，取目标编号的spectrogram

    def __init__(self, bs, feat_train_blocks, spec_train_blocks):
        print("trainDataSet: feature blocks: ", len(feat_train_blocks))
        super(trainDataSet, self).__init__(TRAIN_DIR, feat_train_blocks, spec_train_blocks, gen_fab_random_mode=True)

    def __len__(self):
        return ENTRIES_PER_JSON * RANDOM_SAMPLES_PER_ENTRY * len(spec_train_blocks) // bs

    def __getitem__(self, dummy_index): # index is dummy, cuz doing ordered traverse
        '''
        数据规格协议：
        - block块，标号为self.curr_json_index；需支持 get_next_entry ，内部方法： get_next_block
            - entry流，标号为self.entry_index；需支持 gen_fab
        - fab块：通过 self.curr_fab_index 取下标；需支持 get_next_batch ，内部方法： get_next_entry 与拼接
            - batch流：顺序遍历无标号
        '''
        # to next batch

        fab = None
        if self.curr_fab_index + bs <= self.f_a_b.shape[1]:
            fab = self.f_a_b[:, self.curr_fab_index : self.curr_fab_index + bs]
            self.curr_fab_index += bs
        else: # load next entry
            self.curr_entry_index += 1

            if self.curr_entry_index == ENTRIES_PER_JSON: # load next block
                new_json_index = -1
                if self.curr_json_index + 1 < len(spec_train_blocks):
                    new_json_index += 1
                else:
                    new_json_index = 0

                if not new_json_index == self.curr_json_index:
                    self.curr_json_index = new_json_index
                    self.spec_block = np.array(
                            json.load(open(os.path.join(TRAIN_DIR, spec_train_blocks[new_json_index]), "r"))
                    ).transpose(1,0,2,3)

                self.curr_entry_index = 0

            if self.curr_fab_index < self.f_a_b.shape[1]:
                self.f_a_b = np.concatenate(
                    (   self.f_a_b[:, self.curr_fab_index:self.f_a_b.shape[1]], 
                        gen_f_a_b(self.spec_block, self.curr_entry_index, self.feat_block)  ),
                    axis=1
                )
            else:
                self.f_a_b = gen_f_a_b(self.spec_block, self.curr_entry_index, self.feat_block)

            self.curr_fab_index = 0
            fab = self.f_a_b[:, self.curr_fab_index : self.curr_fab_index + bs]

            self.curr_fab_index += bs

        f = torch.Tensor(fab[0]).view(bs, 256, 128)
        a = torch.Tensor(fab[1]).view(bs, 256, 128)
        b = torch.Tensor(fab[2]).view(bs, 256, 128)

        return f, a, b


def gen_all_pairs():
    all_pairs = []
    for i in range(CLASSES):
        for j in range(CLASSES):
            if(i==j): continue
            all_pairs.append([i, j])
    return np.array(all_pairs)

all_combinations = gen_all_pairs()
all_combination_indices = np.arange(CLASSES * (CLASSES-1) // 2)

def gen_rand_pairs(num_pairs):
    ''' 至多C(10,2)对组合 '''
    assert(2 * num_pairs <= CLASSES * (CLASSES - 1))
    ''' 长为 num_pairs 的 list ，为 [0,CLASSES-1]x[0,CLASSES-1] 中的序偶 '''
    chosen = all_combinations[ 
        np.array( np.random.choice(all_combination_indices, num_pairs, replace=False) ) 
    ]
    return chosen

def gen_f_a_b(spec_block, entry_index, feat_block, random_mode=True):
    if random_mode: 
        samples_selected = RANDOM_SAMPLES_PER_ENTRY
    else:
        samples_selected = ALL_SAMPLES_PER_ENTRY
    a_b_indexes = gen_rand_pairs(samples_selected).transpose()
    a_index_list, b_index_list = a_b_indexes[0], a_b_indexes[1]

    a_b = np.array([
        spec_block[entry_index, a_index_list], 
        spec_block[entry_index, b_index_list]
    ])
    feats = feat_block[
                np.random.randint(feat_block.shape[0]),
                a_index_list
            ].reshape(1, samples_selected, 256, 128)
    return np.concatenate((feats, a_b), axis=0)
