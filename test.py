import numpy as np

CLASSES = 10

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
    chosen = all_combinations[ np.array(np.random.choice(all_combination_indices, num_pairs, replace=False)) ]
    return chosen

print(gen_rand_pairs(2))