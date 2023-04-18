import pickle
from pathlib import Path 
from tqdm import tqdm 
from typing import Dict 
import pandas as pd 


BASE_DIR = Path('..').resolve() 
assert BASE_DIR.name == 'DeepInteract'
DATASET_DIR = BASE_DIR / 'project/datasets/DIPS/final/processed'

def inspect_lengths():
    train_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/pairs-postprocessed-train.txt'
    val_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/pairs-postprocessed-val.txt'
    assert train_list_file.exists(), train_list_file.resolve()
    assert val_list_file.exists(), val_list_file.resolve() 
    
    cache_files = []    
    with open(train_list_file, 'r') as f:
        cache_files += [DATASET_DIR / line.strip() for line in f]
    with open(val_list_file, 'r') as f:
        cache_files += [DATASET_DIR / line.strip() for line in f]
    
    nums = {
        'both<=500': 0,
        'both>500': 0,
        'mixed,sum<=1000': 0,
        'mixed,sum>1000': 0
    }
    for cache_file in tqdm(cache_files):
        d = pd.read_pickle(cache_file)
        len1 = len(d['graph1'].ndata['f'])
        len2 = len(d['graph2'].ndata['f'])
        if len1 <= 500 and len2 <= 500:
            nums['both<=500'] += 1
        elif len1 > 500 and len2 > 500:
            nums['both>500'] += 1
        elif len1 + len2 <= 1000:
            nums['mixed,sum<=1000'] += 1
        elif len1 + len2 > 1000:
            nums['mixed,sum>1000'] += 1
        else:
            raise Exception('Cannot happen')
    
    print(nums)
    
    
if __name__ == '__main__':
    inspect_lengths()