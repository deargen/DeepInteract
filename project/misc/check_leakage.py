import pickle
from pathlib import Path 
from tqdm import tqdm 
from typing import Dict 
import pandas as pd 


BASE_DIR = Path('..').resolve() 
assert BASE_DIR.name == 'DeepInteract'

SAVE_DIR = Path(__file__).parent / 'tmp'
SAVE_DIR.mkdir(exist_ok=True)

GLUE_ALN_FILE = BASE_DIR / 'custom_ppi_model' / 'dataset' / 'alignments' / 'dips-glue.aln'
assert GLUE_ALN_FILE.exists() 

def save_trainval():
    train_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/pairs-postprocessed-train.txt'
    val_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/pairs-postprocessed-val.txt'
    assert train_list_file.exists(), train_list_file.resolve()
    assert val_list_file.exists(), val_list_file.resolve() 
    s = set([])
    with open(train_list_file, 'r') as f:
        for line in f:
            a = line.split('/')[1]
            assert a[4:8] == '.pdb'
            s.add(a[:4])
    with open(val_list_file, 'r') as f:
        for line in f:
            a = line.split('/')[1]
            assert a[4:8] == '.pdb'
            s.add(a[:4])
    
    with open(SAVE_DIR / 'trainval.pkl', 'wb') as f:
        pickle.dump(s, f)
        
def save_glue():
    s = set([])
    with open(GLUE_ALN_FILE, 'r') as f:
        for line in f:
            chain_code, _, sim = line.split()
            pdb_code = chain_code.split('_')[0]
            sim = float(sim)
            if sim > 30:
                s.add(pdb_code)
    
    with open(SAVE_DIR / 'glue.pkl', 'wb') as f:
        pickle.dump(s, f)
        
def compare():
    trainval_set = pickle.load(open(SAVE_DIR / 'trainval.pkl', 'rb'))
    glue_set = pickle.load(open(SAVE_DIR / 'glue.pkl', 'rb'))
    print(len(set.intersection(trainval_set, glue_set)))
        
if __name__ == '__main__':
    #save_glue()
    #save_trainval() 
    compare()
    