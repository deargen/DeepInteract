import pickle
from pathlib import Path 
from tqdm import tqdm 
from typing import Dict 
import pandas as pd 
from project.misc.check_leakage import save_glue

BASE_DIR = Path('..').resolve() 
assert BASE_DIR.name == 'DeepInteract'
DATASET_DIR = BASE_DIR / 'project/datasets/DIPS/final/processed'

SAVE_DIR = Path(__file__).parent / 'tmp'
SAVE_DIR.mkdir(exist_ok=True)

GLUE_ALN_FILE = BASE_DIR / 'custom_ppi_model' / 'dataset' / 'alignments' / 'dips-glue.aln'
assert GLUE_ALN_FILE.exists() 



def generate_dips_500():
    def _do(orig_list_file:Path, list_file:Path):
        if list_file.exists():
            print(f'File {list_file} already exists. Skip.')
            return 
        print(f'Generating {list_file}..')
        
        with open(orig_list_file, 'r') as f:
            orig_lines = f.readlines()
        
        lines = [] 
        for line in tqdm(orig_lines):
            cache_file = DATASET_DIR / line.strip()
            d = pd.read_pickle(cache_file)
            len1 = len(d['graph1'].ndata['f'])
            len2 = len(d['graph2'].ndata['f'])
            if len1 <= 500 and len2 <= 500:
                lines.append(line)
        
        list_file.parent.mkdir(parents=True, exist_ok=True)
        with open(list_file, 'w') as f: 
            f.writelines(lines)
            
            
    orig_train_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/pairs-postprocessed-train.txt'
    train_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/dips_500/pairs-postprocessed-train.txt'
    _do(orig_train_list_file, train_list_file)
    
    orig_val_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/pairs-postprocessed-val.txt'
    val_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/dips_500/pairs-postprocessed-val.txt'
    _do(orig_val_list_file, val_list_file)

def generate_dips_500_noglue():
    glue_codes_file = SAVE_DIR / 'glue.pkl'
    if not glue_codes_file.exists():
        save_glue()
    glue_pdb_codes = pickle.load(open(glue_codes_file, 'rb'))
    
    def _do(orig_list_file:Path, list_file:Path):
        if list_file.exists():
            print(f'File {list_file} already exists. Skip.')
            return 
        print(f'Generating {list_file}..')
        
        with open(orig_list_file, 'r') as f:
            orig_lines = f.readlines()
        
        lines = [] 
        for line in tqdm(orig_lines):
            a = line.split('/')[1]
            assert a[4:8] == '.pdb'
            pdb_code = a[:4]
            if not pdb_code in glue_pdb_codes:
                lines.append(line)
        
        list_file.parent.mkdir(parents=True, exist_ok=True)
        with open(list_file, 'w') as f:
            f.writelines(lines)
    
    dips_500_train_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/dips_500/pairs-postprocessed-train.txt'
    train_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/dips_500_noglue/pairs-postprocessed-train.txt'
    _do(dips_500_train_list_file, train_list_file)

    dips_500_val_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/dips_500/pairs-postprocessed-val.txt'
    val_list_file = BASE_DIR / 'project/datasets/DIPS/final/raw/dips_500_noglue/pairs-postprocessed-val.txt'
    _do(dips_500_val_list_file, val_list_file)    


if __name__ == '__main__':
    generate_dips_500()
    generate_dips_500_noglue()