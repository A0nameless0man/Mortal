import prelude

import logging
import os
from glob import glob
from libriichi.dataset import Grp
from common import tqdm
from config import config

def train():
    cfg = config['grp']

    train_globs = cfg['dataset']['train_globs']
    logging.info('building file index...')
    train_file_list = []
    for pat in train_globs:
        train_file_list.extend(glob(pat, recursive=True))
    for file in tqdm(train_file_list):
        try:
            Grp.load_gz_log_files([file])
        except:
            os.remove(file)
        
if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
