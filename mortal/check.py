from libriichi.dataset import GameplayLoader,Grp
from glob import glob
from config import config
from multiprocessing import Pool
from tqdm import tqdm
import os

loader = GameplayLoader(oracle=True)
def check_single(file:str):
    try:
        Grp.load_gz_log_files([file])
        loader.load_gz_log_files([file])
        return True
    except:
        os.remove(file)
        print(f"Error loading log file {file}")
        return False

def main():
    file_list = []
    for pat in config["dataset"]["globs"]:
        file_list.extend(glob(pat, recursive=True))
    file_list.sort(reverse=True)
    with Pool(16) as pool,tqdm(total=len(file_list),desc="Checking") as bar,tqdm(desc="Error",position=1) as error:
        for res in pool.imap_unordered(check_single,file_list):
            bar.update()
            if not res:
                error.update()

if __name__ == "__main__":
    main()