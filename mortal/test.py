

def gen():
    import prelude

    import logging
    import torch
    import numpy as np
    from datetime import datetime
    from os import path
    from glob import glob
    from torch import optim
    from torch.cuda import amp
    from torch import optim, nn
    from tqdm.auto import tqdm
    from torch.cuda.amp import GradScaler
    from torch_tools import parameter_count
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from model import Brain, DQN, NextRankPredictor
    from lr_scheduler import stage_scheduler
    from config import config

    f = stage_scheduler(config["norm"]["scheduler"])
    print([f(s) for s in range(0,int(1e7),int(5e4))])


def main():
    gen()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
