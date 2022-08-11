import prelude

import logging
import itertools
import socket
import torch
import numpy as np
import time
from os import path
from model import Brain, DQN
from player import TrainPlayer
from common import send_msg, recv_msg
from config import config


def train_play(oracle: Brain, mortal: Brain, dqn: DQN):
    device = torch.device(config["control"]["device"])

    train_player = TrainPlayer()
    rankings, file_list = train_player.train_play(oracle, mortal, dqn, device)
    avg_rank = (rankings * np.arange(1, 5)).sum() / rankings.sum()
    avg_pt = (rankings * np.array([90, 45, 0, -135])).sum() / rankings.sum()
    logging.info(f"trainee rankings: {rankings} ({avg_rank}, {avg_pt}pt)")

    return file_list