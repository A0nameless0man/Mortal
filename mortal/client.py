
import os
import prelude

import logging
import socket
import torch
import numpy as np
import time
import gc
from os import path,environ

from model import Brain, DQN
from player_online import TrainPlayer
from net_emit import send_msg,recv_msg

def get_config(remote):
    while True:
        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {'type': 'get_test_config'})
            rsp = recv_msg(conn, map_location=torch.device('cpu'))
            if rsp['status'] == 'ok':
                return rsp['cfg']
            time.sleep(3)

def get_remote():
    ip=environ.get('MORTAL_SERVER_ADDR', '127.0.0.1')
    port=int(environ.get('MORTAL_SERVER_PORT', '5000'))
    return (ip,port)

def main():
    profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
    remote = get_remote()
    config = get_config(remote)
    logging.info('config has been loaded')
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']
    norm_config = config.get('norm_layer', None)
    oracle = None
    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).to(device).eval()
    dqn = DQN(version=version).to(device)
    train_player = TrainPlayer(remote, version)
    param_version = -1

    pts = np.array([90, 45, 0, -135])
    history_window = config['online']['history_window']
    history = []

    continues_fail_cnt = 0

    while True:
        while True:
            with socket.socket() as conn:
                conn.connect(remote)
                msg = {
                    'type': 'get_param',
                    'param_version': param_version,
                    'name': profile,
                }
                send_msg(conn, msg)
                rsp = recv_msg(conn, map_location=device)
                if rsp['status'] == 'ok':
                    param_version = rsp['param_version']
                    break
                time.sleep(3)
        train_player = TrainPlayer(remote,version)
        mortal.load_state_dict(rsp['mortal'])
        dqn.load_state_dict(rsp['dqn'])
        logging.info('param has been updated')
        try:
            rankings, file_list = train_player.train_play(oracle, mortal, dqn, device)
            avg_rank = (rankings * np.arange(1, 5)).sum() / rankings.sum()
            avg_pt = (rankings * np.array([90, 45, 0, -135])).sum() / rankings.sum()
            logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
            logs = {}
            for filename in file_list:
                with open(filename, 'rb') as f:
                    logs[path.basename(filename)] = f.read()
            with socket.socket() as conn:
                conn.connect(remote)
                send_msg(conn, {
                    'type': 'submit_replay',
                    'logs': logs,
                })
                logging.info('logs have been submitted')
            continues_fail_cnt = 0
        except Exception as e:
            logging.exception('failed to game：%s',str(e))
            continues_fail_cnt += 1
            logging.info('continues_fail_cnt = %d', continues_fail_cnt)
            train_player.train_seed+=train_player.seed_count

        rankings, file_list = train_player.train_play(oracle, mortal, dqn, device)
        avg_rank = (rankings * np.arange(1, 5)).sum() / rankings.sum()
        avg_pt = (rankings * pts).sum() / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = (sum_rankings * np.arange(1, 5)).sum() / sum_rankings.sum()
        ma_avg_pt = (sum_rankings * pts).sum() / sum_rankings.sum()

        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        logs = {}
        for filename in file_list:
            with open(filename, 'rb') as f:
                logs[path.basename(filename)] = f.read()

        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {
                'type': 'submit_replay',
                'logs': logs,
                'param_version': param_version,
            })
            logging.info('logs have been submitted')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
