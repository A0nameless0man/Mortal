import prelude

import numpy as np
import torch
import secrets
import os
from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree
from config import config

def main():
    key = secrets.randbits(64)

    cfg = config['1v3']
    games_per_iter = cfg['games_per_iter']
    seeds_per_iter = games_per_iter // 4
    iters = cfg['iters']
    log_dir = cfg['log_dir']
    use_akochan = cfg['akochan']['enabled']
    all_akochan = cfg['akochan']['all']

    if use_akochan:
        os.environ['AKOCHAN_DIR'] = cfg['akochan']['dir']
        os.environ['AKOCHAN_TACTICS'] = cfg['akochan']['tactics']
    else:
        mortal = Brain(False, **config['resnet']).eval()
        dqn = DQN().eval()
        state = torch.load(cfg['champion']['state_file'], map_location=torch.device('cpu'))
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        engine_cham = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            stochastic_latent = cfg['champion']['stochastic_latent'],
            device = torch.device(cfg['champion']['device']),
            enable_amp = cfg['champion']['enable_amp'],
            enable_rule_based_agari_guard = cfg['champion']['enable_rule_based_agari_guard'],
            name = cfg['champion']['name'],
        )
    if use_akochan and not all_akochan:
        mortal = Brain(False, **config['resnet']).eval()
        dqn = DQN().eval()
        state = torch.load(cfg['challenger']['state_file'], map_location=torch.device('cpu'))
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            stochastic_latent = cfg['challenger']['stochastic_latent'],
            device = torch.device(cfg['challenger']['device']),
            enable_amp = cfg['challenger']['enable_amp'],
            enable_rule_based_agari_guard = cfg['challenger']['enable_rule_based_agari_guard'],
            name = cfg['challenger']['name'],
        )

    seed_start = 10000
    for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
        print('-' * 50)
        print('#', i)
        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = log_dir,
        )
        if use_akochan:
            if all_akochan:
                rankings = env.ako_vs_ako(
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
                )
            else:
                rankings = env.ako_vs_py(
                    engine = engine_chal,
                    seed_start = (seed, key),
                    seed_count = seeds_per_iter,
                )
        else:
            rankings = env.py_vs_py(
                challenger = engine_chal,
                champion = engine_cham,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        rankings = np.array(rankings)
        avg_rank = (rankings * np.arange(1, 5)).sum() / rankings.sum()
        avg_pt = (rankings * np.array([90, 45, 0, -135])).sum() / rankings.sum()
        print(f'challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
