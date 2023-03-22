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
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from config import config

    device = torch.device(config["control"]["device"])
    version = config['control']['version']
    eps = config['optim']['eps']
    lr = config['optim']['lr']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']

    norm_config = config.get('norm_layer', None)
    logging.info(f"mortal version: {version:,}")
    mortal = Brain(version=version, **config['resnet'], norm_config=norm_config).to(device)
    current_dqn = DQN(version=version).to(device)
    next_rank_pred = NextRankPredictor().to(device)

    logging.info(f"mortal params: {parameter_count(mortal):,}")
    logging.info(f"dqn params: {parameter_count(current_dqn):,}")
    logging.info(f'next_rank_pred params: {parameter_count(next_rank_pred):,}')

    mortal.freeze_bn(config["freeze_bn"]["mortal"])

    decay_params = []
    no_decay_params = []
    for model in (mortal, current_dqn):
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=0, betas=betas, eps=eps)
    scheduler = optim.lr_scheduler.StepLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler()

    steps = 0
    state_file = config["control"]["state_file"]
    optimizer.zero_grad(set_to_none=True)
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    state = {
        'mortal': mortal.state_dict(),
        'current_dqn': current_dqn.state_dict(),
        'next_rank_pred': next_rank_pred.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'steps': steps,
        'timestamp': datetime.now().timestamp(),
        'best_perf': best_perf,
        'config': config,
    }
    torch.save(state, state_file)


def main():
    gen()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
