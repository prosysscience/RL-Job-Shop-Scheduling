import os
import multiprocessing as mp
import time

import numpy as np

from JSS import default_ppo_config
from JSS.multiprocessing_env import SubprocVecEnv
from JSS.ppo import ppo, make_seeded_env
import wandb


def random_worker():
    wandb.init(name='random')
    np.random.seed(0)
    envs = [make_seeded_env(i, config['env_name'], 0, None, config['env_config']) for i in range(mp.cpu_count())]
    envs = SubprocVecEnv(envs)
    total_steps = 0
    episode_nb = 0
    start_time = time.time()
    states = envs.reset()
    legal_actions = envs.get_legal_actions()
    while time.time() < start_time + config['running_sec_time']:
        actions = [np.random.choice(len(legal_action), 1, p=(legal_action / legal_action.sum()))[0] for legal_action in
                   legal_actions]
        states, rewards, dones, _ = envs.step(actions)
        legal_actions = envs.get_legal_actions()
        total_steps += 1
        episode_nb += sum(dones)
    sum_best_scores = 0
    all_best_score = float('-inf')
    all_best_actions = []
    all_best_time_step = float('inf')
    for remote in envs.remotes:
        remote.send(('get_best_actions', None))
        best_score, best_actions = remote.recv()
        sum_best_scores += best_score
        if best_score > all_best_score:
            all_best_score = best_score
            all_best_actions = best_actions
        remote.send(('get_best_timestep', None))
        best_time_step = remote.recv()
        if best_time_step < all_best_time_step:
            all_best_time_step = best_time_step
    avg_best_result = sum_best_scores / len(envs.remotes)
    wandb.log({"nb_episodes": episode_nb, "avg_best_result": avg_best_result, "best_episode": all_best_score,
               "best_timestep": all_best_time_step})

if __name__ == "__main__":
    print("I have detected {} CPUs here, so I'm going to create {} actors".format(mp.cpu_count(), mp.cpu_count()))
    os.environ["WANDB_API_KEY"] = '3487a01956bf67cc7882bca2a38f70c8c95f8463'
    config = default_ppo_config.config

    fake_sweep = {
        'method': 'grid',
        'metric': {
            'name': 'avg_best_result',
            'goal': 'maximize',
        },
        'parameters': {
            'random_agent': {
                'values': [1]
            },
        }
    }

    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'avg_best_result',
            'goal': 'maximize',
        },
        'parameters': {
            'learning_rate': {
                'values': [5e-4, 1e-4, 5e-5]
            },
            'actor_layer_nb': {
                'values': [1, 2]
            },
            'actor_layer_size': {
                'values': [64, 128]
            },
            'critic_layer_size': {
                'values': [64, 128, 256]
            },
            'clipping_param': {
                'values': [0.5, 0.2, 0.1]
            },
            'entropy_regularization': {
                'values': [0, 1e-4]
            },
            'ppo_epoch': {
                'values': [2, 4]
            },
            'n_steps': {
                'values': [32, 64]
            },
            'gradient_norm_clipping': {
                'values': [1.0, 0.5]
            },
        }
    }

    sweep_id = wandb.sweep(fake_sweep, project="JSS_FCN_PPO_CPU", )
    wandb.agent(sweep_id, function=lambda: random_worker())
    sweep_id = wandb.sweep(sweep_config, project="JSS_FCN_PPO_CPU")
    wandb.agent(sweep_id,  function=lambda: ppo(config))