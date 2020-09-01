import os
import multiprocessing as mp
import numpy as np

from JSS import default_ppo_config
from JSS.env import JSS
from JSS.ppo import ppo
import wandb


if __name__ == "__main__":
    print("I have detected {} CPUs here, so I'm going to create {} actors".format(mp.cpu_count(), 2 * mp.cpu_count()))
    os.environ["WANDB_API_KEY"] = '3487a01956bf67cc7882bca2a38f70c8c95f8463'
    '''
    env = JSS()
    done = False
    state = env.reset()
    legal_actions = state['action_mask']
    while not done:
        proba = legal_actions/np.sum(legal_actions)
        action = np.random.choice(env.action_space.n, 1, p=proba)[0]
        state, reward, done, infos = env.step(action)
        legal_actions = state['action_mask']
        '''
    config = default_ppo_config.config
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
    sweep_id = wandb.sweep(sweep_config, project="JSS_FCN_PPO_CPU")
    wandb.agent(sweep_id,  function=lambda: ppo(config))
    '''
    all_configs = generate_variants(config)
    best_avg = float('-inf')
    best_config = {}
    for candidate_config in all_configs:
        fix_config = candidate_config[1]
        grid_config = candidate_config[0]
        for attribute in grid_config.keys():
            fix_config[attribute[0]] = grid_config[attribute]
        wandb.config.update(fix_config)
        episode_nb, all_best_score, avg_best_result, all_best_actions = ppo(fix_config)
        if best_avg < avg_best_result:
            best_config = candidate_config
    print("the best config is:\n{}".format(best_config))'''
    #print('\rEpisode {}\tAll time best score: {:.2f}\tAvg best score: {:.2f},\tAll best actions: {}'.format(episode_nb, all_best_score, avg_best_score, all_best_actions), end="")
