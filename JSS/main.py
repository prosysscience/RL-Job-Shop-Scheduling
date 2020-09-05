import os
import multiprocessing as mp
import time
from PIL import Image
import io

import gym
import numpy as np

from JSS import default_ppo_config, default_dqn_config
from JSS.dqn import dqn
from JSS.multiprocessing_env import SubprocVecEnv
from JSS.ppo import ppo, make_seeded_env
import wandb


def random_worker(config):
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
    env_gantt = gym.make(config['env_name'], env_config=config['env_config'])
    state = env_gantt.reset()
    done = False
    legal_actions = env_gantt.get_legal_actions()
    current_step = 0
    # we can't just iterate throught all the actions because of the automatic action taking
    while current_step < len(all_best_actions):
        action = all_best_actions[current_step]
        assert legal_actions[action]
        state, reward, done, action_performed = env_gantt.step(action)
        current_step += len(action_performed)
    assert done
    figure = env_gantt.render()
    img_bytes = figure.to_image(format="png")
    image = Image.open(io.BytesIO(img_bytes))
    wandb.log({"nb_episodes": episode_nb, "avg_best_result": avg_best_result, "best_episode": all_best_score,
               "best_timestep": all_best_time_step, 'gantt': [wandb.Image(image)]})


if __name__ == "__main__":
    print("I have detected {} CPUs here, so I'm going to create {} actors".format(mp.cpu_count(), mp.cpu_count()))
    os.environ["WANDB_API_KEY"] = '3487a01956bf67cc7882bca2a38f70c8c95f8463'
    config = default_dqn_config.config

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
            'clipping_gradient': {
                'values': [1.0, 10.0]
            },
            'update_network_step': {
                'values': [8, 16]
            },
            'batch_size': {
                'values': [32, 64]
            },
            'tau': {
                'values': [1e-3, 1e-4]
            },
            'nb_steps': {
                'values': [4, 8, 16]
            },
            'layer_size': {
                'values': [64, 128, 256]
            },
        }
    }
    sweep_id = wandb.sweep(fake_sweep, project="JSS_FCN_DQN_CPU")
    wandb.agent(sweep_id, function=lambda: random_worker(config))
    sweep_id = wandb.sweep(sweep_config, project="JSS_FCN_DQN_CPU")
    wandb.agent(sweep_id,  function=lambda: dqn(config))