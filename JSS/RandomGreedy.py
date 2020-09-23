import gym
import numpy as np
import time
import random

import wandb

from JSS import default_dqn_config
from JSS.multiprocessing_env import SubprocVecEnv, make_seeded_env

import multiprocessing as mp


def random_worker(default_config):
    wandb.init(name='random', config=default_config)
    config = wandb.config
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    envs = [make_seeded_env(i, config['env_name'], config['seed'], None, {'instance_path': config['instance']}) for i in range(mp.cpu_count())]
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
        current_step += 1
    assert done
    '''
    figure = env_gantt.render()
    img_bytes = figure.to_image(format="png")
    image = Image.open(io.BytesIO(img_bytes))
    wandb.log({"nb_episodes": episode_nb, "avg_best_result": avg_best_result, "best_episode": all_best_score,
               "best_timestep": all_best_time_step, 'gantt': [wandb.Image(image)]})'''
    wandb.log({"nb_episodes": episode_nb, "avg_best_result": avg_best_result, "best_episode": all_best_score,
               "best_timestep": all_best_time_step})


if __name__ == "__main__":
    random_worker(default_dqn_config.config)