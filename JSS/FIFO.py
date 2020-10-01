import random
import numpy as np

import gym
import wandb

from JSS import default_dqn_config
from JSS.env_wrapper import BestActionsWrapper


def FIFO_worker(default_config):
    wandb.init(name='FIFO', config=default_config)
    config = wandb.config
    env = BestActionsWrapper(gym.make(config['env_name'], env_config={'instance_path': config['instances']}))
    env.seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    done = False
    states = env.reset()
    legal_actions = env.get_legal_actions()
    while not done:
        waiting_time = np.reshape(states, (env.jobs, 7))[:, 5]
        illegal_actions = np.invert(legal_actions)
        mask = illegal_actions[:-1] * -1e8
        waiting_time += mask
        fifo_action = np.argmax(waiting_time)
        assert legal_actions[fifo_action]
        state, reward, done, _ = env.step(fifo_action)
        legal_actions = env.get_legal_actions()

    env.reset()
    all_best_score = env.best_score
    all_best_actions = env.best_actions
    all_best_time_step = env.best_time_step
    wandb.log({"nb_episodes": 1, "avg_best_result": all_best_score, "best_episode": all_best_score,
               "best_timestep": all_best_time_step})

if __name__ == "__main__":
    FIFO_worker(default_dqn_config.config)