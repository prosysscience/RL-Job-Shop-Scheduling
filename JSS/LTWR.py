import random
import numpy as np

import wandb

from JSS.default_config import default_config
from JSS.env import JSS
from JSS.env_wrapper import BestActionsWrapper


def LTWR_worker(default_config):
    wandb.init(name='LTWR', config=default_config)
    config = wandb.config
    env = BestActionsWrapper(JSS(env_config={'instance_path': config['instance_path']}))
    env.seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    done = False
    state = env.reset()
    while not done:
        real_state = np.copy(state['real_obs'])
        legal_actions = state['action_mask'][:-1]
        reshaped = np.reshape(real_state, (env.jobs, 7))
        remaining_time = (reshaped[:, 3] * env.max_time_jobs) / env.jobs_length
        illegal_actions = np.invert(legal_actions)
        mask = illegal_actions * -1e8
        remaining_time += mask
        LTWR_action = np.argmax(remaining_time)
        assert legal_actions[LTWR_action]
        state, reward, done, _ = env.step(LTWR_action)
    env.reset()
    all_best_score = env.best_score
    all_best_actions = env.best_actions
    all_best_time_step = env.best_time_step
    wandb.log({"nb_episodes": 1, "avg_best_result": all_best_score, "best_episode": all_best_score,
               "best_timestep": all_best_time_step})

if __name__ == "__main__":
    LTWR_worker(default_config)