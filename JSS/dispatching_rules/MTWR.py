import random
import wandb
import gym
import numpy as np

from JSS.default_config import default_config


def MTWR_worker(default_config):
    wandb.init(config=default_config)
    config = wandb.config
    env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': config['instance_path']})
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
        mask = illegal_actions * 1e8
        remaining_time += mask
        MTWR_action = np.argmin(remaining_time)
        assert legal_actions[MTWR_action]
        state, reward, done, _ = env.step(MTWR_action)
    env.reset()
    make_span = env.last_time_step
    wandb.log({"nb_episodes": 1, "make_span": make_span})


if __name__ == "__main__":
    MTWR_worker(default_config)
