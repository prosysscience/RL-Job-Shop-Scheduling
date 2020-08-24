import default_ppo_config
from ppo import *

if __name__ == "__main__":
    config = default_ppo_config.config
    episode, score = ppo(config)
    print('\rEpisode {}\tMean current return: {:.2f}'.format(episode, score), end="")