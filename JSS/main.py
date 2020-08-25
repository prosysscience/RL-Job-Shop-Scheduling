from JSS import default_ppo_config
from JSS.ppo import ppo
import multiprocessing as mp

if __name__ == "__main__":
    print("I've detected {} cpu in this machine, I'm going to create {} actors".format(mp.cpu_count(), mp.cpu_count()))
    config = default_ppo_config.config
    episode_nb, all_best_score, avg_best_score, all_best_actions, model = ppo(config)
    print('\rEpisode {}\tAll time best score: {:.2f}\tAvg best score: {:.2f},\tAll best actions: {}'.format(episode_nb, all_best_score, avg_best_score, all_best_actions), end="")