from JSS import default_ppo_config
from JSS.ppo import ppo

if __name__ == "__main__":
    config = default_ppo_config.config
    episode_nb, all_best_score, avg_best_score, all_best_actions, model = ppo(config)
    print('\rEpisode {}\tAll time best score: {:.2f}\tAvg best score: {:.2f},\tAll best actions: {}'.format(episode_nb, all_best_score, avg_best_score, all_best_actions), end="")