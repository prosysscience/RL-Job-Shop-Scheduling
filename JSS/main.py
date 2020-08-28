import multiprocessing as mp
import ray
from ray.tune import CLIReporter

from JSS import default_ppo_config
from JSS.ppo import ppo
from ray import tune

if __name__ == "__main__":
    ray.init(local_mode=True, num_cpus=1)
    config = default_ppo_config.config
    config['learning_rate'] = tune.grid_search([5e-4, 1e-4, 5e-5])
    config['actor_config'] = tune.grid_search([[64, 64], [128, 128], [256, 256]])
    config['critic_config'] = tune.grid_search([[256, 256], [512, 512]])
    config['n_steps'] = 8
    config['clipping_param'] = tune.grid_search([0.3, 0.2, 0.1])
    config['entropy_regularization'] = tune.grid_search([0, 1e-4])
    reporter = CLIReporter(max_progress_rows=15)
    reporter.add_metric_column("avg_best_result")
    reporter.add_metric_column("best_episode")
    reporter.add_metric_column("nb_episodes")
    analysis = tune.run(
        ppo,
        config=config,
        progress_reporter=reporter,
        fail_fast=True,
        checkpoint_at_end=True)
    best_trained_config = analysis.get_best_config(metric="avg_best_result")
    print("Best config: ", best_trained_config)
    best_trial = analysis.get_best_trial(metric="episode_reward_max")
    print("Best acc reward: ", best_trial.metric_analysis['episode_reward_max']['max'])
    checkpoints = analysis.get_trial_checkpoints_paths(trial=best_trial,
                                                       metric='episode_reward_max')
    print("Checkpoint :", checkpoints)
    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()
    df.to_csv('results.csv', index=False, header=True)
    #print('\rEpisode {}\tAll time best score: {:.2f}\tAvg best score: {:.2f},\tAll best actions: {}'.format(episode_nb, all_best_score, avg_best_score, all_best_actions), end="")
