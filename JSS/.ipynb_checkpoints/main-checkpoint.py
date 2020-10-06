import os
import pickle
import time

import plotly.io as pio
import ray
from ray import tune

from JSS.CustomCallbacks import CustomCallbacks
from JSS.env.JSS import JSS
from ray.rllib.agents.ppo import ppo, PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_torch
from ray.tune import CLIReporter, register_env

from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger


from JSS.models import FCMaskedActionsModel

pio.orca.config.use_xvfb = True


def env_creator(env_config):
    return JSS(env_config)


register_env("jss_env", env_creator)


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = '3487a01956bf67cc7882bca2a38f70c8c95f8463'
    ray.init()
    torch, nn = try_import_torch()
    ModelCatalog.register_custom_model("fc_masked_model", FCMaskedActionsModel)
    config = ppo.DEFAULT_CONFIG.copy()
    fcnet_architectures = [[1024, 1024], [2048, 2048]]
    config['seed'] = 0
    config['framework'] = 'torch'
    config['env'] = 'jss_env'
    config['env_config'] = {'instance_path': '/home/jovyan/JSS/JSS/env/instances/ta51'}
    config['num_envs_per_worker'] = 64
    config['num_workers'] = 79
    config['log_level'] = 'INFO'
    tune.run(PPOTrainer, config=config)
    config['model'] = {
        "fcnet_activation": tune.grid_search(["relu", "tanh"]),
        "custom_model": "fc_masked_model",
        "fcnet_hiddens": tune.grid_search(fcnet_architectures),
    }
    config['entropy_coeff'] = tune.grid_search([0.0, 1e-4])
    config['kl_coeff'] = tune.grid_search([0.3, 0.2, 0.1])
    # config['train_batch_size'] = tune.grid_search([2048, 4096])
    # config['rollout_fragment_length'] = tune.grid_search([256, 512])
    # config['sgd_minibatch_size'] = tune.grid_search([256, 512])
    config['lr'] = tune.grid_search([5e-4, 1e-4])
    config['evaluation_interval'] = None
    config['metrics_smoothing_episodes'] = 100000
    config['num_envs_per_worker'] = tune.grid_search([64, 128])
    config['callbacks'] = CustomCallbacks
    config['num_workers'] = 79
    wandb_dict = {
            'project': 'JSSRay_PPO',
            'api_key': '3487a01956bf67cc7882bca2a38f70c8c95f8463'
        }
    config["logger_config"] = {
        'wandb': wandb_dict
    }
    stop = {
        "time_total_s": 600,
    }
    reporter = CLIReporter()
    reporter.add_metric_column("episode_reward_max")
    reporter.add_metric_column("custom_metric_time_step_min")
    analysis = tune.run(PPOTrainer, config=config, stop=stop, progress_reporter=reporter,
                        fail_fast=True,
                        checkpoint_at_end=True,
                        loggers=[WandbLogger])
    best_trained_config = analysis.get_best_config(metric="episode_reward_max")
    print("Best config: ", best_trained_config)
    save_config = open("{}_{}.json".format(time.time(), config['env_config']['instance_path']), "w+")
    pickle.dump(best_trained_config, save_config)
    save_config.close()
    best_trial = analysis.get_best_trial(metric="time_step_min")
    print("Best acc reward: ", best_trial.metric_analysis['episode_reward_max']['max'])
    checkpoints = analysis.get_trial_checkpoints_paths(trial=best_trial,
                                                       metric='episode_reward_max')
    print("Checkpoint :", checkpoints)
    ray.shutdown()

