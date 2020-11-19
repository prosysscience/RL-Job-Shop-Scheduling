import ray
import wandb

import numpy as np

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.suggest.optuna import OptunaSearch

from JSS.CustomCallbacks import CustomCallbacks

import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from JSS.env_wrapper import BestActionsWrapper
from JSS.env.JSS import JSS

from JSS.models import FCMaskedActionsModel

from ray.tune.integration.wandb import WandbLogger

from ray.tune.suggest.bayesopt import BayesOptSearch


def env_creator(env_config):
    return BestActionsWrapper(JSS(env_config))


register_env("jss_env", env_creator)


def train_func():
    ModelCatalog.register_custom_model("fc_masked_model", FCMaskedActionsModel)

    config = {
        'env': 'jss_env',
        'seed': 0,
        'framework': 'torch',
        'log_level': 'WARN',
        'num_gpus': 0,
        'instance_path': '/home/local/IWAS/pierre/PycharmProjects/JSS/JSS/env/instances/ta51',
        'num_envs_per_worker': 2,
        'rollout_fragment_length': 1024,
        'num_workers': mp.cpu_count() - 1,
        'sgd_minibatch_size': 256,
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 1000,
        'gamma': 1.0,
        'layer_size': 1024,
        'layer_nb': 2,
        'lr_start': tune.uniform(5e-5, 1e-4),
        'lr_end': tune.uniform(3e-5, 5e-5),
        'clip_param': tune.uniform(0.2, 0.4),
        'vf_clip_param': tune.uniform(5.0, 15.0),
        'kl_target': tune.uniform(0.005, 0.2),
        'num_sgd_iter': 25,
        'lambda': tune.uniform(0.8, 1.0),
        "use_critic": True,
        "use_gae": True,
        "kl_coeff": tune.uniform(0.2, 0.4),
        "shuffle_sequences": True,
        "vf_share_layers": False,
        "vf_loss_coeff": tune.uniform(0.5, 1.5),
        "entropy_coeff": 5e-4,
        "entropy_coeff_schedule": None,
        "grad_clip": None,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "simple_optimizer": False,
        "_fake_gpus": False,
        "logger_config": {
            "wandb": {
                "project": "JSS_WANDB_PPO",
                "api_key": "3487a01956bf67cc7882bca2a38f70c8c95f8463"
            }
        },
    }

    config['model'] = {
        "fcnet_activation": "tanh",
        "custom_model": "fc_masked_model",
        'fcnet_hiddens': [config['layer_size'] for k in range(config['layer_nb'])],
    }
    config['env_config'] = {
        'instance_path': config['instance_path']
    }

    config['train_batch_size'] = config['num_workers'] * config['num_envs_per_worker'] * config['rollout_fragment_length']
    config = with_common_config(config)
    config['callbacks'] = CustomCallbacks

    config['lr'] = config['lr_start']
    config['lr_schedule'] = [[0, config['lr_start']], [4000000, config['lr_end']]]

    config.pop('lr_start', None)
    config.pop('lr_end', None)
    config.pop('instance_path', None)
    config.pop('layer_size', None)
    config.pop('layer_nb', None)

    ray.init()

    stop = {
        "episodes_total": 50,
    }

    analysis = tune.run(PPOTrainer,
                        config=config,
                        stop=stop,
                        metric='episode_reward_mean',
                        mode='max',
                        num_samples=1000,
                        name="ppo-jss",
                        checkpoint_freq=10,
                        loggers=[WandbLogger])
    wandb.log(analysis.best_config)

    ray.shutdown()


if __name__ == "__main__":
    train_func()
