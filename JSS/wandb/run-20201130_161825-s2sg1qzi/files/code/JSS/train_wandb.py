import os
import time

import ray
import wandb

import ray.tune.integration.wandb as wandb_tune

from ray.rllib.agents.ppo import PPOTrainer

from JSS.CustomCallbacks import CustomCallbacks

from typing import Callable, Dict, List, Tuple


import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from JSS.env_wrapper import BestActionsWrapper
from JSS.env.JSS import JSS

from JSS.models import FCMaskedActionsModelTF
from ray.tune.utils import flatten_dict

def env_creator(env_config):
    return BestActionsWrapper(JSS(env_config))


register_env("jss_env", env_creator)

_exclude_results = ["done", "should_checkpoint", "config"]

# Use these result keys to update `wandb.config`
_config_results = [
        "trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname",
        "pid", "date",
]


def _handle_result(result: Dict) -> Tuple[Dict, Dict]:
    config_update = result.get("config", {}).copy()
    log = {}
    flat_result = flatten_dict(result, delimiter="/")

    for k, v in flat_result.items():
        if any(
                k.startswith(item + "/") or k == item
                for item in _config_results):
            config_update[k] = v
        elif any(
                k.startswith(item + "/") or k == item
                for item in _exclude_results):
            continue
        elif not wandb_tune._is_allowed_type(v):
            continue
        else:
            log[k] = v

    config_update.pop("callbacks", None)  # Remove callbacks
    return log, config_update


def train_func():
    default_config = {
        'env': 'jss_env',
        'seed': 0,
        'framework': 'tf',
        'log_level': 'WARN',
        'num_gpus': 1,
        'instance_path': '/JSS/JSS/env/instances/ta51',
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 1000,
        'gamma': 1.0,
        'num_workers': mp.cpu_count(),
        'layer_nb': 2,
        'train_batch_size': 32001,
        'num_envs_per_worker': 2, # TO TUNE
        'rollout_fragment_length': 1064, # TO TUNE
        'sgd_minibatch_size': 5443, # TO TUNE
        'layer_size': 853, # TO TUNE
        'lr': 0.0001798, # TO TUNE
        'entropy_coeff': 0.00007394, # TO TUNE
        'clip_param': 0.2604, # TO TUNE
        'vf_clip_param': 20.0, # TO TUNE
        'kl_target': 0.2088, # TO TUNE
        'num_sgd_iter': 22, # TO TUNE
        "vf_loss_coeff": 0.9394, # TO TUNE
        "kl_coeff": 0.5021, # TO TUNE
        "batch_mode": "truncate_episodes", # TO TUNE
        'lambda': 1.0,
        "grad_clip": None,
        "use_critic": True,
        "use_gae": True,
        "shuffle_sequences": True,
        "vf_share_layers": False,
        "observation_filter": "NoFilter",
        "simple_optimizer": False,
        "_fake_gpus": False,
    }

    wandb.init(config=default_config)
    ray.init()

    config = wandb.config

    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    config['model'] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        'fcnet_hiddens': [config['layer_size'] for k in range(config['layer_nb'])],
    }
    config['env_config'] = {
        'instance_path': config['instance_path']
    }

    config = with_common_config(config)
    config['callbacks'] = CustomCallbacks

    config.pop('instance_path', None)
    config.pop('layer_size', None)
    config.pop('layer_nb', None)

    stop = {
        "time_total_s": 600,
    }

    start_time = time.time()
    trainer = PPOTrainer(config=config)
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        result = wandb_tune._clean_log(result)
        log, config_update = _handle_result(result)
        wandb.log(log)
        wandb.config.update(config_update, allow_val_change=True)

    ray.shutdown()


if __name__ == "__main__":
    train_func()
