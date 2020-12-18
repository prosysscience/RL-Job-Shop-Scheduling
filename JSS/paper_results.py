import os
import time

import ray
import wandb

import ray.tune.integration.wandb as wandb_tune

from ray.rllib.agents.ppo import PPOTrainer

from JSS.CustomCallbacks import CustomCallbacks

from typing import Dict, Tuple

import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from JSS.env.JSS import JSS

from JSS.models import FCMaskedActionsModelTF
from ray.tune.utils import flatten_dict


def env_creator(env_config):
    return JSS(env_config)


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
        'instance_path': '/home/jupyter/JSS/JSS/env/instances/ta41',
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 20000,
        'gamma': 1.0,
        'num_workers': mp.cpu_count(),
        'layer_nb': 2,
        'train_batch_size': 80 * 4 * 704,
        'num_envs_per_worker': 4,
        'rollout_fragment_length': 704,  # TO TUNE
        'sgd_minibatch_size': 33000,
        'layer_size': 470,
        'lr': 0.000479,  # TO TUNE
        'lr_start': 0.000479,  # TO TUNE
        'lr_end': 0.00002832,  # TO TUNE
        'clip_param': 0.5337,  # TO TUNE
        'vf_clip_param': 18.0,  # TO TUNE
        'num_sgd_iter': 10,  # TO TUNE
        "vf_loss_coeff": 0.75,
        "kl_coeff": 0.15,
        'kl_target': 0.1202,  # TO TUNE
        'lambda': 1.0,
        'entropy_coeff': 0.0,  # TUNE LATER
        "batch_mode": "truncate_episodes",
        "grad_clip": None,
        "use_critic": True,
        "use_gae": True,
        "shuffle_sequences": True,
        "vf_share_layers": False,
        "observation_filter": "NoFilter",
        "simple_optimizer": False,
        "_fake_gpus": False,
    }
    wandb.init(config=default_config, group='10Minutes', project='PPOJss')
    ray.init()

    config = wandb.config

    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    config['model'] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        'fcnet_hiddens': [config['layer_size'] for k in range(config['layer_nb'])],
        "vf_share_layers": False,
    }
    config['env_config'] = {
        'instance_path': config['instance_path']
    }

    config = with_common_config(config)
    config['callbacks'] = CustomCallbacks
    config['train_batch_size'] = config['sgd_minibatch_size']

    config['lr'] = config['lr_start']
    config['lr_schedule'] = [[0, config['lr_start']], [15000000, config['lr_end']]]

    config.pop('instance_path', None)
    config.pop('layer_size', None)
    config.pop('layer_nb', None)
    config.pop('lr_start', None)
    config.pop('lr_end', None)

    stop = {
        "time_total_s": 1000,
    }

    start_time = time.time()
    trainer = PPOTrainer(config=config)
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        result = wandb_tune._clean_log(result)
        log, config_update = _handle_result(result)
        wandb.log(log)
        #wandb.config.update(config_update, allow_val_change=True)

    ray.shutdown()


if __name__ == "__main__":
    train_func()
