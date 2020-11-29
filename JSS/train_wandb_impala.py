import os
import time

import ray
import wandb

import ray.tune.integration.wandb as wandb_tune

from ray.rllib.agents.impala import ImpalaTrainer

from JSS.CustomCallbacks import CustomCallbacks

from typing import Callable, Dict, List, Tuple


import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from JSS.env_wrapper import BestActionsWrapper
from JSS.env.JSS import JSS

from JSS.models import FCMaskedActionsModelV1, FCMaskedActionsModelV2
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
    #os.environ["NVIDIA_VISIBLE_DEVICES"] = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = 2
    default_config = {
        'env': 'jss_env',
        'seed': 0,
        'framework': 'torch',
        'log_level': 'WARN',
        'num_gpus': 0,
        'instance_path': '/JSS/JSS/env/instances/ta51',
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 1000,
        'gamma': 1.0,
        'num_workers': mp.cpu_count(),
        'layer_nb': 2,
        "min_iter_time_s": 0,
        # set >1 to load data into GPUs in parallel. Increases GPU memory usage
        # proportionally with the number of buffers.
        "num_data_loader_buffers": 1,
        # V-trace params (see vtrace_tf/torch.py).
        "vtrace": True,
        "vtrace_clip_rho_threshold": 1.0,
        "vtrace_clip_pg_rho_threshold": 1.0,
        # System params.
        #
        # == Overview of data flow in IMPALA ==
        # 1. Policy evaluation in parallel across `num_workers` actors produces
        #    batches of size `rollout_fragment_length * num_envs_per_worker`.
        # 2. If enabled, the replay buffer stores and produces batches of size
        #    `rollout_fragment_length * num_envs_per_worker`.
        # 3. If enabled, the minibatch ring buffer stores and replays batches of
        #    size `train_batch_size` up to `num_sgd_iter` times per batch.
        # 4. The learner thread executes data parallel SGD across `num_gpus` GPUs
        #    on batches of size `train_batch_size`.
        #
        "rollout_fragment_length": 1024,
        "train_batch_size": 48000,
        # how many train batches should be retained for minibatching. This conf
        # only has an effect if `num_sgd_iter > 1`.
        "minibatch_buffer_size": 8196,
        # number of passes to make over each train batch
        "num_sgd_iter": 15,
        # set >0 to enable experience replay. Saved samples will be replayed with
        # a p:1 proportion to new data samples.
        "replay_proportion": 0.1,
        # number of sample batches to store for replay. The number of transitions
        # saved total will be (replay_buffer_num_slots * rollout_fragment_length).
        "replay_buffer_num_slots": mp.cpu_count(),
        # max queue size for train batches feeding into the learner
        "learner_queue_size": 16,
        # wait for train batches to be available in minibatch buffer queue
        # this many seconds. This may need to be increased e.g. when training
        # with a slow environment
        "learner_queue_timeout": 30,
        # level of queuing for sampling.
        "max_sample_requests_in_flight_per_worker": 2,
        # max number of workers to broadcast one set of weights to
        "broadcast_interval": 1,
        # use intermediate actors for multi-level aggregation. This can make sense
        # if ingesting >2GB/s of samples, or if the data requires decompression.
        "num_aggregation_workers": 0,

        # Learning params.
        "grad_clip": 40.0,
        # either "adam" or "rmsprop"
        "opt_type": "adam",
        "lr": 0.0005,
        "lr_schedule": None,
        # rmsprop considered
        "decay": 0.99,
        "momentum": 0.0,
        "epsilon": 0.1,
        # balancing the three losses
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01,
        "entropy_coeff_schedule": None,

        # Callback for APPO to use to update KL, target network periodically.
        # The input to the callback is the learner fetches dict.
        "after_train_step": None,

        # Use the new "trajectory view API" to collect samples and produce
        # model- and policy inputs.
        "_use_trajectory_view_api": False,
    }

    wandb.init(config=default_config)
    ray.init()

    config = wandb.config

    ModelCatalog.register_custom_model("fc_masked_model_v1", FCMaskedActionsModelV1)
    ModelCatalog.register_custom_model("fc_masked_model_v2", FCMaskedActionsModelV2)

    config['model'] = {
        "fcnet_activation": config["activation_fn"],
        "custom_model": config["model_net"],
        'fcnet_hiddens': [config['layer_size'] for k in range(config['layer_nb'])],
    }
    config['env_config'] = {
        'instance_path': config['instance_path']
    }

    config['train_batch_size'] = config['num_workers'] * config['num_envs_per_worker'] * config['rollout_fragment_length']
    config = with_common_config(config)
    config['callbacks'] = CustomCallbacks
    config['lr'] = config['lr_start']
    config['lr_schedule'] = [[0, config['lr_start']], [10000000, config['lr_end']]]
    config['entropy_coeff'] = config['entropy_coeff_start']
    config['entropy_coeff_schedule'] = [[0, config['entropy_coeff_start']], [10000000, config['entropy_coeff_end']]]

    config.pop('instance_path', None)
    config.pop('layer_size', None)
    config.pop('layer_nb', None)
    config.pop('lr_start', None)
    config.pop('lr_end', None)
    config.pop('entropy_coeff_start', None)
    config.pop('entropy_coeff_end', None)
    config.pop('activation_fn', None)
    config.pop('model_net', None)

    stop = {
        "time_total_s": 1000,
    }

    start_time = time.time()
    trainer = ImpalaTrainer(config=config)
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()
        result = wandb_tune._clean_log(result)
        log, config_update = _handle_result(result)
        wandb.log(log)
        wandb.config.update(config_update, allow_val_change=True)

    ray.shutdown()


if __name__ == "__main__":
    train_func()
