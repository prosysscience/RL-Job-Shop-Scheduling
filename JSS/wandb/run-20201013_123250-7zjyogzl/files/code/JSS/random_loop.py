import ray
import wandb
from ray import tune

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.trainer import COMMON_CONFIG

from JSS.CustomCallbacks import CustomCallbacks
from JSS.RandomRLLib import RandomMaskedTrainer

from JSS.default_config import default_config
from ray.rllib.agents import with_common_config
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.tune import CLIReporter, register_env
from JSS.env_wrapper import BestActionsWrapper
from JSS.env.JSS import JSS

import multiprocessing as mp

from JSS.models import FCMaskedActionsModel


def env_creator(env_config):
    return BestActionsWrapper(JSS(env_config))


register_env("jss_env", env_creator)


def rand_func():
    torch, nn = try_import_torch()
    wandb.init(config={})
    config = with_common_config(wandb.config)

    config['evaluation_num_episodes'] = 1000
    config['num_workers'] = mp.cpu_count() - 1
    config['num_envs_per_worker'] = 3
    config['metrics_smoothing_episodes'] = 999999999
    config['observation_filter'] = 'NoFilter'
    config['env'] = default_config['env']
    config['env_config'] = {
        'instance_path': default_config['instance_path']
    }
    config.pop('instance_path', None)

    config['callbacks'] = CustomCallbacks

    ray.init()

    stop = {
        "time_total_s": 600,
    }

    analysis = tune.run(RandomMaskedTrainer, config=config, stop=stop, name="ppo-jss")
    result = analysis.results_df.to_dict('index')
    last_run_id = list(result.keys())[0]
    result = result[last_run_id]
    wandb.log({'time_step_min': result['custom_metrics.time_step_min']})
    if result['custom_metrics.time_step_max'] != float('inf'):
        wandb.log({'time_step_max': result['custom_metrics.time_step_max']})
        wandb.log({'time_step_mean': result['custom_metrics.time_step_mean']})
    wandb.log({'episode_reward_max': result['episode_reward_max']})
    wandb.log({'episode_reward_min': result['episode_reward_min']})
    wandb.log({'episode_reward_mean': result['episode_reward_mean']})
    wandb.log({'episodes_total': result['episodes_total']})
    wandb.log({'training_iteration': result['training_iteration']})

    ray.shutdown()


if __name__ == "__main__":
    rand_func()
