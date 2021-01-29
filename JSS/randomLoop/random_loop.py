import ray
import wandb
from ray import tune

from JSS.CustomCallbacks import CustomCallbacks
from JSS.randomLoop.RandomRLLib import RandomMaskedTrainer

from JSS.default_config import default_config
from ray.rllib.agents import with_common_config

import multiprocessing as mp


def rand_func():
    wandb.init(config={})
    config = with_common_config(wandb.config)
    config['evaluation_num_episodes'] = 10
    config['num_workers'] = mp.cpu_count() - 1
    config['num_envs_per_worker'] = 4
    config['metrics_smoothing_episodes'] = 2000
    config['observation_filter'] = 'NoFilter'
    config['env'] = default_config['env']
    config['env_config'] = {
        'env_config': {'instance_path': config['instance_path']}
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
