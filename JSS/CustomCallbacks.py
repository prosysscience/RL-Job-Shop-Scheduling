from typing import Dict
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
import wandb


class CustomCallbacks(DefaultCallbacks):
    
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super(CustomCallbacks, self).__init__(legacy_callbacks_dict)

    def on_episode_end(self, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        env = base_env.get_unwrapped()[0]
        episode.custom_metrics['time_step'] = env.last_time_step

    def on_train_result(self, trainer, result: dict, **kwargs):
        print(result)
        my_custom_metric = result['custom_metrics']
        wandb.log(my_custom_metric)
        wandb.log({'episode_reward_max': result['episode_reward_max']})
        wandb.log({'episode_reward_min': result['episode_reward_min']})
        wandb.log({'episode_reward_mean': result['episode_reward_mean']})
        wandb.log({'episodes_total': result['episodes_total']})
        wandb.log({'training_iteration': result['training_iteration']})


