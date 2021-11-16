from typing import Dict

from ray.experimental.client import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

@ray.remote
class Storage:
    def __init__(self):
      self.best_makespan = float('inf')
      self.best_solution = None

    def save_sol(self, makespan, solution):
        if makespan < self.best_makespan:
            self.best_solution = solution
            self.best_makespan = makespan

    def get_best_solution(self):
        return self.best_makespan, self.best_solution


class CustomCallbacks(DefaultCallbacks):
    
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super(CustomCallbacks, self).__init__(legacy_callbacks_dict)

    def on_episode_end(self, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        env = base_env.get_unwrapped()[0]
        if env.last_time_step != float('inf'):
            storage = ray.get_actor("global_storage")
            storage.save_sol.remote(env.last_time_step, env.last_solution)
            episode.custom_metrics['make_span'] = env.last_time_step

