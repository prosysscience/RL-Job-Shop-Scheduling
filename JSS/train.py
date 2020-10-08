import ray
import wandb
from ray import tune

from ray.rllib.agents.ppo import PPOTrainer

from JSS.default_config import default_config


def train_func():
    wandb.init(config=default_config)
    config = wandb.config
    config['model']['fcnet_hiddens'] = config['fcnet_hiddens']
    config['train_batch_size'] = config['num_workers'] * config['num_envs_per_worker'] * config['rollout_fragment_length']
    ray.init()
    stop = {
        "time_total_s": 600,
    }
    tune.run(PPOTrainer, config=config, stop=stop)
    ray.shutdown()


if __name__ == "__main__":
    train_func()
