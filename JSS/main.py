import os
import multiprocessing as mp

import plotly.io as pio
import ray
from ray import tune

from JSS import train
from JSS.CustomCallbacks import CustomCallbacks
from JSS.env.JSS import JSS
from ray.rllib.agents.ppo import ppo, PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_torch
from ray.tune import CLIReporter, register_env

from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger

from JSS.env_wrapper import BestActionsWrapper

from JSS.models import FCMaskedActionsModel

pio.orca.config.use_xvfb = True


torch, nn = try_import_torch()
ModelCatalog.register_custom_model("fc_masked_model", FCMaskedActionsModel)

if __name__ == "__main__":
    print("I have detected {} CPUs here, so I'm going to create {} actors".format(mp.cpu_count(), mp.cpu_count() - 1))
    os.environ["WANDB_API_KEY"] = '3487a01956bf67cc7882bca2a38f70c8c95f8463'
    sweep_config = {
        'program': 'train.py',
        'method': 'grid',
        'metric': {
            'name': 'time_step_min',
            'goal': 'minimize',
        },
        'parameters': {
            'num_envs_per_worker': {
                'values': [2, 4]
            },
            'sgd_minibatch_size': {
                'values': [2**12, 2**13, 2**14, 2**15]
            },
            'lr': {
                'values': [1e-4, 5e-5, 1e-5]
            },
            'lambda': {
                'values': [0.95, 1.0]
            },
            'clip_param': {
                'values': [0.2, 0.3, 0.4]
            },
            'num_sgd_iter': {
                'values': [20, 30, 40]
            },
        }
    }

