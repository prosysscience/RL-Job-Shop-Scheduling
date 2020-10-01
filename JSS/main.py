import os
import multiprocessing as mp
import plotly.io as pio

from JSS import default_dqn_config
import wandb

from JSS.dqn import dqn

pio.orca.config.use_xvfb = True

if __name__ == "__main__":
    print("I have detected {} CPUs here, so I'm going to create {} actors".format(mp.cpu_count(), mp.cpu_count()))
    os.environ["WANDB_API_KEY"] = '3487a01956bf67cc7882bca2a38f70c8c95f8463'
    config = default_dqn_config.config

    sweep_config = {
        'program': 'dqn.py',
        'method': 'grid',
        'metric': {
            'name': 'best_timestep',
            'goal': 'minimize',
        },
        'parameters': {
            'layer_size': [1024, 2048, 4096],
            'layer_nb': [2, 3],
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="PAPER_JSS")
    wandb.agent(sweep_id, function=lambda: dqn(config))
    #sweep_id = wandb.sweep(sweep_config, project="PAPER_JSS")
    '''
    sweep_id = wandb.sweep(fake_sweep_fifo, project="PAPER_JSS")
    wandb.agent(sweep_id, function=lambda: FIFO_worker(config))
    sweep_id = wandb.sweep(fake_sweep_random, project="PAPER_JSS")
    wandb.agent(sweep_id, function=lambda: random_worker(config))
    sweep_id = wandb.sweep(sweep_config, project="PAPER_JSS")
    wandb.agent(sweep_id, function=lambda: dqn(config))
    '''