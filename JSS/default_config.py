import multiprocessing as mp

from ray.rllib.agents.ppo import ppo

default_config = ppo.DEFAULT_CONFIG.copy()
default_config.update({
    'env': 'jss_env',
    'seed': 0,
    'framework': 'torch',
    'log_level': 'WARM',
    'num_gpus': 1,
    'env_config': {
        'instance_path': '/JSS/JSS/env/instances/ta51'
    },
    'num_envs_per_worker': 2,
    'rollout_fragment_length': 1024,
    'num_workers': mp.cpu_count() - 1,
    'sgd_minibatch_size': 16112,
    'evaluation_interval': None,
    'metrics_smoothing_episodes': 100000,
    'fcnet_hiddens': [1024, 1024],
    'model':{
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model",
        "fcnet_hiddens": [1024, 1024],
    },
    'lr': 5e-5,
    'clip_param': 0.3,
    'vf_clip_param': 10.0,
    'kl_target': 0.01,
    'num_sgd_iter': 30,
    'lambda': 1.0,
})