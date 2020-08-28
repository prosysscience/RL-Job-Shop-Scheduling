import multiprocessing as mp

config = {
    'seed': 0,
    'learning_rate': 1e-4,
    'n_steps': 128,
    'tau': 0.98,
    'gamma': 0.999,
    'running_sec_time': 10 * 60, # 10 minutes
    'max_steps_per_episode': None,  # None if we don't want to limit the number of steps
    'value_coefficient': 1.0,
    'entropy_regularization': 1e-4,
    #'nb_actors': 32,
    'nb_actors': 2 * mp.cpu_count(),
    'env_name': 'job-shop-v0',
    'ppo_epoch': 10,
    'clipping_param': 0.2,
    'clipping_param_vf': None,  # None to avoid clipping the value estimation
    'minibatch_size': 32,
    'gradient_norm_clipping': 0.5,
    'max_kl_div': None,
    'actor_layer_nb' : 1,
    'actor_layer_size': 64,
    'critic_layer_nb' : 2,
    'critic_layer_size': 64,
    'env_config': {'instance_path': '/home/local/IWAS/pierre/PycharmProjects/JSS/JSS/env/instances/ta51'},
}
