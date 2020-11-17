import multiprocessing as mp

default_config = {
    'env': 'jss_env',
    'seed': 0,
    'framework': 'torch',
    'log_level': 'WARN',
    'num_gpus': 1,
    'instance_path': '/JSS/JSS/env/instances/ta51',
    'num_envs_per_worker': 2,
    'rollout_fragment_length': 64,
    'num_workers': 79,
    'layer_size': 2048,
    'layer_nb': 2,
    'evaluation_interval': None,
    'metrics_smoothing_episodes': 100000, 
    # V-trace params (see vtrace_tf/torch.py).
    "vtrace": True,
    "vtrace_clip_rho_threshold": 1.0,
    "vtrace_clip_pg_rho_threshold": 1.0,
    "min_iter_time_s": 10,
    # set >1 to load data into GPUs in parallel. Increases GPU memory usage
    # proportionally with the number of buffers.
    "num_data_loader_buffers": 1,
    # how many train batches should be retained for minibatching. This conf
    # only has an effect if `num_sgd_iter > 1`.
    "minibatch_buffer_size": 38240,
    # number of passes to make over each train batch
    "num_sgd_iter": 10,
    # set >0 to enable experience replay. Saved samples will be replayed with
    # a p:1 proportion to new data samples.
    "replay_proportion": 0.05,
    # number of sample batches to store for replay. The number of transitions
    # saved total will be (replay_buffer_num_slots * rollout_fragment_length).
    "replay_buffer_num_slots": 1024,
    # max queue size for train batches feeding into the learner
    "learner_queue_size": 10240,
    # wait for train batches to be available in minibatch buffer queue
    # this many seconds. This may need to be increased e.g. when training
    # with a slow environment
    "learner_queue_timeout": 300,
    # level of queuing for sampling.
    "max_sample_requests_in_flight_per_worker": 2,
    # max number of workers to broadcast one set of weights to
    "broadcast_interval": 1,
    # use intermediate actors for multi-level aggregation. This can make sense
    # if ingesting >2GB/s of samples, or if the data requires decompression.
    "num_aggregation_workers": 0,

    # Learning params.
    "grad_clip": 1.0,
    # either "adam" or "rmsprop"
    "opt_type": "adam",
    "lr": 0.0005,
    "lr_schedule": None,
    # rmsprop considered
    "decay": 0.99,
    "momentum": 0.0,
    "epsilon": 0.1,
    # balancing the three losses
    "vf_loss_coeff": 1.0,
    "entropy_coeff": 0.0001,
    "entropy_coeff_schedule": None,

    # Callback for APPO to use to update KL, target network periodically.
    # The input to the callback is the learner fetches dict.
    "after_train_step": None,

    # Use the new "trajectory view API" to collect samples and produce
    # model- and policy inputs.
    "_use_trajectory_view_api": True,
}