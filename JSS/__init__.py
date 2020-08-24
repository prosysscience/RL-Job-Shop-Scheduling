from gym.envs.registration import register

register(
    id='job-shop-v0',
    entry_point='JSS.env:JSS',
    kwargs={'env_config': {}}
)