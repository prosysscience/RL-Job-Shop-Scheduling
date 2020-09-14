from gym.envs.registration import register
from JSS.default_dqn_config import config

from JSS.PrioritizedReplayBuffer import PrioritizedReplayBuffer, Experience

register(
    id='job-shop-v0',
    entry_point='JSS.env:JSS',
    kwargs={'env_config': {}}
)