import numpy as np

from ray.rllib import Policy


class RandomLegalPolicy(Policy):
    """Just pick a random legal action"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        def pick_legal_action(x):
            legal_action = x['action_mask']
            return np.random.choice(len(legal_action), 1, p=(legal_action / legal_action.sum()))[0]

        return [pick_legal_action(x) for x in obs_batch], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
