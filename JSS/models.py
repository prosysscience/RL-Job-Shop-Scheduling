import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class FCMaskedActionsModelV1(TorchModelV2, nn.Module):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        nn.Module.__init__(self)
        super(FCMaskedActionsModelV1, self).__init__(obs_space, action_space, action_space.n, model_config, name)
        true_obs_space = gym.spaces.MultiBinary(n=obs_space.shape[0] - action_space.n)
        self.action_model = FullyConnectedNetwork(
            obs_space=true_obs_space, action_space=action_space, num_outputs=action_space.n,
            model_config=model_config, name=name + "action_model")

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        # Compute the predicted action embedding

        raw_actions, state = self.action_model({
            "obs": input_dict["obs"]["real_obs"]
        })
        return raw_actions.masked_fill(action_mask == 0, torch.finfo(torch.float).min), state

    def value_function(self):
        return self.action_model.value_function()


class FCMaskedActionsModelV2(TorchModelV2, nn.Module):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        nn.Module.__init__(self)
        super(FCMaskedActionsModelV2, self).__init__(obs_space, action_space, action_space.n, model_config, name)
        true_obs_space = gym.spaces.MultiBinary(n=obs_space.shape[0] - action_space.n)
        self.action_model = FullyConnectedNetwork(
            obs_space=true_obs_space, action_space=action_space, num_outputs=action_space.n,
            model_config=model_config, name=name + "action_model")

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        # Compute the predicted action embedding

        raw_actions, state = self.action_model({
            "obs": input_dict["obs"]["real_obs"]
        })
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e10)

        return raw_actions + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()