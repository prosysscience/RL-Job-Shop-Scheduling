import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

from ray.rllib.utils.framework import try_import_tf, try_import_torch

torch, nn = try_import_torch()


from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork as FullyConnectedNetworkTF
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


tf1, tf, tfv = try_import_tf()


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

        raw_actions, _ = self.action_model({
            "obs": input_dict["obs"]["real_obs"]
        })
        return raw_actions.masked_fill(action_mask == 0, torch.finfo(torch.float).min), state

    def value_function(self):
        return self.action_model.value_function()

class FCMaskedActionsModelTF(DistributionalQTFModel, TFModelV2):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):
        super(FCMaskedActionsModelTF, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        true_obs_space = gym.spaces.MultiBinary(n=obs_space.shape[0] - action_space.n)
        self.action_embed_model = FullyConnectedNetworkTF(
            true_obs_space, action_space, action_space.n,
            model_config, name + "_action_embed")
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        raw_actions, _ = self.action_embed_model({
            "obs": input_dict["obs"]["real_obs"]
        })
        return tf.where(action_mask == 1, tf.float32.min, raw_actions), state

    def value_function(self):
        return self.action_embed_model.value_function()