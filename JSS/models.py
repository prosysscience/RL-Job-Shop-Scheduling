import gym
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel

from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

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
        self.action_embed_model = FullyConnectedNetwork(
            obs_space=true_obs_space, action_space=action_space, num_outputs=action_space.n,
            model_config=model_config, name=name + "action_model")
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        raw_actions, _ = self.action_embed_model({
            "obs": input_dict["obs"]["real_obs"]
        })
        #inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        logits = tf.where(tf.math.equal(action_mask, 1), raw_actions,  tf.float32.min)
        return logits, state

    def value_function(self):
        return self.action_embed_model.value_function()

class FCMaskedActionsModelTorch(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        true_obs_space = gym.spaces.MultiBinary(n=obs_space.shape[0] - action_space.n)
        self.action_embed_model = TorchFC(
            obs_space=true_obs_space, action_space=action_space, num_outputs=action_space.n,
            model_config=model_config, name=name + "action_model")

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        raw_actions, _ = self.action_embed_model({
            "obs": input_dict["obs"]["real_obs"]
        })
        # inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return raw_actions.masked_fill_(action_mask == 0, FLOAT_MIN), state

    def value_function(self):
        return self.action_embed_model.value_function()
