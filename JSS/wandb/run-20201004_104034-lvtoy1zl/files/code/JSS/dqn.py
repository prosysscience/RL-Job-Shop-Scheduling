from collections import OrderedDict

import os
import gym
import torch
import multiprocessing as mp
import random
import time
import numpy as np
import io
import torch.nn as nn
import torch.optim as optim
import wandb
from PIL import Image
import plotly.io as pio

from JSS import default_dqn_config
from JSS.env_wrapper import BestActionsWrapper, MaxStepWrapper
from JSS.multiprocessing_env import SubprocVecEnv

pio.orca.config.use_xvfb = True

from JSS import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # We train on a GPU if available

loss_fn = nn.SmoothL1Loss()


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, config):
        super(QNetwork, self).__init__()
        self.action_size = action_size
        self.layer1 = nn.Linear(state_size, 64)

        self.common_layer = []
        self.layers_advantage = []
        self.layers_value = []
        input_size = state_size
        for i, layer_config in enumerate(config):
            if i == 0:
                layer = nn.Linear(input_size, layer_config)
                torch.nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                self.common_layer.append(('layer_{}'.format(i), layer))
                self.common_layer.append(('tanh_{}'.format(i), nn.Tanh()))
            else:
                layer = nn.Linear(input_size, layer_config)
                torch.nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                self.layers_advantage.append(('layer_{}_advantage'.format(i), layer))
                self.layers_advantage.append(('tanh_{}_advantage'.format(i), nn.Tanh()))

                layer = nn.Linear(input_size, layer_config)
                torch.nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                self.layers_value.append(('layer_{}_value'.format(i), layer))
                self.layers_value.append(('tanh_{}_value'.format(i), nn.Tanh()))
            input_size = layer_config

        layer = nn.Linear(input_size, action_size)
        torch.nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh') * 0.1)
        self.layers_advantage.append(('layer_{}_advantage'.format(len(config)), layer))
        self.advantage_model = nn.Sequential(OrderedDict(self.layers_advantage))

        layer = nn.Linear(input_size, 1)
        torch.nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh') * 0.1)
        self.layers_value.append(('layer_{}_value'.format(len(config)), layer))
        self.value_model = nn.Sequential(OrderedDict(self.layers_value))

        self.common_model = nn.Sequential(OrderedDict(self.common_layer))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = self.common_model(state)

        value = self.value_model(state)
        adv = self.advantage_model(state)

        x = value + adv - adv.mean()
        return x


def make_seeded_env(i: int, env_name: str, seed: int, max_steps_per_episode: int, env_config: dict = {}):
    def _anon():
        if max_steps_per_episode is None:
            env = BestActionsWrapper(gym.make(env_name, env_config=env_config))
        else:
            env = BestActionsWrapper(MaxStepWrapper(gym.make(env_name, env_config=env_config), max_steps_per_episode))
        env.seed(seed + i)
        return env

    return _anon


def dqn(default_config=default_dqn_config.config):
    wandb.init(config=default_config)

    config = wandb.config

    start = time.time()

    seed = config['seed']
    gamma = config['gamma']  # Discount rate
    max_steps_per_episode = config['max_steps_per_episode']
    replay_buffer_size = config['replay_buffer_size']
    epsilon = config['epsilon']  # Exploration vs Exploitation trade off
    epsilon_decay = config['epsilon_decay']  # We reduce the epsilon parameter at each iteration
    minimal_epsilon = config['minimal_epsilon']
    update_network_step = config['update_network_step']  # Update Q-Network periodicity
    batch_size = config['batch_size']  # Batch of experiences to get from the replay buffer
    learning_rate = config['learning_rate']
    tau = config['tau']
    nb_steps = config['nb_steps']

    env_name = config['env_name']
    actor_per_cpu = config['actors_per_cpu']
    running_sec_time = config['running_sec_time']
    network_config = [config['layer_size'] for _ in range(config['layer_nb'])]

    nb_actors = actor_per_cpu * mp.cpu_count()
    envs = [make_seeded_env(i, env_name, seed, max_steps_per_episode, {'instance_path': config['instance']}) for i in range(nb_actors)]
    envs = SubprocVecEnv(envs)

    env_best = BestActionsWrapper(gym.make(env_name, env_config={'instance_path': config['instance']}))
    env_info = gym.make(env_name, env_config={'instance_path': config['instance']})

    env_info.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    local_net = QNetwork(env_info.observation_space.shape[0], env_info.action_space.n, network_config)
    # local_net = local_net.to(device)
    target_net = QNetwork(env_info.observation_space.shape[0], env_info.action_space.n, network_config)
    # target_net = target_net.to(device)
    optimizer = optim.Adam(local_net.parameters(), lr=learning_rate)
    memory = PrioritizedReplayBuffer(replay_buffer_size)
    wandb.watch(local_net)

    episode_nb = 0
    previous_nb_episode = 0
    total_steps = 0

    states = envs.reset()
    legal_actions = envs.get_legal_actions()
    masks = np.invert(legal_actions) * -1e10
    state_tensor = torch.FloatTensor(states)
    with torch.no_grad():
        action_values = local_net(state_tensor) + masks
        value_current_states, actions = torch.max(action_values, dim=-1)
    while time.time() < start + running_sec_time:
        experience_stack = [[] for _ in range(nb_actors)]
        for step in range(nb_steps):
            actions = [np.random.choice(len(legal_action), 1, p=(legal_action / legal_action.sum()))[
                           0] if random.random() <= epsilon else actions[actor_nb].item() for actor_nb, legal_action in
                       enumerate(legal_actions)]
            next_states_env, rewards, dones, _ = envs.step(actions)
            legal_actions = envs.get_legal_actions()
            masks = np.invert(legal_actions) * -1e10

            total_steps += 1
            episode_nb += sum(dones)

            for actor_nb in range(nb_actors):
                state = states[actor_nb]
                action = actions[actor_nb]
                reward = rewards[actor_nb]
                done = dones[actor_nb]
                next_state = next_states_env[actor_nb]
                mask = masks[actor_nb]
                value = value_current_states[actor_nb]
                experience = Experience(state, action, reward, next_state, done, mask)
                experience.current_state_value = value
                experience_stack[actor_nb].append(experience)

            if total_steps % update_network_step == 0 and len(memory) > batch_size:
                optimizer.zero_grad()
                experiences, indices, weights = memory.sample(batch_size)
                states = torch.FloatTensor(np.vstack([e.state for e in experiences if e is not None]))
                steps = torch.FloatTensor(np.vstack([e.step for e in experiences if e is not None]))
                actions = torch.LongTensor(np.vstack([e.action for e in experiences if e is not None]))
                rewards = torch.FloatTensor(np.vstack([e.reward for e in experiences if e is not None]))
                next_states = torch.FloatTensor(np.vstack([e.next_state for e in experiences if e is not None]))
                dones = torch.FloatTensor(np.vstack([int(e.done) for e in experiences if e is not None]))
                all_masks = torch.FloatTensor(np.vstack([e.legal_actions for e in experiences if e is not None]))
                q_pred = local_net(states).gather(1, actions)
                with torch.no_grad():
                    target_next = target_net(next_states) + all_masks
                    action_next = torch.argmax(target_next, dim=1).unsqueeze(1)
                    q_pred_next = local_net(next_states).gather(1, action_next)
                    td = rewards + ((gamma ** steps) * q_pred_next * (1 - dones))
                loss = (torch.FloatTensor(weights) * loss_fn(td, q_pred)).mean()
                wandb.log({"loss": loss})
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
                        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
                    errors = td - q_pred
                memory.update(indices, errors)
            if previous_nb_episode != episode_nb:
                epsilon = max(minimal_epsilon, epsilon * epsilon_decay)
                previous_nb_episode = episode_nb

            states = next_states_env
            state_tensor = torch.FloatTensor(states)
            with torch.no_grad():
                action_values = local_net(state_tensor) + masks
                value_current_states, actions = torch.max(action_values, dim=-1)

        acc_return = 0
        step = 1
        done_exp = 0
        for actor_nb in range(nb_actors):
            while len(experience_stack[actor_nb]) > 0:
                experience = experience_stack[actor_nb].pop()
                acc_return = experience.reward + (gamma * acc_return * (1 - experience.done))
                if experience.done == 1:
                    done_exp = 1
                experience.done = done_exp
                experience.reward = acc_return
                experience.next_state = states[actor_nb]
                experience.step = step
                with torch.no_grad():
                    td = experience.reward + (
                            (gamma ** experience.step) * value_current_states[actor_nb] * (1 - experience.done))
                    error = experience.current_state_value - td
                experience.error = error
                memory.add(experience, error)
                step += 1

    sum_best_scores = 0
    all_best_score = float('-inf')
    all_best_actions = []
    all_best_time_step = float('inf')

    for remote in envs.remotes:
        remote.send(('get_best_actions', None))
        best_score, best_actions = remote.recv()
        sum_best_scores += best_score
        remote.send(('get_best_timestep', None))
        best_time_step = remote.recv()
        if best_time_step < all_best_time_step:
            all_best_time_step = best_time_step
            all_best_score = best_score
            all_best_actions = best_actions
    avg_best_result = sum_best_scores / len(envs.remotes)

    # We need to do an iteration without greedy
    state = env_best.reset()
    done = False
    while not done:
        legal_actions = env_best.get_legal_actions()
        masks = np.invert(legal_actions) * -1e10
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_values = local_net(state_tensor) + masks
            action = torch.argmax(action_values).item()
        state, reward, done, action_performed = env_best.step(action)
    state = env_best.reset()

    if env_best.best_time_step < all_best_time_step:
        all_best_score = env_best.best_score
        all_best_actions = env_best.best_actions
        all_best_time_step = env_best.best_time_step

    state = env_info.reset()
    done = False
    legal_actions = env_info.get_legal_actions()
    current_step = 0
    # we can't just iterate throught all the actions because of the automatic action taking
    while current_step < len(all_best_actions):
        action = all_best_actions[current_step]
        assert legal_actions[action]
        state, reward, done, _ = env_info.step(action)
        legal_actions = env_info.get_legal_actions()
        current_step += 1
    assert done
    '''
    figure = env_info.render()
    img_bytes = figure.to_image(format="png")
    image = Image.open(io.BytesIO(img_bytes))
    '''
    # wandb.log({"nb_episodes": episode_nb, "avg_best_result": avg_best_result, "best_episode": all_best_score,
    #           "best_timestep": all_best_time_step, 'gantt': [wandb.Image(image)]})
    wandb.log({"nb_episodes": episode_nb, "avg_best_result": avg_best_result, "best_episode": all_best_score,
               "best_timestep": all_best_time_step})
    torch.save(local_net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    return episode_nb, all_best_score, avg_best_result, all_best_actions


if __name__ == "__main__":
    dqn()