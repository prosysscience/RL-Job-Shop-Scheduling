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
import torch.nn.functional as F

from JSS.PrioritizedReplayBuffer import PrioritizedReplayBuffer, Experience
from JSS.env_wrapper import BestActionsWrapper, MaxStepWrapper
from JSS.multiprocessing_env import SubprocVecEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # We train on a GPU if available

loss_fn = nn.SmoothL1Loss()


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.action_size = action_size
        self.layer1 = nn.Linear(state_size, 64)

        # value
        self.layer1_value = nn.Linear(64, 64)
        self.layer2_value = nn.Linear(64, 1)

        # advantage
        self.layer1_advantage = nn.Linear(64, 64)
        self.layer2_advantage = nn.Linear(64, action_size)

        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer1_value.weight)
        torch.nn.init.xavier_uniform_(self.layer2_value.weight)
        torch.nn.init.xavier_uniform_(self.layer1_advantage.weight)
        torch.nn.init.xavier_uniform_(self.layer2_advantage.weight)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = torch.tanh(self.layer1(state))

        value = torch.tanh(self.layer1_value(x))
        value = self.layer2_value(value)

        adv = torch.tanh(self.layer1_advantage(x))
        adv = self.layer2_advantage(adv)

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


def dqn(config):
    #config_defaults = default_ppo_config.config

    wandb.init()

    #config = wandb.config

    start = time.time()

    seed = 0
    gamma = 0.99  # Discount rate
    max_steps_per_episode = 1000
    number_episodes = 2000
    replay_buffer_size = 2000000
    epsilon = 0.9  # Exploration vs Exploitation trade off
    epsilon_decay = 0.995  # We reduce the epsilon parameter at each iteration
    minimal_epsilon = 0.1
    clipping_gradient = 1.0
    minimal_clipping = 0.1
    clipping_decay = 0.995
    update_network_step = 1  # Update Q-Network periodicity
    batch_size = 64  # Batch of experiences to get from the replay buffer
    learning_rate = 5e-4
    tau = 1e-3  # Define the step of the soft update
    nb_steps = 3
    env_name = config['env_name']
    env_config = config['env_config']
    actor_per_cpu = config['actors_per_cpu']
    running_sec_time = config['running_sec_time']

    nb_actors = actor_per_cpu * mp.cpu_count()
    envs = [make_seeded_env(i, env_name, seed, max_steps_per_episode, env_config) for i in range(nb_actors)]
    envs = SubprocVecEnv(envs)

    env = BestActionsWrapper(gym.make(env_name, env_config=env_config))

    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


    local_net = QNetwork(env.observation_space.shape[0], env.action_space.n)
    #local_net = local_net.to(device)
    target_net = QNetwork(env.observation_space.shape[0], env.action_space.n)
    #target_net = target_net.to(device)
    optimizer = optim.Adam(local_net.parameters(), lr=learning_rate)
    memory = PrioritizedReplayBuffer(replay_buffer_size)


    episode_nb = 0
    total_steps = 0


    state = env.reset()
    legal_actions = env.get_legal_actions()
    mask = (1 - legal_actions) * -1e10
    done = False
    state_tensor = torch.FloatTensor(state)
    with torch.no_grad():
        action_values = local_net(state_tensor) + mask
        value_current_state = torch.max(action_values).item()
    while time.time() < start + running_sec_time:
        experience_stack = []
        for step in range(nb_steps):
            total_steps += 1
            if random.random() > epsilon:
                with torch.no_grad():
                    action = torch.argmax(action_values, dim=0).item()
            else:
                #actions = [np.random.choice(len(legal_action), 1, p=(legal_action / legal_action.sum()))[0] for legal_action in legal_actions]
                #action = env.action_space.sample()
                action = np.random.choice(len(legal_actions), 1, p=(legal_actions / legal_actions.sum()))[0]
            next_state, reward, done, _ = env.step(action)
            legal_actions = env.get_legal_actions()
            mask = (1 - legal_actions) * -1e10

            experience = Experience(state, action, reward, next_state, done, mask)
            experience.current_state_value = value_current_state
            experience_stack.append(experience)

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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_net.parameters(), clipping_gradient)
                optimizer.step()
                for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
                    target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
                with torch.no_grad():
                    errors = td - q_pred
                memory.update(indices, errors)

            state = next_state
            if done:
                episode_nb += 1
                state = env.reset()
                done = False
                legal_actions = env.get_legal_actions()
                epsilon = max(minimal_epsilon, epsilon * epsilon_decay)
                clipping_gradient = max(minimal_clipping, clipping_gradient * clipping_decay)
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                action_values = local_net(state_tensor) + mask
                value_current_state = torch.max(action_values).item()

        acc_return = 0
        step = 1
        done = 0
        while len(experience_stack) > 0:
            experience = experience_stack.pop()
            acc_return = experience.reward + (gamma * acc_return * (1 - experience.done))
            if experience.done == 1:
                done = 1
            experience.done = done
            experience.reward = acc_return
            experience.next_state = state
            experience.step = step
            with torch.no_grad():
                td = experience.reward + ((gamma ** experience.step) * value_current_state * (1 - experience.done))
                error = experience.current_state_value - td
            experience.error = error
            memory.add(experience, error)
            step += 1

    sum_best_scores = 0
    all_best_score = float('-inf')
    all_best_actions = []
    all_best_time_step = float('inf')
    best_score = env.best_score
    best_actions = env.best_actions
    sum_best_scores += best_score
    if best_score > all_best_score:
        all_best_score = best_score
        all_best_actions = best_actions
    best_time_step = env.best_time_step
    if best_time_step < all_best_time_step:
        all_best_time_step = best_time_step
    avg_best_result = sum_best_scores / len(envs.remotes)


    env_test = gym.make(env_name, env_config=env_config)
    state = env_test.reset()
    done = False
    legal_actions = env_test.get_legal_actions()
    current_step = 0
    # we can't just iterate throught all the actions because of the automatic action taking
    while current_step < len(all_best_actions):
        action = all_best_actions[current_step]
        assert legal_actions[action]
        state, reward, done, action_performed = env_test.step(action)
        current_step += len(action_performed)
    assert done
    figure = env.render()
    img_bytes = figure.to_image(format="png")
    image = Image.open(io.BytesIO(img_bytes))
    wandb.log({"nb_episodes": episode_nb, "avg_best_result": avg_best_result, "best_episode": all_best_score,
               "best_timestep": all_best_time_step, 'gantt': [wandb.Image(image)]})
    return episode_nb, all_best_score, avg_best_result, all_best_actions


