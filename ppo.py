import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from multiprocessing_env import SubprocVecEnv
from max_step_wrapper import MaxStepWrapper
from torch.distributions import Categorical


class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.linear_1 = nn.Linear(state_size, 64)
        self.linear_2 = nn.Linear(64, action_size)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, x):
        x = torch.tanh(self.linear_1(x))
        x = F.softmax(self.linear_2(x), dim=-1)
        return x


class Critic(nn.Module):

    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.linear_1 = nn.Linear(state_size, 64)
        self.linear_2 = nn.Linear(64, 64)
        self.linear_3 = nn.Linear(64, 1)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        torch.nn.init.xavier_uniform_(self.linear_3.weight, gain=0.1)

    def forward(self, x):
        x = torch.tanh(self.linear_1(x))
        x = torch.tanh(self.linear_2(x))
        x = self.linear_3(x)
        return x


class ActorCritic(nn.Module):

    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)

    def forward(self, x):
        return self.actor(x), self.critic(x)


def make_seeded_env(i: int, env_name: str, seed: int, max_steps_per_episode: int):
    def _anon():
        if max_steps_per_episode is None:
            env = gym.make(env_name)
        else:
            env = MaxStepWrapper(gym.make(env_name), max_steps_per_episode)
        env.seed(seed + i)
        return env

    return _anon


def ppo(config):
    seed = config['seed']
    learning_rate = config['learning_rate']
    n_steps = config['n_steps']
    tau = config['tau']
    gamma = config['gamma']
    number_episodes = config['number_episodes']
    max_steps_per_episode = config['max_steps_per_episode']
    value_coefficient = config['value_coefficient']
    entropy_regularization = config['entropy_regularization']
    nb_actors = config['nb_actors']
    env_name = config['env_name']
    ppo_epoch = config['ppo_epoch']
    clipping_param = config['clipping_param']
    clipping_param_vf = config['clipping_param_vf']
    minibatch_size = config['minibatch_size']
    gradient_norm_clipping = config['gradient_norm_clipping']
    max_kl_div = config['max_kl_div']

    min_reward = -10
    max_reward = 10

    STEP_BATCH = nb_actors * n_steps
    assert minibatch_size <= STEP_BATCH

    envs = [make_seeded_env(i, env_name, seed, max_steps_per_episode) for i in range(nb_actors)]
    envs = SubprocVecEnv(envs)

    env_infos = gym.make(env_name)

    env_infos.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = ActorCritic(env_infos.observation_space.shape[0], env_infos.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    states = envs.reset()

    episode_nb = 0

    while episode_nb < number_episodes:

        states_reps = []
        state_rewards = []
        state_dones = []
        log_probabilities = []
        states_values = []
        actions_used = []

        for step in range(n_steps):
            # we compute the state value and the probability distribution
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states)
                prob_state, value_state = model(states_tensor)
                categorical = Categorical(prob_state)
                actions_sampled = categorical.sample()
                log_prob = categorical.log_prob(actions_sampled)
                # gym env needs a numpy object
                actions = actions_sampled.cpu().numpy()

            # we act in the environments
            states, rewards, dones, _ = envs.step(actions)
            rewards = np.array([max(min(max_reward, reward), min_reward) for reward in rewards])

            '''
            reward_scaler.extend(rewards)
            if len(reward_scaler) > 2:
                rewards = rewards / statistics.stdev(reward_scaler)
            '''
            # we store the datas
            states_reps.append(states_tensor)
            actions_used.append(actions_sampled)
            states_values.append(value_state)
            state_rewards.append(torch.FloatTensor(rewards).unsqueeze(1))
            state_dones.append(torch.IntTensor(dones).unsqueeze(1))
            log_probabilities.append(log_prob)

        with torch.no_grad():
            # we also compute the next_state value
            states_tensor = torch.FloatTensor(states)
            _, next_value = model(states_tensor)
            states_values.append(next_value)
            gae = 0
            for step in reversed(range(n_steps)):
                current_step = (state_rewards[step] + gamma * states_values[step + 1] * (1 - state_dones[step])) - \
                               states_values[step]
                gae = current_step + (gamma * tau * (1 - state_dones[step]) * gae)
                state_rewards[step] = gae

        states_reps = torch.cat(states_reps)
        states_values = torch.cat(states_values[:-1])
        state_rewards = torch.cat(state_rewards)
        state_dones = torch.cat(state_dones)
        log_probabilities = torch.cat(log_probabilities)
        actions_used = torch.cat(actions_used)

        with torch.no_grad():
            advantage = state_rewards - states_values
        # we compute the clipped loss function
        kl_divergeance = []
        for epoch in range(ppo_epoch):

            permutation = torch.randperm(STEP_BATCH)

            # we normalize the advantage
            with torch.no_grad():
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            for i in range(0, STEP_BATCH, minibatch_size):

                optimizer.zero_grad()

                indices = permutation[i: i + minibatch_size]

                advantage_indicies = advantage[indices]

                new_probabilities, new_states_values = model(states_reps[indices])
                categorical = Categorical(new_probabilities)
                new_probabilities = categorical.log_prob(actions_used[indices])

                # minus because we get log probabilities
                ratio = (new_probabilities - log_probabilities[indices]).exp()
                entropy = categorical.entropy().mean()

                surrogate_non_clipp = (ratio * advantage_indicies).mean()
                surrogate_clipp = (
                            torch.clamp(ratio, 1 - clipping_param, 1 + clipping_param) * advantage_indicies).mean()
                actor_loss = -torch.min(surrogate_non_clipp, surrogate_clipp)

                # non clipped advantage
                new_advantage = state_rewards[indices] - new_states_values
                advantage_loss = new_advantage.pow(2).mean()
                if clipping_param_vf is not None:
                    # clipped advantage
                    clipped_new_states_values = torch.max(
                        torch.min(new_states_values, states_values[indices] + clipping_param_vf),
                        states_values[indices] - clipping_param_vf)
                    clipped_new_advantage = state_rewards[indices] - clipped_new_states_values
                    clipped_advantage_loss = clipped_new_advantage.pow(2).mean()
                    critic_loss = torch.min(advantage_loss, clipped_advantage_loss)
                else:
                    critic_loss = advantage_loss

                advantage[indices] = new_advantage.detach()

                loss = actor_loss + value_coefficient * critic_loss - entropy_regularization * entropy

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm_clipping)
                optimizer.step()
                kl_divergeance.append(torch.mean(ratio))

        if max_kl_div is not None and sum(kl_divergeance) / len(kl_divergeance) > max_kl_div:
            print("We have exceed the maximum KL divergeance alowed, we risk to have a policy crash")
            episode_nb += number_episodes
            break

        episode_nb += torch.sum(state_dones).item()
        print('\rEpisode {}'.format(episode_nb), end="")

    # to check that it's a new episode
    total_reward = 0
    for i in range(100):
        done = False
        state = env_infos.reset()
        ep = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                distribution, _ = model(state_tensor)
                categorical = Categorical(distribution)
                action = categorical.sample().numpy()
            if i % 33 == 0:
                env_infos.render()
            state, reward, done, _ = env_infos.step(action)
            total_reward += reward
            ep += 1
            if done:
                env_infos.close()
    mean_last_eval_scores = total_reward / 100.0
    torch.save(model.state_dict(), 'model.pt')
    return episode_nb, mean_last_eval_scores


