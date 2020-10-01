import random

import gym

from JSS.env import JSS


class MaxStepWrapper(gym.Wrapper):

    def __init__(self, env, max_steps=1000):
        super(MaxStepWrapper, self).__init__(env)
        self.current_step = 0
        self.max_steps = max_steps

    def reset(self, **kwargs):
        observation = super(MaxStepWrapper, self).reset(**kwargs)
        self.current_step = 0
        return observation

    def step(self, action):
        observation, reward, done, info = super(MaxStepWrapper, self).step(action)
        self.current_step += 1
        if self.max_steps <= self.current_step:
            done = True
        return observation, reward, done, info


class BestActionsWrapper(gym.Wrapper):

    def __init__(self, env):
        super(BestActionsWrapper, self).__init__(env)
        self.best_actions = []
        self.current_actions = []
        self.best_score = float('-inf')
        self.current_score = 0
        self.best_time_step = float('inf')

    def reset(self, **kwargs):
        if self.action_step >= self.max_action_step - 1:
            if self.current_time_step < self.best_time_step:
                self.best_time_step = self.current_time_step
            observation = super(BestActionsWrapper, self).reset(**kwargs)
            if self.current_score > self.best_score:
                self.best_score = self.current_score
                self.best_actions = self.current_actions
            self.current_score = 0
            self.current_actions = []
            return observation
        return super(BestActionsWrapper, self).reset(**kwargs)

    def step(self, action):
        observation, reward, done, real_action_performed = super(BestActionsWrapper, self).step(action)
        self.current_actions.append(action)
        self.current_score += reward
        return observation, reward, done, {}


class JSSMultiple(gym.Env):

    def __init__(self, instances=[]):
        self.envs = []
        self.last_score = float('-inf')
        self.current_env = None
        self.iteration = 0
        for instance in instances:
            env = BestActionsWrapper(JSS(env_config={'instance_path': instance}))
            self.envs.append(env)

    def reset(self, **kwargs):
        if self.iteration != 0:
            self.last_score = self.current_env.current_score
        self.current_env = random.choice(self.envs)
        self.iteration += 1
        return self.current_env.reset(**kwargs)

    def step(self, action):
        return self.current_env.step(action)

    def render(self, mode='human'):
        return self.current_env.render()

    def seed(self, seed=None):
        random.seed(seed)

    def get_legal_actions(self):
        return self.current_env.get_legal_actions()