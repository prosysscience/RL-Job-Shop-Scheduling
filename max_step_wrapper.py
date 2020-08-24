import gym

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