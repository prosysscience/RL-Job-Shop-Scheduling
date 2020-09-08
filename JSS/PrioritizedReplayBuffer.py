import numpy as np

class Experience(object):

    def __init__(self, state, action, reward, next_state, done, legal_actions):
        self.step = 0
        self.error = 0
        self.current_state_value = 0
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.legal_actions = legal_actions
        
    def __str__(self):
        return 'step {} error {} current_state {} state {} action {} reward {} next_state {} done {}'.format(self.step,
        self.error,
        self.current_state_value,
        self.state,
        self.action,
        self.reward,
        self.next_state,
        self.done,
        self.legal_actions)


class PrioritizedReplayBuffer(object):

    def __init__(self, max_size: int=20000, epsilon: float=1e-2, alpha: float=0.6, beta: float=0.4, beta_step: float=1e-3):
        self.max_size = max_size
        self.size = 0
        self.epsilon = epsilon
        self.priorities = np.zeros((self.max_size, ), dtype=float)
        self.experiences = np.empty((self.max_size, ), dtype=type(Experience))
        self.alpha = alpha
        self.beta = beta
        self.beta_step = beta_step
        self.priority_sum = 0.0

    def _get_priority(self, error):
        return np.power(np.abs(error) + self.epsilon, self.alpha)

    def add(self, experience: Experience, error: float):
        priority = self._get_priority(error)
        if self.size < self.max_size:
            self.experiences[self.size] = experience
            self.priorities[self.size] = priority
            self.size += 1
        else:
            index = np.random.randint(0, self.size)
            self.priorities[index] = priority
            self.experiences[index] = experience


    def sample(self, batch_size: int):
        current_priorities = self.priorities[:self.size]
        probabilities = current_priorities / current_priorities.sum()
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=True)
        batch = np.empty((batch_size, ), dtype=type(Experience))
        weights = np.empty((batch_size, ), dtype=np.float)
        max_weight = -1
        for i, idx in enumerate(indices):
            probability = probabilities[idx]
            idx_weight = np.power(probability * self.size, -self.beta)
            max_weight = max(max_weight, idx_weight)
            weights[i] = idx_weight
            batch[i] = self.experiences[idx]
        self.beta = min(1.0, self.beta + self.beta_step)
        weights = weights / max_weight
        return batch, indices, weights

    def update(self, idxes, errors):
        #assert len(idxes) == len(errors)
        self.priorities[idxes] = self._get_priority(errors.reshape(-1))
        '''
        for idx, error in zip(idxes, errors):
            priority = self._get_priority(error)
            self.priorities[idx] = priority
        '''

    def __len__(self):
        return self.size
