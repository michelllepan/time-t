import numpy as np

from time_travel.envs.maze import *

class MazeAgent:

    def __init__(self, env: MazeEnv, lr: float = 1e-2):
        self.env = env
        self.q_values = np.zeros((np.prod(self.env.observation_space.nvec) + 1, self.env.action_space.n))
        self.lr = lr

        self.truncated_obs_idx = self.q_values.shape[0] - 1

    def _obs_to_idx(self, obs: Observation | ObservationWithTrapPos):
        if obs is None:
            return self.truncated_obs_idx
        
        return obs.to_idx()
    
    def softmax_stable(self, x):
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    
    def act(self, obs: Observation | ObservationWithTrapPos, epsilon: float = 0, deterministic: bool = True):
        obs_idx = self._obs_to_idx(obs)
        obs_qs = self.q_values[obs_idx]

        if deterministic:
            action_idx = np.argmax(obs_qs)
        elif np.random.rand() > epsilon:
            action_idx = np.random.choice(np.arange(self.env.action_space.n), p=self.softmax_stable(obs_qs))
        else:
            action_idx = np.random.choice(np.arange(self.env.action_space.n))

        return Action(value=action_idx)
    
    def update(self, obs: Observation | ObservationWithTrapPos, action: Action, next_obs: Observation | ObservationWithTrapPos, reward: float):
        obs_idx = self._obs_to_idx(obs)
        next_obs_idx = self._obs_to_idx(next_obs)

        next_q = np.max(self.q_values[next_obs_idx])
        this_q = self.q_values[obs_idx][action.value]
        self.q_values[obs_idx][action.value] += self.lr * (reward + next_q - this_q)