import numpy as np

from time_travel.envs.door import *

class DoorAgent:

    def __init__(self, env: DoorEnv, lr: float = 1e-2):
        self.env = env
        self.q_values = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def _obs_to_idx(self, obs: Observation):
        return (obs.door0.value * (len(DoorState) ** 2) +
                obs.door1.value * (len(DoorState)) +
                obs.agent_type.value)

    def act(self, obs: Observation, deterministic: bool = False):
        obs_idx = self._obs_to_idx(obs)
        obs_qs = self.q_values[obs_idx]
        if deterministic:
            return np.argmax(obs_qs)
        else:
            # TODO
            return np.argmax(obs_qs)
    
    def update(self, obs: Observation, action: Action, next_obs: Observation, reward: float):
        obs_idx = self._obs_to_idx(obs)
        next_obs_idx = self._obs_to_idx(next_obs)

        next_q = np.max(self.q_values[next_obs_idx])
        this_q = self.q_values[obs_idx][action.value]
        self.q_values[obs_idx][action.value] += self.lr * (reward + next_q - this_q)