import numpy as np

from time_travel.envs.door import *

class DoorAgent:

    def __init__(self, env: DoorEnv, lr: float = 1e-2):
        self.env = env
        self.q_values = np.zeros((np.prod(self.env.observation_space.nvec) + 2, self.env.action_space.n))
        self.lr = lr

        self.truncated_obs_idx = self.q_values.shape[0] - 2
        self.do_nothing_obs_idx = self.q_values.shape[0] - 1

    def _obs_to_idx(self, obs: Observation):
        if obs is None:
            return self.truncated_obs_idx
        
        if (self.env.t == 0 and obs.agent_type == AgentType.NORMAL or
            self.env.t == 1 and obs.agent_type == AgentType.TIME_TRAVELING):
            return self.do_nothing_obs_idx
        
        return (obs.door0.value * (len(DoorState) * len(AgentType)) +
                obs.door1.value * (len(AgentType)) +
                obs.agent_type.value)

    def act(self, obs: Observation, epsilon: float = 0, deterministic: bool = True):
        obs_idx = self._obs_to_idx(obs)
        obs_qs = self.q_values[obs_idx]

        if deterministic:
            action_idx = np.argmax(obs_qs)
        elif np.random.rand() > epsilon:
            action_idx = np.random.choice(np.arange(self.env.action_space.n), p=(np.exp(obs_qs) / np.sum(np.exp(obs_qs))))
        else:
            action_idx = np.random.choice(np.arange(self.env.action_space.n))

        return Action(value=action_idx)
    
    def update(self, obs: Observation, action: Action, next_obs: Observation, reward: float):
        obs_idx = self._obs_to_idx(obs)
        next_obs_idx = self._obs_to_idx(next_obs)

        next_q = np.max(self.q_values[next_obs_idx])
        this_q = self.q_values[obs_idx][action.value]
        self.q_values[obs_idx][action.value] += self.lr * (reward + next_q - this_q)