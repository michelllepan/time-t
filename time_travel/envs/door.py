from dataclasses import dataclass
from enum import Enum

import random

import gymnasium as gym
from gymnasium import spaces

R = 199
BAD_ACTION_R = int(-200)

class Action(Enum):
    OPEN_DOOR_0 = 0
    OPEN_DOOR_1 = 1
    LOCK_DOOR_0 = 2
    LOCK_DOOR_1 = 3
    TIME_TRAVEL = 4
    DO_NOTHING = 5

class AgentType(Enum):
    NORMAL = 0
    TIME_TRAVELING = 1

class DoorState(Enum):
    LOCKED = 0
    CLOSED = 1
    OPEN_GOOD = 2
    OPEN_BAD = 3

@dataclass
class Door:
    reward: int
    state: DoorState

@dataclass
class Observation:
    door0: DoorState
    door1: DoorState
    agent_type: AgentType

    def __str__(self):
        return f"Observation(door0={self.door0.name}, door1={self.door1.name}, agent_type={self.agent_type.name})"

class DoorEnv(gym.Env):
    """A class for the door environment. (2 doors)
    """
    
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.MultiDiscrete([len(DoorState), len(DoorState), len(AgentType)], dtype=int)

    def reset(self, is_original_timeline=True):
        if is_original_timeline:
            # only re-generate reward door if we are resetting in the original timeline
            # if we are resetting 
            self.reward_door = random.randint(0, 1)
        self.t = 0
        self.doors = [
            Door(reward=R * (1 - self.reward_door), state=DoorState.CLOSED),
            Door(reward=R * self.reward_door, state=DoorState.CLOSED),
        ]
        self.is_original_timeline = is_original_timeline
        return self._get_obs()

    def step(self, joint_action: tuple[Action, Action]):
        normal_action, time_travel_action = joint_action

        obs = (None, None)
        reward = 0
        terminated = False
        truncated = False
        info = {"t": self.t}

        if (not self._check_valid_action(normal_action, AgentType.NORMAL) or
            not self._check_valid_action(time_travel_action, AgentType.TIME_TRAVELING)):
            # print("Invalid action")
            truncated = True
            reward = BAD_ACTION_R
            return obs, reward, terminated, truncated, info
        
        self.t += 1

        match time_travel_action:
            case Action.LOCK_DOOR_0:
                self.doors[0].state = DoorState.LOCKED
            case Action.LOCK_DOOR_1:
                self.doors[1].state = DoorState.LOCKED
        
        match normal_action:
            case Action.OPEN_DOOR_0 | Action.OPEN_DOOR_1:
                door_to_open = normal_action.value
                reward = self.doors[door_to_open].reward

                self.doors[door_to_open].state = DoorState.OPEN_GOOD if reward > 0 else DoorState.OPEN_BAD
                terminated = not self.is_original_timeline
            case Action.TIME_TRAVEL:
                reward = 0
                if (self.doors[0].state == DoorState.OPEN_GOOD or
                    self.doors[1].state == DoorState.OPEN_GOOD):
                    reward = -R
                self.reset(is_original_timeline=False)
            case Action.DO_NOTHING:
                if self.t == 3 and self.is_original_timeline:
                    terminated = True
                if self.t == 2 and not self.is_original_timeline:
                    terminated = True

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        normal_obs = Observation(door0=self.doors[0].state, door1=self.doors[1].state, agent_type=AgentType.NORMAL)
        if self.is_original_timeline:
            time_travel_obs = None
        else:
            time_travel_obs = Observation(door0=self.doors[0].state, door1=self.doors[1].state, agent_type=AgentType.TIME_TRAVELING)
        return normal_obs, time_travel_obs
    
    def _check_valid_action(self, action: Action, agent_type: AgentType):
        # print(f"Checking valid action {action} for agent type {agent_type} at t={self.t}")
        if agent_type == AgentType.TIME_TRAVELING and self.is_original_timeline:
            return action is None
        
        if agent_type == AgentType.NORMAL:
            match self.t:
                case 0:
                    valid_actions = {Action.DO_NOTHING}
                case 1:
                    valid_actions = {Action.OPEN_DOOR_0, Action.OPEN_DOOR_1}
                    if self.doors[0].state == DoorState.LOCKED:
                        valid_actions -= {Action.OPEN_DOOR_0}
                    elif self.doors[1].state == DoorState.LOCKED:
                        valid_actions -= {Action.OPEN_DOOR_1}
                case 2:
                    valid_actions = {Action.TIME_TRAVEL, Action.DO_NOTHING}
        elif agent_type == AgentType.TIME_TRAVELING:
            match self.t:
                case 0:
                    valid_actions = {Action.LOCK_DOOR_0, Action.LOCK_DOOR_1, Action.DO_NOTHING}
                case 1:
                    valid_actions = {Action.DO_NOTHING}
        return action in valid_actions
    
    def render(self):
        print("-" * 30)
        print(f"t = {self.t}")
        print(f"In original timeline: {self.is_original_timeline}")

        door0 = self.doors[0].state.name
        door1 = self.doors[1].state.name
        print(f"State: door0: {door0}, door1: {door1}")

        obs = self._get_obs()
        print(f"Normal obs: {obs[0]}")
        print(f"Time travel obs: {obs[1]}")

        print("#" * 20)