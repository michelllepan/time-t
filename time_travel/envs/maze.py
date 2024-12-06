from dataclasses import dataclass
from enum import Enum

import random

import gymnasium as gym
from gymnasium import spaces

GRID_SIZE = 5
VISIBILITY = 2

MAX_EPISODE_LEN = 100

GOAL_R = 199
BAD_ACTION_R = -200
TRAP_R = -200
AGENTS_CLOSE_R = -100
TIME_R = -1

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    LEFT_WALL = 4
    RIGHT_WALL = 5
    UP_WALL = 6
    DOWN_WALL = 7
    TIME_TRAVEL = 8
    DO_NOTHING = 9
    
class AgentType(Enum):
    NORMAL = 0
    TIME_TRAVELING = 1
    
class CellState(Enum):
    GOAL = 0
    TRAP = 1
    WALL = 2
    EMPTY = 3

@dataclass
class Observation:
    position: tuple[int, int]
    cells: list[CellState]
    agent_type: AgentType

class MazeEnv(gym.Env):
    """A class for the maze environment.
    """
    
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.MultiDiscrete([GRID_SIZE, GRID_SIZE] + [len(CellState) for _ in range(5)] + [len(AgentType)])

    def reset(self, is_original_timeline=True):
        self.t = 0

        if is_original_timeline:
            self.grid = {(i, j): CellState.EMPTY for i in range(GRID_SIZE) for j in range(GRID_SIZE)}
            for i in range(1, 4):
                for j in range(1, 4):
                    self.grid[(i, j)] = CellState.WALL

            for i in range(-1, GRID_SIZE + 1):
                self.grid[(-1, i)] = CellState.WALL
                self.grid[(GRID_SIZE, i)] = CellState.WALL
                self.grid[(i, -1)] = CellState.WALL
                self.grid[(i, GRID_SIZE)] = CellState.WALL

            self.grid[(GRID_SIZE-1, GRID_SIZE-1)] = CellState.GOAL

            trap_is_below = random.randint(0, 1) == 0
            if trap_is_below:
                self.grid[(GRID_SIZE-2, GRID_SIZE-1)] = CellState.TRAP
            else:
                self.grid[(GRID_SIZE-1, GRID_SIZE-2)] = CellState.TRAP
        
        self.is_original_timeline = is_original_timeline
        self.normal_agent_pos = (0, 0)
        self.time_travel_agent_pos = (GRID_SIZE-1, GRID_SIZE-1)

        return self._get_obs()
    
    def action_to_dx_dy(self, action: Action):
        match action:
            case Action.LEFT | Action.LEFT_WALL:
                return (-1, 0)
            case Action.RIGHT | Action.RIGHT_WALL:
                return (1, 0)
            case Action.UP | Action.UP_WALL:
                return (0, 1)
            case Action.DOWN | Action.DOWN_WALL:
                return (0, -1)
        
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
            # print(normal_action, self._check_valid_action(normal_action, AgentType.NORMAL))
            # print(time_travel_action, self._check_valid_action(time_travel_action, AgentType.TIME_TRAVELING))
            truncated = True
            reward = BAD_ACTION_R
            return obs, reward, terminated, truncated, info

        self.t += 1
        reward += TIME_R

        if self.t >= MAX_EPISODE_LEN:
            truncated = True
            return obs, reward, terminated, truncated, info

        # normal agent move
        if normal_action in {Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN}:
            x, y = self.normal_agent_pos
            dx, dy = self.action_to_dx_dy(normal_action)
            proposed_new_position = (x + dx, y + dy)
            if self.grid[proposed_new_position] != CellState.WALL:
                self.normal_agent_pos = proposed_new_position

        if self.normal_agent_pos == (GRID_SIZE-1, GRID_SIZE-1):
            if normal_action == Action.TIME_TRAVEL:
                reward += -1 * TIME_R * self.t  # undo time rewards
                reward -= GOAL_R  # undo goal reward
                self.reset(is_original_timeline=False)
                return self._get_obs(), reward, terminated, truncated, info
            elif normal_action == Action.DO_NOTHING:
                reward = 0
                terminated = True
                return self._get_obs(), reward, terminated, truncated, info
            elif not self.is_original_timeline:
                terminated = True
                reward += GOAL_R
                return self._get_obs(), reward, terminated, truncated, info

        if not self.is_original_timeline:
            # time travel agent move
            if time_travel_action in {Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN}:
                x, y = self.time_travel_agent_pos
                dx, dy = self.action_to_dx_dy(time_travel_action)
                proposed_new_position = (x + dx, y + dy)
                if self.grid[proposed_new_position] != CellState.WALL:
                    self.time_travel_agent_pos = proposed_new_position
            
            # check closeness
            if (abs(self.normal_agent_pos[0] - self.time_travel_agent_pos[0]) +
                abs(self.normal_agent_pos[1] - self.time_travel_agent_pos[1])) <= 1:
                terminated = True
                reward += AGENTS_CLOSE_R
                return obs, reward, terminated, truncated, info

            # time travel agent place wall
            if time_travel_action in {Action.LEFT_WALL, Action.RIGHT_WALL, Action.UP_WALL, Action.DOWN_WALL}:
                x, y = self.time_travel_agent_pos
                dx, dy = self.action_to_dx_dy(time_travel_action)
                proposed_wall_pos = (x + dx, y + dy)
                if self.grid[proposed_wall_pos] == CellState.EMPTY:
                    self.grid[proposed_wall_pos] = CellState.WALL
        
        match self.grid[self.normal_agent_pos]:
            case CellState.GOAL:
                reward += GOAL_R
            case CellState.TRAP:
                reward += TRAP_R
                terminated = True
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs = []
        for pos, agent_type in zip([self.normal_agent_pos, self.time_travel_agent_pos],
                                   [AgentType.NORMAL, AgentType.TIME_TRAVELING]):
            x, y = pos
            cells = [self.grid[(x, y)]]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                cells.append(self.grid[(x + dx, y + dy)])
            obs.append(Observation(pos, cells, agent_type))
        return tuple(obs)

    def _check_valid_action(self, action: Action, agent_type: AgentType):
        if agent_type == AgentType.NORMAL:
            valid_actions = {Action.DO_NOTHING}
            if self.normal_agent_pos == (GRID_SIZE-1, GRID_SIZE-1):
                if self.is_original_timeline:
                    valid_actions |= {Action.TIME_TRAVEL}
            else:
                valid_actions |= {Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN}
            return action in valid_actions
        elif agent_type == AgentType.TIME_TRAVELING:
            return action != Action.TIME_TRAVEL
    
    def render(self):
        print("-" * 30)
        print(f"t = {self.t}")
        print(f"In original timeline: {self.is_original_timeline}")
        
        for y in reversed(range(GRID_SIZE)):
            for x in range(GRID_SIZE):
                display = " "
                match self.grid[(x, y)]:
                    case CellState.EMPTY:
                        display = "."
                    case CellState.WALL:
                        display = "#"
                    case CellState.GOAL:
                        display = "G"
                    case CellState.TRAP:
                        display = "T"
                if self.normal_agent_pos == (x, y):
                    display = "n"
                elif self.time_travel_agent_pos == (x, y) and not self.is_original_timeline:
                    display = "t"
                print(display, end=" ")
            print()