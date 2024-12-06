from dataclasses import dataclass
from enum import Enum

import random

import gymnasium as gym
from gymnasium import spaces

GRID_SIZE = 5
VISIBILITY = 2

GOAL_R = 199
BAD_ACTION_R = -200
TRAP_R = -200
TIME_R = -1

class Action(Enum):
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, 1)
    DOWN = (0, -1)
    TIME_TRAVEL = 4
    DO_NOTHING = 5
    # TODO: add actions for wall moving
    
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
    agent_type:AgentType

class MazeEnv(gym.Env):
    """A class for the maze environment.
    """
    
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.MultiDiscrete([GRID_SIZE, GRID_SIZE] + [len(CellState) for _ in range(8)])

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
        # TODO: time travel pos?

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

        if normal_action in {Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN}:
            x, y = self.normal_agent_pos
            dx, dy = normal_action.value
            proposed_new_position = (x + dx, y + dy)
            if self.grid[proposed_new_position] != CellState.WALL:
                self.normal_agent_pos = proposed_new_position

        match self.grid[self.normal_agent_pos]:
            case CellState.GOAL:
                reward = GOAL_R
                terminated = True
            case CellState.TRAP:
                reward = TRAP_R
                terminated = True

        reward += TIME_R
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        surrounding_cells = []
        x, y = self.normal_agent_pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                surrounding_cells.append(self.grid[(x + dx, y + dy)])
        
        return Observation(self.normal_agent_pos, surrounding_cells, AgentType.NORMAL), None
        # TODO: time travel obs

    def _check_valid_action(self, action: Action, agent_type: AgentType):
        if agent_type == AgentType.NORMAL:
            agent_pos = self.normal_agent_pos
        # else:
        #     agent_pos = self.time_travel_agent_pos
            
        # if action in {Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN}:
        #     x, y = agent_pos
        #     dx, dy = action.value
        #     new_x, new_y = x + dx, y + dy

        #     if self.grid[(new_x, new_y)] == CellState.WALL:
        #         return False
            
        return True
    
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
                print(display, end=" ")
            print()