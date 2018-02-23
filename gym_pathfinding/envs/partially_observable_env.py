

import numpy as np
import gym

from gym_pathfinding.envs.pathfinding_env import PathFindingEnv


class PartiallyObservableEnv(gym.Env):
    """ PartiallyObservableEnv
        -1 = unknown
    """

    def __init__(self, width, height, observable_depth, screen_size=(640, 480), seed=None):
        self.env = PathFindingEnv(width, height, screen_size=screen_size, seed=seed)
        self.observable_depth = observable_depth

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        state = self.env.reset()
        return self.partial_state(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self.partial_state(state), reward, done, info

    def seed(self):
        return self.env.seed()

    def render(self, mode='human'):
        grid = self.env.game.get_state()
        grid = self.partial_state(grid)

        if (mode == 'human'):
            self.env.viewer.draw(grid)
        elif (mode == 'array'):
            return grid

    def close(self):
        self.env.close()

    def partial_state(self, state):
        return partial_grid(state, self.env.game.player, self.observable_depth)

    
def partial_grid(grid, center, observable_depth):
    """return the centered partial state, place -1 to non-visible cells"""

    x, y = center
    offset = observable_depth

    mask = np.ones_like(grid, dtype=bool)
    mask[max(0, x - offset): x + offset + 1, max(0, y - offset): y + offset + 1] = False

    grid[mask] = -1
    return grid


class Env(PartiallyObservableEnv):
    id="mdrmdr-v0"

    """docstring for Env"""
    def __init__(self):
        super(Env, self).__init__(9, 9, 2)
        



        
