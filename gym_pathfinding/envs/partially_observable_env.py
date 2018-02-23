

import numpy as np
import gym

from gym_pathfinding.envs.pathfinding_env import PathFindingEnv


class PartiallyObservablePathFindingEnv(gym.Env):
    """ PartiallyObservableEnv
        -1 = unknown
    """

    def __init__(self, width, height, observable_depth, *, screen_size=(640, 640), generation_seed=None, spawn_seed=None):
        self.env = PathFindingEnv(width, height, screen_size=screen_size, generation_seed=generation_seed, spawn_seed=spawn_seed)
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

def create_partially_observable_pathfinding_env(id, name, width, height, observable_depth, seed=None):

    def constructor(self):
        PartiallyObservablePathFindingEnv.__init__(self, width, height, observable_depth, seed=seed)
    
    env_class = type(name, (PartiallyObservablePathFindingEnv,), {
            "id" : id,
            "__init__": constructor
        })
    return env_class


# Create classes 

sizes = list(range(9, 20, 2)) + [25, 35, 55]
envs = [create_partially_observable_pathfinding_env("partially_observable_pathfinding-{i}x{i}-v0".format(i=i), "PartiallyObservablePathFinding{i}x{i}Env".format(i=i), i, i, 2) for i in sizes]

for env_class in envs:
    globals()[env_class.__name__] = env_class

def get_env_classes():
    return envs


        
