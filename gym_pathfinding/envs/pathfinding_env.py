import pygame
import numpy as np
import gym
from gym import error, spaces, utils

from gym_pathfinding.games.pathfinding import PathFindingGame
from gym_pathfinding.rendering import GridViewer


class PathFindingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'array']}

    def __init__(self, width, height, screen_size=(640, 480), seed=None):
        self.game = PathFindingGame(width, height, seed)

        self.viewer = GridViewer(screen_size[0], screen_size[1])
        self.viewer.start(width, height)

        self.observation_space = spaces.MultiDiscrete(self.game.get_state().shape)
        self.action_space = spaces.Discrete(4)
    
    def reset(self):
        return self.game.reset()

    def step(self, action):
        return self.game.step(action)

    def seed(self):
        return self.game.seed

    def render(self, mode='human'):
        grid = self.game.get_state()

        if (mode == 'human'):
            self.viewer.draw(grid)
        elif (mode == 'array'):
            return grid

    def close(self):
        self.viewer.stop()


def create_pathfinding_env(id, name, width, height, seed=None):

    def constructor(self):
        PathFindingEnv.__init__(self, width, height, seed=seed)
    
    pathfinding_env_class = type(name, (PathFindingEnv,), {
            "id" : id,
            "__init__": constructor
        })
    return pathfinding_env_class


# Create classes 

sizes = list(range(9, 20, 2)) + [25, 35, 55]
envs = [create_pathfinding_env("pathfinding-{i}x{i}-v0".format(i=i), "PathFinding{i}x{i}Env".format(i=i), i, i) for i in sizes]

for env_class in envs:
    globals()[env_class.__name__] = env_class


def get_env_classes():
    return envs
