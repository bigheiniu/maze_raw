import gym
import os
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
from itertools import chain

from gym_pathfinding.games.maze import MazeGame


class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    id = "maze-v0"

    def __init__(self, width, height, state_type, seed=None, full_deterministic=False):

        self.env = MazeGame(width, height, 640, 480, state_type, 80, 80, seed=seed, seed_both=full_deterministic)

        self.observation_space = spaces.MultiDiscrete(self.env.get_state().shape)
        self.action_space = spaces.Discrete(4)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def seed(self):
        return self.env.seed

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.quit()


def create_maze_env(id, name, width, height, state_type, seed=None, full_deterministic=False):

    def constructor(self):
        MazeEnv.__init__(self, width, height, state_type, seed, full_deterministic)
    
    maze_env_class = type(name, (MazeEnv,), {
            "id" : id,
            "__init__": constructor
        })
    return maze_env_class


# Create Classes 

sizes = list(range(9, 20, 2)) + [25, 35, 55]
envs = list(chain(
    [create_maze_env("maze-arr-{i}x{i}-full-deterministic-v0".format(i=i), "MazeArrFull{i}x{i}Env".format(i=i), i, i, "array", 1337, full_deterministic=True) for i in sizes],
    [create_maze_env("maze-img-{i}x{i}-full-deterministic-v0".format(i=i), "MazeImgFull{i}x{i}Env".format(i=i), i, i, "image", 1337, full_deterministic=True) for i in sizes],
    [create_maze_env("maze-arr-{i}x{i}-deterministic-v0".format(i=i), "MazeArr{i}x{i}Env".format(i=i), i, i, "array", 1337) for i in sizes],
    [create_maze_env("maze-img-{i}x{i}-deterministic-v0".format(i=i), "MazeImg{i}x{i}Env".format(i=i), i, i, "image", 1337) for i in sizes],
    [create_maze_env("maze-arr-{i}x{i}-stochatic-v0".format(i=i), "MazeArrRnd{i}x{i}Env".format(i=i), i, i, "array") for i in sizes],
    [create_maze_env("maze-img-{i}x{i}-stochatic-v0".format(i=i), "MazeImgRnd{i}x{i}Env".format(i=i), i, i, "image") for i in sizes],
))

for env_class in envs:
    globals()[env_class.__name__] = env_class

def get_env_classes():
    return envs
    