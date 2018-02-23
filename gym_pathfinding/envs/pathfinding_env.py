import pygame
import numpy as np
import gym
from gym import error, spaces, utils

from gym_pathfinding.games.pathfinding import PathFindingGame
from gym_pathfinding.rendering import GridViewer


class PathFindingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'array']}

    def __init__(self, lines, columns, *, grid_type="free", screen_size=(640, 640), generation_seed=None, spawn_seed=None):
        self.game = PathFindingGame(lines, columns, 
            grid_type=grid_type, 
            generation_seed=generation_seed, 
            spawn_seed=spawn_seed
        )
        self.game.reset()
        
        self.viewer = GridViewer(screen_size[0], screen_size[1], lines, columns)

        shape = self.game.get_state().shape
        self.observation_space = spaces.Box(low=0, high=3, shape=shape, dtype=np.int8)
        self.action_space = spaces.Discrete(4)
    
    def reset(self):
        return self.game.reset()

    def step(self, action):
        return self.game.step(action)

    def seed(self):
        return self.game.generation_seed or self.game.spawn_seed or None

    def render(self, mode='human'):
        grid = self.game.get_state()

        if (mode == 'human'):
            self.viewer.draw(grid)
        elif (mode == 'array'):
            return grid

    def close(self):
        self.viewer.stop()


def create_pathfinding_env(id, name, lines, columns, grid_type="free", generation_seed=None, spawn_seed=None):

    def constructor(self):
        PathFindingEnv.__init__(self, lines, columns, 
            grid_type=grid_type,
            generation_seed=generation_seed, 
            spawn_seed=spawn_seed
        )
    
    pathfinding_env_class = type(name, (PathFindingEnv,), {
        "id" : id,
        "__init__": constructor
    })
    return pathfinding_env_class


# Create classes 

sizes = list(range(9, 20, 2)) + [25, 35, 55]
envs = [
    create_pathfinding_env(
        id="pathfinding-{type}-{n}x{n}{deterministic}-v0".format(
            type=grid_type, n=size,
            deterministic="-deterministic" if seed else ""
        ),
        name="PathFinding{type}{n}x{n}{deterministic}Env".format(
            type=grid_type.capitalize(), n=size,
            deterministic="Deterministic" if seed else ""
        ),
        grid_type=grid_type,
        lines=size, 
        columns=size,
        generation_seed=seed
    ) 
    for grid_type in ["free", "obstacle", "maze"]
    for seed in [None, 1]
    for size in sizes 
]

for env_class in envs:
    globals()[env_class.__name__] = env_class


def get_env_classes():
    return envs
