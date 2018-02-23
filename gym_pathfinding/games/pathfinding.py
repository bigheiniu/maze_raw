# -*- coding: utf-8 -*-
import random
from queue import PriorityQueue

import numpy as np
import scipy.misc

from gym_pathfinding.games.grid_generation import generate_grid

class PathFindingGame(object):
    """
    A PathFinding games
    state : 
        0 = nothing
        1 = wall
        2 = player
        3 = goal
    """

    def __init__(self, width=15, height=15, *, grid_type="free", generation_seed=None, spawn_seed=None):
        self.width = width
        self.height = height
        self.shape = (width, height)

        self.grid_type = grid_type
        
        self.generation_seed = generation_seed
        self.spawn_seed = spawn_seed

        self.terminal = True

        self.grid = self.player = self.target = None

    def reset(self):

        self.terminal = False

        self.grid, self.player, self.target = generate_grid(
            self.shape, 
            grid_type=self.grid_type,
            generation_seed=self.generation_seed, 
            spawn_seed=self.spawn_seed
        )

        return self.get_state()

    def get_state(self):
        state = np.array(self.grid, copy=True)
        state[self.player[0], self.player[1]] = 2
        state[self.target[0], self.target[1]] = 3
        return state


    def step(self, a):
        if self.terminal:
            return self.step_return(1)
        
        assert 0 <= a and a < 4

        dx, dy = MOUVEMENT[a]
        px, py = self.player

        next_x, next_y = px + dx, py + dy

        if is_legal(self.grid, next_x, next_y):
            self.player = (next_x, next_y)

        if self.player == self.target:
            self.terminal = True
            return self.step_return(1)

        return self.step_return(-0.01)

    def step_return(self, reward):
        return self.get_state(), reward, self.terminal, ""


# North, South, East, West
MOUVEMENT = [(0, -1), (0, 1), (1, 0), (-1, 0)]

def is_legal(grid, next_x, next_y):
    return grid[next_x, next_y] == 0



