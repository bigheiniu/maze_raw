# -*- coding: utf-8 -*-
import random
from queue import PriorityQueue

import numpy as np
import scipy.misc


class PathFindingGame(object):
    """
    A PathFinding games
    state : 
        0 = nothing
        1 = wall
        2 = player
        3 = goal
    """
    # North, South, East, West
    MOUVEMENT = [(0, -1), (0, 1), (1, 0), (-1, 0)]

    def __init__(self, width=15, height=15, seed=None):
        self.width = width
        self.height = height
        self.shape = (width, height)

        self.seed = seed

        self.reset()

    def reset(self):
        # Reinitialize RNG
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.RandomState(self.seed)

        self.terminal = False

        self.grid = np.zeros(self.shape, dtype=np.int8)
        # Add borders
        self.grid[0, :] = self.grid[-1, :] = 1
        self.grid[:, 0] = self.grid[:, -1] = 1

        self.player, self.target = self.spawn_players()

        return self.get_state()

    def spawn_players(self):
        """Returns two random position on the grid."""

        xs, ys = np.where(self.grid == 0)
        free_positions = list(zip(xs, ys))

        target_spawn, player_spawn = self.rng.sample(free_positions, 2)

        # self.optimal_path_length, self.optimal_path = self.dfs(player_spawn, target_spawn)

        return player_spawn, target_spawn

    def get_state(self):
        state = np.array(self.grid, copy=True)
        state[self.player[0], self.player[1]] = 2
        state[self.target[0], self.target[1]] = 3
        return state


    def step(self, a):
        if self.terminal:
            return self.step_return(1)
        
        assert 0 <= a and a < 4

        dx, dy = self.MOUVEMENT[a]
        px, py = self.player

        next_x, next_y = px + dx, py + dy

        if self.is_legal(next_x, next_y):
            self.player = (next_x, next_y)

        if self.player == self.target:
            self.terminal = True
            return self.step_return(1)

        return self.step_return(-0.01)

    def step_return(self, reward):
        return self.get_state(), reward, self.terminal, {}
        #{"optimal_path": self.optimal_path_length}


    def is_legal(self, next_x, next_y):
        return self.grid[next_x, next_y] == 0


    # def dfs(self, start, goal):
    #     stack = [(start, [start])]

    #     possible_path = PriorityQueue()

    #     while stack:
    #         (vertex, path) = stack.pop()
    #         legal_cells = set(self.legal_directions(*vertex)) - set(path)
    #         for next in legal_cells:
    #             if next == goal:
    #                 full_path = path + [next]
    #                 length = len(path)
    #                 possible_path.put((length, full_path))
    #             else:
    #                 stack.append((next, path + [next]))

    #     return possible_path.get()

    # def legal_directions(self, posx, posy):
    #     possible_moves = [(posx + dx, posy + dy) for dx, dy in self.MOUVEMENT]
    #     return [(next_x, next_y) for next_x, next_y in possible_moves if self.is_legal(next_x, next_y)]


