# -*- coding: utf-8 -*-
import random
from queue import PriorityQueue

import numpy as np
import scipy.misc
from .astar import astar
from .gridworld import generate_grid, MOUVEMENT
from gym_pathfinding.games.gridworld import generate_grid, is_legal, MOUVEMENT
import operator

ACTION = {mouvement: action for action, mouvement in dict(enumerate(MOUVEMENT)).items()}

class PathFindingGame(object):
    """
    A PathFinding games
    state : 
        0 = nothing
        1 = wall
        2 = player
        3 = goal
    """

    def __init__(self, lines=15, columns=15, *, grid_type="free"):
        self.lines = lines
        self.columns = columns
        self.shape = (lines, columns)

        self.grid_type = grid_type
        
        self.terminal = True

        self.generation_seed = self.spawn_seed = None
        self.grid = self.player = self.target = None

    def seed(self, generation_seed=None, spawn_seed=None):
        self.generation_seed = generation_seed
        self.spawn_seed = spawn_seed

    # def reset(self):
    #
    #     self.terminal = False
    #
    #     self.grid, self.player, self.target = generate_grid(
    #         self.shape,
    #         grid_type=self.grid_type,
    #         generation_seed=self.generation_seed,
    #         spawn_seed=self.spawn_seed
    #     )
    #
    #     return self.get_state()

    def reset(self):

        self.terminal = False

        self.grid, self.player, self.target = generate_grid(
            self.shape,
            grid_type=self.grid_type,
            generation_seed=self.generation_seed,
            spawn_seed=self.spawn_seed
        )
        path, action_planning = self.compute_action_planning(self.grid, self.player, self.target)
        sign_list = self.build_sign(action_planning)
        self.grid = self.add_sign(action_planning, path, self.grid, sign_list)
        return self.get_state()

    def get_state(self):
        """ return a (n, n) grid """
        state = np.array(self.grid, copy=True)
        state[self.player[0], self.player[1]] = 2
        state[self.target[0], self.target[1]] = 3
        return state


    def step(self, a):
        if self.terminal:
            return self.step_return(1)
        
        assert 0 <= a and a < 4

        di, dj = MOUVEMENT[a]
        pi, pj = self.player

        next_i, next_j = pi + di, pj + dj

        if is_legal(self.grid, next_i, next_j):
            self.player = (next_i, next_j)

        if self.player == self.target:
            self.terminal = True
            return self.step_return(1)

        return self.step_return(-0.01)

    def step_return(self, reward):
        return self.get_state(), reward, self.terminal, ""

    def build_sign(self, action_planning):
        '''
        random put a direction sign at the observation area when change direction
        :param action_planning:
        :return:
        '''
        space = [0]
        action_wanted = [action_planning[0]]
        result = np.zeros(len(action_planning), dtype=np.int)
        for i in range(len(action_planning) - 1):
            if action_planning[i] != action_planning[i + 1]:
                space.append(i + 1)
                action_wanted.append(action_planning[i + 1])
        for i in range(len(space) - 1):
            index = np.random.randint(low=space[i], high=space[i + 1], size=1)
            # 5 left turn
            # 6 right turn
            # action: 0 -> up, 1 -> down, 2 -> left, 3 -> right
            # mouvement = (-1, 0), (1, 0), (0, -1), (0, 1)
            action_cur = action_wanted[i]
            action_next = action_wanted[i + 1]
            if ((action_cur == 2 and action_next == 1) or (action_cur == 1 and action_next == 3) or (
                    action_cur == 3 and action_next == 0) or (action_cur == 0 and action_next == 2)):
                # make left turn
                result[index] = 5
            else:
                # make right turn
                result[index] = 6
        return result

    def add_sign(self, action_planning, path, grid, sign_list):
        # ATTENTION: manupulate ob depth tp 5
        observable_depth = 3
        for timestep in range(len(action_planning)):
            # at the end, pad the episode with the last action
            action = action_planning[timestep]
            position = path[timestep]
            sign = sign_list[timestep]
            # Compute the partial grid
            # -1: unseen place
            _partial_grid = self.partial_grid(grid, position, observable_depth)
            if sign != 0:
                # 2 up, 3 down, 4 left, 5 right
                # add sign to where the item can seen and should be in obstacle
                pos_loc = np.argwhere(_partial_grid == 0)
                np.random.shuffle(pos_loc)
                pos_loc_x = pos_loc[0][0]
                pos_loc_y = pos_loc[0][1]
                grid[pos_loc_x, pos_loc_y] = sign
        return grid

    def get_state(self):
        """ return a (n, n) grid """
        state = np.array(self.grid, copy=True)
        state[self.player[0], self.player[1]] = 2
        state[self.target[0], self.target[1]] = 3
        return state

    def compute_action_planning(self, grid, start, goal):
        path = astar(grid, start, goal)

        action_planning = []
        for i in range(len(path) - 1):
            pos = path[i]
            next_pos = path[i + 1]

            # mouvement = (-1, 0), (1, 0), (0, -1), (0, 1)
            mouvement = tuple(map(operator.sub, next_pos, pos))

            action_planning.append(ACTION[mouvement])

        return path, action_planning

    def partial_grid(self, grid, center, observable_depth):
        """return the centered partial state, place -1 to non-visible cells"""

        i, j = center
        offset = observable_depth

        mask = np.ones_like(grid, dtype=bool)
        mask[max(0, i - offset): i + offset + 1, max(0, j - offset): j + offset + 1] = False

        _grid = np.array(grid, copy=True)
        _grid[mask] = -1
        return _grid

