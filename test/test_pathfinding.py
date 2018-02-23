
import pytest

import numpy as np
from gym_pathfinding.games.pathfinding import dfs

def test_dfs():
    grid = np.array([
        [1,1,1,1,1,1,1],
        [1,0,0,0,0,0,1],
        [1,0,1,1,1,1,1],
        [1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1],
    ])

    start = (1, 1)
    goal = (3, 5)

    optimal_path_length, optimal_path = dfs(grid, start, goal)

    result_path = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)]
    
    assert optimal_path == result_path
    assert optimal_path_length == len(result_path) - 1 
    

from gym_pathfinding.envs.partially_observable_env import partial_grid

def test_partial_state():
	state = np.array([
		[1,1,1,1,1,1,1,1],
		[1,0,2,0,0,0,0,1],
		[1,0,0,0,0,0,3,1],
		[1,0,0,0,0,0,0,1],
		[1,1,1,1,1,1,1,1]
	])

	partial_state = partial_grid(state, (1, 2), 2)

	assert np.all(partial_state == np.array([
		[ 1, 1, 1, 1, 1,-1,-1,-1],
		[ 1, 0, 2, 0, 0,-1,-1,-1],
		[ 1, 0, 0, 0, 0,-1,-1,-1],
		[ 1, 0, 0, 0, 0,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1,-1]
	]))

