
import pytest

import numpy as np
from gym_pathfinding.games.grid_generation import path_exists

def test_path_exists():
    grid = np.array([
        [1,1,1,1,1,1,1],
        [1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1],
        [1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1],
    ])

    assert path_exists(grid, (1, 1), (1, 5))
    assert not path_exists(grid, (1, 1), (3, 5))


from gym_pathfinding.games.grid_generation import generate_grid




