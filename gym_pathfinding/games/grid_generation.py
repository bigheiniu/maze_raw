

import random

import numpy as np
# from gym_pathfinding.games.pathfinding import path_exists


def generate_grid(shape, *, grid_type="free", generation_seed=None, spawn_seed=None):
    """ 
    Generate a grid

    shape : (width, height)
    grid_type : {"free", "with_obstacle", "maze") 
    """

    # while True:
    #     create_with_obstacle
    #     if path_exists(grid, start, goal):
    #         return grid, start, goal
        
    grid = create_grid(shape, grid_type, generation_seed=generation_seed)
    start, goal = spawn_start_goal(grid, spawn_seed=spawn_seed)

    return grid, start, goal


def create_grid(shape, grid_type, generation_seed=None):
    return {
        "free" : init_grid(shape),
        "maze" : create_maze(shape, generation_seed),
        # "with_obstacle" : create_with_obstacle(shape, generation_seed),
    }[grid_type]


def spawn_start_goal(grid, spawn_seed=None):
    """Returns two random position on the grid."""

    xs, ys = np.where(grid == 0)
    free_positions = list(zip(xs, ys))

    start, goal = random.Random(spawn_seed).sample(free_positions, 2)

    return start, goal

def init_grid(shape):
    grid = np.zeros(shape, dtype=np.int8)

    # Add borders
    grid[0, :] = grid[-1, :] = 1
    grid[:, 0] = grid[:, -1] = 1
    return grid


# def create_with_obstacle(shape, generation_seed=None):
    # pass    

def create_maze(shape, generation_seed=None, complexity=.75, density=.50):
    # Only odd shapes
    shape = ((shape[0] // 2) * 2 + 1, (shape[1] // 2) * 2 + 1)

    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * ((shape[0] // 2) * (shape[1] // 2)))

    rng = np.random.RandomState(generation_seed)

    grid = init_grid(shape)

    # Make aisles
    for i in range(density):
        x, y = rng.random_integers(0, shape[1] // 2) * 2, rng.random_integers(0, shape[0] // 2) * 2
        grid[y, x] = 1

        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[rng.random_integers(0, len(neighbours) - 1)]

                if grid[y_, x_] == 0:
                    grid[y_, x_] = 1
                    grid[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    return grid


# North, South, East, West
MOUVEMENT = [(0, -1), (0, 1), (1, 0), (-1, 0)]

def is_legal(grid, next_x, next_y):
    return grid[next_x, next_y] == 0


def path_exists(grid, start, goal):
    """ 
    Test if a path exist from start to goal
    It's a Depth-first Search
    """

    stack = [(start, [start])]

    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        visited.add(vertex)

        legal_cells = set(legal_directions(grid, *vertex)) - visited
        for next in legal_cells:
            if next == goal:
                return True
            stack.append((next, path + [next]))

    return False

def legal_directions(grid, posx, posy):
    possible_moves = [(posx + dx, posy + dy) for dx, dy in MOUVEMENT]
    return [(next_x, next_y) for next_x, next_y in possible_moves if is_legal(grid, next_x, next_y)]
