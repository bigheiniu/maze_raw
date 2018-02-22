import numpy as np

from gym_pathfinding.envs.pathfinding_env import PathFindingEnv



class PartiallyObservableEnv(PathFindingEnv):
    """ PartiallyObservableEnv
        -1 = unknown
    """

    def __init__(self, width, height, observable_depth, screen_size=(640, 480), seed=None):
        super(PartiallyObservableEnv, self).__init__(width, height, screen_size, seed)

        self.observable_depth = observable_depth


    # def step(self, action):
    #     state, reward, done, info = super(PartiallyObservableEnv, self).step(action)

    #     state = partial_state(state, self.game.player, self.observable_depth)

    #     return state, reward, done, info


def partial_state(state, center, observable_depth):
    """return the centered partial state"""

    x, y = center
    offset = observable_depth

    mask = np.ones_like(state, dtype=bool)
    mask[max(0, x - offset): x + offset + 1, max(0, y - offset): y + offset + 1] = False

    state[mask] = -1
    return state


class Env(PartiallyObservableEnv):
    id="mdrmdr-v0"

    """docstring for Env"""
    def __init__(self):
        super(Env, self).__init__(9, 9, 2)
        



        
