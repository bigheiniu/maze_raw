import pygame
import numpy as np
import gym

from gym_pathfinding.games.pathfinding import PathFindingGame


class PathFindingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'array']}
    id = "pathfinding-v0"

    def __init__(self, width, height, screen_width=640, screen_height=480, seed=None):
        self.game = PathFindingGame(width, height, seed)

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.render_init = False
    
    def reset(self):
        return self.game.reset()

    def step(self, action):
        return self.game.step(action)

    def seed(self):
        return self.game.seed

    def render(self, mode='human'):
        if (mode == 'human'):
            if not self.render_init:
                self.init_render()
                self.render_init = True

            self.draw()

        elif (mode == 'array'):
            print(self.game.get_state())

    def close(self):
        self.game.quit()





    def init_render():
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("PathFindingGame")

        # self.font = pygame.font.SysFont("Arial", size=16)
        self.screen = pygame.display.set_mode((self.screen_width + 5, self.screen_height + 5), 0, 32)
        self.surface = pygame.Surface(self.screen.get_size())
        self.surface = self.surface.convert()
        self.surface.fill((255, 255, 255))
        self.tile_w = (self.screen_width + 5) / width
        self.tile_h = (self.screen_height + 5) / height

    def draw(self):

        self.surface.fill((0, 0, 0))
        
        for (x, y), value in np.ndenumerate(self.game.get_state()):
            quad = self.screen_position(x, y)
            color = self.get_color(value)

            pygame.draw.rect(self.surface, color, quad)

        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def get_color(self, value):
        return {
            0 : 0xFFFFFF,
            1 : 0x000000,
            2 : 0x00FF00,
            3 : 0xFF0000
        }[value]

    def screen_quad_position(self, x, y):
        return x * self.tile_w, y * self.tile_h, self.tile_w + 1, self.tile_h + 1

    # def entity_quad(self, position):
    #     x, y = position
    #     return x * self.tile_w, y * self.tile_h, self.tile_w, self.tile_h



def create_pathfinding_env(id, name, width, height, state_type, seed=None):

    def constructor(self):
        PathFindingEnv.__init__(self, width, height, seed=seed)
    
    pathfinding_env_class = type(name, (PathFindingEnv,), {
            "id" : id,
            "__init__": constructor
        })
    return pathfinding_env_class


# Create classes 

sizes = list(range(9, 20, 2)) + [25, 35, 55]
envs = [create_pathfinding_env("pathfinding-{i}x{i}-v0".format(i=i), "PathFinding{i}x{i}Env".format(i=i), i, i, "image") for i in sizes]

for env_class in envs:
    globals()[env_class.__name__] = env_class


def get_env_classes():
    return envs
