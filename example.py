import gym
import gym_pathfinding
from time import sleep

env = gym.make('partially-observable-pathfinding-obstacle-25x25-d5-v0')
env.seed(1) # for full-deterministic environment

for episode in range(5):
    s = env.reset()
    
    for timestep in range(20):
        env.render()
        sleep(0.05)
        
        s, r, done, _ = env.step(env.action_space.sample())
        
        if done:
            break

env.close()