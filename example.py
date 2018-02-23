import gym
import gym_pathfinding
from time import sleep

env = gym.make('pathfinding-obstacle-25x25-deterministic-v0')

for episode in range(5):
    s = env.reset()
    
    for timestep in range(20):
        env.render()
        sleep(0.05)
        
        s, r, done, _ = env.step(env.action_space.sample())
        
        if done:
            break

env.close()