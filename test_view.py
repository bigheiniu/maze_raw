import gym
import gym_pathfinding
from time import sleep

env = gym.make('mdrmdr-v0')

for episode in range(5):
    s = env.reset()
    
    for timestep in range(50):
        env.render()
        sleep(0.2)
        
        s, r, done, _ = env.step(env.action_space.sample())
        
        if done:
            break

env.close()