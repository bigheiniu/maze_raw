import numpy as np

import gym
import gym_pathfinding

import time
import json

from tensorforce.agents import DQNAgent, Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

env = OpenAIGym('pathfinding-free-5x5-v0', visualize=False)

# network_spec = [    
#     {"type": "conv1d", "size": 8, "window": 3, "stride": 1, "padding": "SAME"},
#     {"type": "conv1d", "size": 8, "window": 3, "stride": 2, "padding": "VALID"},
#     {"type": "conv1d", "size": 8, "window": 3, "stride": 2, "padding": "VALID"},
#     {"type": "flatten"},
#     {"type": "dense", "size": 256, "activation": "relu"},
#     {"type": "dense", "size": 32, "activation": "tanh"},
#     {"type": "internal_lstm", "size": 32}

#     {"type": "conv2d", "size": 8, "window": 3, "stride": 1},
#     {"type": "flatten"},
#     {"type": "dense", "size": 32, "activation": "relu"},
#     {"type": "dense", "size": 32, "activation": "relu"}
# ]



with open("agent.json", 'r') as fp:
    agent = json.load(fp=fp)
with open("net.json", 'r') as fp:
    network = json.load(fp=fp)


agent = Agent.from_spec(
    spec=agent,
    kwargs=dict(
        states_spec=env.states,
        actions_spec=env.actions,
        network_spec=network
    )
)


# summary_spec={
#     "directory" : "./summaries/", 
#     "seconds": 1,
#     "labels": [
#         'configuration',
#         'gradients_scalar',
#         'regularization',
#         'inputs',
#         'losses',
#         'variables'
#     ]
# }

# saver_spec = {
#     # "load" : True,
#     # "file": "model.ckpt-230516.meta",
#     "directory": "./model2/",
#     "seconds": 60
# }
# agent = DQNAgent(
#     states_spec=env.states,
#     actions_spec=env.actions,
#     batched_observe=None,
#     scope='dqn',

#     # parameters specific to LearningAgents
#     summary_spec=summary_spec,
#     network_spec=network_spec,
#     discount=0.99,

#     saver_spec=saver_spec,
#     optimizer={
#         "type": "clipped_step",
#         "clipping_value": 0.1,
#         "optimizer": {
#             "type": "adam",
#             "learning_rate": 1e-3
#         }
#     },
#     entropy_regularization=None,

#     explorations_spec={
#         "type": "epsilon_anneal",
#         "initial_epsilon": 1.0,
#         "final_epsilon": 0.1,
#         "timesteps": 10000
#     },
#     states_preprocessing_spec=None,
#     reward_preprocessing_spec=None,

#     # parameters specific to MemoryAgents
#     batch_size=32,
#     memory={
#         "type": "replay", 
#         "capacity": 10000
#     },
#     first_update=10000,
#     update_frequency=4,
#     repeat_update=1,

#     # parameters specific to dqn-agents
#     target_sync_frequency=10000,
#     target_update_weight=1.0,
#     double_q_model=False,
#     huber_loss=None
# )


# Create the runner
runner = Runner(agent=agent, environment=env)

# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward:.2f})".format(
        ep=r.episode, 
        ts=r.episode_timestep, 
        reward=r.episode_rewards[-1]
    ))
    return True

report_episodes = 100

def episode_finished2(r):
    if r.episode % report_episodes == 0:
        steps_per_second = r.timestep / (time.time() - r.start_time)
        average_500 = sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))
        average_100 = sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))

        print("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}. Average reward last 500 rewards = {:0.2f}, last 100 = {:0.2f}".format(
            r.agent.episode, r.episode_timestep, steps_per_second, average_500, average_100
        ))

    return True

# Start learning
runner.run(episodes=20000, max_episode_timesteps=100, episode_finished=episode_finished2)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)