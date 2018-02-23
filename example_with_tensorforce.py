import numpy as np

import gym
import gym_pathfinding

from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# Create an OpenAIgym environment.
env = OpenAIGym('pathfinding-free-9x9-deterministic-v0', visualize=False)

# Network as list of layers
# - Embedding layer:
#   - For Gym environments utilizing a discrete observation space, an
#     "embedding" layer should be inserted at the head of the network spec.
#     Such environments are usually identified by either:
#     - class ...Env(discrete.DiscreteEnv):
#     - self.observation_space = spaces.Discrete(...)

network_spec = [    
    {"type": "conv1d", "size": 8, "window": 3, "stride": 1, "padding": "SAME"},
    {"type": "conv1d", "size": 8, "window": 3, "stride": 2, "padding": "VALID"},
    {"type": "conv1d", "size": 8, "window": 3, "stride": 2, "padding": "VALID"},
    {"type": "flatten"},
    {"type": "dense", "size": 256, "activation": "relu"},
    {"type": "dense", "size": 32, "activation": "tanh"},
    {"type": "internal_lstm", "size": 32}
]

summary_spec={
    "directory" : "./summaries/", 
    "seconds": 60,
    "labels": [
        'configuration',
        'gradients_scalar',
        'regularization',
        'inputs',
        'losses',
        'variables'
    ]
}


saver_spec = {
    "directory": "./models/",
    "seconds": 60
}

agent = DQNAgent(
    states_spec=env.states,
    actions_spec=env.actions,
    batched_observe=None,
    scope='dqn',

    # parameters specific to LearningAgents
    summary_spec=summary_spec,
    network_spec=network_spec,
    discount=0.99,

    saver_spec=saver_spec,
    optimizer={
        "type": "clipped_step",
        "clipping_value": 0.1,
        "optimizer": {
            "type": "adam",
            "learning_rate": 1e-3
        }
    },
    entropy_regularization=None,

    explorations_spec={
        "type": "epsilon_anneal",
        "initial_epsilon": 0.5,
        "final_epsilon": 0.0,
        "timesteps": 10000
    },
    states_preprocessing_spec=None,
    reward_preprocessing_spec=None,

    # parameters specific to MemoryAgents
    batch_size=32,
    memory={
        "type": "replay", 
        "capacity": 10000
    },
    first_update=10000,
    update_frequency=4,
    repeat_update=1,

    # parameters specific to dqn-agents
    target_sync_frequency=10000,
    target_update_weight=1.0,
    double_q_model=False,
    huber_loss=None
)


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


# Start learning
runner.run(episodes=3000, max_episode_timesteps=100, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)