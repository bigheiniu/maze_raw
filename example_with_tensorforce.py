import numpy as np

import gym
import gym_pathfinding

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# Create an OpenAIgym environment.
environment = OpenAIGym('pathfinding-free-9x9-deterministic-v0', visualize=False)

# Network as list of layers
# - Embedding layer:
#   - For Gym environments utilizing a discrete observation space, an
#     "embedding" layer should be inserted at the head of the network spec.
#     Such environments are usually identified by either:
#     - class ...Env(discrete.DiscreteEnv):
#     - self.observation_space = spaces.Discrete(...)

network_spec = [
    # dict(type='embedding', indices=100, size=32),
    dict(type='dense', size=32),
    dict(type='dense', size=32)
]

agent = PPOAgent(
    states=environment.states,
    actions=environment.actions,
    network=network_spec,
    # Agent
    states_preprocessing=None,
    actions_exploration=None,
    reward_preprocessing=None,
    # MemoryModel
    update_mode=dict(
        unit='episodes',
        # 10 episodes per update
        batch_size=20,
        # Every 10 episodes
        frequency=20
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=5000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode='states',
    baseline=dict(
        type='mlp',
        sizes=[32, 32]
    ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        num_steps=5
    ),
    gae_lambda=0.97,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    subsampling_fraction=0.2,
    optimization_steps=25
)

# Create the runner
runner = Runner(agent=agent, environment=environment)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=3000, max_episode_timesteps=200, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)