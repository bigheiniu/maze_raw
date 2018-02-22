from gym.envs.registration import register
import gym_pathfinding.envs.maze_env

for env in dir(maze_env):

    if "Maze" not in env or "Env" not in env:
        continue

    env_class = getattr(maze_env, env)

    register(
        id=env_class.id,
        entry_point='gym_pathfinding.envs.maze_env:{env}'.format(env=env)
    )
