from gym.envs.registration import register
import gym_pathfinding.envs.pathfinding_env
import gym_pathfinding.envs.maze_env


for env_class in maze_env.get_env_classes():
    register(
        id=env_class.id,
        entry_point='gym_pathfinding.envs.maze_env:{name}'.format(name=env_class.__name__)
    )

for env_class in pathfinding_env.get_env_classes():
    register(
        id=env_class.id,
        entry_point='gym_pathfinding.envs.pathfinding_env:{name}'.format(name=env_class.__name__)
    )



# for env in dir(maze_env):
#     print(env)

#     if "Maze" not in env or "Env" not in env:
#         continue

#     env_class = getattr(maze_env, env)

#     register(
#         id=env_class.id,
#         entry_point='gym_pathfinding.envs.maze_env:{env}'.format(env=env)
#     )
