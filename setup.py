from setuptools import setup

setup(name='gym_pathfinding',
      version='0.0.1',
      install_requires=['gym', 'cython', 'numpy', 'scipy', 'pygame'],
      packages=['gym_pathfinding', 'gym_pathfinding.envs']
      )
