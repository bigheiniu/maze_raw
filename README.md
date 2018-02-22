# gym-pathfinding

A gym implementation for the pathfinding problem.

## Installation
```bash
git clone [url_repo]
cd gym-pathfinding
pip install .
```

## Basic Usage
```python
import gym
import gym_pathfinding
from time import sleep

env = gym.make('pathfinding-9x9-v0')

for episode in range(10):
    s = env.reset()
    
    for timestep in range(50):
        env.render()
        sleep(0.05)
        
        s, r, done, _ = env.step(env.action_space.sample())

        if done:
            break

env.close()
```

## Environnement


## Information
The initiale project come from https://github.com/cair/gym-maze

## Licence
Copyright 2017 Per-Arne Andersen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
