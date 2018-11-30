
import numpy as np
import random


def one_hot(state):
    # get 4x4 state with the following being represented
    # 0 - floor (may be ice or hole)
    # 1 - agent
    # 2 - goal
    row = int(np.ceil(state / 4.0)-1)
    col = int((state % 4) - 1)

    obs_space = np.zeros([4,4])
    goal_position = 3

    obs_space[goal_position][goal_position] = 2
    obs_space[row][col] = 1
    return obs_space

obs = one_hot(1)
print(obs)
