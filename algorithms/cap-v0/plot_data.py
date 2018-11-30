import numpy as np
import os
print(os.getcwd())
path = './checkpoints/episode_data.txt'
with open(path, 'r') as f:
    data = np.loadtxt(f)

print(data)