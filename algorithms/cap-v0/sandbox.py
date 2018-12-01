import numpy as np
import math
import matplotlib.pyplot as plt

epsilon_start = 1.0
epsilon_final = 0.02
epsilon_decay = 200000
def epsilon_by_frame(frame_idx):
    epsilon_curr = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
    
    return epsilon_curr


num_frames = 10000
e_list = np.zeros(num_frames)
for frame_idx in range(0, num_frames):
    e_list[frame_idx] = epsilon_by_frame(frame_idx)


plt.plot(e_list)
plt.show()

