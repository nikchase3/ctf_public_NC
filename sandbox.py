import numpy as np
from numpy import shape
import matplotlib.pyplot as plt
# regular python stuff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import gym
from gym import wrappers
import gym_cap

with open('./checkpoints/episode_data.txt', 'r') as f:
    episode_data = np.loadtxt(f)

episode_arr = episode_data[0, :]
reward_arr = episode_data[1,:]
episode_duration_arr = episode_data[2,:]

avg_duration = np.mean(episode_duration_arr)
print(avg_duration, ' frames')
#plt.plot(episode_arr, episode_duration_arr)
#plt.show()