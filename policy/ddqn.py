# regular python stuff
import os
import numpy as np
from numpy import shape
import time
import math
import matplotlib.pyplot as plt
from collections import deque
import random

# torch neural net stuff
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

# classes for DQN
class myDQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(myDQN, self).__init__()
        # nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(1, 10, 3, 1, 0)
        self.bn1 = nn.BatchNorm2d(10)
        
        self.conv2 = nn.Conv2d(10, 10, 3, 1, 0)
        self.bn2 = nn.BatchNorm2d(10)

        self.conv3 = nn.Conv2d(10, 10, 3, 1, 0)
        self.bn3 = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(90, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 5)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(4)
        
    def forward(self, x):
        # x.size() = [batch_size, 400]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.pool(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
