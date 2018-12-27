import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

###############################
## network architectures
class DQN(nn.Module):
    def __init__(self, num_states, num_actions, batch_size):
        super(DQN, self).__init__()
        self.batch_size = batch_size
        # this CNN architecture will maintain the size of the input throughout the convolutions
        self.conv1 = nn.Conv2d(6, 6, 3, padding = 1) 
        self.conv2 = nn.Conv2d(6, 6, 3, padding = 1) 
        self.conv3 = nn.Conv2d(6, 6, 3, padding = 1)
        self.relu = nn.ReLU(inplace=True)
        #TODO figure out a better way to get this dimension number
        self.fc = nn.Linear(2166, num_actions)
    
    def forward(self, state_batch):
        '''
        inputs{
            state (fully observable state) - square np array of integers representing the grid-world
        } 
        outputs{
            out - Q values for the actions corresponding to the input state
        }
        '''
        # input to CNN needs to be of the following form:
        # dims_state = batch_size, num_channels, map_size_x, map_size_y

        if np.shape(state_batch)[0] == self.batch_size:
            state = np.swapaxes(state_batch, 2, 4)
            state = torch.from_numpy(state).type(torch.cuda.FloatTensor).squeeze(1)
        
        else:
            state = np.swapaxes(state_batch, 1, 3)
            state = torch.from_numpy(state).type(torch.cuda.FloatTensor)

        out = self.conv1(state)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        
        q_values = self.fc(out)

        return q_values.cpu()

###############################
## other useful classes
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        action, reward, done = np.asarray(action), np.asarray(reward), np.asarray(done)
        
        action = torch.from_numpy(action).type(torch.LongTensor).unsqueeze(1)
        reward = torch.from_numpy(reward).type(torch.FloatTensor)
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
