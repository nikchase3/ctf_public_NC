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
        #TODO add more channels may help training
        self.conv1 = nn.Conv2d(6, 6, 3, padding = 1) 
        self.conv2 = nn.Conv2d(6, 6, 3, padding = 1) 
        self.conv3 = nn.Conv2d(6, 6, 3, padding = 1)
        self.relu = nn.ReLU(inplace=True)
        #TODO figure out a better way to get this dimension number
        self.fc = nn.Linear(2646, num_actions)
    
    def forward(self, state):
        '''
        Inputs{
            state: array of integers representing the grid-world (torch.Tensor)
                - shape(state_batch) must be of the form: (batch_size, num_channels, num_agents, map_x, map_y)
        } 
        Outputs{
            out: Q values for the actions corresponding to the input state (torch.Tensor)
        }
        '''
        # TODO make it work for bool array (or convert bool to int / float)
        # TODO for easier implementation, I have taken each observation on it's own (for 4 agents, and batch size 100, we 
        # have 400 'states' going through the network)
        #  
        # However, I think having a 3D convolution with the agent observations "stacked" for each transition
        # would also work and could be a cool thing to check out (for 4 agents, and batch size 100, we 
        # have 100 'stacked states' going through the network)
        
        out = self.conv1(state)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        #TODO to optimize and allow all observations to be passed through the network at once, split into 4 vectors here?
        out = out.view(out.size(0), -1)
        
        q_values = self.fc(out)

        return q_values.cpu()

###############################
## other useful classes
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        '''
        Inputs{
            state: numpy array with shape (num_agents, map_x, map_y, num_channels)
            action: numpy array with shape (num_agents,)
            reward: int
            next_state: numpy array with shape (num_agents, map_x, map_y, num_channels)
            done: bool
        }
        Actions{
            resizes a state transition and appends to the replay buffer  
        }
        '''
        
        # swap dimensions so we have (batch_size, num_agents, num_channels, map_x, map_y)
        state = np.swapaxes(np.swapaxes(state, 1, 3), 2, 3)
        next_state = np.swapaxes(np.swapaxes(next_state, 1, 3), 2, 3)

        state = np.expand_dims(np.asarray(state), 0)
        action = np.expand_dims(np.asarray(action), 0)
        next_state = np.expand_dims(np.asarray(next_state), 0)

        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        '''
        Inputs{
            batch_size: number of transitions to be sampled during a single round of training
        }
        
        Outputs{
            state: sampled states (torch.FloatTensor) 
                - shape(state) is of the form (batch_size*num_agents, num_channels, map_x, map_y)
            action: sampled actions (torch.LongTensor)
                - shape(action) is of the form (batch_size, num_agents)
            reward: sampled rewards (numpy.ndarray)
            next_state: sampled next_states (torch.FloatTensor)
                - np.shape(next_state) is of the form (batch_size*num_agents, num_channels, map_x, map_y)
            done: whether the sampled transition was the final frame for an episode (numpy.ndarray)
        }
        '''
        # state, action, reward, next_state, done = np.asarray(random.sample(self.buffer, batch_size))
        sample = np.asarray(random.sample(self.buffer, batch_size))
        
        state = np.vstack(sample[:, 0]) # gives (batch_size, num_agents, num_channels, map_x, map_y)
        action = np.vstack(sample[:, 1])
        reward = np.array(sample[:, 2], dtype = 'float')
        next_state = np.vstack(sample[:, 3])
        done = np.array(sample[:, 4])

        #TODO for the optimized version
        # state = np.concatenate(np.vstack(sample[:, 0]))  # gives (batch_size*num_agents, num_channels, map_x, map_y)      
        # action = np.expand_dims(np.concatenate(np.vstack(sample[:, 1])), axis= 1)
        # reward = np.array(sample[:, 2], dtype = 'float')
        # next_state = np.concatenate(np.vstack(sample[:, 3]))
        # done = np.array(sample[:, 4])
        
        state = torch.from_numpy(state).type(torch.cuda.FloatTensor)
        next_state = torch.from_numpy(next_state).type(torch.cuda.FloatTensor)
        action = torch.from_numpy(action).type(torch.LongTensor)
        reward = torch.from_numpy(reward).type(torch.FloatTensor)
        
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
