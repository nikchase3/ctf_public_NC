'''
TODO: look at this 
https://gist.github.com/simoninithomas/7611db5d8a6f3edde269e18b97fa4d0c#file-deep-q-learning-with-doom-ipynb
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#sphx-glr-download-intermediate-reinforcement-q-learning-py

Implementation of Q-Learning for the CTF environment

Created Date: Wednesday, October 3rd 2018, 5:52:26 pm
Author: Jacob Heglund
Questions:

- Assumptions
1. Partial Observability
    - at each timestep, the each team team recieves all information that their respective team members observe
    - this aggregate information is then put together into a of the environment, which is sent into the Q-network

2. fully deterministic environment
    - an action has a 100% probability of occuring if specified in the state vector

X1. full state observability (didn't use b/c I would need to figure out how to get the 'top down view' as the input to the NN)
    - an agent takes in the raw pixel values for the entire map as input
X2. a centralized agent controls each team (didn't use b/c I would need to figure out how to get a tuple of actions from the network)
    - the agent's state at any timestep is a tuple containing 4, (x,y) positions in the map

- Implementation Notes v0
1. Uses Q-Learning by taking in full observation space and spitting out actions for each of the 4 team members
2. Utilizes a CNN to process spatial information
    - the partially observable map is fed directly into the CNN 
'''


#TODO - high level things 
# for now we'll have Q-Learning for only 1 agent, the rest will stay still
# do frozen lake with function approximation, then port the code over to this environment

###########################################
# regular python stuff
import os
import numpy as np
from numpy import shape
import time
import matplotlib.pyplot as plt

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

###########################################
class PolicyGen:
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    
    Methods:
        gen_action: Required method to generate a list of actions.
    """
    
    def __init__(self, free_map, agent_list):
        """Constuctor for policy class.
        
        This class can be used as a template for policy generator.
        
        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.
        """
        
    def gen_action(self, agent_list, observation, free_map=None):
        """Action generation method.

        This is a required method that generates list of actions corresponding 
        to the list of units. 
        
        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            free_map (np.array): 2d map of static environment (optional).
            
        Returns:
            action_out (list): list of integers as actions selected for team.
        """
        '''
        ##################################
        # visualization code
        #TODO: plots for loss over time
        #TODO: plots for reward over time
        #TODO: duration of each training episode over tiem

        ##################################
        # learning model 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #TODO: implement TD Loss -> allows us to update the Q function 
        # in a way consistent with the Bellman Equation
        
        #TODO: choose an optimizer
        optim
        #TODO: double check experience replay implementation
        # i have two papers to read about this  
        class ReplayMemory(object):
            def __init__(self, capacity):
                self.capacity = capacity
                self.memory = []
                self.position = 0

            def push(self, *args):
                """Saves a transition."""
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = Transition(*args)
                self.position = (self.position + 1) % self.capacity

            def sample(self, batch_size):
                return random.sample(self.memory, batch_size)

            def __len__(self):
                return len(self.memory)

        # model
        #I need to figure out how to get the Q-network to output multiple values
        # to control multiple agents at the same time
        class DQN(nn.Module):
            def __init__(self):
                #TODO: this seems like a relatively common DQN network architecture, but
                # check around to see if there are any better
                # extending to a deep network is super easy now using Pytorch!
                super(DQN, self).__init__()
                self.conv1 = nn.Conv2d(1, 16, 5, 2)
                self.bn1 = nn.BatchNorm2d(16)
                self.conv2 = nn.Conv2d(16, 32, 5, 2)
                self.bn2 = nn.BatchNorm2d(32)
                self.conv3 = nn.Conv2d(32, 32, 5, 2)
                self.bn3 = nn.BatchNorm2d(32)
                #TODO: check this is the right size
                self.fc = nn.Linear(448, self.sizeOutput)
            
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        
        ###########################
        # functions for training
        def obs2vec(x):
            #take the input observation and return a float vector
            return x.astype(np.float).ravel()

        
        #TODO: implement an epsilon greedy action 
        def chooseAction(state):
            print('stuff')
            #TODO: set a threshold for epsilon, if above, take the "optimal action" as specified by Q values
            # otherwise, take a random action
            
            #TODO: decay epsilon over time
        ###########################
        # define model, hyperparameters, and environment parameters
        action_out = []
        self.sizeInput = 20 * 20 # size of the environment
        self.sizeOutput = 5 # size of the action space (for one agent)

        model = DQN().to(device)

        
            
        ###########################

        def train():
            model.train()

            #TODO batch memory

            # 
            
            # compute Q values for current state and action

            # compute value function for next timestep

            # computer Huber Loss

            # optimize model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        '''
        action_out = [0, 0, 0, 0] # TODO: remove this, do nothing at each timestep
        return action_out