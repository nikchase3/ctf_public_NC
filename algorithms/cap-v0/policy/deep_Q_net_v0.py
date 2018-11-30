'''
https://gist.github.com/simoninithomas/7611db5d8a6f3edde269e18b97fa4d0c#file-deep-q-learning-with-doom-ipynb
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#sphx-glr-download-intermediate-reinforcement-q-learning-py

Implementation of Q-Learning for the CTF environment

Created Date: Wednesday, October 3rd 2018, 5:52:26 pm
Author: Jacob Heglund

- for an easier first implementation, just deal with 1 blue team agent, the rest are removed from the sim
- also, give the agent full state observability
'''
###########################################
from research_training import myDQN, ReplayBuffer
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

###########################################
# class for generating actions from observations
class PolicyGen:
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    
    Methods:
        gen_action: Required method to generate a list of actions.
    """
    
    def __init__(self, env, device, free_map, agent_list, hyperparam_dict):
        super().__init__()
        """Constuctor for policy class.
        
        This class can be used as a template for policy generator.
        
        Args:
            free_map (np.array): 2d map of static environment (use as observation for the fully observable case)
            agent_list (list): list of all friendly units.
        """
        self.env = env
        self.epsilon_start = hyperparam_dict["epsilon_start"]
        self.epsilon_final = hyperparam_dict["epsilon_final"]
        self.epsilon_decay = hyperparam_dict["epsilon_decay"]
        self.gamma = hyperparam_dict["gamma"]
        
        self.num_states = int(self.env.observation_space_blue.shape[0] * self.env.observation_space_blue.shape[1])
        self.num_actions = int(self.env.action_space.n)

        self.device = device
        #TODO generalize to red and blue
        #TODO generalize sizes for the number of agents, different maps
        #TODO do this init in the training file, not here!
        self.current_model = myDQN(self.num_states, self.num_actions)
        self.target_model  = myDQN(self.num_states, self.num_actions)

        # send to GPU if available, otherwise keep on CPU
        #TODO do this init in the training file, not here!
        self.current_model = self.current_model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        self.replay_buffer = ReplayBuffer(hyperparam_dict["replay_buffer_size"])

        self.optimizer = optim.Adam(self.current_model.parameters())

    def gen_action(self, agent_list, observation, frame_idx, train, free_map=None):
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
        #TODO add multiple agent functionality with a for loop 
        
        if train == True:
            epsilon = self.epsilon_by_frame(frame_idx)
            if random.random() > epsilon:
                state = observation
                state = torch.FloatTensor(np.float32(state))
                state = state.to(self.device).unsqueeze(0).unsqueeze(0)
                q_value =  self.current_model.forward(state)
                max_q, action = q_value[0].max(0)
                max_q = float(max_q)
                action = int(action)
                
            else:
                action = random.randrange(self.num_actions)
        
        # for evaluation
        elif train == False:
            #TODO fix the CNN input dimensions
            state = observation.flatten()
            state = torch.FloatTensor(np.float32(state))
            state = state.to(self.device)
                
            q_value =  self.current_model.forward(state)
            max_q, action = q_value.max(0)

        #TODO get all agent actions for one team here
        action_out = []
        action_out.append(action)
        return action_out
                
    #######################   
    # functions for DDQN
    def update_target_network(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device).unsqueeze(1)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        q_values = self.current_model.forward(state)
        next_q_values = self.current_model.forward(next_state)
        next_q_state_values = self.target_model.forward(next_state)

        q_value       = q_values.gather(1, action).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def epsilon_by_frame(self, frame_idx):
        epsilon_curr = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)
        
        return epsilon_curr
    








