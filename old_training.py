import math, random

import gym
from gym import wrappers
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import random
import gym_cap

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
from torch.autograd import Variable
import torch.nn.functional as F
from collections import deque
from IPython.display import clear_output

#################################
env_id = "cap-v0"
env = gym.make(env_id)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args**kwargs)

#################################
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

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space_blue.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data.item()
        else:
            action = random.randrange(env.action_space.n)
        return action

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state) 

    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def plot(frame_idx, rewards, losses):
    #clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

#################################
if __name__ == '__main__': 
    #TODO change to take in only the observation of a single unit at a time
    current_model = DQN(env.observation_space_blue.shape[0], env.action_space.n)
    target_model  = DQN(env.observation_space_blue.shape[0], env.action_space.n)

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()
        
    optimizer = optim.Adam(current_model.parameters())

    replay_buffer = ReplayBuffer(1000)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    num_frames = 1000
    batch_size = 100
    gamma      = 0.99
    update_target(current_model, target_model)

    losses = []
    all_rewards = []
    episode_reward = 0
    
    #################################
    # set checkpoint save directory
    ckpt_dir = './checkpoints/'
    ckpt_paths = []
    dir_exist = os.path.exists(ckpt_dir)
    if not dir_exist:
        os.mkdir(ckpt_dir)
    
    ckpt_env_id_dir = os.path.join(ckpt_dir, env_id)
    dir_exist = os.path.exists(ckpt_env_id_dir)
    if not dir_exist:
        os.mkdir(ckpt_env_id_dir)

    ###################################
    # define the DDQN  in this file, having a different policy file is annoying as shit 
    state = env.reset(map_size=20, policy_blue = None, policy_red = None)

    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(state, epsilon)
        env.render()
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
            
        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(batch_size)
            losses.append(loss.data[0])
            
        if frame_idx % 200 == 0:
            print('Frame_idx: {} / {}'.format(frame_idx, num_frames))
            #plot(frame_idx, all_rewards, losses)
            
        if frame_idx % 100 == 0:
            update_target(current_model, target_model)
        
        # save checkpoints every so often
        if frame_idx % 200 == 0:
            ckpt_fn = 'frame_' + str(frame_idx) + '.ckpt'
            save_dir = os.path.join(ckpt_env_id_dir, ckpt_fn)
            torch.save(current_model.state_dict(), save_dir)
            print('saving checkpoint to: ', save_dir)
            # save a list of checkpoint names for easy loading later 
            ckpt_paths.append(save_dir)

    ckpt_names_fn = 'checkpoint_paths.txt'
    ckpt_names_path = os.path.join(ckpt_env_id_dir, ckpt_names_fn)

    with open(ckpt_names_path, 'w') as f:
        json.dump(ckpt_paths, f)
    


