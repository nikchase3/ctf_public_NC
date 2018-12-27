######################
## import custom modules
import policy.random_actions
import gym_cap

######################
## regular imports
import sys
import argparse
import os
import gym
import numpy as np
from numpy import shape
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from collections import deque
import random

## Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

## Data Processing Module
from utility.dataModule import one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import Experience_buffer, discount_rewards

######################
parser = argparse.ArgumentParser(description = 'Set training parameters')
parser.add_argument('--run', type = int, help = 'set the run number within a batch of sims', default = 0)
parser.add_argument('--epi', type = int, help = 'set the episode of the checkpoint to load', default = 0)
args = vars(parser.parse_args())

run_number = args['run']
load_ckpt = args['epi']

print('run number:', run_number)

# set checkpoint save directory
ckpt_dir = './checkpoints_' + str(run_number)
dir_exist = os.path.exists(ckpt_dir)
if not dir_exist:
    os.mkdir(ckpt_dir)

######################
#TODO if loading a checkpoint, make sure epsilon and frame count is reflected in that!
frame_count = 0

######################
## "deep" Q-network 
#TODO only has a single channel, will increasing channels help for this?

class myDQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(myDQN, self).__init__()
        # this CNN architecture will maintain the size of the input throughout the convolutions
        self.conv1 = nn.Conv2d(1, 1, 3, padding = 1) 
        self.conv2 = nn.Conv2d(1, 1, 3, padding = 1) 
        self.conv3 = nn.Conv2d(1, 1, 3, padding = 1)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(num_states, num_actions)
    
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
        # dims_state = batch_size, num_channels (1), map_size_x, map_size_y
        if np.shape(state_batch)[0] == batch_size:
            state = torch.from_numpy(state_batch).type(torch.cuda.FloatTensor).unsqueeze(1)
        else:
            state = torch.from_numpy(state_batch).type(torch.cuda.FloatTensor).unsqueeze(0).unsqueeze(0)
        
        out = self.conv1(state)
        out = self.relu(out)
       
        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = out.view(out.size(0), -1)
        q_values = self.fc(out)
       
        return q_values.cpu()

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

######################
# file / data management
def save_data(step_list, reward_list, loss_list, epsilon_list):
    # window = int(num_episodes/10)
    window = 100
    episode_fn = 'episode_data.txt'
    episode_path = os.path.join(ckpt_dir, episode_fn)

    step_list = np.asarray(step_list)
    reward_list = np.asarray(reward_list)
    loss_list = np.asarray(loss_list) 
    epsilon_list = np.asarray(epsilon_list)
    episode_save = np.vstack((step_list, reward_list, loss_list, epsilon_list))

    with open(episode_path, 'w') as f:
        np.savetxt(f, episode_save)

    plt.figure(figsize = [10,8])
    plt.subplot(211)
    plt.plot(pd.Series(step_list).rolling(window).mean())
    plt.title('Step Moving Average ({}-episode window)'.format(window))
    plt.ylabel('Moves')
    plt.xlabel('Episode')

    plt.subplot(212)
    plt.plot(pd.Series(reward_list).rolling(window).mean())
    plt.title('Reward Moving Average ({}-episode window)'.format(window))
    plt.ylabel('Reward')
    plt.xlabel('Episode')

    plt.tight_layout(pad=2)
    # plt.show()
    file_name = 'training_data.png'
    file_path = os.path.join(ckpt_dir, file_name)
    plt.savefig(file_path, dpi=300)
    plt.close()
    
def save_model(episode):
    model_fn = 'dqn_episode_' + str(episode) + '.model'
    save_path = os.path.join(ckpt_dir, model_fn)
    torch.save(online_model, save_path)

def load_model(run_number, episode):
    if (episode == 0):
        online_model = myDQN(num_states, num_actions)
        online_model = online_model.to(device)
    else:
        online_model = myDQN(num_states, num_actions)
        
        # load only the state dict
        model_fn = 'dqn_episode_' + str(episode) + '.model'

        model_path = os.path.join(ckpt_dir, model_fn)
        load_model = torch.load(model_path)
        online_model.load_state_dict(load_model.state_dict())
        online_model = online_model.to(device)
    return online_model

def setup_storage(run_number, load_ckpt):
    #TODO setup an archive folder for all runs with the folders as timestamps of when the run began
    # -> this won't depend on the 'run_number', and is a good way to make sure data isn't being overwritten
    if load_ckpt == 0:
        step_list = []
        reward_list = []
        loss_list = []
        epsilon_list = []

    else:
        data_path = os.path.join(ckpt_dir, 'episode_data.txt')
        with open(data_path, 'r') as f:
            data = np.loadtxt(f)
        
        step_list = np.ndarray.tolist(data[0, 0:load_ckpt])
        reward_list = np.ndarray.tolist(data[1, 0:load_ckpt])
        
        #TODO get rid of this garbage for future runs
        loss_list = np.ndarray.tolist(np.zeros([1, load_ckpt]))
        epsilon_list = np.ndarray.tolist(np.zeros([1, load_ckpt]))

        #TODO uncomment for future runs where these things are being saved
        # loss_list = np.ndarray.tolist(data[2, :])
        # epsilon_list = np.ndarray.tolist(data[3, :])

    return step_list, reward_list, loss_list, epsilon_list

######################
# RL functions
def gen_action(state, epsilon):
    '''
    TODO: make it work for multiple agents
    action_list = []
    
    for agent in len(blue_team_agent_list):
        if np.random.rand(1) < epsilon:
            q_values = online_model(state)
            action = env.action_space.sample()

        else:
            q_values = online_model(state)
            _, action = torch.max(q_values, 1)
            action = action.item()
        
        action_list.append(action)
    '''    
    if np.random.rand(1) < epsilon:
        action = env.action_space.sample()

    else:
        with torch.no_grad():
            q_values = online_model(state)
            _, action = torch.max(q_values, 1)
            action = action.item()

    return action

def epsilon_by_frame(frame_count):
    epsilon_curr = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_count / epsilon_decay)
    return epsilon_curr

def train_online_network(batch_size):
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
    
    # get all the q-values for actions at state and next state
    q_values = online_model(state_batch)
    next_q_values = online_model(next_state_batch)
    
    # find Q(s, a) for the action taken during the sampled transition
    state_action_value = q_values.gather(1, action_batch).squeeze(1)
    
    # Find Q(s_next, a_next) for an optimal agent (take the action with max q-value in s_next)
    next_state_action_value = next_q_values.max(1)[0]
    
    # if done, multiply next_state_action_value by 0, else multiply by 1
    one = np.ones(batch_size)
    done_mask = torch.from_numpy(one-done_batch).type(torch.FloatTensor)
    
    discounted_next_value = (gamma * next_state_action_value)
    discounted_next_value = discounted_next_value.type(torch.FloatTensor)
    
    # Compute the target of current q-values
    target_value = reward_batch + discounted_next_value * done_mask

    loss = criterion(state_action_value, target_value)
    
    online_model.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def play_episode():
    global frame_count

    # this gives the partially observable state
    # state = env.reset(map_size = map_size, policy_red = policy_red)
    env.reset(map_size = map_size, policy_red = policy_red)

    episode_length = 0.
    episode_loss = 0.
    done = 0

    while (done == 0):
        # set exploration rate for this frame
        epsilon = epsilon_by_frame(frame_count)
        
        state = env.get_full_state
        #TODO get centering working!!!
        action = gen_action(state, epsilon)
        next_state, reward, done, _ = env.step(entities_action = [action])
        episode_length += 1
        frame_count += 1
                
        # stop the episode if it goes too long
        if episode_length >= max_episode_length:
            reward = -100.
            done = True

        # store the transition in replay buffer
        replay_buffer.push(state, action, reward/100. , next_state, done)

        # train the network
        if len(replay_buffer) > replay_buffer_init:
            if (frame_count % train_online_network_frame) == 0:
                loss = train_online_network(batch_size)
                episode_loss += loss

        # end the episode         
        if done:
            return episode_loss, episode_length, reward, epsilon
        
######################
# make environment
env_id = 'cap-v0'
env = gym.make(env_id)
map_size = 5
blue_team_agent_list = env.get_team_blue
policy_red = policy.random_actions.PolicyGen(env.get_map, env.get_team_red)
env.reset(map_size = map_size, policy_red = policy_red)

# set hyperparameters
# exploration rate (decays based on number of frames that have passed)
#TODO have exploration based on number of successful episodes?
epsilon_start = 1.0
epsilon_final = 0.02
epsilon_decay = 10000

gamma = 0.99 # future reward discount

num_episodes = 200000
max_episode_length = 100
learning_rate = 10**-4

batch_size = 100 # number of transitions to sample from replay buffer
replay_buffer_capacity = 1000
replay_buffer_init = 20000 # number of frames to simulate before we start sampling from the buffer
train_online_network_frame = 4 # number of frames between training of the online network (see Hasselt 2016 - DDQN)
replay_buffer = ReplayBuffer(replay_buffer_capacity)

# storage for plots
step_list, reward_list, loss_list, epsilon_list = setup_storage(run_number, load_ckpt)

# get fully observable state
obs_space = env.get_full_state

num_states = np.shape(obs_space)[0] * np.shape(obs_space)[1] 
num_actions = env.action_space.n

# setup neural net q-function approximator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
online_model = load_model(run_number, load_ckpt)
criterion = nn.MSELoss()
optimizer = optim.Adam(online_model.parameters(), lr=learning_rate)

######################
if __name__ == '__main__':
    time1 = time.time()    
    for episode in range(load_ckpt, load_ckpt+num_episodes):
        loss, length, reward, epsilon = play_episode()
        
        # save episode data after the episode is done
        step_list.append(length)
        loss_list.append(loss / length)
        reward_list.append(reward)
        epsilon_list.append(epsilon)

        if episode % 10 == 0:
            print('Run: {} ---- Episode: {}/{} ({}) ---- Runtime: {} '.format(run_number, episode, num_episodes, round(float(episode) / float(num_episodes), 3), round(time.time()-time1, 3)))

        if episode % 2000 == 0 and episode != 0:
            save_model(episode)
            save_data(step_list, reward_list, loss_list, epsilon_list)

