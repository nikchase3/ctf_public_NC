######################
## program controls
load_datetime = ''
load_episode = 0

def setup_hyperparameters():
    train_params = {}
    ## game hyperparameters
    train_params['num_episodes'] = 40000
    train_params['map_size'] = 20
    train_params['vision_radius'] = 9
    train_params['max_episode_length'] = 100

    ## training hyperparameters
    #TODO have exploration based on number of successful episodes?
    train_params['epsilon_start'] = 1.0
    train_params['epsilon_final'] = 0.02
    train_params['epsilon_decay'] = 10000
    train_params['gamma'] = 0.99 # future reward discount
    train_params['learning_rate'] = 10**-4
    train_params['batch_size'] = 100 # number of transitions to sample from replay buffer
    train_params['replay_buffer_capacity'] = 2000 # number of frames to simulate before we start sampling from the buffer
    train_params['replay_buffer_init'] = 1000
    train_params['train_online_model_frame'] = 4 # number of frames between training the online network (see Hasselt 2016 - DDQN)
    
    return train_params

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
import datetime
import time
from collections import deque
import random
import json

## Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

## custom modules
import policy.random_actions
import gym_cap
from utility.dataModule import one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import Experience_buffer, discount_rewards
from utility.utilityRL import DQN, ReplayBuffer

######################
# file / data management
def save_data(episode, step_list, reward_list, loss_list, epsilon_list):
    '''
    - save model weights
    - save hyperparameters
    - save training data
    - make a plot 
    '''
    # save weights
    fn = 'episode_' + str(episode) + '.model'
    fp = os.path.join(ckpt_dir, fn)
    torch.save(online_model, fp)

    # save hyperparameters
    fn = 'train_params.json'
    fp = os.path.join(ckpt_dir, fn)

    if episode == 0:
        with open(fp, 'w') as f:
            json.dump(train_params, f)

    # save training data
    step_list = np.asarray(step_list)
    reward_list = np.asarray(reward_list)
    loss_list = np.asarray(loss_list) 
    epsilon_list = np.asarray(epsilon_list)
    episode_save = np.vstack((step_list, reward_list, loss_list, epsilon_list))

    window = 100
    fn = 'episode_data.txt'
    fp = os.path.join(ckpt_dir, fn)
    with open(fp, 'w') as f:
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
    fn = 'training_data.png'
    fp = os.path.join(ckpt_dir, fn)
    plt.savefig(fp, dpi=300)
    plt.close()
    
def load_model(load_episode):
    if (load_datetime == ''):
        #TODO make this generalize to multiple network architectures, save network type in train_params
        online_model = DQN(num_states, num_actions, train_params['batch_size'])
        online_model = online_model.to(device)
  
    else:
        #TODO make this generalize to multiple network architectures
        online_model = DQN(num_states, num_actions, train_params['batch_size'])
        
        # load only the state dict
        fn = 'episode_' + str(load_episode) + '.model'
        fp = os.path.join(ckpt_dir, fn)
        load_model = torch.load(fp)
        online_model.load_state_dict(load_model.state_dict())
        online_model = online_model.to(device)
    return online_model

def setup_data_storage(load_episode):
    '''
    - init ckpt_dir for saving data
    - init train_params
    - init frame_count
    - init lists for saving training data to disk
    '''
    
    if (load_datetime == ''):
        # set checkpoint save directory
        time = datetime.datetime.now()
        ckpt_dir = './data/' + str(time).replace(' ', '--').replace(':', '')
        dir_exist = os.path.exists(ckpt_dir)
        if not dir_exist:
            os.mkdir(ckpt_dir)
        
        # setup hyperparameters
        train_params = setup_hyperparameters()

        # init frame count
        frame_count = 0

        # init lists for training data
        step_list = []
        reward_list = []
        loss_list = []
        epsilon_list = []
        
    else:
        # set checkpoint save directory
        ckpt_dir = './data/' + load_datetime

        # setup hyperparameters
        fn = 'train_params.json'
        fp = os.path.join(ckpt_dir, fn)
        with open(fp, 'r') as f:
            train_params = json.load(f)
        
        train_params['num_episodes'] = num_episodes
        
        # init lists for training data
        fn = 'episode_data.txt'
        fp = os.path.join(ckpt_dir, fn)
        with open(fp, 'r') as f:
            data = np.loadtxt(f)
        
        step_list = np.ndarray.tolist(data[0, 0:load_episode])
        reward_list = np.ndarray.tolist(data[1, 0:load_episode])
        loss_list = np.ndarray.tolist(data[2, 0:load_episode])
        epsilon_list = np.ndarray.tolist(data[3, 0:load_episode])

        # init frame_count
        #TODO make sure step_list is correct, and make sure this is right too!
        frame_count = np.sum(step_list) 

    return ckpt_dir, train_params, frame_count, step_list, reward_list, loss_list, epsilon_list

######################
# RL functions
#TODO have gen_action, epsilon_by_frame, and train_online_model as functions in the RL algorithm class
#NOTE this cannot be easily done, for pytorch to save the model, the DQN class can only have init and forward functions

def gen_action(state, epsilon):
    #TODO: make it work for multiple agents
    if np.random.rand(1) < epsilon:
        action = env.action_space.sample()

    else:
        with torch.no_grad():
            q_values = online_model.forward(state)
            _, action = torch.max(q_values, 1)
            action = action.item()

    return action

def epsilon_by_frame(frame_count):
    epsilon_curr = train_params['epsilon_final'] + (train_params['epsilon_start'] - train_params['epsilon_final']) * math.exp(-1. * frame_count / train_params['epsilon_decay'])
    
    return epsilon_curr

def train_online_model(batch_size):
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
    
    # get all the q-values for actions at state and next state
    q_values = online_model.forward(state_batch)
    next_q_values = online_model.forward(next_state_batch)
    
    # find Q(s, a) for the action taken during the sampled transition
    state_action_value = q_values.gather(1, action_batch).squeeze(1)
    
    # Find Q(s_next, a_next) for an optimal agent (take the action with max q-value in s_next)
    next_state_action_value = next_q_values.max(1)[0]
    
    # if done, multiply next_state_action_value by 0, else multiply by 1
    one = np.ones(batch_size)
    done_mask = torch.from_numpy(one-done_batch).type(torch.FloatTensor)
    
    discounted_next_value = (train_params['gamma'] * next_state_action_value)
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

    env.reset(map_size = train_params['map_size'], policy_red = policy_red)

    episode_length = 0.
    episode_loss = 0.
    done = 0
    
    #TODO simplify by replacing with a for loop
    while (done == 0):
        # set exploration rate for this frame
        epsilon = epsilon_by_frame(frame_count)
        
        state = one_hot_encoder(env._env, env.get_team_blue, vision_radius = train_params['vision_radius'])
        action = gen_action(state, epsilon)
        next_state, reward, done, _ = env.step(entities_action = [action])
        
        next_state = one_hot_encoder(env._env, env.get_team_blue, vision_radius = train_params['vision_radius'])
        episode_length += 1
        frame_count += 1
                
        # stop the episode if it goes too long
        if episode_length >= train_params['max_episode_length']:
            reward = -100.
            done = True

        # store the transition in replay buffer
        replay_buffer.push(state, action, reward/100. , next_state, done)

        # train the network
        if len(replay_buffer) > train_params['replay_buffer_init']:
            if (frame_count % train_params['train_online_model_frame']) == 0:
                loss = train_online_model(train_params['batch_size'])
                episode_loss += loss

        # end the episode         
        if done:
            return episode_loss, episode_length, reward, epsilon
    
######################
## storage for training data
ckpt_dir, train_params, frame_count, step_list, reward_list, loss_list, epsilon_list = setup_data_storage(load_episode)

######################
## setup for training
# init environment
env_id = 'cap-v0'
env = gym.make(env_id)
blue_team_agent_list = env.get_team_blue
policy_red = policy.random_actions.PolicyGen(env.get_map, env.get_team_red)
env.reset(map_size = train_params['map_size'], policy_red = policy_red)

# init replay buffer
replay_buffer = ReplayBuffer(train_params['replay_buffer_capacity'])

# get fully observable state
obs_space = env.get_full_state
num_states = np.shape(obs_space)[0] * np.shape(obs_space)[1] 
num_actions = env.action_space.n

# setup neural net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
online_model = load_model(load_episode)
criterion = nn.MSELoss()
optimizer = optim.Adam(online_model.parameters(), lr = train_params['learning_rate'])

######################
if __name__ == '__main__':
    time1 = time.time()    
    for episode in range(load_episode, load_episode + train_params['num_episodes']+1):
        loss, length, reward, epsilon = play_episode()
        
        # save episode data after the episode is done
        step_list.append(length)
        loss_list.append(loss / length)
        reward_list.append(reward)
        epsilon_list.append(epsilon)

        if episode % 100 == 0:
            print('Episode: {}/{} ({}) ---- Runtime: {} '.format(episode, train_params['num_episodes'], round(float(episode) / float(train_params['num_episodes']), 3), round(time.time()-time1, 3)))

        if episode % 2000 == 0:
            save_data(episode, step_list, reward_list, loss_list, epsilon_list)
