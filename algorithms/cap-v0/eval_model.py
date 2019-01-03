######################
## program controls
load_ckpt_dir = '2018-12-24--185256.843035'
load_episode = '38000'
eval_episodes = 2000 # choose nice, even numbers please
window = int(eval_episodes / 25)

# renders only for recorded episodes (50 videos is around 2 minutes of footage)
record_video = 0
num_videos = 0

# renders at every stimestep
render_model = 0 

#TODO record agent actions and q-values during evaluation
#TODO visualize actions at each timestep and the associated q-values (https://www.youtube.com/watch?v=XjsY8-P4WHM)

######################
## regular imports
import sys
import argparse
import os
import gym
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import moviepy.editor as mp
from moviepy.video.fx.all import speedx
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
    - save episode data
    - make a plot 
    '''
    # save evaluation data
    step_list = np.asarray(step_list)
    reward_list = np.asarray(reward_list)
    loss_list = np.asarray(loss_list) 
    epsilon_list = np.asarray(epsilon_list)
    episode_save = np.vstack((step_list, reward_list, loss_list, epsilon_list))

    # get averages
    step_avg = np.mean(step_list)
    step_list_avg = step_avg*np.ones(np.shape(step_list))
    reward_avg = np.mean(reward_list)
    reward_list_avg = reward_avg*np.ones(np.shape(reward_list))
    num_success = np.count_nonzero(reward_list == 100)
    percent_success = round((num_success / eval_episodes)*100, 3)

    fn = 'episode_data.txt'
    fp = os.path.join(data_dir, fn)
    with open(fp, 'w') as f:
        np.savetxt(f, episode_save)

    plt.figure(figsize = [10,8])
    plt.subplot(211)
    plt.plot(pd.Series(step_list).rolling(window).mean(), label = 'Length (frames)')
    plt.plot(step_list_avg, label = 'Mean Episode Length = {}'.format(round(step_avg, 1)), linewidth = .7)
    plt.title('Frames per Episode (Moving Average {}-episode Window)'.format(window))
    plt.ylabel('Frames')
    plt.xlabel('Episode')
    plt.legend(loc = 'upper right')

    plt.subplot(212)
    plt.plot(pd.Series(reward_list).rolling(window).mean(), label = 'Reward')
    plt.plot(reward_list_avg, label = 'Mean Reward = {}'.format(round(reward_avg, 1)), linewidth = .7)
    plt.title('Reward per Episode (Moving Average, {}-episode Window)'.format(window))
    plt.ylabel('Reward')
    
    
    success_str = 'Succesful Eval. Episodes: {}/{} ({}%)\n'.format(num_success, eval_episodes, percent_success)

    training_string = 'Network: DQN\nTraining Episodes: {}\n--------------------------------------------------\n'.format(load_episode)

    num_UGV_red, num_UAV_red = count_team_units(env.get_team_red)
    num_UGV_blue, num_UAV_blue = count_team_units(env.get_team_blue)
    env_param_string = 'Map Size: {}\nMax # Frames per Episode: {}\nVision Radius: {}\n# Blue UGVs: {}\n# Blue UAVs: {}\n# Red UGVs: {}\n# Red UAVs: {}'.format(train_params['map_size'], train_params['max_episode_length'], train_params['vision_radius'], num_UGV_blue, num_UAV_blue,num_UGV_red, num_UAV_red)

    text = success_str + training_string + env_param_string
    bbox_props = dict(boxstyle='square', fc = 'white')
    plt.xlabel(text, bbox = bbox_props)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.legend(loc = 'upper right')

    fn = 'eval_data.png'
    fp = os.path.join(data_dir, fn)
    plt.savefig(fp, dpi=300)
    plt.close()
    
def load_model(load_episode):  
    #TODO make this generalize to multiple network architectures, save network type in train_params
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
    # set checkpoint save directory
    ckpt_dir = './data/' + load_ckpt_dir
    
    eval_dir = os.path.join(ckpt_dir, 'eval')
    dir_exist = os.path.exists(eval_dir)
    if not dir_exist:
        os.mkdir(eval_dir)

    data_dir =  os.path.join(eval_dir, 'ep_' + str(load_episode))
    dir_exist = os.path.exists(data_dir)
    if not dir_exist:
        os.mkdir(data_dir)

    # setup hyperparameters
    #TODO right now, this will load the map parameters used during training, 
    # if you want to evaluate on a different sized map, change it here
    fn = 'train_params.json'
    fp = os.path.join(ckpt_dir, fn)
    with open(fp, 'r') as f:
        train_params = json.load(f)
    
    # init lists for training data
    step_list = []
    reward_list = []
    loss_list = []
    epsilon_list = [] 

    # init frame_count
    frame_count = 0

    return ckpt_dir, data_dir, train_params, frame_count, step_list, reward_list, loss_list, epsilon_list

######################
# RL functions

def count_team_units(team_list):
    '''
    Inputs{
        team_list - use env.get_team_(red or blue) as input
        }
    Outputs{
        num_UGV, num_UAV
    }
    '''
    num_UAV = 0
    num_UGV = 0
    for i in range(len(team_list)):
        if isinstance(team_list[i], gym_cap.envs.agent.GroundVehicle):
            num_UGV += 1
        elif isinstance(team_list[i], gym_cap.envs.agent.AerialVehicle):
            num_UAV += 1
        else:
            continue
    return num_UGV, num_UAV

def gen_action(state, epsilon):
    #TODO: make it work for multiple agents
    if np.random.rand(1) < epsilon:
        action = env.unwrapped.action_space.sample()

    else:
        with torch.no_grad():
            q_values = online_model.forward(state)
            _, action = torch.max(q_values, 1)
            action = action.item()

    return action

def play_episode():
    global frame_count

    if record_video and (episode % int(eval_episodes/num_videos) == 0):
        video_dir = os.path.join(data_dir, 'raw_videos')
    
        dir_exist = os.path.exists(video_dir)
        if not dir_exist:
            os.mkdir(video_dir)

        video_fn = 'episode_' + str(episode) + '.mp4'
        video_path = os.path.join(video_dir, video_fn)
        
        video_recorder = VideoRecorder(env, video_path, enabled = record_video)

    env.reset(map_size = train_params['map_size'], policy_red = policy_red)

    episode_length = 0.
    episode_loss = 0.
    done = 0
    
    #TODO simplify by replacing with a for loop
    while (done == 0):
        if render_model:
            env.unwrapped.render()
        
        # set exploration rate for this frame
        epsilon = 0
        if record_video and (episode % int(eval_episodes/num_videos) == 0):
            video_recorder.capture_frame()

        state = one_hot_encoder(env.unwrapped._env, env.unwrapped.get_team_blue, vision_radius = train_params['vision_radius'])
        action = gen_action(state, epsilon)
        next_state, reward, done, _ = env.unwrapped.step(entities_action = [action])
        
        episode_length += 1
        frame_count += 1
                
        # stop the episode if it goes too long
        if episode_length >= train_params['max_episode_length']:
            reward = -100.
            done = True

        # end the episode         
        if done:
            if record_video and (episode % int(eval_episodes/num_videos) == 0):
                video_recorder.close()
                vid = mp.VideoFileClip(video_path)
                vid_list.append(vid)

            return episode_loss, episode_length, reward, epsilon
    
######################
## storage for training data-
ckpt_dir, data_dir, train_params, frame_count, step_list, reward_list, loss_list, epsilon_list = setup_data_storage(load_episode)

######################
## setup for training
# init environment
env_id = 'cap-v0'
env = gym.make(env_id)
blue_team_agent_list = env.unwrapped.get_team_blue
policy_red = policy.random_actions.PolicyGen(env.unwrapped.get_map, env.unwrapped.get_team_red)
env.reset(map_size = train_params['map_size'], policy_red = policy_red)

# get fully observable state
obs_space = env.unwrapped.get_full_state
num_states = np.shape(obs_space)[0] * np.shape(obs_space)[1] 
num_actions = env.unwrapped.action_space.n

# setup neural net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
online_model = load_model(load_episode)

######################
if __name__ == '__main__':
    vid_list = []
    time1 = time.time()    
    for episode in range(eval_episodes):
        loss, length, reward, epsilon = play_episode()
        
        # save episode data after the episode is done
        step_list.append(length)
        loss_list.append(loss / length)
        reward_list.append(reward)
        epsilon_list.append(epsilon)

        if episode % (eval_episodes/20) == 0:
            print('Episode: {}/{} ({}) ---- Runtime: {} '.format(episode, eval_episodes, round(float(episode) / float(eval_episodes), 3), round(time.time()-time1, 3)))

save_data(episode, step_list, reward_list, loss_list, epsilon_list)

if record_video:
    vid_render = mp.concatenate_videoclips(vid_list)
    vid_render = speedx(vid_render, 0.25)
    fp_final = os.path.join(data_dir, 'eval.mp4')
    vid_render.write_videofile(fp_final)

env.close()