import math
import gym
from gym import wrappers
import gym_cap
import numpy as np
from numpy import shape
import os
import json
import matplotlib.pyplot as plt
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
from torch.autograd import Variable
import torch.nn.functional as F
from IPython.display import clear_output

import policy.roomba
import policy.random
import policy.deep_Q_net_v0

#################################
env_id = 'cap-v0'
env = gym.make(env_id)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#################################
def plot_training_data(frame_idx, loss_list, frame_list, episode_list, episode_duration_list, reward_list):
    fig, axs = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    ax = axs[0]
    ax.plot(frame_list, loss_list)
    ax.set_xlabel('Frames')
    ax.set_ylabel('Loss')
    
    ax = axs[1]
    ax.plot(episode_list, episode_duration_list)
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Duration (frames)')

    ax = axs[2]
    ax.plot(episode_list, reward_list)
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Episode Reward')
    
    fig_fn = 'training_data_' + str(frame_idx) + '.png'
    fig_path = os.path.join('./checkpoints', fig_fn)
    plt.savefig(fig_path, dpi = 400)
    plt.close()

def save_training_data(frame_idx, model, loss_list, frame_list, episode_list, episode_duration_list, reward_list):
    ckpt_fn = 'checkpoint_frame_' + str(frame_idx) + '.ckpt'
    save_path = os.path.join(ckpt_dir, ckpt_fn)
    torch.save(model.state_dict(), save_path)
    # save a list of checkpoint names for easy loading later 
    ckpt_paths.append(save_path)

    # also save the episode data up to this point
    # save loss-by-frame
    loss_fn = 'loss_data.txt'
    loss_list = np.array(loss_list)
    frame_list = np.array(frame_list)
    loss_save = np.vstack((frame_list, loss_list))

    loss_path = os.path.join(ckpt_dir, loss_fn)
    with open(loss_path, 'w') as f:
        np.savetxt(f, loss_save)

    # save data for each episode
    episode_fn = 'episode_data.txt'
    
    episode_save = np.vstack((episode_list, reward_list, episode_duration_list))

    episode_path = os.path.join(ckpt_dir, episode_fn)
    with open(episode_path, 'w') as f:
        np.savetxt(f, episode_save)
    
    plot_training_data(frame_idx, loss_list, frame_list, episode_list, episode_duration_list, reward_list)

#################################
# this file contains a training loop that interfaces with DDQN to train it
if __name__ == '__main__': 
    #TODO consolidate all hyperparameters into one area of the code
    batch_size = 100
    max_frames_per_episode = 150

    # set checkpoint save directory
    ckpt_paths = []
    ckpt_dir = './checkpoints/'
    dir_exist = os.path.exists(ckpt_dir)
    if not dir_exist:
        os.mkdir(ckpt_dir)
    
    # set policies for each team
    policy_blue =  policy.deep_Q_net_v0.PolicyGen(env, device, env.get_map, env.get_team_blue)
    policy_red = policy.random.PolicyGen(env.get_map, env.get_team_red)
    
    env.reset(map_size=20, policy_blue = policy_blue, policy_red = policy_red)
    
    agent_list_blue = env.get_team_blue

    # set current_model and target_model to be identical before starting training
    policy_blue.update_target_network()

    # episode_arr = np.arange(0, num_episodes, 1).reshape(1, -1)
    # episode_duration_list = np.zeros([1, num_episodes])

    # recording model performance
    #TODO add features to allow for stopping of the sim, then picking up where we left off 
    #TODO includes saving current episode list, durations, frames, and losses when the sim is stopped

    num_episodes = 100000
    episode_list = np.arange(0, num_episodes, 1)
    episode_duration_list = np.zeros(num_episodes) # number of frames it takes to complete each episode 
    reward_list = np.zeros(num_episodes) # reward for each episode
        
    frame_list = [] # frames for which loss is computed
    loss_list = []

    def play_episode(frame_idx):
        env.reset(map_size=20, policy_blue = policy_blue, policy_red = policy_red)
        done = 0
        episode_reward = 0
        episode_duration = 0 # amount of frames played for this episode
        while not done:
            episode_duration += 1
            frame_idx += 1
            state = env.get_full_state
            action = policy_blue.gen_action(agent_list_blue, state, frame_idx, train = True)
            next_state, reward, done, _ = env.step(action)
            policy_blue.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if episode_duration > max_frames_per_episode:
                done = 1

            if done:
                return frame_idx, episode_reward, episode_duration

            if len(policy_blue.replay_buffer) > batch_size:
                loss = policy_blue.compute_td_loss(batch_size)
                loss_list.append(loss.item())
                frame_list.append(frame_idx)

            # update target network
            if frame_idx % 100 == 0:
                policy_blue.update_target_network()
            
            # save checkpoints and training data
            if frame_idx % 25000 == 0:
                model = policy_blue.current_model
                save_training_data(frame_idx, model, loss_list, frame_list, episode_list, episode_duration_list, reward_list)

    ###################################
    frame_idx = 1
    time_start = time.time()
    for episode in range(num_episodes):
        print(episode)
        # print updates to console
        if episode % 10 == 0:
            print('Episode: {} / {} ---- Runtime: {}'.format(episode, num_episodes, round(time.time()-time_start, 3)))
            #TODO plot stuff?
        
        frame_idx, episode_reward, episode_duration = play_episode(frame_idx)
        reward_list[episode] = episode_reward
        episode_duration_list[episode] = episode_duration

    # save filepaths of all checkpoints
    ckpt_names_fn = 'checkpoint_paths.txt'
    ckpt_names_path = os.path.join(ckpt_dir, ckpt_names_fn)

    with open(ckpt_names_path, 'w') as f:
        json.dump(ckpt_paths, f)
    
