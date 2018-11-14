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
def plot_training_data(episode_list, mean_reward_list, mean_duration_list):
    fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    ax = axs[0]
    ax.plot(episode_list, mean_duration_list)
    ax.set_xlabel('Episode Number\n*moving average taken over past {} frames'.format(num_avg_episodes))
    ax.set_ylabel('Duration (frames)')

    ax = axs[1]
    ax.plot(episode_list, mean_reward_list)
    ax.set_xlabel('Episode Number\n*moving average over past {} frames'.format(num_avg_episodes))
    ax.set_ylabel('Reward')
    
    fig_fn = 'training_data_' + str(episode) + '.png'
    fig_path = os.path.join('./checkpoints', fig_fn)
    plt.savefig(fig_path, dpi = 300)
    plt.close()

#TODO make so it saves the best performing model
def save_checkpoint(episode, model):
    ckpt_fn = 'checkpoint_episode_' + str(episode) + '.ckpt'
    save_path = os.path.join(ckpt_dir, ckpt_fn)
    torch.save(model.state_dict(), save_path)
    # save a list of checkpoint names for easy loading later 
    ckpt_paths.append(save_path)

def save_training_data():
    # save the episode data up to this point
    episode_fn = 'episode_data.txt'    
    episode_save = np.vstack((episode_list, episode_duration_list, reward_list))
    episode_path = os.path.join(ckpt_dir, episode_fn)
    with open(episode_path, 'w') as f:
        np.savetxt(f, episode_save)
    
#################################
if __name__ == '__main__': 
    # set hyperparameters
    map_size = 10
    batch_size = 32
    num_episodes = 200000
    max_frames_per_episode = 50
    save_checkpoint_episode = 1000
    print_update_episode = 1000
    num_avg_episodes = 50

    buffer_init = 100000 # number of frames to simulate before we start sampling from the buffer
    train_online_network_frame = 4 # number of frames between training of the online network (see Hasselt 2016 - DDQN)
    update_target_network_frame = 100 # number of frames between updating the target network

    # a jank way of giving reward as -100 if the agent fails to reach the flag in the alloted number of frames
    failure_reward = -100
    
    # hyperparameters used by the network or policy
    hyperparam_dict = {
        "epsilon_start": 1.0,
        "epsilon_final": 0.05,
        "epsilon_decay": 200000,
        "gamma": 0.99,
        "replay_buffer_size": 1000000,
    }

    # set checkpoint save directory
    ckpt_paths = []
    ckpt_dir = './checkpoints/'
    dir_exist = os.path.exists(ckpt_dir)
    if not dir_exist:
        os.mkdir(ckpt_dir)
    
    # set policies for each team
    policy_blue =  policy.deep_Q_net_v0.PolicyGen(env, device, env.get_map, env.get_team_blue, hyperparam_dict)
    policy_red = policy.random.PolicyGen(env.get_map, env.get_team_red)
    
    env.reset(map_size = map_size, policy_blue = policy_blue, policy_red = policy_red)
    
    agent_list_blue = env.get_team_blue

    # set current_model and target_model to be identical before starting training
    policy_blue.update_target_network()

    # recording model performance
    #TODO add features to allow for stopping of the sim, then picking up where we left off 
    #TODO includes saving current episode list, durations, frames when the sim is stopped
    episode_list = np.arange(0, num_episodes, 1)
    episode_duration_list = np.zeros(num_episodes) # number of frames it takes to complete each episode 
    reward_list = np.zeros(num_episodes) # reward for each episode

    mean_reward_list = np.zeros(num_episodes)
    mean_duration_list = np.zeros(num_episodes)
    
    def play_episode(frame_idx):
        env.reset(map_size = map_size, policy_blue = policy_blue, policy_red = policy_red)
        done = 0
        episode_reward = 0
        episode_duration = 0 # amount of frames played for this episode
        while not done:
            episode_duration += 1
            frame_idx += 1
            state = env.get_full_state
            action = policy_blue.gen_action(agent_list_blue, state, frame_idx, train = True)
            next_state, reward, done, _ = env.step(action)

            if episode_duration >= max_frames_per_episode:
                reward = failure_reward
                done = 1

            policy_blue.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if done:
                return frame_idx, episode_reward, episode_duration
            
            # take a training step with the online network
            if len(policy_blue.replay_buffer) > buffer_init:
                if (frame_idx % train_online_network_frame) == 0:
                    policy_blue.compute_td_loss(batch_size)

            # update target network
            if (frame_idx % update_target_network_frame) == 0:
                policy_blue.update_target_network()

    ###################################
    frame_idx = 1
    time_start = time.time()
    for episode in range(num_episodes):
        frame_idx, episode_reward, episode_duration = play_episode(frame_idx)
        
        reward_list[episode] = episode_reward
        episode_duration_list[episode] = episode_duration

        curr_idx = np.argmax(episode_duration_list == 0)
        episode_list_curr = episode_list[0:curr_idx]
        episode_duration_list_curr = episode_duration_list[0:curr_idx]
        reward_list_curr = reward_list[0:curr_idx]

        # get a moving average of rewards and durations
        if episode >= num_avg_episodes:
            reward_mean = np.mean(reward_list_curr[-num_avg_episodes:])
            duration_mean = np.mean(episode_duration_list_curr[-num_avg_episodes:])

            mean_reward_list[episode] = reward_mean
            mean_duration_list[episode] = duration_mean
        mean_episode_list_curr = episode_list_curr[num_avg_episodes:]
        mean_reward_list_curr = mean_reward_list[num_avg_episodes:curr_idx]
        mean_duration_list_curr = mean_duration_list[num_avg_episodes:curr_idx]


        # take the average over this array
        # save that data along with the episode

        if (episode % print_update_episode) == 0:
            print('Episode: {} / {} ---- Runtime: {}'.format(episode, num_episodes, round(time.time()-time_start, 3)))
        
        if (episode % save_checkpoint_episode) == 0:
            if episode != 0:
                curr_model = policy_blue.current_model
                save_checkpoint(episode, curr_model)
                save_training_data()
                plot_training_data(mean_episode_list_curr, mean_reward_list_curr, mean_duration_list_curr)
                

    # save filepaths of all checkpoints
    ckpt_names_fn = 'checkpoint_paths.txt'
    ckpt_names_path = os.path.join(ckpt_dir, ckpt_names_fn)

    with open(ckpt_names_path, 'w') as f:
        json.dump(ckpt_paths, f)
    
