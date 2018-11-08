from policy.ddqn import myDQN, ReplayBuffer

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env_id = "cap-v0"
env = gym.make(env_id)
num_states = int(env.observation_space_blue.shape[0] * env.observation_space_blue.shape[1])
num_actions = int(env.action_space.n)
policy_blue =  policy.deep_Q_net_v0.PolicyGen(env, device, env.get_map, env.get_team_blue)
policy_red = policy.random.PolicyGen(env.get_map, env.get_team_red)

agent_list_blue = env.get_team_blue

env.reset()
# No exploring, only playing to the model for Testing
epsilon=0

# set checkpoint load directory
# easily generalize by setting env_id as a variable
ckpt_names_fn = 'checkpoint_paths.txt'
ckpt_dir = './checkpoints'
ckpt_names_path = os.path.join(ckpt_dir, ckpt_names_fn)

# with open(ckpt_names_path, 'r') as f:
#     ckpt_paths = json.load(f)

vid_dir = os.path.join('./videos')
dir_exist = os.path.exists(vid_dir)
if not dir_exist:
    os.mkdir(vid_dir)

vid_env_id_dir = os.path.join('./videos', env_id)
dir_exist = os.path.exists(vid_env_id_dir)
if not dir_exist:
    os.mkdir(vid_env_id_dir)

def test(model, video_path, env):
    episode_reward = 0
    done=False
    model.eval()
    

    testReward = 0
    env = gym.wrappers.Monitor(env, video_path, video_callable = False, force = True)
    env.reset(map_size=20, policy_blue = policy_blue, policy_red = policy_red)
    frame_count = 0
    while not done:
        frame_idx = 0
        # fully observable state
        state = env.env.get_full_state

        action = policy_blue.gen_action(agent_list_blue, state, frame_idx, train = False)
        next_state, reward, done, _ = env.step(action)
        if frame_count > 2000:
            done = 1
            
        episode_reward += reward
        state = next_state
        testReward += reward
        frame_count += 1
        if (frame_count % 10) == 0:
            print(frame_count)
        env.render()
        print(action)

    state = env.reset()
    if done:
        env.env.close()

#################################
#for ckpt in ckpt_paths:
for i in range(1):
    #ckpt_path = ckpt
    #ckpt_fn = ckpt_path.split('/')[-1]
    ckpt_fn = 'frame_1000000.ckpt'
    print(ckpt_fn)
    ckpt_path = os.path.join(ckpt_dir, ckpt_fn)
    
    video_fn = ckpt_fn[:-5]
    video_path = os.path.join('./videos', video_fn)
    
    dir_exist = os.path.exists(video_path)
    if (dir_exist == 0):
        os.mkdir(video_path)

    ckpt_path = 'C:/dev/research/ctf_public/checkpoints/frame_600000.ckpt'
    # Load the Model
    model = myDQN(num_states, num_actions).to(device)
    model.load_state_dict(torch.load(ckpt_path))
    
    # run the loaded model, save the output
    test(model, video_path, env = env)
#################################
