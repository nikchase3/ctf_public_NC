import sys
import argparse
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from collections import deque
import random

######################
def gen_plots(data_path, plot_path):
    '''
    actions{
        saves a plot of data_path to plot_path
    }

    inputs{
        data_path: path to a text file containing a numpy array with
        episode_duration = array[0, :]
        episode_reward = array[1, :]
        episode_loss = array[2, :]
        epsilon_by_frame = array[3, :]
        
        plot_path: path at which the data plot will be saved 
    }
    '''

    with open(data_path, 'r') as f:
        data = np.loadtxt(f)

    num_episodes = np.shape(data)[1]
    window = int(num_episodes / 100.)

    step_list = data[0, :]
    reward_list = data[1, :]
    loss_list  = data[2, :]
    epsilon_list = data[3, :]

    step_avg = np.mean(step_list)
    step_list_avg = step_avg*np.ones(np.shape(step_list))

    reward_avg = np.mean(reward_list)
    reward_list_avg = reward_avg*np.ones(np.shape(reward_list))
    success_count = np.count_nonzero(reward_list == 100)
    percent_success = round((success_count/num_episodes)*100., 2)
    
    loss_avg = np.mean(loss_list)
    loss_list_avg = loss_avg*np.ones(np.shape(loss_list))

    plt.figure(figsize = [10,8])
    plt.subplot(411)
    plt.plot(pd.Series(step_list).rolling(window).mean(), label = 'Length (frames)', linewidth = .5)
    plt.plot(step_list_avg, label = 'Mean Length = {}'.format(round(step_avg, 1)))
    plt.title('Frames per Episode (Moving Average, {}-episode window)'.format(window))
    plt.ylabel('Frames')
    plt.legend(loc = 'best')

    plt.subplot(412)
    plt.plot(pd.Series(reward_list).rolling(window).mean(), label = 'Reward', linewidth = .5)
    plt.plot(reward_list_avg, label = 'Mean Reward = {}'.format(round(reward_avg, 1)))
    plt.title('Reward per Episode (Moving Average, {}-episode window\n Percent Success: '.format(window) + r"$\bf{" + str(percent_success) + "}$")
    plt.ylabel('Reward')
    plt.legend(loc = 'lower right')

    plt.subplot(413)
    plt.plot(pd.Series(loss_list).rolling(window).mean(), label = 'Episode Loss', linewidth = .5)
    plt.plot(loss_list_avg, label = 'Mean Episode Loss = {}'.format(round(loss_avg, 1)))
    plt.title('Loss per Episode (Moving Average, {}-episode window)'.format(window))
    plt.ylabel('Loss')
    plt.legend(loc = 'best')

    plt.subplot(414)
    plt.plot(epsilon_list, label = 'Exploration Rate per Frame', linewidth = .5)
    plt.title('Epsilon per Frame')
    plt.ylabel('Random Action Prob.')
    plt.xlabel('Episode')
    plt.legend(loc = 'best')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('./data/my_plot.png', dpi=300)


def load_data(data_path):
    with open(data_path, 'r') as f:
        data = np.loadtxt(f)

    num_episodes = np.shape(data)[1]
    window = int(num_episodes / 100.)

    step_list = data[0, :]
    reward_list = data[1, :]
    loss_list  = data[2, :]
    epsilon_list = data[3, :]


if __name__ == '__main__':
    data_path = './data/checkpoints_1/episode_data.txt'
    plot_path = './data/my_plot.png'
    # load_data(data_path)
    gen_plots(data_path, plot_path)
