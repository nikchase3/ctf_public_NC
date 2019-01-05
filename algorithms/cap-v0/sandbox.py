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
from moviepy.editor import VideoFileClip, clips_array, vfx

#################################

vid = mp.VideoFileClip('success.mp4')
legend = mp.ImageClip('legend.png', duration = vid.duration)

final_vid = mp.clips_array([[legend, vid]])
final_vid_fn = 'vid_with_legend.mp4'
final_vid.write_videofile(final_vid_fn)