"""
Simple (Policy Gradient) agents policy generator
"""

import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from pathlib import Path

# features in observation
INVISIBLE = -1
TEAM1_BACKGROUND = 0
TEAM2_BACKGROUND = 1
TEAM1_UGV = 2
TEAM2_UGV = 4
TEAM1_FLAG = 6
TEAM2_FLAG = 7
OBSTACLE = 8
DEAD = 9

# actions
STAY = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
ACTION_SPACE = [STAY, UP, RIGHT, DOWN, LEFT]

    def parse_obs(self, obs, x, y):
        # the channel number for different features
        # Channel 0: INVISIBLE: 1
        # Channel 1: Team1_Background VS Team2_Background: 1 VS -1
        # Channel 2: Team1_UGV VS Team2_UGV: 1 VS -1
        # Channel 3: Team1_Flag VS Team2_Flag: 1 VS -1
        # Channel 4: Obstacle + Everything out of Boundary: 1
        # Ignore DEAD
        switcher = {
            INVISIBLE:(0,  1),
            TEAM1_BACKGROUND:(1,  1),
            TEAM2_BACKGROUND:(1, -1),
            TEAM1_UGV:(2,  1),
            TEAM2_UGV:(2, -1),
            TEAM1_FLAG:(3,  1),
            TEAM2_FLAG:(3, -1),
            OBSTACLE:(4,  1)
        }

        parsed_obs = np.zeros((len(obs),len(obs[0]),5))

        # Shift the active unit to the center of the observation
        x_shift = int(len(obs)/2 - x)
        y_shift = int(len(obs[0])/2 - y)

        for i in range(max(0, int(x-len(obs)/2)), min(len(obs), int(x+len(obs)/2))):
            for j in range(max(0, int(y-len(obs[0]))/2), min(len(obs[0]), int(y+len(obs[0])/2))):

                # if obs[i][j] != INVISIBLE:
                #     parsed_obs[i+x_shift][j+y_shift][0] = 1
                result = switcher.get(obs[i][j], 'nothing')
                if result != 'nothing':
                    parsed_obs[i+x_shift][j+y_shift][result[0]] = result[1]

        # add the background of the current location to channel 1
        if self.free_map[x][y] == TEAM1_BACKGROUND:
            parsed_obs[x+x_shift][y+y_shift][1] = 1
        else:
            parsed_obs[x+x_shift][y+y_shift][1] = -1

        # add the enemy flag location to channel 3
        flag_loc_x, flag_loc_y = self.flag_loc[0]+x_shift, self.flag_loc[1]+y_shift
        if flag_loc_x >= 0 and flag_loc_x < len(obs) and flag_loc_y >= 0 and flag_loc_y < len(obs[0]):
            #parsed_obs[flag_loc_x][flag_loc_y][0] = 1
            parsed_obs[flag_loc_x][flag_loc_y][1] = -1
            parsed_obs[flag_loc_x][flag_loc_y][3] = -1

        # add padding to Channel 4 for everything out of boundary
        for i in range(len(obs)):
            for j in range(len(obs[i])):
                ori_i, ori_j = i - x_shift, j - y_shift
                if ori_i < 0 or ori_i >= len(obs) or ori_j < 0 or ori_j >= len(obs[i]):
                    parsed_obs[i][j][4] = 1

        return parsed_obs
