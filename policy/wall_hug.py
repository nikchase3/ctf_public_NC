"""Attack Policy"""

import numpy as np


class PolicyGen:
    def __init__(self, free_map, agent_list):

        self.free_map = free_map
        self.free_map_old = free_map
        self.team = agent_list[0].team

        self.flag_location = None
        self.random = np.random
        self.exploration = 0.3
        # set flag_code for enemy flag
        self.flag_code = 7 if self.team == 0 else 6

    def gen_action(self, agent_list, observation, free_map=None):

        action_out = []

        # search for a flag until finds it
        if self.flag_location == None:

            loc = self.scan_obs(observation,self.flag_code)
            if len(loc) is not 0:
                self.flag_location = loc[0]

            for idx,agent in enumerate(agent_list):
                if idx < int(len(agent_list)/2): # divides agents in 2
                    a = self.search(agent, idx, observation, True)
                else:
                    a = self.search(agent, idx, observation, False)
                action_out.append(a)

            return action_out

        # go to the flag
        for idx,agent in enumerate(agent_list):
            a = self.flag_approach(agent, idx, observation)
            action_out.append(a)

        return action_out

    def search(self, agent, index, obs, half):

        x,y = agent.get_loc()
        #hug wall
        if half:
            #finds wall

        else:


        # approach the boarder.
        if (y > len(self.free_map[0]) / 2 and
                self.free_map[x][y - 1] == self.free_map[x][y]):
            action = 1
        elif (y < len(self.free_map[0]) / 2 - 1 and
              self.free_map[x][y + 1] == self.free_map[x][y]):
            action = 3
        else:
            action = self.random.randint(0, 5)


        return action

    def flag_approach(self, agent, index, obs):
        x,y = agent.get_loc()
        action = 0

        if self.flag_location[0] > x:
            action = 2
        elif self.flag_location[0] < x:
            action = 4
        elif self.flag_location[1] > y:
            action = 3
        elif self.flag_location[1] < y:
            action = 1

        if self.random.random() < self.exploration:
            action = self.random.randint(0, 5)

        return action

    def scan_obs(self, obs, value):
        location = []

        for y in range(len(obs)):
            for x in range(len(obs[0])):
                if obs[x][y] == value:
                    location.append([x,y])

        return location
