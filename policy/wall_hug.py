import numpy as np
from collections import defaultdict


class PolicyGen:

    def __init__(self, free_map, agent_list):

        self.random = np.random
        self.previous_move = self.random.randint(0, 5, len(agent_list)).tolist()

        self.team = agent_list[0].team

        self.flag_location = None
        self.enemy_flag_code = 7 if self.team == 0 else 6
        self.enemy_code = 4 if self.team == 0 else 2

    def gen_action(self, agent_list, observation, free_map=None):

        action_out = []

        for idx, agent in enumerate(agent_list):
            action = self.policy(agent,observation, idx)
            action_out.append(action)

        return action_out

    def policy(self, agent, obs, agent_id):

        # Expand the observation with wall
        # - in order to avoid dealing with the boundary
        obsx, obsy = obs.shape
        padding = agent.range
        _obs = np.ones((obsx+2*padding, obsy+2*padding)) * 8
        _obs[padding:obsx+padding, padding:obsy+padding] = obs
        obs = _obs

        # Initialize Variables
        x, y = agent.get_loc()
        x += padding
        y += padding
        view = obs[x+1-padding:x+padding,
                    y+1-padding:y+padding] # limited view for the agent

        # Continue the previous action
        action = self.previous_move[agent_id]

        # Checking obstacle
        dir_x = [0, 0, 1, 0, -1] # dx for [stay, up, right, down, left]
        dir_y = [0,-1, 0, 1,  0] # dy for [stay, up, right, down ,left]
        is_possible_to_move = lambda d: obs[x+dir_x[d]][y+dir_y[d]] not in [2,4,8]
        if not is_possible_to_move(action): # Wall or other obstacle
            action_pool = [0]
            for movement in range(1,5):
                if is_possible_to_move(movement):
                    action_pool.append(movement)
            action = np.random.choice(action_pool) # pick from possible movements

        # Obtain information based on the vision
        field = self.scan(view)
        elements = field.keys()

        if self.enemy_flag_code in elements: # Flag Found
            # move towards the flag
            fx, fy = field[self.enemy_flag_code][0] # flag location (coord. of 'view')
            if fy > 2: # move down
                action = 3
            elif fy < 2: # move up
                action = 1
            elif fx > 2: # move left
                action = 2
            elif fx < 2: # move right
                action = 4


        return action

    def scan(self, view):
        """
        This function returns the dictionary of locations for each element by its type.
            key : field element (int)
            value : list of element's coordinate ( (x,y) tuple )
        """

        objects = defaultdict(list)
        dx, dy = len(view), len(view[0])
        for i in range(dx):
            for j in range(dy):
                objects[view[i][j]].append((i,j))

        return objects

