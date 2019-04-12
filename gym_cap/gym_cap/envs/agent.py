# from gym import error, spaces, utils
# from gym.utils import seeding
# from .cap_view2d import CaptureView2D
from .const import *
# from .create_map import CreateMap
#from .enemy_ai import EnemyAI

from math import cos, sin, radians


class Agent:
    """This is a parent class for all agents.
    It creates an instance of agent in specific location"""

    def __init__(self, loc, loct, map_only, team_number, full_type):
        """
        Constructor

        Parameters
        ----------
        self    : object
            Agent object
        loc     : list
            [X,Y] location of unit
        """
        self.isAlive = True
        self.x, self.y = loc
        self.xt, self.yt = loct #True (float) location.
        self.heading = 0
        self.step = UGV_STEP
        self.range = UGV_RANGE
        self.a_range = UGV_A_RANGE
        self.size = UGV_SIZE
        self.air = False
        #self.ai = EnemyAI(map_only)
        self.team = team_number
        self.move_selected = False
        self.full_type = full_type

    def collision (self, env, team_home, agents, new_loc):
        
        def in_range (loc1, loc2, a_range):

            '''
            Returns whether the two locations are less than or equal to some distance apart.
            '''

            distance = ((loc1[0] - loc2[0]) ** 2) + ((loc1[1] - loc2[1]) ** 2)
            
            if distance <= a_range:
                return true
            else:   
                return false

        def static_collision (self):

            overlapping_tiles = []
            size_mod = int(self.size * 2)

            new_locx, new_locy = new_loc

            for x in range (-size_mod, size_mod + 1):
                for y in range(-size_mod, size_mod + 1):
                    overlapping_tiles.append(int(new_loc[0]) + x, int(new_loc[1]) + y) 

            for tile in overlapping_tiles:
                if env[tile] == OBSTACLE:
                    if in_range(new_loc, (tile[0] + .5, tile[1] + .5), self.size * 2):
                        return True

            return False

        def entity_collision (self):

            overlapping_tiles = []
            size_mod = int(self.size * 2)

            new_locx, new_locy = new_loc

            for x in range (-size_mod, size_mod + 1):
                for y in range(-size_mod, size_mod + 1):
                    overlapping_tiles.append(int(new_loc[0]) + x, int(new_loc[1]) + y)     

            for tile in overlapping_tiles:
                for entity in agents[tile]:
                    if entity is not self:
                        if in_range(new_loc, entity.get_loc(), self.size * 2):
                            return True

            return False

        if not self.static_collision() and not self.entity_collision():
            return False
        else:
            return True

    def move(self, action, env, team_home, agents):
        """
        Moves each unit individually. Checks if action is valid first.

        Parameters
        ----------
        self        : object
            CapEnv object
        action      : string
            Action the unit is to take
        env         : list
            the environment to move units in
        team_home   : list
            easily place the correct home tiles
        """
        if not self.isAlive:
            return
        if action[0] == 0 or action[1] == 0:
            pass
        if action[0] < -MAX_DIRECTION or action[0] > MAX_DIRECTION or action[1] < MIN_MAGNITUDE or action[1] > MAX_MAGNITUDE:
            raise ValueError("Action not within bounds.")
        
        direction, magnitude = action
        magnitude *= self.step

        oldx = self.x
        oldy = self.y 

        self.heading += direction
        heading_rad = radians(self.heading)
        step_size = magnitude / (DISCRETE_SIZE * magnitude)

        for _ in range(DISCRETE_SIZE * magnitude):
            newx = self.xt + (step_size * cos(heading_rad))
            newy = self.yt + (step_size * sin(heading_rad))

            if not self.collision(env, team_home, agents, (newx, newy)):
                self.xt = newx
                self.yt = newy
            else:
                break

        self.x = int(self.xt)
        self.y = int(self.yt)

        env[oldx, oldy].remove(self)
        env[self.x, self.y].append(self)

        agents[(oldx, oldy)].remove(self)
        agents[(self.x, self.y)].append(self)

    def individual_reward(self, env):
        """
        Generates reward for individual
        :param self:
        :return:
        """
        # Small reward range [-1, 1]
        lx, ly = self.get_loc()
        small_observation = [[-1 for i in range(2 * self.range + 1)] for j in range(2 * self.range + 1)]
        small_reward = 0
        if self.air:
            for x in range(lx - self.range, lx + self.range + 1):
                for y in range(ly - self.range, ly + self.range + 1):
                    if ((x - lx) ** 2 + (y - ly) ** 2 <= self.range ** 2) and \
                            0 <= x < self.map_size[0] and \
                            0 <= y < self.map_size[1]:
                        small_observation[x - lx + self.range][y - ly + self.range] = self._env[x][y]
                        # Max reward for finding red flag
                        if env[x][y] == TEAM2_FLAG:
                            small_reward = .5
                        # Reward for UAV finding enemy wherever
                        elif env[x][y] == TEAM2_UGV:
                            small_reward += .5 / NUM_RED
        else:
            if env[lx][ly] == TEAM2_FLAG:
                small_reward = 1
            elif not self.isAlive:
                small_reward = -1
        return small_reward

    def get_loc(self):
        return self.x, self.y

    def get_loc(self):
        return self.xt, self.yt

    def report_loc(self):
        print("report: position x:%d, y:%d" % (self.x, self.y))
    
    def report_loc(self):
        print("report: position x:%d, y:%d" % (self.xt, self.yt))


class GroundVehicle(Agent):
    """This is a child class for ground agents. Inherited from Agent class.
    It creates an instance of UGV in specific location"""

    def __init__(self, loc, loct, map_only, team_number, full_type):
        """
        Constructor

        Parameters
        ----------
        self    : object
            CapEnv object
        """
        Agent.__init__(self, loc, loct, map_only, team_number, full_type)


# noinspection PyCallByClass
class AerialVehicle(Agent):
    """This is a child class for aerial agents. Inherited from Agent class.
    It creates an instance of UAV in specific location"""

    def __init__(self, loc, loct, map_only, team_number, full_type):
        """
        Constructor

        Parameters
        ----------
        self    : object
            CapEnv object
        """
        Agent.__init__(self, loc, loct, map_only, team_number, full_type)
        self.step = UAV_STEP
        self.range = UAV_RANGE
        self.a_range = UAV_A_RANGE
        self.size = UAV_SIZE
        self.air = True


class CivilAgent(GroundVehicle):
    """This is a child class for civil agents. Inherited from UGV class.
    It creates an instance of civil in specific location"""

    def __init__(self, loc, loct, map_only, team_number, full_type):
        """
        Constructor

        Parameters
        ----------
        self    : object
            CapEnv object
        """
        Agent.__init__(self, loc, loct, map_only, team_number, full_type)
        self.direction = [0, 0]
        self.isDone = False
