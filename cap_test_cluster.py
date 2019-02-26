import time
import gym
import gym_cap
import numpy as np


# the modules that you can use to generate the policy.
import policy.roomba
import policy.random_actions
import policy.patrol
import policy.defense
import policy.stay_still

start_time = time.time()
env = gym.make("cap-v0") # initialize the environment

done = False
t = 0
rscore = []

policy_blue = policy.random_actions.PolicyGen(env.get_map, env.get_team_blue)
policy_red = policy.stay_still.PolicyGen(env.get_map, env.get_team_red)

def count_team_units(team_list):
    '''
    Counts total UAVs and UGVs for a team.

    Args:
        team_list (list): list of team members.  Use env.get_team_(red or blue) as input.

    Returns:
        num_UGV (int): number of ground vehicles
        num_UAV (int): number of aerial vehicles
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

num_UGV_red, num_UAV_red = count_team_units(env.get_team_red)
num_UGV_blue, num_UAV_blue = count_team_units(env.get_team_blue)
agent_str = 'Blue UGVs: {}\nBlue UAVs: {}\nRed UGVs: {}\nRed UAVs: {}'.format(num_UGV_blue, num_UAV_blue, num_UGV_red, num_UAV_red)
print(agent_str)

# reset the environment and select the policies for each of the team
observation = env.reset(map_size=10,
                        policy_blue=policy_blue,
                        policy_red=policy_red)

while True:
    while not done:

        #you are free to select a random action
        # or generate an action using the policy
        # or select an action manually
        # and the apply the selected action to blue team
        # or use the policy selected and provided in env.reset
        #action = env.action_space.sample()  # choose random action
        #action = policy_blue.gen_action(env.team1,observation,map_only=env.team_home)
        #action = [0, 0, 0, 0]
        #observation, reward, done, info = env.step(action)

        observation, reward, done, info = env.step()  # feedback from environment

        # render and sleep are not needed for score analysis
        # env.render()
        # time.sleep(.1)

        t += 1
        if t == 100:
            break

    env.reset()
    done = False
    rscore.append(reward)
    print("Time: %.2f s, score: %.2f" %
        ((time.time() - start_time),np.asarray(rscore).mean()))
