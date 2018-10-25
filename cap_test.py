import time
import gym
import gym_cap
import numpy as np
from numpy import shape

# import policy generating modules
import policy.patrol 
import policy.random
import policy.stay_still
import policy.deep_Q_net_v0
############################################
start_time = time.time()
env = gym.make("cap-v0") # initialize the environment

#TODO: can we get a fully observable version of the map?
# Or at least a version where we have a view of the environment, but not the enemy or the flag
# This only gives what the blue team would see, and makes the problem a lot harder
print(env.observation_space_blue) # [Output: ] Box(4,)

done = False
t = 0
total_score = 0

# reset the environment and select the policies for each of the team
# actions must be specified in the policies, which are then passed to the environment here
observation = env.reset(map_size=20,
                        render_mode="env",
                        policy_blue = policy.deep_Q_net_v0.PolicyGen(env.get_map, env.get_team_blue),
                        policy_red = policy.random.PolicyGen(env.get_map, env.get_team_red))

# we only control the blue team, the red team can be considered a hostile part of the environment
#action_size = env.action_space.n
#state_space = env.observation_space_blue
#print(action_size)
#print(np.shape(state_space))




while True:
    while not done:
        observation, reward, done, info = env.step()  # feedback from environment

        # render and sleep are not needed for score analysis
        env.render(mode="fast")
        time.sleep(.1)
        
        t += 1
        if t == 100000:
            break
        
    total_score += reward
    env.reset()
    done = False
    print("Total time: %s s, score: %s" % ((time.time() - start_time),total_score))

