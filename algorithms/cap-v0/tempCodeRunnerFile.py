_path = os.path.join(project_root, 'policy') 
# sys.path.insert(0, my_path)
# sys.path.insert(0, project_root)

# # import dqn_policy
# import random_actions

# import gym_cap
# #TODO run multiple sims at the same time -> avoid bad random seed
# ######################
# ## regular imports
# import gym
# import numpy as np
# from numpy import shape
# import matplotlib.pyplot as plt
# import pandas as pd
# import time

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torchvision.transforms as T


# ######################
# ## "deep" Q-network 
# #TODO only designed to take 1 image at a time -> upgrade to batches along with implementing replay buffer
# #TODO only has a single channel, will increasing channels help for this?

# class myDQN(nn.Module):
#     def __init__(self, num_states, num_actions):
#         super(myDQN, self).__init__()
#         # this CNN architecture will maintain the size of the input throughout the convolutions
#         self.conv1 = nn.Conv2d(1, 1, 3, padding = 1) 
#         self.conv2 = nn.Conv2d(1, 1, 3, padding = 1) 
#         self.conv3 = nn.Conv2d(1, 1, 3, padding = 1)
#         self.relu = nn.ReLU(inplace=True)

#         self.fc = nn.Linear(num_states, num_actions)
#     def forward(self, state):
#         '''
#         inputs{
#             state (fully observable state) - np array of values representing the world
#         } 
#         outputs{
#             out - Q values for the actions corresponding to the input state
#         }
#         '''
#         state = torch.from_numpy(state)
#         state = state.type(torch.cuda.FloatTensor)
#         state = state.unsqueeze(0).unsqueeze(0)

#         out = self.conv1(state)
#         out = self.relu(out)
       
#         out = self.conv2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.relu(out)

#         out = out.view(out.size(0), -1)
#         q_values = self.fc(out)
       
#         return q_values

# ######################
# def one_hot(x, l):
#     x = torch.LongTensor([[x]])
#     one_hot = torch.FloatTensor(1,l)
#     return one_hot.zero_().scatter_(1,x,1)


# def save_data(step_list, reward_list):
#     # window = int(num_episodes/10)
#     window = 1
#     episode_fn = 'episode_data.txt'
#     episode_path = os.path.join(ckpt_dir, episode_fn)

#     step_list = np.asarray(step_list)
#     reward_list = np.asarray(reward_list)
#     episode_save = np.vstack((step_list, reward_list))

#     with open(episode_path, 'w') as f:
#         np.savetxt(f, episode_save)

#     plt.figure(figsize=[16,9])
#     plt.subplot(211)
#     plt.plot(pd.Series(reward_list).rolling(window).mean())
#     plt.title('Reward Moving Average ({}-episode window)'.format(window))
#     plt.ylabel('Reward')
#     plt.xlabel('Episode')

#     plt.subplot(212)
#     plt.plot(pd.Series(step_list).rolling(window).mean())
#     plt.title('Step Moving Average ({}-episode window)'.format(window))
#     plt.ylabel('Moves')
#     plt.xlabel('Episode')

#     # plt.subplot(413)
#     # plt.plot(pd.Series(loss_list).rolling(window).mean())
#     # plt.title('Loss Moving Average ({}-episode window)'.format(window))
#     # plt.ylabel('Loss')
#     # plt.xlabel('Episode')

#     # plt.subplot(414)
#     # plt.plot(epsilon_list)
#     # plt.title('Random Action Parameter')
#     # plt.ylabel('Chance Random Action')
#     # plt.xlabel('Episode')

#     plt.tight_layout(pad=2)
#     # plt.show()
#     file_name = 'training_data.png'
#     file_path = os.path.join(ckpt_dir, file_name)
#     plt.savefig(file_path)
#     plt.close()
    
# def save_model(episode):
#     model_fn = 'dqn_episode_' + str(episode) + '.model'
#     save_path = os.path.join(ckpt_dir, model_fn)
#     torch.save(online_model, save_path)

# def gen_action(state):
#     '''
#     TODO: make it work for multiple agents
#     action_list = []
    
#     for agent in len(blue_team_agent_list):
#         if np.random.rand(1) < epsilon:
#             q_values = online_model(state)
#             action = env.action_space.sample()

#         else:
#             q_values = online_model(state)
#             _, action = torch.max(q_values, 1)
#             action = action.item()
        
#         action_list.append(action)
#     '''    
#     if np.random.rand(1) < epsilon:
#         q_values = online_model(state)
#         action = env.action_space.sample()

#     else:
#         q_values = online_model(state)
#         _, action = torch.max(q_values, 1)
#         action = action.item()

#     return action, q_values


# def play_episode():
#     global epsilon

#     #this this gives the partially observable state
#     # state = env.reset(map_size = map_size, policy_red = policy_red)
#     env.reset(map_size = map_size, policy_red = policy_red)

#     episode_length = 0.
#     episode_reward = 0.
#     episode_loss = 0.
#     done = 0
    
#     while (done == 0):
#         state = env.get_full_state
#         action, q_value = gen_action(state)
#         next_state, reward, done, _ = env.step(entities_action = [action])
#         episode_length += 1

#         if episode_length >= max_episode_length:
#             done = True

#         if done:
#             if reward == 0.0:
#                 reward = -100
#             if reward > 0.0:
#                 epsilon = 1./((episode/50)+10)
#             # end the episode
#             return episode_loss, episode_length, reward, epsilon

#         next_q_value = online_model(next_state).cpu()
#         max_next_q_value, _ = torch.max(next_q_value.data, 1)
#         max_next_q_value = torch.FloatTensor(max_next_q_value)
        
#         with torch.no_grad():
#             target_q_value = q_value.data
#             target_q_value[0,action] = reward + torch.mul(max_next_q_value, gamma)

#         output = online_model(state)
#         loss = criterion(output, target_q_value)
#         episode_loss += loss.item()
        
#         online_model.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         episode_reward += reward
#         # state = next_state


# ######################
# # make environment
# env_id = 'cap-v0'
# env = gym.make(env_id)
# map_size = 10
# blue_team_agent_list = env.get_team_blue
# policy_red = random_actions.PolicyGen(env.get_map, env.get_team_red)
# env.reset(map_size = map_size, policy_red = policy_red)

# # set hyperparameters
# # exploration rate
# epsilon = 0.05

# # future reward discount
# gamma = 0.99

# num_episodes = 100
# max_episode_length = 150
# learning_rate = 0.001

# # storage for plots
# step_list = []
# reward_list = []
# loss_list = []
# epsilon_list = []

# # get fully observable state
# obs_space = env.get_full_state

# num_states = np.shape(obs_space)[0] * np.shape(obs_space)[1] 
# num_actions = env.action_space.n

# # setup neural net q-function approximator
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# online_model = myDQN(num_states, num_actions)
# online_model.to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(online_model.parameters(), lr=learning_rate)

# # set checkpoint save directory
# ckpt_dir = './algorithms/cap-v0/checkpoints'
# dir_exist = os.path.exists(ckpt_dir)
# if not dir_exist:
#     os.mkdir(ckpt_dir)

# if __name__ == '__main__':
#     for episode in range(num_episodes):
#         loss, length, reward, epsilon = play_episode()
        
#         # save episode data after the episode is done
#         loss_list.append(loss / length)
#         step_list.append(length)
#         reward_list.append(reward)
#         epsilon_list.append(epsilon)

#         if episode % 10 == 0:
#             print('episode: {}/{}'.format(episode, num_episodes))

#         if episode % 10 == 0 and episode != 0:
#             save_model(episode)
#             save_data(step_list, reward_list)

# save_data(step_list, reward_list)
# save_model(episode = 'final')

# # print('\nSuccessful episodes: {}'.format(np.sum(np.array(reward_list)>0.0)/num_episodes))

