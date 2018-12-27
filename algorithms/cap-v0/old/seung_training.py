def policy_rollout(PARTIAL=False):
    
    # get full observation, convert to channels for features
    obs_next = one_hot_encoder(env._env, env.get_team_blue, VISION_RANGE)
    
    ep_history = []
    indv_history = [[] for _ in range(len(env.get_team_blue))]
    
    was_alive = [ag.isAlive for ag in env.get_team_blue]
    prev_reward=0
    frame=0
    
    for frame in range(max_ep+1):
        obs = obs_next
        
        with tf.device('/cpu:0'):
            act_prob = sess.run(myAgent.output, feed_dict={myAgent.state_input:obs})
        
        act = [np.random.choice(action_space, p=act_prob[x]/sum(act_prob[x])) for x in range(n_agent)] # divide by sum : normalize
            
        s,r1,d,_ = env.step(act) #Get our reward for taking an action given a bandit.

        r = r1-prev_reward

        if frame == max_ep and d == False:
            #r -= frame * (30/1000)
            r = -100
            r1 = -100

        obs_next = one_hot_encoder(env._env, env.get_team_blue, VISION_RANGE) # Full observation

        # Push history for individual that 'was' alive previous frame
        for idx, agent in enumerate(env.get_team_blue):
            if was_alive[idx]:
                indv_history[idx].append([obs[idx],act[idx],r])
        
        # State Transition
        prev_reward = r1
        was_alive = [ag.isAlive for ag in env.get_team_blue]
        
        if d:
            break

    for idx, history in enumerate(indv_history):
        if len(history) == 0:
             continue
        
        _history = np.array(history)
        _history[:,2] = discount_rewards(_history[:,2], gamma)
        ep_history.extend(_history)
            
    if len(ep_history) > 0:        
        ep_history = np.stack(ep_history)
    
    return [frame, ep_history, r1, env.blue_win, obs]