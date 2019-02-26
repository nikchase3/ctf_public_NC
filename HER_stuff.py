# training

#TODO get goal from env or from observation
state, goal = env.reset()

num_states = 400 # this is like the state space size

# number of pseudo-goals we will use to augment stored trajectories
num_new_goals = 5

# list of all transition tuples during an episode (the trajectory the agent took)
episode = []

while not done:
    action = get_action(model, state, goal)
    next_state, reward, done, _ = env.step(action)
    episode.append((state, reward, done, next_state, goal))
    replay_buffer.push(state, action, reward, next_state, done, goal)
    state = next_state

    for state, reward, done, next_state, goal in episode:
        # grab a random subset of states that we accomplished during our trajectory
        # we can also grab all of them if we want :P
        # t: randomly chosen timesteps
        for t in np.random.choice(num_states, num_new_goals, replace = False)
            try:
                episode[t]
            except:
                continue

            #TODO how to incorporate "visual" and "global" observations of the state?
            # set the pseudo-goal as a state that was reached during the trajectory
            #TODO we need a global 2d coordinate from the CTF env as part of our state
            # (the agent can't use pseudo-goals that would cause them to get captured)

            #TODO (later) moving the location of the flag for a pseudo-goal will change the perceived behavior of the red team during hindsight training (i.e., they won't be playing defensively for the pseudo-flag)
            #TODO (later)we need to check for enemies along the trajectory too
            #TODO (later) for CTF, we want to force the pseudo-goals to be in enemy territory

