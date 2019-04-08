#testing code

#movement tests
def test_movements(grid, currentX, currentY, action):
    #individual movement test
    if action is Left:
        newGrid = env.move(grid, action)
        assert newGrid.getLocation(currentX-1, currentY) == agent
    else if action is Right
        newGrid = env.move(grid, action)
        assert newGrid.getLocation(currentX + 1, currentY) == agent
    else if action is Up
        newGrid = env.move(grid, action)
        assert newGrid.getLocation(currentX, currentY+1) == agent
    else action is Down
        newGrid = env.move(grid, action)
        assert newGrid.getLocation(currentX, currentY-1) == agent

    #test ground agent vs UAV agent

    #continous test
    #env is divided up continuously, location based on 360 deg, test location
    #create new grid, set agent to corner
    #test move() never goes offgrid

    #test move() never hits wall
    #ensure option for trapped agent

#multiple agent interaction
def test_interaction(agent1, agent2, action1, action2):
    #agent capture
    if agent1.action1.location == agent2.action2.location:
        if agent1.location.home is True:
            #policy test
        else if agent2.location.home is True:
            #policy test
    else:
        #random policy interaction test

    #nothing happens, agents don't move


#policy tests
def test_policy(grid, policy):
    #random policy
    if policy == randomPolicy:


#env testing
def test_env(grid):
    #flag exists
    red = False
    blue = False
    for x in range(0, grid.size):
        for y in range(0, grid.size):
            if grid[x][y].redFlag is True:
                red = True
            if grid[x][y].blueFlag is True:
                blue = True
    assert red and blue
def test_gameOver(grid, action, agent):
    #when game over
    if agent.action.location is flag.location:
        assert nextEpisode() = grid.episode++

#break general tests into unit