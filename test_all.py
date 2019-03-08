#testing code

#movement tests
def test_movements(grid, currentX, currentY, action):
    #individual movement test
    if action is Left:
        newGrid = env.move(grid, action)
        assert(newGrid.getLocation(currentX-1, currentY) == agent)
    else if action is Right
        newGrid = env.move(grid, action)
        assert (newGrid.getLocation(currentX + 1, currentY) == agent)
    else if action is Up
        newGrid = env.move(grid, action)
        assert (newGrid.getLocation(currentX, currentY+1) == agent)
    else action is Down
        newGrid = env.move(grid, action)
        assert (newGrid.getLocation(currentX, currentY-1) == agent)

#multiple agent interaction
def test_interaction(grid, action1, action2):
    #agent capture

    #nothing happens

#env testing
def test_env(grid):
    #confirm creation of continuous world