import numpy

#Library noise: https://github.com/caseman/noise/
from noise import SimplexNoise

class Continuous ():

    '''
    To be used by create_map for a continuous data. Sandboxes functionality from old discrete codebase.
    Uses Simplex Noise as a coherent-noise method. Information on this method can be found by http://libnoise.sourceforge.net/glossary/index.html#octave
    and http://staffwww.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf. Essentially just a smooth random function.
    '''

    def __init__ (period, discrete_size, view_radius, seed = None, filter = None, cutoff = None):

        '''
        TODO: Add changes parameters as necessary. 
        TODO: Update parameter list "TODO" instances.

        @param              period                 The number of points after which to pattern terrain. Should be related to discrete_size (generally greater). Currently used for randomising grid.
        @param              discrete_size          Currently unused. TODO: Size of existing discrete grid retained for fetching discrete observations. 
        @param              view_radius            Currently unused. TODO: Use to create numpy grid of possible observation space and memoise data?
        @param              granularity            Currently unused. TODO: Use to create numpy grid of possible observation space and memoise data?
        @param              seed                   Currently unused. TODO: Random seed used to randomise the environment.
        @param              filter                 The type of filter to use on underlying structure. Currently supports "walls" (which creates walls)
        @param              cutoff                 The threshold at which to make a points a wall.
        '''

        if (filter != "wall" and cutoff is None) or (filter is None and cutoff is not None):
            raise ValueError("'filter' and 'cutoff' must be specified together.")

        self.filter = filter
        self.cutoff = cutoff

        self.simplexNoise = SimplexNoise(period)
        

    def getObservations (x, y, view_radius, point_density):

        '''
        Returns ...

        @param              x                       X coordinate the observation is centred about.
        @param              y                       Y coordinate the observation is centred about.
        @param              view_radius             Number of discrete blocks to be seen in any cardinal direction.
        @param              point_density           Number of data points to provide in a single discrete block.      
        '''

        #TODO: Implement how to return observations for out-of-bounds coordinates in view_radius.
        #TODO: Implement returning NUMPY grid of appropriate size.
        #TODO: Add comments exmplaining below code.
        #TODO: Finish method.

        point_radius = view_radius * point_density
        step_size = 1 / point_density

        #TODO: Determine whether a consistnatly float grid should be used.
        if self.filter == "wall":
            grid_type = int
        else     
            grid_type = float
        
        observationGrid = numpy.empty((point_radius + 1 + point_radius, point_radius + 1 + point_radius), dtype = grid_type)

        #TODO: Determine ranges to traverse over. Will be symettric for X and Y.
        for X in ...:
            for Y in ...:

                point = self.simplexNoise.noise2(X, Y)

                if self.filter == "wall":
                    if point > self.cutoff:
                        point = 1
                    else:
                        point = 0

                observationGrid[X][Y] = point

        return observationGrid 

    def move (path)

        '''
        TODO: Implement.
        TODO: Add some way to determine step size (how often to check if a point on the walk is untraversable). 
        TODO: Additional parameters. Agent?

        Takes a path of tuples (coordinates) and tries and traverse it. If it every gets stuck (hits a wall or other untraversable terrain)
        it will terminate. Linearizes between points. 
        '''

    def valid ():
        # returns whether a point is valid for unit placement

    