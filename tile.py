"""Tilecoding based machine learning"""

# Authors:  Neal Hughes <neal.hughes@anu.edu.au>

from __future__ import division
import numpy as np
from time import time
import sys
import pylab
from tilecode import Tilecode
from tilecode import buildgrid 

class TilecodeSamplegrid:

    """
    Construct a sample grid (sample of approximately equidistant points) from 
    a  large data set, using a tilecoding data structure

    Parameters
    -----------

    D : int,
        Number of input dimensions

    L : int,
        Number of tilings or 'layers'

    mem_max : float, optional (default = 1)
        Tile array size, values less than 1 turn on hashing

    cores : int, optional (default=1)
        Number of CPU cores to use (fitting stage is parallelized)

    offset : {'optimal', 'random', 'uniform'}, optional
        Type of displacement vector used

    Examples
    --------

    See also
    --------

    Notes
    -----

    This is an approximate method: it is possible that the resulting sample will contain
    some points less than ``radius`` distance apart. The accuracy improves with the number 
    of layers ``L``.

    Currently the tile widths are defined as ``int((b - a)  / radius)**-1``, so small changes in 
    radius may have no effect.

    """
 
    def __init__(self, D, L, mem_max=1, cores=1, offset='optimal'):

        if D == 1 and offset == 'optimal':
            offset = 'uniform'
        
        self.D = D
        self.L = L
        self.mem_max = mem_max
        self.cores = cores
        self.offset= offset
        
    def fit(self, X, radius, prop=1):
        
        """
        Fit a density function to X and return a sample grid with a maximum of M points

        Parameters
        ----------

        X : array of shape [N, D]
            Input data (unscaled)
    
        radius : float
            minimum distance between points. This determines tile widths.

        prop : float in (0, 1), optional (default=1.0)
            Proportion of sample points to return (lowest density points are excluded)
        
        Returns
        -------

        GRID, array of shape [M, D]
            The sample grid with M < N points

        """

        a = np.min(X, axis=0)
        b = np.max(X, axis=0)
        #Tr = int(1 / radius)
        #T = [Tr + 1] * self.D

        T = [int((b[i] - a[i]) / radius) + 1 for i in range(self.D)] 

        self.tile = Tilecode(self.D, T, self.L, mem_max=self.mem_max , cores=self.cores, offset = self.offset) 
        
        N = X.shape[0]
        GRID, max_points =  self.tile.fit_samplegrid(X, prop)
        self.max_points = max_points

        return GRID


class TilecodeRegressor:

    """    
    Tile coding for function approximation (Supervised Learning).  
    Fits by averaging and/or Stochastic Gradient Descent.
    Supports multi-core fit and predict. Options for uniform, random or 'optimal' displacement vectors.
    Provides option for linear spline extrapolation / filling

    Parameters
    -----------
    
    D : integer
        Total number of input dimensions 
    
    T : list of integers, length D
        Number of tiles per dimension 
    
    L : integer
        Number of tiling 'layers'

    mem_max : double, (default=1)
        Proportion of tiles to store in memory: less than 1 means hashing is used.
    
    min_sample : integer, (default=50) 
        Minimum number of observations per tile

    offset : string, (default='uniform')
        Type of displacement vector, one of 'uniform', 'random' or 'optimal'

    lin_spline : boolean, (default=False)
        Use sparse linear spline model to extrapolate / fill empty tiles

    linT : integer, (default=6)
        Number of linear spline knots per dimension
    
    Attributes
    -----------

    tile : Cython Tilecode instance
    
    """

    def __init__(self, D, T, L, mem_max = 1, min_sample=1, offset='optimal', lin_spline=False, linT=7, cores=4):
        
        if D == 1 and offset == 'optimal':
            offset = 'uniform'

        self.tile = Tilecode(D, T, L, mem_max, min_sample, offset, lin_spline, linT, cores)

    def fit(self, X, Y, method='A', score=False, copy=True, a=0, b=0, pc_samp=1, eta=0.01, n_iters=1, scale=0):

        """    
        Estimate tilecode weights. 
        Supports `Averaging', Stochastic Gradient Descent (SGD) and Averaged SGD.

        Parameters
        -----------
        X : array, shape=(N, D) 
            Input data (unscaled)

        Y : array, shape=(N) 
            Output data (unscaled)

        method : string (default='A')
            Estimation method, one of 'A' (for Averaging), 'SGD' or 'ASGD'.

        score : boolean, (default=False)
            Calculate R-squared

        copy : boolean (default=False)
            Store X and Y

        a : array, optional shape=(D) 
            Percentile to use for minimum tiling range (if not provided set to 0)
        
        b : array, optional, shape=(D) 
            Percentile to use for maximum tiling range (if not provided set to 100)

        pc_samp : float, optional, (default=1)
            Proportion of sample to use when calculating percentile ranges

        eta : float (default=.01)
            SGD Learning rate

        n_iters : int (default=1)
            Number of passes over the data set in SGD

        scale : float (default=0)
            Learning rate scaling factor in SGD
        """

        if method == 'A':
            sgd = False
            asgd = False
        elif method == 'SGD':
            sgd = True
            asgd = False
        elif method == 'ASGD':
            sgd = True
            asgd = True
        
        if X.ndim == 1:
            X = X.reshape([X.shape[0], 1])

        self.tile.fit(X, Y, score=score, copy=copy, a=a, b=b, pc_samp=pc_samp, sgd=sgd, asgd=asgd, eta=eta, scale=scale, n_iters=n_iters)

    def check_memory(self, ):
        
        """
        Provides information on the current memory usage of the tilecoding scheme.
        If memory usage is an issue call this function after fitting and then consider rebuilding the scheme with a lower `mem_max` parameter.
        """

        print 'Number of Layers: ' + str(self.tile.L)
        print 'Tiles per layer: ' + str(self.tile.SIZE)
        print 'Total tiles: ' + str(self.tile.L * self.tile.SIZE)
        print 'Weight array size after hashing: ' + str(self.tile.mem_max)
        temp = np.count_nonzero(self.tile.count) / self.tile.mem_max
        print 'Percentage of weight array active: ' + str(np.count_nonzero(self.tile.count) / self.tile.mem_max)
        mem_max = self.tile.mem_max / (self.tile.L*self.tile.SIZE)
        print '----------------------------------------------'
        print 'Estimated current memory usage (Mb): ' + str((self.tile.mem_max * 2 * 8)/(1024**2))
        print '----------------------------------------------'
        print 'Current hashing parameter (mem_max): ' + str(mem_max)
        print 'Minimum hashing parameter (mem_max): ' + str(temp*mem_max)

    def predict(self, X):
        
        """    
        Return tilecode predicted value 

        Parameters
        -----------
        X : array, shape=(N, D) or (D,)
            Input data

        Returns
        --------
    
        Y : array, shape=(N,)
            Predicted values
        """

        return self.tile.predict(X)

    def plot(self, xargs=['x'], showdata=True):

        """
        Plot the function on along one dimension, holding others fixed 

        Parameters
        -----------
        xargs : list, length = D
            List of variable default values, set plotting dimension to 'x'
            Not required if D = 1

        showdata : boolean, (default=False)
            Scatter training points
        """

        self.tile.plot(xargs=xargs, showdata=showdata)
        pylab.show()

class TilecodeDensity:

    """    
    Tile coding approximation of the pdf of X  
    Fits by averaging. Supports multi-core fit and predict.
    Options for uniform, random or 'optimal' displacement vectors.

    Parameters
    -----------
    
    D : integer
        Total number of input dimensions 
    
    T : list of integers, length D
        Number of tiles per dimension 
    
    L : integer
        Number of tiling 'layers'

    mem_max : double, (default=1)
        Proportion of tiles to store in memory: less than 1 means hashing is used.
    
    min_sample : integer, (default=50) 
        Minimum number of observations per tile

    offset : string, (default='uniform')
        Type of displacement vector, one of 'uniform', 'random' or 'optimal'

    Attributes
    -----------

    tile : Tilecode instance
    
    Examples
    --------

    See also
    --------

    Notes
    -----

    """

    def __init__(self, D, T, L, mem_max = 1, offset='optimal', cores=1):

        if D == 1 and offset == 'optimal':
            offset = 'uniform'
        
        self.tile = Tilecode(D, T, L, mem_max=mem_max, min_sample=1, offset=offset, cores=cores)
    
    def fit(self, X, cdf=False):
        
        N = X.shape[0]

        if X.ndim == 1:
            X = X.reshape([X.shape[0], 1])
        
        self.tile.fit(X, np.zeros(N))
        d = np.array(self.tile.d)
        w = (d**-1) / np.array(self.tile.T) 
        adj = np.product(w)**-1  
        self.tile.countsuminv = (1 / N) * adj

    def predict(self, X):

        return self.tile.predict_prob(X)

    def plot(self, xargs=['x']):

        """
        Plot the pdf along one dimension, holding others fixed 

        Parameters
        -----------
        xargs : list, length = D
            List of variable default values, set plotting dimension to 'x'
            Not required if D = 1
        """

        self.tile.plot_prob(xargs=xargs)
        pylab.show()


class TilecodeNearestNeighbour:

    """
    Fast approximate nearest neighbour search using tile coding data structure  
    

    Parameters
    -----------

    D : int,
        Number of input dimensions

    L : int,
        Number of tilings or 'layers'

    mem_max : float, optional (default = 1)
        Tile array size, values less than 1 turn on hashing

    cores : int, optional (default=1)
        Number of CPU cores to use (fitting stage is parallelized)

    offset : {'optimal', 'random', 'uniform'}, (default='optimal') optional
        Type of displacement vector used

    Examples
    --------

    See also
    --------

    Notes
    -----

    This is an approximate method: it is possible that some points > than radius may be included 
    and some < than radius may be excluded.

    """
 
    def __init__(self, D, L, mem_max=1, cores=1, offset='optimal'):

        if D == 1 and offset == 'optimal':
            offset = 'uniform'
        
        self.D = D
        self.L = L
        self.mem_max = mem_max
        self.cores = cores
        self.offset= offset
        
    def fit(self, X, radius, prop=1):
        
        """
        Fit a tile coding data structure to X

        Parameters
        ----------

        X : array of shape [N, D]
            Input data (unscaled)
    
        radius : float 
            radius for nearest neighbor queries. Tile widths for each dimension
            of X are int((b[i] - a[i]) / radius) where b and a are the
            max and min values of X[:,i].
        """

        a = np.min(X, axis=0)
        b = np.max(X, axis=0)

        T = [int((b[i] - a[i]) / radius) + 1 for i in range(self.D)] 
        
        self.tile = Tilecode(self.D, T, self.L, mem_max=self.mem_max , cores=self.cores, offset = self.offset) 
        
        self.tile.fit(X, np.ones(X.shape[0]), unsupervised=True, copy=True)

    def predict(self, X, thresh = 1):

        """
        Obtain nearest neighbors (points within distance radius)

        Parameters
        ----------

        X : array of shape [N, D]
            Query points

        thresh : int, (default=1)
            Only include points if they are active in at least thresh layers (max is L)
            Higher thresh values will tend to exclude the points furthest from the query point

        Returns
        -------

        Y : list of arrays (length = N)
            Nearest neighbors for each query point
        """
        N = X.shape[0]
        Y = np.array(self.tile.nearest(X, N, thresh))
        #idx = Y > -0.1

        #return [Y[i, idx[i,:]] for i in range(N)]


class TilecodeQVIteration:

    """
    Solve a MDP with 1 policy variable and D state variables by Q-V Iteration
    
    Parameters
    -----------

    D : int, 
        Number of state variables
    
    T : list of integers, length D
        Number of tiles per dimension 

    L : int,
        Number of tilings or 'layers'

    radius : float,
        Radius for state space sample grid

    beta : float in (0, 1),
        Discount rate

    ms : int, optional (default = 1)
        Minimum samples per tile for the Q function
    
    mem_max : float, optional (default = 1)
        Tile array size, values less than 1 turns on hashing

    cores : int, optional (default=1)
        Number of CPU cores to use 
 
    ASGD : boolean, optional (default=True)
        Fit Q function by ASGD

    offset : {'optimal', 'random', 'uniform'}, (default='optimal')
        Type of displacement vector used

    linT : integer, optional (default=6)
        Number of linear spline knots per dimension

    Examples
    --------

    See also
    --------

    Notes
    -----


    """
 
 
    def __init__(self, D, T, L, radius, beta, ms=1, mem_max=1, cores=1,  ASGD=True, linT=6):

        self.Q_f = Tilecode(D + 1, T, L, mem_max, min_sample=ms, cores=cores)
        
        self.radius = radius
        
        Twv = int((1 / self.radius) / 2)
        T = [Twv for t in range(D)]
        L = int(130 / Twv)
        
        # Initialize policy function
        self.A_f = Tilecode(D, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=cores)
        
        # Initialize value function
        self.V_f = Tilecode(D, T, L, mem_max = 1, lin_spline=True, linT=linT, cores=cores)
        
        self.first = True
        
        self.D = D
        self.beta = beta
        self.CORES = cores
        self.asgd = ASGD

    def resetQ(self, D, T, L, mem_max=1, ms=1):
    
        """
        Reset the Q function
        
        Parameters
        -----------

        D : int, 
            Number of state variables
        
        T : list of integers, length D
            Number of tiles per dimension 

        L : int,
            Number of tilings or 'layers'

        mem_max : float, optional (default = 1)
            Tile array size, values less than 1 turns on hashing
        
        ms : int, optional (default = 1)
            Minimum samples per tile for the Q function

        """

        self.Q_f = Tilecode(D + 1, T, L, mem_max, min_sample=ms, cores=self.CORES)
    
    
    def iterate(self, XA, X1, R, A_low, A_high, ITER=50, plot=False, plotdim=0, output=True, a=0, b=0, pc_samp=1, eta=0.8, maxT=60000, tilesg=False, sg_points=100, sg_prop=0.96,  sg_samp=1,  sgmem_max=0.4):

        """
        Perform QV iteration given a set of training data (N state-action and state transition samples) 
        to derive optimal value and policy functions
        
        Parameters
        -----------

        XA : array of shape [N, D + 1]
            State-action samples (i.e., actions in first column, then state variables) 
        
        X1 : array of shape [N, D]
            State transition samples (i.e., state at t+1)
        
        R : array of shape [N,]
            Payoff samples
        
        A_low : array of shape [N,], 
            Lower feasible bound for action A conditional on X

        A_high : array of shape [N,], 
            Upper feasible bound for action A conditional on X

        ITER : int, optional (default = 50)
            Number of iterations
        
        plot : boolean, optional (default = True)
            Whether to generate plots of the final value and policy function. 
        
        plotdim : int in [0, D],  optional (default = 0)
            Which state dimension to plot 
            (other dimensions are held fixed at their mean values). 
        
        a : array, optional, shape=(D) 
            Percentile to use for minimum tiling domain (if not provided set to 0)
        
        b : array, optional, shape=(D) 
            Percentile to use for maximum tiling domain (if not provided set to 100)

        pc_samp : float, optional, (default=1)
            Proportion of sample to use when calculating percentile ranges

        output : boolean, optional (default=True)
            Whether to print value function change updates each iteration
        
        eta : float (default=.01)
            ASGD / SGD learning rate
        
        maxT : int, default (default=60000)
            ASGD / SGD learning rate parameter

        tilesg : boolean, (default=False)
            If True then will use tilecoding to build state space sample grid
            else will use distance method. Tilecoding is preferred for large samples.
        
        sg_points : int, (default=100)
            If tilesg=False, then the number of points in the state space sample grid
        
        sg_prop : float, (default=0.96)
            If tilesg=True, then the proportion of points to include in the 
            state space sample grid (set less than 1 to exclude outliers)
        
        sg_samp : float, (default=0.5)  
            If tilesg=True, then the proportion of the sample points to use for the 
            state space sample grid 
        
        sgmem_max : float, (default = 0.4)
            If tilesg=True, then the mem_max (hashing) parameter of the sample grid
            tilecode scheme.  

        """
        tic = time()
        
        self.v_e = 0        # Value function error
        self.p_e = 0        # Policy function error
        
        T = XA.shape[0]
        
        self.value_error = np.zeros(ITER)
        
        if not(tilesg):
            grid, m = buildgrid(X1, sg_points, self.radius, scale=True, stopnum=X1.shape[0])
        else: 
            nn = int(X1.shape[0]*sg_samp)
            tic = time()
            tile = TilecodeSamplegrid(X1.shape[1], 25, mem_max=sgmem_max, cores=self.CORES)
            grid = tile.fit(X1[0:nn], self.radius, prop=sg_prop)
            toc = time()
            if output:
                print 'State grid points: ' + str(grid.shape[0]) + ', of maximum: ' + str(tile.max_points) + ', Time taken: ' + str(toc - tic)
            del tile
         
        points = grid.shape[0]

        if self.first:
            self.A_f.fit(grid, np.zeros(points))
            self.V_f.fit(grid, np.zeros(points))
            self.first = False
        
        minpol = np.min(A_low)
        maxpol = np.max(A_high)
        
        if ITER == 1:
            precompute = False
        else:
            precompute = True
        
        # ------------------
        #   Q-learning
        # ------------------

        ############ First iteration ================
        j = 0

        # Q values
        ticfit = time()
        Q = R + self.beta * self.V_f.predict(X1, store_XS=precompute)
        tocfit = time()
        if output:
            print 'Initial V prediction time: ' + str(tocfit - ticfit)
        
        # Fit Q function
        ticfit = time()
        self.Q_f.fit(XA, Q, pa=minpol, pb=maxpol , copy=True, unsupervised=precompute, sgd=self.asgd, asgd=self.asgd, eta=eta, n_iters=1, scale=1* (1 / min(T, maxT)), storeindex=(self.asgd and precompute), a=a, b=b, pc_samp=pc_samp)
        tocfit = time()
        if output:
            print 'Initial Q Fitting time: ' + str(tocfit - ticfit)

        # Optimise Q function
        self.value_error[0], W_opt, state = self.maximise(grid, A_low, A_high, output=output)
        ########### =================================

        ## Remaining iterations

        for j in range(1, ITER):
            # Q values
            Q = R + self.beta * self.V_f.fast_values()
            
            # Fit Q function
            self.Q_f.partial_fit(Q, 0)

            # Optimise Q function
            self.value_error[j], A_opt, state = self.maximise(grid, A_low, A_high, output=output)
        
        # Final policy function

        #A_opt_old = self.A_f.predict(grid)
        self.A_f.fit(state, A_opt, sgd=0, eta=0.1, n_iters=5, scale=0)
        A_opt_new = self.A_f.predict(grid)
        #self.pe = np.mean((W_opt_old - W_opt_new))/np.mean(W_opt_old)
        
        toc = time()

        if output:
            print 'Solve time: ' + str(toc - tic)
        
        if plot:
            xargs = [np.mean(X1[:, i]) for i in range(self.D)]
            xargs[plotdim] = 'x' 
            xargstemp = xargs
            self.A_f.plot(xargs, showdata=True)
            pylab.show()
            self.V_f.plot(xargstemp, showdata=True)
            pylab.show()

    def maximise(self, grid, A_low, A_high, output=True):

        """
        Maximises current Q-function for a subset of state space points and returns new value and policy functions

        Parameters
        -----------

        grid : array, shape=(N, D)
            State space grid

        A_low : array, shape=(N,)
            action lower bound

        A_high : array, shape=(N,)
            action upper bound
        
        Returns
        -----------

        ERROR: float
            Mean absolute deviation

        """
        
        tic = time()

        A_old = self.A_f.predict(grid)
        
        X =  np.vstack([A_old, grid.T]).T
        
        [W_opt, V, state, idx] = self.Q_f.opt(X, A_low, A_high)
        nidx = np.array([not(i) for i in idx])
        if output:
            print 'Number of optimisation points: ' + str(len(W_opt))

        V_old = self.V_f.predict(state)
        
        self.V_f.fit(state, V, sgd=0, eta=0.1, n_iters=5, scale=0)
        
        if np.count_nonzero(V_old) < V_old.shape[0]:
            self.ve = 1
        else:
            self.ve = np.mean(abs(V_old - V)/V_old)

        toc = time()
        
        if output:
            print 'Value change: ' + str(round(self.ve, 3)) + '\t---\tMaximisation time: ' + str(round(toc - tic, 4))

        return [self.ve, W_opt, state]
