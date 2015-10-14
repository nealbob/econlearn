#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=False, initializedcheck=False
#
# Multiple tilecodings - fit by SGD / ASGD 
# Adaptive tilecoding - Two Tilecodings (switch to fine tilecoding scheme when sample above threshold)
#
#
# Need multi w arrays, multi count (offset, flat) can't scale to * T
# Multi hashing

from __future__ import division
import pylab
cimport numpy as np
import numpy as np
import time
cimport cython
from sklearn.linear_model import LinearRegression as OLS
from displace import Displace
from cython.parallel import prange, parallel
#from cython_gsl cimport *

from libc.math cimport fmin as c_min
from libc.math cimport fmax as c_max
from libc.math cimport log as c_log
from libc.math cimport exp as c_exp
from libc.math cimport sin as c_sin
from libc.math cimport cos as c_cos

# ===============================================
#   Inline functions
# ===============================================

cdef inline int getindex(int j, int i, double[:, :] XS, int D, double[:,:] offset, int[:] flat, int size, int dohash, int mem_max, int[:] key) nogil:
    
    cdef int idx = 0
    
    idx = index(j, i, XS, D, offset, flat, size)
    
    if dohash == 1:
        idx = hash_idx(idx, key, mem_max)

    return idx

cdef inline int index(int j, int i, double[:, :] XS, int D, double[:,:] offset, int[:] flat, int size) nogil:

    cdef int k, idx = 0
    
    for k in range(D):
        idx +=  c_int(XS[i, k] + offset[j, k]) * flat[k]
    idx += size * j
    
    return idx

cdef inline int hash_idx(int idx, int[:] key, int mem_max) nogil:

    cdef int hashidx = 0
    cdef int n = 0

    hashidx = idx * (idx + 3)   #hash function f(x) = x(x + 3) 
    
    hashidx = c_abs(hashidx % mem_max)

    # Open addressing with linear probing
    while key[hashidx] != idx and n < mem_max:
        hashidx += 1
        hashidx = hashidx % mem_max
        if key[hashidx] == 0:
            key[hashidx] = idx
            break

    return hashidx

cdef inline double linear_value(int i, double[:,:] XS, int D, double[:] Tinv, int[:] lin_T, double[:,:] lin_w, double[:,:] lin_c) nogil:

    cdef int k = 0
    cdef double val = 0
    cdef int idx = 0
    cdef double x = 0

    for k in range(D):
        x = XS[i, k] * Tinv[k]
        idx = c_int(c_min(c_max(x, 0.000001), 0.99999) * lin_T[k])
        val += lin_w[k, idx] * x + lin_c[k, idx]

    return val
        
cdef inline double predict(int D, int i, double[:,:]  XS, int extrap, double[:] Tinv, double[:,:] offset, int[:] flat, double Linv, double[:] w, int[:] count, int min_count, int linval, double[:,:] linw, double[:,:] linc, int[:] linT, int size, int L,  int dohash, int mem_max, int[:] key) nogil:

    cdef int j, k
    cdef double y = 0
    cdef int idx = 0
    cdef int nosample = 0

    if extrap == 0:
        for j in range(L): 
        
            idx = getindex(j, i, XS, D, offset, flat, size, dohash, mem_max, key)
           
            if count[idx] >= min_count:
                y += w[idx]
            else:
                if linval == 1:
                    y += linear_value(i, XS, D, Tinv, linT, linw, linc)
                else:
                    y = 0
                    break
        y =  y * Linv
    else:
        if linval == 1:
            y = linear_value(i, XS, D, Tinv, linT, linw, linc)
    
    return y

cdef inline double predict_fast(int D, int i, double[:,:]  XS, int extrap, double[:] Tinv, double Linv, double[:] w, int[:] count, int min_count, int linval, double[:,:] linw, double[:,:] linc, int[:] linT, int L, int[:,:] datastruct_reverse) nogil:

    cdef int j, k
    cdef double y = 0
    cdef int idx = 0
    cdef int nosample = 0

    if extrap == 0:
        for j in range(L): 
        
            idx = datastruct_reverse[i, j]
           
            if count[idx] >= min_count:
                y += w[idx]
            else:
                if linval == 1:
                    y += linear_value(i, XS, D, Tinv, linT, linw, linc)
                else:
                    y = 0
                    break
        y =  y * Linv
    else:
        if linval == 1:
            y = linear_value(i, XS, D, Tinv, linT, linw, linc)
    
    return y

cdef class Tilecode:
    
    """    
    Tile coding for function approximation (Supervised Learning).  
    Fits by averaging and/or Stochastic Gradient Descent.
    Supports multi-core fit and predict. Options for uniform, random or 'optimal' displacement vectors.
    Provides option for linear spline extrapolation / filling (suitable for small data sets)
    Optimises over first dimension (by grid search) subject to supplied constraints.

    Parameters
    -----------
    
    D : integer
        Total number of dimensions (e.g., action + state dimensions)
    
    T : list of integers, length D
        Number of tiles per dimension 
    
    L : integer
        Number of tiling 'layers'

    mem_max : double, (default=1)
        Proportion of total tiles to store in memory: less than 1 means hashing is used.
    
    min_sample : integer, (default=50) 
        Minimum number of observations per tile

    offset : string, (default='uniform')
        Type of displacement vector one of 'uniform', 'random' or 'optimal'

    lin_spline : boolean, (default=False)
        Use sparse linear spline model to extrapolate / fill empty tiles

    linT : integer, (default=6)
        Number of linear spline knots per dimension
    
    Attributes
    -----------

    N : integer
        Number of samples

    X : array, shape=(N, D)
        Input data (unscaled) (e.g., state-action samples, with action as first dimension X[:,0])

    Y : array, shape=(N) 
        Output data (unscaled), (e.g., Q samples)

    a : array, shape=(D) 
        Minimum of tiling range in each dimension (observations outside of range are ignored)
    
    b : array, shape=(D) 
        Maximum of tiling range in each dimension (observations outside of range are ignored)
    
    R2 : array, shape=(D) 
        Score (r-squared)
    
    """ 

    def __init__(self, D, T, L, mem_max = 1, min_sample=1, offset='optimal', lin_spline=False, linT=7, cores=4):

        self.D = D
        self.L = L
        self.Linv = 1.0 / L
        self.min_sample = min_sample
        self.T = np.array(T, dtype='int32')
        self.Tinv = np.zeros(D)
        self.flat = np.ones([self.D], dtype='int32')
        self.SIZE = 1
        width = np.zeros(self.D)
        self.offset = np.zeros([self.L, self.D])
        self.CORES = cores


        # Calculate tile widths and total number of tiles
        cdef int k, j
        for k in range(self.D):
            self.Tinv[k] = 1.0 / self.T[k]
            width[k] = 1.0 / self.T[k]
            self.SIZE *= (self.T[k] + 1)
            for j in range(k + 1, self.D):
                self.flat[k] *= (self.T[j] + 1)

        # Build displacement (offset) vectors
        if offset == 'uniform':
            for k in range(self.D):
                for j in range(self.L):
                    self.offset[j, k] = (width[k] - (j + 1) * (width[k] / self.L) ) * self.T[k]
        
        elif offset == 'random':
            temp = range(self.L)
            for k in range(self.D):
                np.random.shuffle(temp)
                off = 1.0 * np.array(temp) / self.L + (1/self.L)
                for j in range(self.L):
                    self.offset[j, k] = (width[k] - off[j] * width[k] ) * self.T[k]
            for k in range(self.D):
                self.offset[0, k] = (width[k] - 0.5 * width[k] ) * self.T[k]
        
        elif offset == 'optimal':
            d = np.array(Displace().table[self.D][self.L])
            for j in range(self.L):
                for k in range(self.D):
                    self.offset[j, k] = (width[k] - (width[k] / self.L) * (1 + (((j + 1) * d[k]) % self.L ) ) ) * self.T[k]
        
        if mem_max == 1:
            self.dohash = 0
        else:
            self.dohash = 1
        
        self.mem_max = int(self.L * self.SIZE * mem_max)

        self.w = np.zeros(self.mem_max)
        self.count = np.zeros(self.mem_max, dtype='int32')
        self.key = np.zeros(self.mem_max, dtype='int32')
        self.xs = np.zeros([1, self.D])

        # Linear Spline (for extrapolation)
        self.lin_spline = 0
        self.lin_T = np.ones(self.D, dtype='int32') * linT
        self.lin_width = np.zeros(self.D)
        self.lin_SIZE = self.D * linT
        self.lin_w = np.zeros([self.D, linT])
        self.lin_c = np.zeros([self.D, linT])
        if lin_spline:
            self.lin_spline = 1
            for k in range(self.D):
                self.lin_width[k] = 1.0 / self.lin_T[k]
        
    def score(self, X, Y):

        X = np.array(X)
        Y = np.array(Y)
        Y_hat = self.predict(X)
        index = Y_hat != 0

        SS_tot = np.sum((Y[index] - np.mean(Y[index]))**2)
        SS_res = np.sum((Y[index] - Y_hat[index])**2)

        R2 = 1 -  SS_res / SS_tot

        return R2
        
    def fit(self, double[:,:] X, double[:] Y, policy=0, pa=-1, pb=-1, unsupervised=False, score=False, copy=True, a=0, b=0, 
            pc_samp=1, sgd=False, eta=0.01, n_iters=1, scale=0, asgd=False, storeindex=False, samplegrid=False, M=0, NS=False):

        """    
        Fit tilecode function by averaging (then by SGD if sgd=True)

        Parameters
        -----------
        X : array, shape=(N, D) 
            Input data (unscaled), policy variable is first dimension X[:,0]

        Y : array, shape=(N) 
            Output data (unscaled), (e.g., Q samples)

        score : boolean, (default=False)
            Calculate R-squared

        cores : integer, (default=1)
            Number of CPU cores / jobs to use for fitting

        copy : boolean (default=False)
            Store X and Y

        setrange : boolean (default=False)
            Restrict tiling to a percentile range (if False then use min-max of data)

        a : array, optional shape=(D) 
            Percentile to use for minimum tiling range
        
        b : array, optional, shape=(D) 
            Percentile to use for maximum tiling range

        pc_samp : float, (default=0.05
            Proportion of sample to use when calculating percentile ranges (only needed when setrange = True)

        R2 : array, shape=(D)
            Score (r-squared)
        
        sgd : boolean (default=False)
            Fit by Stochastic Gradient Descent (SGD) (using averaging for starting values)

        eta : float (default=.01)
            SGD Learning rate

        n_iters : int (default=1)
            Number of passes over the data set in SGD

        scale : float (default=0)
            Learning rate scaling factor in SGD
        """
        ##################      Initialize      ####################

        # Number of data points
        self.N = X.shape[0]

        # Store data
        if copy:
            self.X = X 
            self.Y = Y

        # Set min-max tiling range
        if a == 0:
            a = [0]*self.D
        if b == 0:
            b = [100]*self.D
        atemp = np.zeros(self.D)
        btemp = np.zeros(self.D)
        n = int(pc_samp * self.N)
        for i in range(self.D):
            atemp[i] = np.percentile(X[1:n, i], a[i])
            btemp[i] = np.percentile(X[1:n, i], b[i])
        
        if not(pa == -1):
            atemp[policy] = pa
        if not(pb == -1):
            btemp[policy] = pb

        dtemp = 1 / (btemp - atemp)
        # Memoryviews
        self.a = atemp
        self.b = btemp
        self.d = dtemp

        # Zero weights
        self.w = np.zeros([self.mem_max])
        self.wav = np.zeros([self.mem_max])
        self.count = np.zeros([self.mem_max], dtype='int32')
        self.key = np.zeros([self.mem_max], dtype='int32')
        
        #if self.dohash == 0:
        #    self.key = np.array(range(self.mem_max), dtype='int32')

        self.extrap = np.zeros(self.N, dtype='int32')
        cdef double[:,:] XS = np.zeros([self.N, self.D])
        cdef int z = 0
        ###############################################################
        
        XS = self.scale_X(X, self.N, XS)
        self.extrap = self.check_extrap(XS, self.N, self.extrap)
        
        if samplegrid:
            return self.fit_samplegrid(X, XS, self.N, M)
        else:

            self.fit_tiles(XS, Y)
            
            if storeindex:
                self.datastruct_reverse = np.zeros([self.N, self.L], dtype='int32')
                self.fit_data_reverse(XS, self.N, self.extrap)
            
            self.sgd = 0 
            if sgd:
                self.sgd = 1
                if asgd == True:
                    self.asgd = 1
                    self.wav = np.zeros([self.mem_max])
                else:
                    self.asgd = 0
                self.eta = eta 
                self.scale = scale 
                
                self.tempidx = np.zeros(self.L, dtype='int32')
                self.fit_sgd(XS, Y, eta, scale, n_iters, self.asgd)
                self.wav = np.zeros([self.mem_max])
            
            
            
            if unsupervised:
                self.datastruct = np.ones(int(np.sum(self.count) + 10), dtype='int32')
                self.datahash = np.zeros(self.mem_max, dtype='int32')
                self.fit_data(XS, np.zeros(self.mem_max, dtype='int32'))
                self.max_neigh = min(np.max(self.count) * self.L, self.N)
            
            if self.lin_spline == 1:
                if NS:
                    for z in range(self.N):
                        XS[z, 0] = 0.0

                self.fit_linear(XS, Y)
            
            if score:
                self.R2 = self.score(X, Y)
                print 'R-squared: ' + str(self.R2)

            mem_usage = np.count_nonzero(self.count) / self.mem_max
            if mem_usage < 0.025:
                print 'Tilecoding memory usage: ' + str(mem_usage)
                print 'On average ' + str(np.count_nonzero(self.count)/self.L) + ' of ' + str(self.SIZE) + ' tiles are active per layer'
    
    cdef int[:] check_extrap(self, double[:,:] XS, int N, int[:] extrap):

        cdef int i, k
        cdef double xs = 0
        cdef double xmax = 0

        for i in prange(N, nogil=True, num_threads=self.CORES, schedule=dynamic):
            for k in range(self.D):
                xs = XS[i, k]
                xmax = self.T[k] + 0.0001
                if xs < -0.0001 or xs > xmax:
                    extrap[i] = 1

        return extrap

    cdef double[:,:] scale_X(self, double[:,:] X, int N, double[:,:] XS):

        cdef int i, k

        for i in prange(N, nogil=True, num_threads=self.CORES, schedule=dynamic):
            for k in range(self.D):
                XS[i, k] = ((X[i, k] - self.a[k]) * self.d[k]) * self.T[k]

        return XS
         

    cdef void fit_tiles(self, double[:,:] X, double[:] Y):
    
        cdef int idx = 0 
        cdef int i, j
        cdef double n = 0

        for j in prange(self.L, nogil=True, num_threads=self.CORES, schedule=guided):
            for i in range(self.N): 
                if self.extrap[i] == 0:
                    idx = getindex(j, i, X, self.D, self.offset, self.flat, self.SIZE, self.dohash, self.mem_max, self.key)
                    self.count[idx] += 1
                    self.w[idx] += Y[i]

        for j in prange(self.mem_max, nogil=True, num_threads=self.CORES, schedule=dynamic):
            n = self.count[j]
            if n > 0:
                self.w[j] = self.w[j] * (n**-1)

    cdef void fit_data(self, double[:,:] X, int[:] countsofar):
        
        """ fit data structure mapping tiles to data points """

        cdef int idx, dataidx = 0
        cdef int i, j
        cdef double n = 0
        
        for i in range(1, self.mem_max):
            self.datahash[i] = self.datahash[i-1] + self.count[i-1] 

        for i in prange(self.N, num_threads=self.CORES, nogil=True, schedule=guided):
            if self.extrap[i] == 0:
                for j in range(self.L):
                    idx = getindex(j, i, X, self.D, self.offset, self.flat, self.SIZE, self.dohash, self.mem_max, self.key)
                    dataidx = self.datahash[idx]
                    self.datastruct[dataidx + countsofar[idx]] = i
                    countsofar[idx] += 1
        
    cdef void fit_data_reverse(self, double[:,:] X, int N, int[:] extrap):

        """ fit data structure mapping data points to tiles """

        cdef int idx, dataidx = 0
        cdef int i, j
        cdef double n = 0
        
        for i in prange(N, num_threads=self.CORES, nogil=True, schedule=guided):
            if extrap[i] == 0:
                for j in range(self.L):
                    idx = getindex(j, i, X, self.D, self.offset, self.flat, self.SIZE, self.dohash, self.mem_max, self.key)
                    self.datastruct_reverse[i, j] = idx
   

    def fit_samplegrid(self, double[:,:] X, double prop):
        self.N = X.shape[0]
        
        cdef int idx = 0
        cdef int i, j, k, h
        cdef int m = 0
        cdef double dens = 0
        cdef double[:,:] grid
        cdef double cmax = 0
        cdef int cmax_idx = 0
        cdef int[:] xc = np.zeros([self.N], dtype='int32')
        cdef double[:] xc_dens = np.zeros([self.N])
        cdef int[:] xc_count
        cdef int M = 0

        atemp = np.min(X, axis=0)
        btemp = np.max(X, axis=0)
        dtemp = 1 / (btemp - atemp)

        # Memoryviews
        self.a = atemp
        self.b = btemp
        self.d = dtemp
        
        # scale X
        cdef double[:,:] XS = np.zeros([self.N, self.D])
        XS = self.scale_X(X, self.N, XS)

        # Zero weights
        self.count = np.zeros([self.mem_max], dtype='int32')
        xc_count = np.zeros(self.mem_max, dtype='int32')
        self.key = np.zeros([self.mem_max], dtype='int32')
        
        self.datastruct_reverse = np.zeros([self.N, self.L], dtype='int32')

        for i in prange(self.N, num_threads=self.CORES, nogil=True, schedule=guided):
            for j in range(self.L):
                idx = getindex(j, i, XS, self.D, self.offset, self.flat, self.SIZE, self.dohash, self.mem_max, self.key)
                self.datastruct_reverse[i, j] = idx
                self.count[idx] += 1
        
        mem_usage = np.count_nonzero(self.count) / self.mem_max
        print 'Tilecoding memory usage: ' + str(mem_usage)
        print 'On average ' + str(np.count_nonzero(self.count)/self.L) + ' of ' + str(self.SIZE) + ' tiles are active per layer'
        
        for i in range(self.N):
            dens = 0
            for j in range(self.L):
                idx = self.datastruct_reverse[i, j]
                dens += xc_count[idx] * self.Linv
            if dens == 0:
                xc[m] = i
                m += 1
                for j in range(self.L):
                    idx = self.datastruct_reverse[i, j] 
                    xc_count[idx] += 1
        
        if prop < 1:           
            M = <int> (prop * m)
            grid = np.zeros([M, self.D])
            for h in range(m):  # Predict density
                i = xc[h]
                dens = 0
                for j in range(self.L):
                    idx = self.datastruct_reverse[i, j]
                    dens += self.count[idx] * self.Linv
                xc_dens[h] = dens

            for j in range(M): # Select highest density points
                cmax = 0
                for h in range(m):
                    if xc_dens[h] > cmax:
                        cmax = xc_dens[h]
                        cmax_idx = h
                xc_dens[cmax_idx] = -1
                for k in range(self.D):
                    grid[j,k] = X[xc[cmax_idx], k]
        else:
            grid = np.zeros([m, self.D])
            for j in range(m): 
                for k in range(self.D):
                    grid[j,k] = X[xc[j], k]
   
        return [grid, m]
    
    def nearest(self, double[:,:] X, int N, int thresh):

        # ==============================
        # Find nearest neighbors
        # ===============================
        
        cdef int i, j
        cdef int[:] neigh_idx = np.zeros(self.max_neigh, dtype='int32')
        cdef int[:] neigh_count = np.zeros(self.max_neigh, dtype='int32')
        cdef int[:] neigh_key = np.zeros(self.N, dtype='int32') 
        cdef int[:] n = np.zeros(N, dtype='int32') 
        cdef int k, pointidx, dataidx, idx, key, data
        cdef int totalcount = 0
        cdef int z = 0
        cdef int[:] pos = np.zeros(N, dtype='int32') 
        cdef double[:,:] XS = np.zeros([N, self.D])

        self.neigh_idx = np.ones([N, self.max_neigh], dtype='int32') * -1
        self.neigh_count= np.zeros(N, dtype='int32')
        
        XS = self.scale_X(X, N, XS)

        for i in prange(N, nogil=True, num_threads=self.CORES, schedule=static):
            data = 1
            for j in range(self.L):
                idx = getindex(j, i, XS, self.D, self.offset, self.flat, self.SIZE, self.dohash, self.mem_max, self.key)
                if self.count[idx] == 0:
                    data = 0
                    break
                dataidx = self.datahash[idx]
                for k in range(self.count[idx]):
                    pointidx = self.datastruct[dataidx + k] 
                    key = neigh_key[pointidx]
                    if key == 0:
                        neigh_idx[n[i]] = pointidx
                        neigh_key[pointidx] = n[i]
                        neigh_count[n[i]] += 1
                        n[i] += 1
                    else:
                        neigh_count[key] += 1
            
            for z in range(n[i]):
                if neigh_count[z] > thresh:
                    self.neigh_idx[i,pos[i]] = neigh_idx[z]
                    pos[i] += 1
                self.neigh_count[i] = pos[i]

        return self.neigh_idx
    
    """ 
    def predict_quad(self, X):
        
        cdef int i, k, N = 0

        if len(X.shape) == 1:
            X = X.reshape([1, self.D])
            N = 1
        else:
            N = X.shape[0]

        cdef double[:,:] XS = np.zeros([N, self.D])

        XS = self.scale_X(X, N, XS)
    
        return np.array(self.local_quadratic(XS, N))

    cdef double[:] local_quadratic(self, double[:, :] X, int N):

        cdef int[:] neigh_idx = np.zeros(self.max_neigh, dtype='int32')
        cdef int[:] neigh_count = np.zeros(self.max_neigh, dtype='int32')
        cdef int[:] neigh_key = np.zeros(self.N, dtype='int32') 
        cdef int[:] n = np.zeros(N, dtype='int32') 
        cdef int[:] col = np.zeros(N, dtype='int32') 
        cdef int i, j, k, pointidx, dataidx, idx, key, h, data, z
        cdef double[:] x = np.zeros(self.D + 1)
        cdef double weight, totalcount = 0
        cdef double[:] yhat = np.zeros(N)
        cdef int cols = <int> (((self.D + 1)*(self.D + 2)) / 2)

        cdef gsl_vector *S
        cdef gsl_vector *beta
        cdef gsl_matrix *V
        cdef gsl_matrix *Xm
        cdef gsl_matrix *XX
        cdef gsl_vector *Xy
        cdef gsl_vector *y
        
        V = gsl_matrix_alloc(cols, cols)
        S = gsl_vector_alloc(cols)
        beta = gsl_vector_calloc(cols)
        Xm = gsl_matrix_calloc(self.max_neigh, cols)
        XX = gsl_matrix_calloc(cols, cols)
        Xy = gsl_vector_calloc(cols)
        y = gsl_vector_calloc(self.max_neigh)

        for i in range(N): #, nogil=True, num_threads=self.CORES, schedule=static):
            
            # ==============================
            # Find nearest neighbors
            #===============================

            totalcount = 0
            data = 1
            for j in range(self.L):
                idx = getindex(j, i, X, self.D, self.offset, self.flat, self.SIZE, self.dohash, self.mem_max, self.key)
                if self.count[idx] == 0:
                    data = 0
                    break
                dataidx = self.datahash[idx]
                totalcount += self.count[idx]
                for k in range(self.count[idx]):
                    pointidx = self.datastruct[dataidx + k] 
                    key = neigh_key[pointidx]
                    if key == 0:
                        neigh_idx[n[i]] = pointidx
                        neigh_key[pointidx] = n[i]
                        neigh_count[n[i]] += 1
                        n[i] += 1
                    else:
                        neigh_count[key] += 1
            
            if n[i] > (2 * cols) and data == 1:
                
                #==============================================
                # Fit quadratic by OLS, using the c GSL library
                #==============================================
                
                #n[i] = <int> c_min(n[i], 1000)

                V = gsl_matrix_calloc(cols, cols)
                S = gsl_vector_calloc(cols)
                beta = gsl_vector_calloc(cols)
                Xm = gsl_matrix_calloc(n[i], cols)
                XX = gsl_matrix_calloc(cols, cols)
                Xy = gsl_vector_calloc(cols)
                y = gsl_vector_calloc(n[i])
                
                #gsl_matrix_set_zero(V)
                #gsl_vector_set_zero(S)
                #gsl_vector_set_zero(beta)
                #gsl_matrix_set_zero(Xm)
                #gsl_matrix_set_zero(XX)
                #gsl_vector_set_zero(Xy)
                #gsl_vector_set_zero(y)
                
                #NN_counter = gsl_vector_view_array(neigh_count, n[i])
                #gsl_sort_vector_index(NN_counter.vector)
                 
                for h in range(n[i]):
                    x[0] = 1
                    weight = (neigh_count[h] * (totalcount**-1)) #** c_max( ((n / 2000)**0.5), 1) 
                    for k in range(self.D):
                        x[k + 1] = ((self.X[neigh_idx[h], k] - self.a[k] ) * self.d[k]) * self.T[k] 
                    gsl_vector_set(y, h, self.Y[neigh_idx[h]] * weight) 
                    col[i] = 0
                    for k in range(self.D + 1):             
                        for z in range(k, self.D + 1):
                            gsl_matrix_set(Xm, h, col[i], x[k] * x[z] * weight) 
                            col[i] += 1
                
                gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Xm, Xm, 0, XX) 
                gsl_blas_dgemv(CblasTrans, 1.0, Xm, y, 0, Xy) 

                gsl_linalg_cholesky_decomp(XX)
                gsl_linalg_cholesky_solve(XX, Xy, beta)

                #gsl_linalg_SV_decomp_jacobi(Xm, V, S)
                #gsl_linalg_SV_solve(Xm, V, S, y, beta) 
                
                #for h in range(cols):
                #    print 'Beta: ' + str(gsl_vector_get(beta, h))

                # ========================================= 
                # Evaluate quadratic
                # =========================================
                
                x[0] = 1
                for k in range(self.D):
                    x[k + 1] = X[i, k]
                
                col[i] = 0
                for k in range(self.D + 1):             
                    for z in range(k, self.D + 1):
                        yhat[i] += gsl_vector_get(beta, col[i]) * x[k] * x[z]
                        col[i] += 1

                # ========================================
                # zero for next iteration
                # ========================================
                
                for j in range(self.max_neigh):
                    neigh_idx[j] = 0 
                    neigh_count[j] = 0 
                
                for j in range(self.N):    
                    neigh_key[j] = 0
            else:
                yhat[i] = 0
        else:
            yhat[i] = 0
        
        gsl_vector_free(beta)
        gsl_vector_free(S)
        gsl_matrix_free(V)
        gsl_matrix_free(Xm)
        gsl_matrix_free(XX)
        gsl_vector_free(Xy)
        gsl_vector_free(y)
        
        return yhat
    """

    cpdef partial_fit(self, double[:] Y, int copy):

        cdef int i, k, pointidx, dataidx, idx, t = 0
        cdef double n, new_w
        cdef double y, error, alpha, power = (2.0/3.0)*-1 
        
        # Fit by averaging (assuming we have data structure)
        for i in prange(self.mem_max, nogil=True, num_threads=self.CORES, schedule=guided):
            dataidx = self.datahash[i] 
            self.w[i] = 0
            for k in range(self.count[i]):
                pointidx = self.datastruct[dataidx + k] 
                self.w[i] += Y[pointidx]
            n = self.count[i]
            if n > 0:
                self.w[i] = self.w[i] * (n**-1)
        
        # Store Y values
        if copy == 1:
            for i in range(self.N):
                self.Y[i] = Y[i]
        
        # Fit by SGD / ASGD (assuming we have - reverse data structure)
        if self.sgd == 1:
            for i in range(self.N):
                if self.extrap[i] == 0:
                    t += 1
                    alpha =  self.eta * (1 +  self.eta * self.scale * t)**power
                    y = 0
                    for j in range(self.L): 
                        idx = self.datastruct_reverse[i, j]
                        y += self.w[idx]
                    y =  y * self.Linv

                    error = (Y[i] - y) * self.Linv

                    for j in range(self.L):
                        idx = self.datastruct_reverse[i, j]
                        self.w[idx] += error * alpha
                        self.wav[idx] += self.w[idx]
            if self.asgd == 1:
                for i in prange(self.mem_max, nogil=True, num_threads=self.CORES, schedule=guided):
                    n = self.count[i]
                    if n > 0:
                        self.w[i] = self.wav[i] * (n**-1)

            self.wav = np.zeros([self.mem_max])

    cdef void fit_sgd(self, double[:,:] X, double[:] Y, double eta, double scale, int n_iters, int ASGD):

        cdef int i, j, k, t, it = 0
        cdef int[:] idx = self.tempidx
        cdef double Yhat
        cdef double y
        cdef double alpha = 0
        cdef double error = 0
        cdef double n = 0
        cdef double power = (2.0/3.0)
        
        for it in range(n_iters):
            t = 0
            for i in range(self.N):
                if self.extrap[i] == 0:
                    t += 1
                    alpha =  eta * (((1 +  eta * scale * t)**power)**-1)

                    # predict yhat
                    y = 0
                    for j in range(self.L): 
                        idx[j] = getindex(j, i, X, self.D, self.offset, self.flat, self.SIZE, self.dohash, self.mem_max, self.key)
                        y += self.w[idx[j]]
                    y =  y * self.Linv

                    # compute error
                    error = (Y[i] - y) * self.Linv

                    for j in range(self.L):
                        self.w[idx[j]] += error * alpha
                        self.wav[idx[j]] += self.w[idx[j]]
                        idx[j] = 0

        if ASGD == 1:           
            for i in prange(self.mem_max, nogil=True, num_threads=self.CORES, schedule=guided):
                n = self.count[i]
                self.w[i] = self.wav[i] * (n**-1)

    def partial_sgd(self, double[:,:] X, double[:] Y, double eta, double scale, int n_iters, int ASGD):

        cdef int i, j, k, t, it = 0
        cdef int[:] idxL = self.tempidx
        cdef int idx = 0
        cdef double Yhat
        cdef double y
        cdef double alpha = 0
        cdef double error = 0
        cdef double n = 0
        cdef double power = (2.0/3.0)
    
        cdef int N = X.shape[0]
        self.count = np.zeros([N, self.mem_max], dtype='int32')

        for it in range(n_iters):
            t = 0
            for i in range(N):
                if self.extrap[i] == 0:
                    t += 1
                    alpha =  eta * (((1 +  eta * scale * t)**power)**-1)

                    # predict yhat
                    y = 0
                    idx = 0 
                    for j in range(self.L): 
                        idx = getindex(j, i, X, self.D, self.offset, self.flat, self.SIZE, self.dohash, self.mem_max, self.key)
                        self.count[idx] += 1
                        y += self.w[idx]
                        idxL[j] = idx
                    y =  y * self.Linv

                    # compute error
                    error = (Y[i] - y) * self.Linv

                    for j in range(self.L):
                        self.w[idxL[j]] += error * alpha
                        self.wav[idxL[j]] += self.w[idxL[j]]
                        idxL[j] = 0

        if ASGD == 1:           
            for i in prange(self.mem_max, nogil=True, num_threads=self.CORES, schedule=guided):
                n = self.count[i]
                self.w[i] = self.wav[i] * (n**-1)
    
    cdef void fit_linear(self,  double[:,:] X, double[:] Y):

        cdef int i, j, k, z = 0
        cdef double x
        cdef double[:,:] Xreg = np.zeros([self.N, self.lin_SIZE])
        cdef double[:] beta = np.zeros(self.lin_SIZE)
        cdef double c = 0

        for i in range(self.N):
            z = 0
            for j in range(self.D):
                x = X[i, j] / self.T[j]
                for k in range(self.lin_T[j]):
                    Xreg[i, z] = c_max(x - self.lin_width[j] * k, 0)
                    z +=1
        
        ols = OLS(fit_intercept=True).fit(Xreg, Y)
        beta = ols.coef_
        c = ols.intercept_ / self.D

        z = 0

        self.lin_w = np.zeros([self.D, self.lin_T[0]])
        self.lin_c = np.zeros([self.D, self.lin_T[0]])

        for j in range(self.D):
            for k in range(self.lin_T[j]):
                if k > 0:
                    self.lin_w[j, k] = beta[z] + self.lin_w[j, k - 1]
                    self.lin_c[j, k] += self.lin_w[j, k - 1] * self.lin_width[j] * k - self.lin_w[j, k] * self.lin_width[j] * k + self.lin_c[j, k -1]
                else:
                    self.lin_w[j, k] = beta[z]
                    self.lin_c[j, k] = c

                z += 1

    def predict(self, X, store_XS=False):
        
        """    
        Return predicted value 

        Parameters
        -----------
        X : array, shape=(N, D) or (D,)
            Input data

        Returns
        --------
    
        Y : array, shape=(N,)
            Predicted values
        """
        
        cdef int i, k, N = 0
        
        if self.D == 1:
            N = X.shape[0]
            X = X.reshape([N, 1])
        elif len(X.shape) == 1:
            X = X.reshape([1, self.D])
            N = 1
        else:
            N = X.shape[0]

        cdef double[:] Y = np.zeros(N)
        cdef double[:,:] XS = np.zeros([N, self.D])
        cdef int[:] extraptemp = np.zeros(N, dtype='int32')
        
        XS = self.scale_X(X, N, XS) 
        extraptemp = self.check_extrap(XS, N, extraptemp)
        
        if store_XS:
            self.XS = np.zeros([N, self.D])
            self.XS[...] = XS
            self.extrapXS = np.zeros(N, dtype='int32')
            self.extrapXS[...] = extraptemp
            self.XSN = N
            self.fastvalues = np.zeros(N)
            self.datastruct_reverse = np.zeros([N, self.L], dtype='int32')
            self.fit_data_reverse(XS, N, extraptemp)

        for i in prange(N, nogil=True, num_threads=self.CORES, schedule=static):
            Y[i] = predict(self.D, i, XS, extraptemp[i], self.Tinv, self.offset, self.flat, self.Linv, self.w, self.count, self.min_sample, self.lin_spline, self.lin_w, self.lin_c, self.lin_T, self.SIZE, self.L, self.dohash, self.mem_max, self.key)
        return np.array(Y)
    
    def predict_prob(self, X):
        
        """    
        Return predicted cdf / pdf value

        Parameters
        -----------
        X : array, shape=(N, D) or (D,)
            Input data

        """
        
        cdef int i, k, N = 0

        if self.D == 1:
            N = X.shape[0]
            X = X.reshape([N, 1])
        elif len(X.shape) == 1:
            X = X.reshape([1, self.D])
            N = 1
        else:
            N = X.shape[0]

        cdef double[:] Y = np.zeros(N)
        cdef double[:,:] XS = np.zeros([N, self.D])
        cdef int[:] extrap = np.zeros(N, dtype='int32')
        
        XS = self.scale_X(X, N, XS) 
        extrap = self.check_extrap(XS, N, extrap)
        
        for i in prange(N, nogil=True, num_threads=self.CORES, schedule=static):
            Y[i] = predict_prob(self.D, i, XS, extrap[i],  self.offset, self.flat, self.Linv, self.count, self.min_sample, self.SIZE, self.L, self.dohash, self.mem_max, self.key, self.countsuminv)
        
        return np.asarray(Y)

    def cross_validate(self, cores = 1):
        
        x1 = self.X[0:self.N / 2, :] 
        y1 = self.Y[0:self.N / 2]
    
        x2 = self.X[self.N / 2::, :]
        y2 = self.Y[self.N / 2::] 

        self.fit(x1, y1, cores = cores)
        R2_1 = self.score(x2, y2)
        self.fit(x2, y2, cores = cores)
        R2_2 = self.score(x1, y1)

        self.fit(self.X, self.Y)

        return (R2_1 + R2_2) / 2

    def opt(self, double[:,:] X, double[:] Al, double[:] Ah):
        
        """    
        Optimise the function over the first dimension (e.g., actions) for a subset of points.

        Subject to feasibility constraints Al <= A* <= Ah

        Parameters
        -----------
        X : array, shape=(N, D) 
            Input points to optimise over (unscaled)
        
        Al : array, shape=(N) 
            Action lower bound
        
        Ah : array, shape=(N) 
            Action upped bound
        
        Returns
        --------
    
        actions : array, shape=(<N,)
            Optimal actions  (e.g. argmax Q(a, s))

        values : array, shape=(<N)
            Optimal Q values, (e.g. max Q(a, s))
        
        points : array, shape(<N, D)
            Points actually used: some points may be ignored if count < min_sample
        
        index : array, shape(<N, D)
            index of points actually used
        """
        
        cdef int N = X.shape[0]
        cdef double v = 0
        cdef double A = 0
        cdef int i, h, j, k
        cdef double[:] v_opt = np.ones(N)*-10e100
        cdef double[:] A_opt = np.zeros(N)
        cdef int GRID = c_int((self.T[0] + 1) * self.L * 1.3)
        cdef double step =  (self.b[0] - self.a[0]) / GRID
        cdef double A_old
        cdef double xt = 0
        cdef double[:,:] XS = np.zeros([N, self.D])
        cdef double AS
        cdef int[:] extrap = np.zeros(N, dtype = 'int32')
        cdef int[:] somedata = np.zeros(N, dtype = 'int32')
        cdef int thisdata = 0
        cdef int idx, nosample
        cdef int temp = 0

        XS = self.scale_X(X, N, XS)
        extrap = self.check_extrap(XS, N, extrap)

        for i in prange(N, nogil=True, num_threads=self.CORES, schedule=static):
            if extrap[i] == 0:
                A = c_max(self.a[0], Al[i])
                while A <= Ah[i]:
                    thisdata = 0
                    # Scale A
                    AS = ((A - self.a[0]) * self.d[0]) * self.T[0]
                    XS[i, 0] = AS 
                    
                    if AS < -0.0001 or AS > self.T[0] + 0.0001:
                        v = 0
                    else:
                        # Predict y
                        #========================================================================
                        v = 0
                        """
                        v = predict(self.D, i, XS, 0, self.Tinv, self.offset, self.flat, self.Linv, self.w, self.count, self.min_sample, 
                        self.lin_spline, self.lin_w, self.lin_c, self.lin_T, self.SIZE, self.L, self.dohash, self.mem_max, self.key)
                        """
                        idx = 0
                        j = 0
                        temp = 0
                        for j in range(self.L): 
                        
                            idx = getindex(j, i, XS, self.D, self.offset, self.flat, self.SIZE, self.dohash, self.mem_max, self.key)
                           
                            if self.count[idx] >= self.min_sample:
                                v += self.w[idx]
                                temp = temp + 1
                            else:
                                v = 0
                                break
                        
                        v =  v * self.Linv
                        if temp == self.L:
                            somedata[i] = 1
                            thisdata = 1
                        #========================================================================
                    
                    if v > v_opt[i] and thisdata == 1:
                        v_opt[i] = v
                        A_opt[i] = A
                    
                    A_old = A
                    A = A + step
                    if A > Ah[i]:
                        A = Ah[i]
                    if A_old == Ah[i]:
                        break
            else:
                v_opt[i] = 0
       
        # return values
        index = np.array(somedata).astype(bool)
        #index = np.array(v_opt) > 0
        values = np.array(v_opt)[index]
        actions = np.array(A_opt)[index]
        state = np.array(X)[index, 1::]

        return [actions, values, state, index]

    def opt_test(self, double[:,:] X, double[:] Al, double[:] Ah):
        
        """    
        Optimise the function over the first dimension (e.g., actions) for a subset of points.

        Subject to feasibility constraints Al <= A* <= Ah

        Parameters
        -----------
        X : array, shape=(N, D) 
            Input points to optimise over (unscaled)
        
        Al : array, shape=(N) 
            Action lower bound
        
        Ah : array, shape=(N) 
            Action upped bound
        
        Returns
        --------
    
        actions : array, shape=(<N,)
            Optimal actions  (e.g. argmax Q(a, s))

        values : array, shape=(<N)
            Optimal Q values, (e.g. max Q(a, s))
        
        points : array, shape(<N, D)
            Points actually used: some points may be ignored if count < min_sample
        
        index : array, shape(<N, D)
            index of points actually used
        """
        
        cdef int N = X.shape[0]
        cdef double v = 0
        cdef double A = 0
        cdef int i, h, j, k
        cdef double[:] v_opt = np.ones(N)*-10e100
        cdef double[:] A_opt = np.zeros(N)
        cdef int GRID = c_int((self.T[0] + 1) * self.L * 1.3)
        cdef double step =  (self.b[0] - self.a[0]) / GRID
        cdef double A_old
        cdef double xt = 0
        cdef double[:,:] XS = np.zeros([N, self.D])
        cdef double AS
        cdef int[:] extrap = np.zeros(N, dtype = 'int32')
        cdef int[:] somedata = np.zeros(N, dtype = 'int32')
        cdef int idx, nosample
        cdef int temp = 0

        XS = self.scale_X(X, N, XS)
        extrap = self.check_extrap(XS, N, extrap)

        for i in range(N): 
            if extrap[i] == 0:
                print 'no extrap'
                A = c_max(self.a[0], Al[i])
                while A <= Ah[i]:
                    print 'A: ' + str(A)
                    
                    # Scale A
                    AS = ((A - self.a[0]) * self.d[0]) * self.T[0]
                    XS[i, 0] = AS 
                    
                    print 'AS: ' + str(AS)
                    
                    if AS < -0.0001 or AS > self.T[0] + 0.0001:
                        v = 0
                    else:
                        print 'predict y'
                        print 'somedata: ' + str(somedata[i])
                        # Predict y
                        #========================================================================
                        v = 0
                        """
                        v = predict(self.D, i, XS, 0, self.Tinv, self.offset, self.flat, self.Linv, self.w, self.count, self.min_sample, 
                        self.lin_spline, self.lin_w, self.lin_c, self.lin_T, self.SIZE, self.L, self.dohash, self.mem_max, self.key)
                        """
                        idx = 0
                        j = 0
                        temp = 0
                        for j in range(self.L): 
                        
                            idx = getindex(j, i, XS, self.D, self.offset, self.flat, self.SIZE, self.dohash, self.mem_max, self.key)
                           
                            if self.count[idx] >= self.min_sample:
                                v += self.w[idx]
                                temp = temp + 1
                            else:
                                v = 0
                                break
                        print 'v: ' + str(v)
                        print 'temp: ' + str(temp)
                        print 'L: ' + str(self.L)

                        print 'somedata: ' + str(somedata[i])
                        v =  v * self.Linv
                        if temp == self.L:
                            somedata[i] = 1
                            print 'somedata = 1'
                        
                        print 'somedata: ' + str(somedata[i])
                        #========================================================================
                    
                    if somedata[i] == 1:
                        if v > v_opt[i]:
                            v_opt[i] = v
                            A_opt[i] = A
                            print 'update optimal'

                    print 'v_opt: ' + str(v_opt[i])
                    print 'A_opt: ' + str(A_opt[i])

                    A_old = A
                    A = A + step
                    if A > Ah[i]:
                        A = Ah[i]
                    if A_old == Ah[i]:
                        break
            else:
                v_opt[i] = 0
       
        # return values
        index = np.array(somedata).astype(bool)
        #index = np.array(v_opt) > 0
        values = np.array(v_opt)[index]
        actions = np.array(A_opt)[index]
        state = np.array(X)[index, 1::]

        return [actions, values, state, index]
    
    def plot(self, xargs=0, showdata=True, label='', showplot=True, quad=False, returndata=False):

        """
        Plot the function on one dimension

        Parameters
        -----------
        xargs : list, length = D
            List of variable default values, set plotting dimension to 'x'

        showdata : boolean, (default=False)
            Scatter training points

        label : string, (default='')
            Series label

        showplot : boolean, (default='')
            > pylab.show()
        """

        cdef int k = 0
        

        if self.D == 1:
            Xsmooth = np.linspace(self.a[0], self.b[0], 300)
            Ysmooth = self.predict(Xsmooth)
            X = self.X
            Y = self.Y
            if showdata:
                pylab.plot(X, Y, 'o', Xsmooth, Ysmooth, label=label)
            else:
                pylab.plot(Xsmooth, Ysmooth, label=label)
        else:
            x = [xargs[i] for i in range(self.D)]
            k = x.index('x')
            Xsmooth = np.linspace(self.a[k], self.b[k]*1.1, 300)

            xpoints = np.ones([300, self.D])
            xpoints[:, k]  = 1
            x[k] = 1

            for i in range(self.D):
                xpoints[:, i] = xpoints[:, i] * x[i]
            xpoints[:, k] = Xsmooth
            Ysmooth = self.predict(xpoints)
            idx = Ysmooth != 0
            if showdata:
                pylab.plot(self.X[:, k], self.Y, 'o', Xsmooth[idx], Ysmooth[idx], label = label)
            else:
                pylab.plot(Xsmooth[idx], Ysmooth[idx], label = label)
            if returndata:
                return [Xsmooth[idx], Ysmooth[idx]]

    def plot_prob(self, xargs=0, showdata=True, label='', showplot=True, quad=False, returndata=False):

        """
        Plot the function on one dimension

        Parameters
        -----------
        xargs : list, length = D
            List of variable default values, set plotting dimension to 'x'

        showdata : boolean, (default=False)
            Scatter training points

        label : string, (default='')
            Series label

        showplot : boolean, (default='')
            > pylab.show()
        """

        cdef int k = 0
        
        x = [xargs[i] for i in range(self.D)]

        if self.D == 1:
            Xsmooth = np.linspace(self.a[0], self.b[0], 300)
            Ysmooth = self.predict_prob(Xsmooth)
            pylab.plot(Xsmooth, Ysmooth)
        else:
            k = x.index('x')
            Xsmooth = np.linspace(self.a[k], self.b[k]*1.1, 300)

            xpoints = np.ones([300, self.D])
            xpoints[:, k]  = 1
            x[k] = 1

            for i in range(self.D):
                xpoints[:, i] = xpoints[:, i] * x[i]
            xpoints[:, k] = Xsmooth
            Ysmooth = self.predict_prob(xpoints)
            idx = Ysmooth != 0
            pylab.plot(Xsmooth[idx], Ysmooth[idx], label = label)
    
    #==========================================================================
    # Convenience functions 
    #===========================================================================

    cdef double one_value(self, double[:] X):
        
        cdef int k
        cdef double xs
        cdef int extrap = 0
        cdef double[:,:] XS = self.xs
        cdef double val

        for k in range(self.D):
            xs = (X[k] - self.a[k]) * self.d[k] 
            if xs < -0.0001 or xs > 1.0001:
                extrap = 1
            XS[0, k] = xs * self.T[k]

        val = predict(self.D, 0, XS, extrap, self.Tinv, self.offset, self.flat, self.Linv, self.w, self.count, self.min_sample, self.lin_spline, self.lin_w, self.lin_c, self.lin_T, self.SIZE, self.L, self.dohash, self.mem_max, self.key)
   
        return val

    cpdef double one_value_pdf(self, double[:] X):
        
        cdef int k
        cdef double xs
        cdef int extrap = 0
        cdef double[:,:] XS = self.xs
        cdef double val

        for k in range(self.D):
            xs = (X[k] - self.a[k]) * self.d[k] 
            if xs < -0.0001 or xs > 1.0001:
                extrap = 1
            XS[0, k] = xs * self.T[k]

        val = predict_prob(self.D, 0, XS, extrap,  self.offset, self.flat, self.Linv, self.count, self.min_sample, self.SIZE, self.L, self.dohash, self.mem_max, self.key, self.countsuminv)
   
        return val

    cdef double[:] N_values(self, double[:,:] X, int N, double[:] values, double[:,:] XS):
        
        cdef int i, k
        
        XS = self.scale_X(X, N, XS)

        for i in prange(N, nogil=True, num_threads=self.CORES, schedule=static):
            values[i] = predict(self.D, i, XS, 0, self.Tinv, self.offset, self.flat, self.Linv, self.w, self.count, self.min_sample, self.lin_spline, self.lin_w, self.lin_c, self.lin_T, self.SIZE, self.L, self.dohash, self.mem_max, self.key)
    
        return values


    cdef double[:] N_values_policy(self, double[:,:] X, int N, double[:] values, int[:] extrap):
        
        cdef int i, k
        cdef double xs, xmax
        
        for i in range(N):
            for k in range(self.D):
                xs = ((X[i, k] - self.a[k]) * self.d[k]) * self.T[k]
                xmax = self.T[k] + 0.0001
                if xs < -0.0001 or xs > xmax:
                    extrap[i] = 1
                X[i, k] = xs

            values[i] = predict(self.D, i, X, extrap[i], self.Tinv, self.offset, self.flat, self.Linv, self.w, self.count, self.min_sample, self.lin_spline, self.lin_w, self.lin_c, self.lin_T, self.SIZE, self.L, self.dohash, self.mem_max, self.key)
    
        return values
    
    def fast_values(self, ):
        
        cdef int i
        
        for i in prange(self.XSN, nogil=True, num_threads=self.CORES, schedule=static):
            
            self.fastvalues[i] = predict_fast(self.D, i, self.XS, self.extrapXS[i], self.Tinv, self.Linv, self.w, self.count, self.min_sample, self.lin_spline, self.lin_w, self.lin_c, self.lin_T, self.L, self.datastruct_reverse)
    
        return np.array(self.fastvalues)

    
    cdef double[:,:] matrix_values(self,  double[:,:,:] X, int N1, int N2, double[:,:] values, double[:,:, :] XS, double[:,:] Xi):

        cdef int i, j, k
        cdef double xs = 0

        for i in prange(N1, nogil=True, num_threads=self.CORES):
            for j in range(N2):
                for k in range(self.D):
                    xs = (X[i,j,k] - self.a[k]) * self.d[k] 
                    XS[i,j,k] = xs * self.T[k]

        for j in prange(N2, nogil=True, num_threads=self.CORES):
            for i in range(N1): 
                for k in range(self.D):
                    Xi[j, k] = XS[i,j,k]
                values[i, j] = predict(self.D, j, Xi, 0, self.Tinv, self.offset, self.flat, self.Linv, self.w, self.count, self.min_sample, self.lin_spline, self.lin_w, self.lin_c, self.lin_T, self.SIZE, self.L, self.dohash, self.mem_max, self.key)

        return values

cdef class Function_Group:

    """ 
    Container for a group of Tilecode instances (user policy functions)
    
    Assumes all Tilecode instances use identical grids
    
    Assumes lin_spline == 1

    Assumes no hashing

        Parameters
        -----------
        N : int 
            Number of users
        
        N_low : array
            Number of low reliability users

        wlow: Tilecode instance
            Low user policy function

        whigh: Tilecode instance
            High user policy function

    """

    def __init__(self, N, N_low, Tilecode wlow, Tilecode whigh):
        
        self.N = N                                   # Number of instances (e.g., users)
        self.N_low = N_low

        # Assuming all these parameters are the same for each instance

        self.D = wlow.D                              # Number of dimensions
        self.L = wlow.L                              # Number of layers
        self.T = wlow.T
        self.flat = wlow.flat
        self.SIZE = wlow.SIZE
        self.offset = wlow.offset
        self.lin_SIZE = wlow.lin_SIZE
        self.lin_T = wlow.lin_T[0]
        self.min_sample = wlow.min_sample
        self.Tinv = wlow.Tinv
        self.Linv = wlow.Linv
        self.mem_max = wlow.mem_max
        self.xs_zeros = np.zeros(self.D)

        # These ones vary across users

        self.a = np.zeros([self.N, self.D]) 
        self.b = np.zeros([self.N, self.D])
        self.d = np.zeros([self.N, self.D])
        self.w = np.zeros([self.N, self.mem_max])
        self.count = np.zeros([self.N, self.mem_max], dtype='int32')
        self.key = np.zeros([self.N, self.mem_max], dtype='int32')
        self.lin_w = np.zeros([self.N, self.D, self.lin_T])
        self.lin_c = np.zeros([self.N, self.D, self.lin_T])

        cdef int i, k, z, j

        # Load parameters - Low users first
        for i in range(self.N_low):
            self.a[i,:] = wlow.a
            self.b[i,:] = wlow.b
            self.d[i,:] = wlow.d
            self.w[i, :] = wlow.w
            self.count[i, :] = wlow.count
            self.key[i, :] = wlow.key
            for k in range(self.D):
                self.lin_w[i, k, :] = wlow.lin_w[k, :]
                self.lin_c[i, k, :] = wlow.lin_c[k, :]
        
        for i in range(self.N_low, self.N):
            self.a[i,:] = whigh.a
            self.b[i,:] = whigh.b
            self.d[i,:] = whigh.d
            self.w[i, :] = whigh.w
            self.count[i, :] = whigh.count
            self.key[i, :] = whigh.key
            for k in range(self.D):
                self.lin_w[i, k, :] = whigh.lin_w[k, :]
                self.lin_c[i, k, :] = whigh.lin_c[k, :]
        
        self.w = np.ascontiguousarray(self.w)
        self.count = np.ascontiguousarray(self.count)
        self.key = np.ascontiguousarray(self.key)
        self.lin_w = np.ascontiguousarray(self.lin_w)
        self.lin_c = np.ascontiguousarray(self.lin_c)
        self.a = np.ascontiguousarray(self.a)
        self.b = np.ascontiguousarray(self.b)
        self.d = np.ascontiguousarray(self.d)
    
        self.XS = np.zeros([self.N, self.D])
        self.extrap = np.zeros([self.N])

    def update(self, int[:] Ilow, Tilecode wlow, int[:] Ihigh, Tilecode whigh):

        cdef int i, k, z, j, h

        # Load parameters - Low users first
        for i in range(self.N_low):
            self.a[i,:] = wlow.a
            self.b[i,:] = wlow.b
            self.d[i,:] = wlow.d
            self.w[i, :] = wlow.w
            self.count[i, :] = wlow.count
            self.key[i, :] = wlow.key
            for k in range(self.D):
                self.lin_w[i, k, :] = wlow.lin_w[k, :]
                self.lin_c[i, k, :] = wlow.lin_c[k, :]
        
        for i in range(self.N_low, self.N):
            self.a[i,:] = whigh.a
            self.b[i,:] = whigh.b
            self.d[i,:] = whigh.d
            self.w[i, :] = whigh.w
            self.count[i, :] = whigh.count
            self.key[i, :] = whigh.key
            for k in range(self.D):
                self.lin_w[i, k, :] = whigh.lin_w[k, :]
                self.lin_c[i, k, :] = whigh.lin_c[k, :]
        
        self.w = np.ascontiguousarray(self.w)
        self.count = np.ascontiguousarray(self.count)
        self.key = np.ascontiguousarray(self.key)
        self.lin_w = np.ascontiguousarray(self.lin_w)
        self.lin_c = np.ascontiguousarray(self.lin_c)
        self.a = np.ascontiguousarray(self.a)
        self.b = np.ascontiguousarray(self.b)
        self.d = np.ascontiguousarray(self.d)

    def predict(self, X):

        return np.array(self.get_values(X, np.zeros(self.N)))

    cdef void scale_X(self, double[:,:] X, int N):

        cdef int i, k
        cdef double xs

        for i in range(N):
            self.extrap[i] = 0
            for k in range(self.D):
                xs = (X[i, k] - self.a[i, k]) * self.d[i, k]
                if xs < -0.0001 or xs > 1.0001:
                    self.extrap[i] = 1
                self.XS[i, k] = xs * self.T[k]
    
    cpdef double[:] get_values(self, double[:,:] X, double[:] values):

        cdef int i, j, k
        cdef double y = 0
        cdef int idx = 0
        cdef double x = 0
        cdef double extrap = 0
        cdef double[:] xs = self.xs_zeros
        #cdef double[:] T = self.T
        #cdef double[:,:] w = self.w
        #cdef double[:,:] offset = self.offset
        #cdef double[:] flat = self.flat
        #cdef int SIZE = self.SIZE
        #cdef double[:,:] count = self.count
        #cdef double[:] a = self.a
        #cdef double[:] d = self.d
        #cdef int L = self.L
        #cdef double[:] Tinv = self.Tinv
        #cdef int lin_T = self.lin_T
        #cdef double[:,:,:] lin_w = self.lin_w[
        #cdef double[:,:,:] lin_c = self.lin_c
        #cdef double Linv = self.Linv
        #cdef int D = self.D

        cdef int T = self.T[0]

        for i in range(self.N):
            
            y = 0
            extrap = 0
            
            for k in range(self.D):
                x = ((X[i, k] - self.a[i, k]) * self.d[i, k]) 
                if x < -0.0001 or x > 1.0001:
                    extrap = 1
                xs[k] = x * T
            
            if extrap == 0:
                for j in range(self.L):
                    idx = 0
                    for k in range(self.D):
                        idx +=  c_int(xs[k] + self.offset[j, k]) * self.flat[k]
                    
                    idx += self.SIZE * j
                    
                    if self.count[i, idx] > 0:
                        y += self.w[i, idx]
                    else:
                    ## Linear value
                        idx = 0
                        for k in range(self.D):
                            x = xs[k] * self.Tinv[k]
                            idx = c_int(c_min(c_max(x, 0.000001), 0.99999) * self.lin_T)
                            y += (self.lin_w[i, k, idx] * x + self.lin_c[i, k, idx]) #*self.L
                            #break 
                            # self.w[i, idx] = y
                            # self.count[i, idx] = 1
                y =  y * self.Linv

            else:
                
                # Linear value
                idx = 0
                for k in range(self.D):
                    x = xs[k] * self.Tinv[k]
                    idx = c_int(c_min(c_max(x, 0.000001), 0.99999) * self.lin_T)
                    y += self.lin_w[i, k, idx] * x + self.lin_c[i, k, idx]
                
            # Apply feasibility constraint
            values[i] = c_min(c_max(y, 0), X[i, 1])
        
        return values

    def plot(self, xargs=0, showdata=True, label='', showplot=True):
        
        """
        Plot all the user functions on one dimension

        Parameters
        -----------
        xargs : list, length = D
            List of variable default values, set plotting dimension to 'x'

        showdata : boolean, (default=False)
            Scatter training points

        label : string, (default='')
            Series label

        showplot : boolean, (default='')
            > pylab.show()
        """


        cdef int i, k = 0
        
        points = 1000
        X = np.zeros([self.N, self.D])
        Xsmooth = np.zeros([self.N, points])
        xpoints = np.ones([self.N, points, self.D])
        Ysmooth = np.zeros([self.N, points])
        k = xargs.index('x')
        for i in range(self.N):
            Xsmooth[i,:] = np.linspace(self.a[i, k]*0.8, self.b[i, k]*1.2, points)
            xval = xargs
            xval[k] = 1
            for h in range(points):
                for j in range(self.D):
                    xpoints[i, h, j] = xpoints[i, h, j] * xval[j]
                xpoints[i, h, k] = Xsmooth[i,h]

        for h in range(points):
            X = xpoints[:, h, :]
            Y = self.predict(X)
            Ysmooth[:, h] = Y

        for i in range(self.N):
            pylab.plot(Xsmooth[i, :], Ysmooth[i, :], label = ("user " + str(i)))


def buildgrid(double[:,:] x, int M, double radius, scale = False, stopnum = 1000):

    """
    Generates an approximately equidistant subset of points. 
    
    Returns at most M points at least radius distance apart. 
    Ranks each point by the number of samples within radius
    If there are more than M grid points found, returns the highest scoring points.
    
    Has option for early stopping (by decreasing stopnum) 
    
    ============
    Parameters
    ============

    """

    import time
    tic = time.time()

    cdef int N = x.shape[0]
    cdef int D = x.shape[1]
    cdef int m = 1
    cdef double[:,:] xc = np.zeros([N, D])
    cdef double[:] xc_counter = np.zeros([N])
    cdef double r = 0
    cdef double r_min = 10
    cdef int i, j, k
    cdef double n = 0
    cdef double[:] a = np.zeros(D)
    cdef double[:] b = np.zeros(D)
    cdef double[:] d = np.ones(D)
    cdef double[:] d1 = np.ones(D)
    cdef double[:] xs = np.zeros(D)
    cdef int scale_1 = 0
    cdef double[:,:] grid
    cdef double cmax
    cdef int cmax_idx = 0
    cdef int j_star = 0

    cdef int stop_counter = 0
    cdef int stop_num = stopnum

    if scale: 
        atemp = np.min(x, axis=0)
        btemp = np.max(x, axis=0)
        dtemp = (btemp - atemp)**-1
        dtemp1 = (btemp - atemp)
        
        a = atemp
        b = btemp
        d = dtemp
        d1 = dtemp1

        scale_1 = 1
        
    for k in range(D):
        xc[0,k] = (x[0,k] - a[k]) * d[k]
    
    for i in range(1, N):
        for k in range(D):
            xs[k] = (x[i,k] - a[k]) * d[k]
        r_min = 10
        j_star = 0
        for j in range(m):
            r = 0
            for k in range(D):
                r += (xs[k] - xc[j, k])**2
            r **= 0.5
            if r < r_min:
                r_min = r
                j_star = j
        if r_min > radius:
            m += 1
            for k in range(D):
                xc[m - 1,k] = xs[k]
            stop_counter = 0
        else:
            xc_counter[j_star] += 1
            stop_counter += 1 
        if stop_counter > stop_num: #and m > M:
            print 'Stopped after: ' + str(i) + ' of ' + str(N) + ' points.'
            break
    
    M = <int> c_min(M, m)
    grid = np.zeros([M, D])

    for j in range(M): 
        for i in range(m):
            if xc_counter[i] > cmax:
                cmax = xc_counter[i]
                cmax_idx = i
        cmax = 0
        xc_counter[cmax_idx] = -1
        if scale_1 == 1: 
            for k in range(D):
                grid[j,k] = xc[cmax_idx, k] * d1[k] + a[k]
        else:
            for k in range(D):
                grid[j,k] = xc[cmax_idx, k]
    
    toc = time.time()
    print 'State grid points: ' + str(grid.shape[0]) + ', of maximum: ' + str(m) + ', Time taken: ' + str(toc - tic)
    
    return [np.asarray(grid), m]
