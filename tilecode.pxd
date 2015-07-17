cimport numpy as np
import numpy as np
cimport cython

from libc.math cimport fmin as c_min
from libc.math cimport fmax as c_max
from libc.math cimport log as c_log
from libc.math cimport exp as c_exp
from libc.math cimport sin as c_sin
from libc.math cimport cos as c_cos
from libc.stdlib cimport abs as c_abs

cdef extern from "fastonebigheader.h":
    double c_log_approx "fastlog" (double)

cdef extern from "vfastexp2.h":
    double c_exp "EXP" (double)

cdef extern from "fastonebigheader.h":
    double c_exp_approx "fastexp" (double)

cdef extern from "fast_floattoint.h" nogil:
    int c_int "Real2Int" (double)

cdef class Tilecode:

    cdef public double countsuminv
    
    cdef public int[:,:] neigh_idx
    cdef public int[:] neigh_count

    cdef public int N, D, L, n, XSN
    cdef public int[:] T, lin_T
    cdef public double Linv
    cdef public double[:] Tinv

    cdef public double[:] Y
    cdef public double[:,:] X
    cdef public double[:] a, b, d

    cdef public double[:, :] lin_w                            # Linear coefficients
    cdef public double[:, :] lin_c

    cdef public double[:] lin_width
    cdef public double[:, :] offset

    cdef public double[:] w
    cdef public double[:] wav
    cdef public int[:] count
    cdef public int[:] key
    cdef public int[:] hashidx
    cdef public int[:] datahash
    cdef public int[:] datastruct
    cdef public int[:,:] datastruct_reverse
    cdef public int max_neigh

    cdef public int SIZE
    cdef public int mem_max
    cdef public int dohash
    cdef public int lin_SIZE
    cdef public int[:] flat
    cdef int[:] tempidx
    cdef public double R2

    cdef double[:,:] XS_zero
    
    cdef public int[:] extrap
    cdef int[:] extrap_zero
    cdef int lin_spline
    cdef int min_sample
    cdef public int CORES
    cdef double[:,:] xs
    cdef public double[:,:] XS
    cdef public int[:] extrapXS
    cdef public double[:] fastvalues

    cdef public int asgd
    cdef public double eta
    cdef public double scale
    cdef public int sgd
    
    cdef double[:,:] scale_X(self, double[:,:] X, int N, double[:,:] XS)
    
    cdef int[:] check_extrap(self, double[:,:] XS, int N, int[:] extrap)
    
    cdef void fit_tiles(self, double[:,:] X, double[:] Y)
    
    cdef void fit_sgd(self, double[:,:] X, double[:] Y, double eta, double scale, int n_iters, int ASGD)
    
    cdef void fit_linear(self,  double[:,:] X, double[:] Y)
    
    cdef double[:,:] matrix_values(self,  double[:,:,:] X, int N1, int N2, double[:,:] values, double[:,:, :] XS, double[:,:] Xi)

    cdef double one_value(self, double[:] x)

    cdef double[:] N_values(self, double[:,:] X, int N, double[:] values, double[:,:] XS)
    
    cdef double[:] N_values_policy(self, double[:,:] X, int N, double[:] values, int[:] extrap)

    cdef void fit_data(self, double[:,:] X, int[:] countsofar)

    #cdef double[:] local_quadratic(self, double[:, :] X, int N)
    
    cdef void fit_data_reverse(self, double[:,:] X, int N, int[:] extrap)
    
    cpdef partial_fit(self, double[:] Y, int copy)
    
    cpdef double one_value_pdf(self, double[:] X)

cdef class Function_Group:

    cdef public int N, D, L, N_low         # Here N is number of users
    cdef double Linv
    cdef public int[:] T
    cdef public double[:] Tinv
    cdef public int lin_T

    cdef public double[:, :] a
    cdef public double[:, :] b
    cdef public double[:, :] d

    cdef public double[:, :, :] lin_w            # Linear coefficients
    cdef public double[:, :, :] lin_c

    cdef public double[:] lin_width
    cdef public double[:,:] offset
    cdef public int linval
    cdef public int min_sample
    cdef public int mem_max

    cdef public double[:,:] w
    cdef public int[:,:] count
    cdef public int[:,:] key

    cdef public int SIZE
    cdef public int lin_SIZE
    cdef public int[:] flat
    cdef double[:] xs_zeros

    cdef double[:,:] XS
    cdef int[:] keyi

    cdef public double[:] extrap

    cdef void scale_X(self, double[:,:] X, int N)

    cpdef double[:] get_values(self, double[:,:] x, double[:] values)

