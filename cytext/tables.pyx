# distutils: language=c++

import  numpy as np
cimport numpy as np

from libc.math cimport (
    log as c_log,
    exp as c_exp
)

from .typedefs  cimport (int32, float32, real)
from .tables cimport ApproximationTablesClass, ApproximationTables


DEF SIGMOID_TABLE_SIZE = 16384
DEF LOG_TABLE_SIZE     = 16384
DEF WARP_TABLE_SIZE    = 16384
DEF MAX_SIGMOID        = 8


# noinspection PyAttributeOutsideInit
cdef class ApproximationTablesClass:

    def __init__(self):
        self.init_log()
        self.init_sigmoid()
        self.init_warp()

    def foobar(self):
        return 'foobar'

    cdef inline float32 sigmoid(ApproximationTablesClass self, float32 x) nogil:
        """ Compute sigmoid function for scalar `x` """

        cdef:
            int32 i

        if x < -MAX_SIGMOID:
          return 0.0

        elif x > MAX_SIGMOID:
          return 1.0

        else:
          i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2)
          return self.SIGMOID_TABLE[i]

    cdef inline float32 log(ApproximationTablesClass self, float32 x) nogil:
        cdef:
            int32 i

        if x > 1.0:
            return 0.0

        i = <int32> (x * LOG_TABLE_SIZE)
        return self.LOG_TABLE[i]

    cdef inline float32 warploss(ApproximationTablesClass self,
                                 int32 n_labels, int32 n_targets, int32 n_trials) nogil:
        cdef:
            int32 idx
            real  div

        div = <real> max(n_trials, 1)
        idx = <int32> ((n_labels - n_targets) / div)
        idx = min(idx, WARP_TABLE_SIZE-1)

        return self.WARP_TABLE[idx]

    cpdef void init_sigmoid(ApproximationTablesClass self):
        """ Initialize the table for sigmoid approximation """
        cdef:
            int32   idx
            float32 x

        self.SIGMOID_TABLE = np.linspace(-MAX_SIGMOID, MAX_SIGMOID, num=SIGMOID_TABLE_SIZE+1, dtype=np.float32)

        for idx in range(SIGMOID_TABLE_SIZE + 1):
            x = self.SIGMOID_TABLE[idx]
            self.SIGMOID_TABLE[idx] = 1. / (1. + c_exp(-x))

    cpdef void init_log(ApproximationTablesClass self):
        """ Initialize the table for logarithm approximation """
        cdef:
            int32   idx
            float32 x

        self.LOG_TABLE = np.zeros((LOG_TABLE_SIZE + 1, ), dtype=np.float32, order='C')

        for idx in range(LOG_TABLE_SIZE + 1):
            x = (<float32> idx + <float32> 1e-5) / LOG_TABLE_SIZE
            self.LOG_TABLE[idx] = c_log(x)

    cpdef void init_warp(ApproximationTablesClass self):
        """ Initialize the table for warp loss """
        cdef:
            int32 idx

        self.WARP_TABLE = np.zeros((WARP_TABLE_SIZE+1, ), dtype=np.float32, order='C')

        for idx in range(1, WARP_TABLE_SIZE+1):
            self.WARP_TABLE[idx] = (1. / idx) + self.WARP_TABLE[idx-1]


ApproximationTables = ApproximationTablesClass()
