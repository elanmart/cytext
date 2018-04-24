# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: embedsignature=True
# cython: language=3
# distutils: language=c++


from __future__ import print_function
from typedefs  cimport (int32, float32, uint8, sparse_row)

import  numpy as np
cimport numpy as np

from libc.math cimport fabs, log as c_log, exp as c_exp


cpdef enum APPROACH:
    sample,
    full

cpdef enum LOSS:
    ns,
    warp,
    sft

cpdef enum MODEL:
    our,
    ft

def str2enum(mapping, s):
    if isinstance(s, str):
        s = mapping[s]
    return s


DEF SIGMOID_TABLE_SIZE = 4096
DEF LOG_TABLE_SIZE     = 4096
DEF WARP_TABLE_SIZE    = 4096
DEF MAX_SIGMOID        = 8
DEF CALLBACK_FREQ      = 1000

cdef np.uint64_t MAX_NEG_TABLE_SZ   = <np.uint64_t> (2**25)
cdef np.uint64_t MAX_EXM_TABLE_SZ   = <np.uint64_t> (2**25)

cdef float32[::1] SIGMOID_TABLE
cdef float32[::1] LOG_TABLE
cdef float32[::1] WARP_TABLE


cpdef void init_sigmoid():
    """ Initialize the table for sigmoid approximation """
    cdef:
        int32 idx
        float32 x

    global SIGMOID_TABLE
    SIGMOID_TABLE = np.linspace(-MAX_SIGMOID, MAX_SIGMOID, num=SIGMOID_TABLE_SIZE+1, dtype=np.float32)

    for idx in range(SIGMOID_TABLE_SIZE + 1):
        x = SIGMOID_TABLE[idx]
        SIGMOID_TABLE[idx] = 1. / (1. + c_exp(-x))


cpdef void init_log():
    """ Initialize the table for logarithm approximation """
    cdef:
        int32 idx
        float32 x

    global LOG_TABLE
    LOG_TABLE = np.zeros((LOG_TABLE_SIZE+1, ), dtype=np.float32, order='C')

    for idx in range(LOG_TABLE_SIZE + 1):
        x = (<float32> idx + <float32> 1e-5) / LOG_TABLE_SIZE
        LOG_TABLE[idx] = c_log(x)


cpdef void init_warp():
    """ Initialize the table for warp loss """
    cdef:
        int32 idx

    global WARP_TABLE
    WARP_TABLE = np.zeros((WARP_TABLE_SIZE+1, ), dtype=np.float32, order='C')

    for idx in range(1, WARP_TABLE_SIZE+1):
        WARP_TABLE[idx] = (1. / idx) + WARP_TABLE[idx-1]


cdef inline float32 sigmoid(float32 x) nogil:
    """ Compute sigmoid function for scalar `x` """

    cdef:
        int32 i

    if x < -MAX_SIGMOID:
      return 0.0

    elif x > MAX_SIGMOID:
      return 1.0

    else:
      i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2)
      return SIGMOID_TABLE[i]


cdef inline float32 log(float32 x) nogil:
  cdef:
      int32 i

  if x > 1.0:
    return 0.0

  i = <int32> (x * LOG_TABLE_SIZE)
  return LOG_TABLE[i]


cdef inline float32 warploss(int32 idx) nogil:
    idx = min(idx, WARP_TABLE_SIZE-1)

    return WARP_TABLE[idx]