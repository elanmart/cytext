# distutils: language=c++

from .typedefs  cimport (int32, float32)

cdef class ApproximationTablesClass:

    cdef:
        float32[::1] SIGMOID_TABLE
        float32[::1] LOG_TABLE
        float32[::1] WARP_TABLE

    cpdef void init_sigmoid(ApproximationTablesClass self)
    cpdef void init_log(ApproximationTablesClass self)
    cpdef void init_warp(ApproximationTablesClass self)
    cdef inline float32 sigmoid(ApproximationTablesClass self, float32 x) nogil
    cdef inline float32 log(ApproximationTablesClass self, float32 x) nogil
    cdef inline float32 warploss(ApproximationTablesClass self,
                                 int32 n_labels, int32 n_targets, int32 n_trials) nogil

cdef ApproximationTablesClass ApproximationTables
