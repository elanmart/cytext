# distutils: language=c++

from .typedefs cimport (int32, float32)
from libcpp.pair cimport pair

# ugly hack to get the equivalent of:
# ctypedef (int32[::1], float32[::1]) SparseRow
ctypedef int32[::1]   int32_c_vec
ctypedef float32[::1] float32_c_vec
ctypedef pair[int32_c_vec, float32_c_vec] SparseRow


# noinspection PyUnresolvedReferences
cdef class CSRMatrix:
    """ Cython-compatible sparse matrix in CSR format.

    Attributes
    ----------
    data      : float32[::1]
    indices   : int32[::1]
    indptr    : int32[::1]
    shape     : int32[::1]
    _row_sums : float32[::1]
    """

    cdef:
        float32[::1]   data
        int32[::1]     indices
        int32[::1]     indptr
        int32[::1]     shape
        float32[::1]   _row_sums

    cdef float32 row_sum(CSRMatrix self, int32 idx) nogil
    cdef SparseRow row(CSRMatrix self, int32 idx) nogil
