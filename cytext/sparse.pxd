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
cimport numpy as np
import numpy as np
from typedefs  cimport (int32, float32, sparse_row)


cdef class FastCSR:
    """ Cython-compatible sparse matrix in CSR format.

    Attributes
    ----------
    data:    float32[::1]
    indices: int32[::1]
    indptr:  int32[::1]
    shape:   Tuple[int, int]

    """
    cdef:
        float32[::1]   data
        int32[::1]     indices
        int32[::1]     indptr
        (int32, int32) shape
        float32[::1]   sums

    def __init__(self, X):
        """ Creates cython-compatible CSR matrix from scipy.sparse.csr_matrix X

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
        """
        cdef:
            int32 i, j, n, m, low, high

        n, m = X.shape
        self.shape   = <int32> n, <int32> m
        self.data    = X.data.astype(np.float32)
        self.indices = X.indices.astype(np.int32)
        self.indptr  = X.indptr.astype(np.int32)
        self.sums    = np.zeros((n, ), dtype=np.float32)

        for i in range(n):
            low, high = self.indptr[i], self.indptr[i+1]
            for j in range(low, high):
                self.sums[i] += self.data[j]

    cdef sparse_row take_row(FastCSR self, int32 idx) nogil:
        """ Returns row at index `idx` as a `sparse_row` type (Tuple[int32[::1], float32[::1])

        Parameters
        ----------
        idx : int
            index of the row to take

        Returns
        -------
        row : sparse_row
            a tuple (indices, weights), where indices are inds of nnz elements, and weights are their weights
        """
        cdef:
            int32 low, high
            int32[::1]   indices
            float32[::1] weights
            float32 _sum

        low, high  = self.indptr[idx], self.indptr[idx+1]
        indices    = self.indices[low:high]
        weights    = self.data[low:high]
        _sum       = self.sums[idx]

        return indices, weights, _sum

    cdef int32[::1] take_nnz(FastCSR self, int32 idx) nogil:
        """ Returns indices of nonzero elements in a row at index `idx`

        Parameters
        ----------
        idx : int
            Index of the row to take

        Returns
        -------
        indices: int32[::1]
            Nonzero indices for this row
        """
        cdef:
            int32 low, high
            int32[::1] indices

        low, high = self.indptr[idx], self.indptr[idx+1]
        indices   = self.indices[low:high]

        return indices
