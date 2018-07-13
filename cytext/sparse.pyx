# distutils: language=c++

from __future__ import print_function
import numpy  as np

from .typedefs cimport (int32, float32)
from .sparse cimport CSRMatrix, SparseRow


cdef class CSRMatrix:

    def __init__(self, X):
        """ Creates cython-compatible CSR matrix from scipy.sparse.csr_matrix X

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
        """
        cdef:
            int32 i, j, n, m, low, high

        X    = X.tocsr()
        n, m = X.shape
        
        self.data    = X.data.astype(np.float32)
        self.indices = X.indices.astype(np.int32)
        self.indptr  = X.indptr.astype(np.int32)
        self.shape   = np.array([n, m], dtype=np.int32)

        # precompute this for efficiency
        self._row_sums = np.zeros((n, ), dtype=np.float32)

        for i in range(n):
            for j in range(self.indptr[i], self.indptr[i + 1]):
                self._row_sums[i] += self.data[j]

    cdef float32 row_sum(CSRMatrix self, int32 idx) nogil:
        return self._row_sums[idx]

    cdef SparseRow row(CSRMatrix self, int32 idx) nogil:
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

        low, high  = self.indptr[idx], self.indptr[idx+1]
        indices    = self.indices[low:high]
        weights    = self.data[low:high]

        return SparseRow(indices, weights)
