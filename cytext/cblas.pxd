# distutils: language=c++

from .typedefs cimport real, int32, real, FloatVector, FloatMatrix

from libc.string cimport memset
from libc.math   cimport (sqrt, exp as c_exp)


cdef extern from "cblas.h" nogil:
    enum CBLAS_ORDER: CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans

    real cblas_sdot(int N, real  *x, int dx, real  *y, int dy)
    void  cblas_sscal(int N, real  alpha, real  *x, int dx)
    void  cblas_saxpy(int N, real  alpha, real  *x, int dx, real  *y, int dy)
    void  cblas_scopy(int N, real  *x, int dx, real  *y, int dy)
    void  cblas_sgemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                                 float  alpha, float  *A, int lda, float  *x, int incX,
                                 float  beta, float  *y, int incY)

    void set_num_threads "openblas_set_num_threads"(int num_threads)


# ---------------------------------------------
cdef inline real dot(FloatVector x, FloatVector y) nogil:
    return cblas_sdot(x.shape[0], &x[0], 1, &y[0], 1)


cdef inline real scale(FloatVector vec, real alpha) nogil:
    cblas_sscal(vec.shape[0], alpha, &vec[0], 1)


cdef inline void axpy(FloatVector x, real a, FloatVector y) nogil:
    cblas_saxpy(x.shape[0], a, &x[0], 1, &y[0], 1)


cdef inline void unsafe_axpy(int N, real* x, real a, real* y) nogil:
    cblas_saxpy(N, a, x, 1, y, 1)


cdef inline void copy(FloatVector src, FloatVector dest) nogil:
    cblas_scopy(src.shape[0], &src[0], 1, &dest[0], 1)


cdef inline void zero(FloatVector x) nogil:
    memset(&x[0], 0, x.shape[0] * sizeof(real))
    
cdef inline real sum(FloatVector x) nogil:
    cdef:
        int32 i
        real _s = 0.

    for i in range(x.shape[0]):
        _s += x[i]

    return _s

cdef inline void mul(FloatMatrix matrix, FloatVector vector, FloatVector out) nogil:
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        matrix.shape[0],
        matrix.shape[1],
        1.,
        &matrix[0, 0],
        matrix.shape[1],
        &vector[0],
        1,
        0.,
        &out[0],
        1
    )


cdef inline void normalize(FloatVector vec, real max_norm) nogil:
    cdef:
        int32 i
        real z
        real scaling_factor

    z = 0.
    for i in range(vec.shape[0]):
        z += vec[i] * vec[i]
    z = sqrt(z)

    if z > max_norm:
        scaling_factor = max_norm / z
        for i in range(vec.shape[0]):
            vec[i] *= scaling_factor


cdef inline void sub(FloatVector v1, FloatVector v2, FloatVector dest) nogil:
    cdef:
        int32 i

    for i in range(v1.shape[0]):
        dest[i] = v1[i] - v2[i]


cdef inline void softmax(FloatVector vec) nogil:
    cdef:
        real _max, z
        int32 i, n

    _max = vec[0]
    n    = vec.shape[0]
    z    = 0.

    for i in range(n):
        _max = max(_max, vec[i])

    for i in range(n):
        vec[i] = c_exp(vec[i] - _max)
        z += vec[i]

    for i in range(n):
        vec[i] /= z


cdef inline void unit_norm(FloatVector vec) nogil:
    cdef:
        int32 i
        real _sum, val

    _sum = 0.
    for i in range(vec.shape[0]):
        val = vec[i]
        _sum += val * val

    if _sum == 0:
        return

    _sum = sqrt(_sum)
    for i in range(vec.shape[0]):
        vec[i] /= _sum
