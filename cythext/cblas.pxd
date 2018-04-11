# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True


from typedef import float32, int32, uint8

from libc.string cimport memset
from libc.math   cimport sqrt, exp as c_exp
from random      cimport mt19937, uniform_int_distribution


cdef extern from "cblas.h" nogil:
    enum CBLAS_ORDER: CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans

    inline float32 cblas_sdot(int N, float32  *x, int dx, float32  *y, int dy)
    inline void  cblas_sscal(int N, float32  alpha, float32  *x, int dx)
    inline void  cblas_saxpy(int N, float32  alpha, float32  *x, int dx, float32  *y, int dy)
    inline void  cblas_scopy(int N, float32  *x, int dx, float32  *y, int dy)
    inline void  cblas_sgemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                                 float  alpha, float  *A, int lda, float  *x, int incX,
                                 float  beta, float  *y, int incY)

    void set_num_threads "openblas_set_num_threads"(int num_threads)


# ---------------------------------------------
cdef inline float32 dot(float32[::1] x, float32[::1] y) nogil:
    return cblas_sdot(x.shape[0], &x[0], 1, &y[0], 1)


cdef inline float32 scale(float32[::1] vec, float32 alpha) nogil:
    cblas_sscal(vec.shape[0], alpha, &vec[0], 1)


cdef inline void axpy(float32[::1] x, float32 a, float32[::1] y) nogil:
    cblas_saxpy(x.shape[0], a, &x[0], 1, &y[0], 1)


cdef inline void copy(float32[::1] src, float32[::1] dest) nogil:
    cblas_scopy(src.shape[0], &src[0], 1, &dest[0], 1)


cdef inline void zero(float32[::1] x) nogil:
    memset(&x[0], 0, x.shape[0] * sizeof(float32))


cdef inline void unsafe_axpy(int N, float32* x, float32 a, float32* y) nogil:
    cblas_saxpy(N, a, x, 1, y, 1)


cdef inline void mul(float32[:,::1] M, float32[::1] x, float32[::1] dest) nogil:
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M.shape[0], M.shape[1], 1., &M[0,0], M.shape[1], &x[0], 1, 0., &dest[0], 1)


cdef inline void constraint(float32[::1] vec, float32 max_norm) nogil:
    cdef:
        int32 i
        float32 z
        float32 scaling_factor

    z = 0.
    for i in range(vec.shape[0]):
        z += vec[i] * vec[i]
    z = sqrt(z)

    if z > max_norm:
        scaling_factor = max_norm / z
        for i in range(vec.shape[0]):
            vec[i] *= scaling_factor


cdef inline void sub(float32[::1] v1, float32[::1] v2, float32[::1] dest) nogil:
    cdef:
        int32 i

    for i in range(v1.shape[0]):
        dest[i] = v1[i] - v2[i]


cdef inline void softmax(float32[::1] vec) nogil:
    cdef:
        float32 _max, z
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


cdef inline void normalize(float32[::1] vec) nogil:
    cdef:
        int32 i
        float32 _sum, val

    _sum = 0.
    for i in range(vec.shape[0]):
        val = vec[i]
        _sum += val * val

    if _sum == 0:
        return

    _sum = sqrt(_sum)
    for i in range(vec.shape[0]):
        vec[i] /= _sum


cdef inline int32 unif(int32 l, int32 h) nogil:
    cdef:
        mt19937 engine
        uniform_int_distribution[int32] uniform
        int32 r

    engine  = mt19937(42)
    uniform = uniform_int_distribution[int32](l, h)

    r = uniform(engine)

    return r
