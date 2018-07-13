# distutils: language=c++

from .typedefs cimport (int32, float32)
from .random cimport (
    IntSampler, RealSampler,
    uniform_real_distribution, uniform_int_distribution, mt19937
)


cdef class IntSampler:
    def __cinit__(self, int32 seed, int32 low, int32 high):
        self.engine  = mt19937(seed)
        self.uniform = uniform_int_distribution[int32](low, high)

    cdef inline int32 sample(self) nogil:
        return self.uniform(self.engine)

    cdef inline int32 sample_range(self, int32 low, int32 high) nogil:
        cdef:
            uniform_int_distribution[int32] _uniform

        _uniform = uniform_int_distribution[int32](low, high)
        return _uniform(self.engine)


cdef class RealSampler:
    def __cinit__(self, int32 seed, float32 low, float32 high):
        self.engine  = mt19937(seed)
        self.uniform = uniform_real_distribution[float32](low, high)

    cdef inline float32 sample(self) nogil:
        return self.uniform(self.engine)

    cdef inline float32 sample_range(self, float32 low, float32 high) nogil:
        cdef:
            uniform_real_distribution[float32] _uniform

        _uniform = uniform_real_distribution[float32](low, high)
        return _uniform(self.engine)
