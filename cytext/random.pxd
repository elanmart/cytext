# distutils: language=c++

from .typedefs cimport (int32, float32)


cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass mt19937:
        mt19937() except +
        mt19937(unsigned int32) except +

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(float32, float32)
        T operator()(mt19937)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution()
        uniform_int_distribution(int32, int32)
        T operator()(mt19937)


cdef class IntSampler:
    cdef:
        mt19937 engine
        uniform_int_distribution[int32] uniform

    cdef inline int32 sample(self) nogil
    cdef inline int32 sample_range(self, int32 low, int32 high) nogil


cdef class RealSampler:
    cdef:
        mt19937 engine
        uniform_real_distribution[float32] uniform

    cdef inline float32 sample(self) nogil
    cdef inline float32 sample_range(self, float32 low, float32 high) nogil
