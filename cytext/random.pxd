# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True


cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass mt19937:
        mt19937() except +
        mt19937(unsigned int) except +

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(float, float)
        T operator()(mt19937)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution()
        uniform_int_distribution(int, int)
        T operator()(mt19937)


cdef class IntSampler:
    cdef:
        mt19937 engine
        uniform_int_distribution[int] uniform

    def __init__(self, int seed, float low, float high):
        self.engine  = mt19937(seed)
        self.uniform = uniform_int_distribution[int](low, high)

    cdef sample(IntSampler self):
        return self.uniform(self.engine)


cdef class RealSampler:
    cdef:
        mt19937 engine
        uniform_real_distribution[float] uniform

    def __init__(self, int seed, float low, float high):
        self.engine  = mt19937(seed)
        self.uniform = uniform_real_distribution[float](low, high)

    cdef sample(RealSampler self):
        return self.uniform(self.engine)
