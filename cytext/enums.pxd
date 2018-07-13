# distutils: language=c++

cdef enum APPROACH:
    sample,
    full

cdef enum LOSS:
    ns,
    warp,
    softmax

cdef enum MODEL:
    zsl,
    ft
