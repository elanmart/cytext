import  numpy as np
cimport numpy as np
np.import_array()

ctypedef (int32[::1], float32[::1], float32) sparse_row
ctypedef np.uint8_t   uint8
ctypedef np.int32_t   int32
ctypedef np.float32_t float32
