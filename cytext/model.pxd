# distutils: language=c++

from .typedefs cimport (int32, real, uint8, bool_t, FloatMatrix, FloatVector, IntVector, BoolVector)
from .random   cimport IntSampler
from .sparse   cimport CSRMatrix
from .random    cimport IntSampler
from .sparse    cimport CSRMatrix, SparseRow
from .tables    cimport ApproximationTables
from .model     cimport Worker
from .enums     cimport APPROACH, LOSS, MODEL
from .cblas     cimport (dot, scale, axpy, copy, zero, mul, sub, sum,
                         softmax, normalize, set_num_threads)


cdef class Worker:
    cdef:
        # rank
        int32 thread_id
        int32 n_threads

        # weights
        FloatMatrix d2h
        FloatMatrix l2h
        real d2h_norm
        real l2h_norm

        # sampling
        IntVector  indices
        IntVector  negatives
        IntSampler int_sampler

        # learning rate
        real lr
        real lr_0
        real lr_low

        # random numbers
        int32 base_seed

        # callbacks
        bool_t callback_available
        int32  callback_frequency
        object callback

        # running thread info
        object thread
        bool_t running

        # flags
        int32 approach
        int32 loss
        int32 model

        # internals
        real  loss_pos
        real  loss_neg
        int32 n_steps
        int32 idx_ptr
        int32 neg_ptr

        # buffers
        IntVector   prev_targets
        FloatVector doc_hidden
        FloatVector lab_hidden
        FloatVector grad
        FloatVector hidden_cache
        FloatVector output
        CSRMatrix   cached_D
        BoolVector  labels_set

        # misc
        int32 num_negatives
        int32 n_labels
        real  margin

        cdef void _c_train_nogil(Worker self, CSRMatrix X, CSRMatrix D, CSRMatrix Y) nogil
        cdef void _update(Worker self, SparseRow document, IntVector targets) nogil
        cdef void update_ns(Worker self, SparseRow document, IntVector targets) nogil
        cdef void update_warp(Worker self, SparseRow document, IntVector targets) nogil
        cdef void update_softmax(Worker self, SparseRow document, IntVector targets) nogil
        cdef real _binary_logistic(Worker self, int32 y, uint8 is_positive) nogil
        cdef void _fprop(Worker self, SparseRow features, FloatMatrix weights, FloatVector out) nogil
        cdef inline int32 _sample_negative(Worker self) nogil
        cdef inline int32 _next_idx(Worker self) nogil
        cdef void _zero_grad(Worker self) nogil
        cdef void _set_labels(Worker self, IntVector targets) nogil
        cdef void _step(Worker self, FloatMatrix weights, SparseRow features, FloatVector gradient, real alpha=?) nogil
        cdef inline FloatVector fprop_target(Worker self, int32 y) nogil
        cdef inline void bprop_target(Worker self, int32 y, FloatVector grad) nogil
