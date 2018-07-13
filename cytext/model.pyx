# distutils: language=c++

import threading

import  numpy as np
cimport numpy as np

from libc.math cimport ceil, fabs

from .typedefs  cimport (int64, int32, real, uint8, bool_t, IntVector, FloatVector, FloatMatrix)
from .random    cimport IntSampler
from .sparse    cimport CSRMatrix, SparseRow
from .tables    cimport ApproximationTables
from .model     cimport Worker
from .enums     cimport APPROACH, LOSS, MODEL
from .cblas     cimport (dot, scale, axpy, copy, zero, mul, sub, sum,
                         softmax, normalize, set_num_threads)


cdef class Worker:
    def __init__(Worker self,
                 thread_id, n_threads,
                 approach, loss, model,
                 d2h, l2h, dh2_norm, l2h_norm,
                 indices, negatives,
                 lr, lr_low,
                 num_negatives, margin,
                 callback_frequency, callback,
                 base_seed):

        set_num_threads(1)

        self.thread_id   = thread_id
        self.n_threads   = n_threads
        self.int_sampler = IntSampler(0, (2 ** 32 - 1))

        # shared arrays
        self.doc2hid          = d2h
        self.lab2hid          = l2h
        self.sampling_indices = negatives
        self.example_indices  = indices
        self.lr_0             = lr
        self.lr_low           = lr_low
        self.lr               = self.lr_0

        # per thread buffers
        self.prev_targets = np.zeros((0, ),            dtype=np.int32)
        self.labels_set   = np.zeros((0, ),            dtype=np.uint8)
        self.doc_hidden   = np.empty((d2h.shape[1], ), dtype=np.float32)
        self.lab_hidden   = np.empty((d2h.shape[1], ), dtype=np.float32)
        self.hidden_cache = np.empty((d2h.shape[1], ), dtype=np.float32)
        self.grad         = np.empty((d2h.shape[1], ), dtype=np.float32)
        self.output       = np.empty((0, ),            dtype=np.float32)
        self.cached_D     = None

        # reporting
        self.callback = callback
        self.callback_available = <bool_t> (callback is not None)
        self.callback_frequency = <int32>  (callback_frequency)

        # multithreading
        self.thread  = None   # type: threading.Thread
        self.running = False

        # internals
        self.n_steps  = 0
        self.loss_pos = 0.
        self.loss_neg = 0.
        self.idx_ptr  = <int32> ((thread_id / n_threads) * indices.shape[0])
        self.neg_ptr  = <int32> ((thread_id / n_threads) * negatives.shape[0])
        self.n_labels = 0

        # flags
        self.approach = {
            'sample': APPROACH.sample,
            'full':   APPROACH.full
        }[approach]

        self.loss = {
            'warp':    LOSS.warp,
            'ns':      LOSS.ns,
            'softmax': LOSS.softmax
        }[loss]

        self.model = {
            'zsl': MODEL.zsl,
            'ft':  MODEL.ft,
        }[model]

        # misc
        self.num_negatives = num_negatives
        self.margin        = margin
        self.d2h_norm      = dh2_norm
        self.l2h_norm      = l2h_norm

    def start(self, X, D, Y):
        """ """
        self.thread = threading.Thread(
            target = self._start,
            args   = (X, D, Y),
            daemon = True,
        )

        self.thread.start()
        self.running = True

        return self

    def wait(self):

        if self.running:
            self.thread.join()

        if (self.thread is not None) and self.thread.is_alive():
            raise RuntimeError('Thread did not exit :(')

        self.running = False
        self.cached_D = None
        return self

    @property
    def is_running(self):
        return self.running

    def _start(self, X, D, Y):
        cdef:
            CSRMatrix _X, _D, _Y

        _X = CSRMatrix(X)
        _D = CSRMatrix(D)
        _Y = CSRMatrix(Y)

        self.output     = np.empty((Y.shape[1], ), dtype=np.float32)
        self.labels_set = np.zeros((Y.shape[1], ), dtype=np.uint8)
        self.cached_D   = _D
        self.n_labels   = _D.shape[0]

        with nogil:
            self._c_train_nogil(_X, _D, _Y)


    cdef void _c_train_nogil(Worker self, CSRMatrix X, CSRMatrix D, CSRMatrix Y) nogil:
        """ """
        cdef:
            int64       step_idx, idx   # indexing vars
            real        progress
            SparseRow   document # currently processed document
            IntVector   targets  # positive labels for currently processed document

        # setup
        self.loss_pos = 0.
        self.loss_neg = 0.
        self.n_steps  = <int32> ceil(X.shape[0] / self.n_threads)

        for step_idx in range(self.n_steps):

            # fetch next example
            idx      = self._next_idx()
            document = X.row(idx)
            targets  = Y.row(idx).first

            # skip empty
            if (document.first.shape[0] == 0) or (targets.shape[0] == 0):
                continue

            # update weights
            self._update(document=document, targets=targets)

            # decay lr
            progress = (step_idx / (<real> self.n_steps))
            self.lr  = (self.lr_0 - (self.lr_0 - self.lr_low) * progress)

            # callback & printing
            if (self.thread_id == 0) and self.callback_available and (step_idx % self.callback_frequency == 0):
                with gil:

                    self.callback(dict(
                        iteration = step_idx,
                        loss_pos  = self.loss_pos / self.callback_frequency,
                        loss_neg  = self.loss_neg / self.callback_frequency,
                        lr        = self.lr
                    ))

                self.loss_pos = 0.
                self.loss_neg = 0.

    cdef void _update(Worker self, SparseRow document, IntVector targets) nogil:
        cdef:
            int32 y_idx, y_sz
            int32 num_documents

        # prepare buffers
        self._zero_grad()
        self._set_labels(targets)

        # sample only one target if neccessary
        if self.approach == APPROACH.sample:
            y_idx   = self.int_sampler.sample_range(0, targets.shape[0] - 1)
            targets = targets[y_idx:y_idx+1]

        # compute doc representation
        self._fprop(
            features = document,
            weights  = self.d2h,
            out      = self.doc_hidden
        )

        if self.loss == LOSS.ns:
            self.update_ns(document=document, targets=targets)

        elif self.loss == LOSS.warp:
            self.update_warp(document=document, targets=targets)

        elif self.loss == LOSS.softmax:
            self.update_softmax(document=document, targets=targets)

    cdef void update_ns(Worker self, SparseRow document, IntVector targets) nogil:
        cdef:
            int32 i
            real  loss_pos, loss_neg
            real  _p, _n,
            real  min_p, max_n
            real  doc_sum, denom
            real  BIG = 99999.

        loss_pos = 0.
        loss_neg = 0.
        min_p    = BIG
        max_n    = -BIG

        for i in range(targets.shape[0]):

            _p = self._binary_logistic(
                y           = targets[i],
                is_positive = 1
            )

            min_p     = min(_p, min_p)
            loss_pos += _p

        for i in range(self.num_negatives):

            _n = self._binary_logistic(
                y           = self._sample_negative(),
                is_positive = 0
            )

            max_n     = max(_n, max_n)
            loss_neg += _n

        # finalize gradient computation
        doc_sum = sum(document.second)
        denom   = 1. / doc_sum
        scale(self.grad, denom)

        # update document matrix
        self._step(
            weights  = self.d2h,
            features = document,
            gradient = self.grad
        )

        # keep track of the loss
        loss_pos /= targets.shape[0]
        loss_neg /= self.num_negatives

        self.loss_pos += (min_p - max_n)
        self.loss_neg += (loss_pos - loss_neg) / (1e-6 + (fabs(loss_pos) + fabs(loss_neg)) / 2)

    cdef void update_warp(Worker self, SparseRow document, IntVector targets) nogil:
        cdef:
            int32 worst_idx, best_idx
            int32 n_trials
            int32 idx, y
            real worst_score, best_score
            real doc_sum
            real score, L
            real BIG = 99999.
            FloatVector label_repr

        worst_idx   = 0
        best_idx    = 0
        worst_score = BIG
        best_score  = -BIG
        n_trials    = 0

        for idx in range(targets.shape[0]):

            y = targets[idx]

            self.fprop_target(y=y)
            score = dot(self.lab_hidden, self.doc_hidden)

            if score < worst_score:
                worst_score = score
                worst_idx   = y
                copy(
                    src  = self.lab_hidden,
                    dest = self.hidden_cache
                )

        while (n_trials < self.num_negatives) and (best_score <= (worst_score - self.margin)):

            y          = self._sample_negative()
            label_repr = self.fprop_target(y=y)
            score      = dot(label_repr, self.doc_hidden)

            if score > best_score:
                best_score  = score
                best_idx    = y

            n_trials += 1

        if best_score > (worst_score - self.margin):
            doc_sum = sum(document.second)

            L = ApproximationTables.warploss(
                n_labels  = self.n_labels,
                n_targets = targets.shape[0],
                n_trials  = n_trials,
            )

            scale(self.doc_hidden, alpha=-L)
            self.bprop_target(
                y    = worst_idx,
                grad = self.doc_hidden
            )

            scale(self.doc_hidden, alpha=-1)
            self.bprop_target(
                y    = best_idx,
                grad = self.doc_hidden
            )

            sub(v1=self.lab_hidden, v2=self.hidden_cache, dest=self.grad)
            scale(self.grad, alpha=(L / doc_sum))

            self._step(
                weights  = self.d2h,
                features = document,
                gradient = self.grad
            )

        self.loss_pos += n_trials
        self.loss_neg += (best_score - worst_score) / (1e-6 + (fabs(best_score) + fabs(worst_score)) / 2)

    cdef void update_softmax(Worker self, SparseRow document, IntVector targets) nogil:
        cdef:
            real alpha, doc_sum
            real score, score_true
            real is_positive
            int32 y_true, y_temp

        mul(matrix=self.l2h, vector=self.doc_hidden, out=self.output)
        softmax(self.output)

        # if we use softmax, targets is a vec of size 1
        y_true = targets[0]

        for y_temp in range(self.output.shape[0]):

            is_positive = <real> (y_true == y_temp)
            score       = self.output[y_temp]

            axpy(
                a = (score - is_positive),
                x = self.l2h[y_temp, :],
                y = self.grad
            )

            copy(
                src  = self.doc_hidden,
                dest = self.hidden_cache
            )

            scale(
                vec   = self.hidden_cache,
                alpha = (score - is_positive)
            )

            self.bprop_target(
                y    = y_temp,
                grad = self.hidden_cache
            )

        scale(
            vec   = self.grad,
            alpha = 1. / sum(document.second)
        )

        self._step(
            weights  = self.d2h,
            features = document,
            gradient = self.grad
        )

        score_true     = self.output[y_true]
        self.loss_pos += (-1 * ApproximationTables.log(score_true))

    cdef real _binary_logistic(Worker self, int32 y, uint8 is_positive) nogil:
        """ Given a label y and a binary indicator if its positive, make a step using binary_logistic loss

        This function ovewrwrites self.hidden_cache, self.lab_hidden
        This function updates self.lab2hid, self.grad

        Parameters
        ----------
        y : int32
            label
        is_positive : uint8
            if 1, label is relevant for currently examined document. Else negative.

        Returns
        -------
        loss: float
            loss suffered for this label.
        """
        cdef:
            real z, score, dLoss_dZ
            real loss = 0.

        # read this labels' description and fprop it into lab_hidden
        self.fprop_target(
            y = y
        )

        # compute score and derivative of the loss
        score = ApproximationTables.sigmoid(
            x = dot(self.doc_hidden, self.lab_hidden)
        )

        dLoss_dZ = score - (<real> is_positive)

        # update grad (g := g + x*a)
        axpy(
            a = dLoss_dZ,
            x = self.lab_hidden,
            y = self.grad
        )

        # compute gradient wrt to labels' description
        copy(
            src  = self.doc_hidden,
            dest = self.hidden_cache
        )

        scale(
            vec   = self.hidden_cache,
            alpha = dLoss_dZ
        )

        self.bprop_target(
            y    = y,
            grad = self.hidden_cache)

        # loss  = -log(score) if label else -log(1. - score)
        return score

    cdef void _fprop(Worker self, SparseRow features, FloatMatrix weights, FloatVector out) nogil:
        """ Forward propagation: average embeddings for each feature using matrix weights, store in destination

        Parameters
        ----------
        features : SparseRow
            features to use for computing emedding
        weights : np.ndarray[float, ndim=2]
            weight matrix to use
        out : np.ndarray[float, ndim=1]
            vector to store resulting embedding
        """
        cdef:
            int32 idx, size
            int32 word_idx
            real  word_cnt

            IntVector   indices
            FloatVector counts

            FloatVector row
            real        _sum

        # unpack features
        indices, counts = features.first, features.second
        _sum            = sum(counts)

        # prepare
        total_count = 0.
        size        = indices.shape[0]
        zero(out)

        # embed
        for i in range(size):

            word_idx = indices[i]
            word_cnt = counts[i]

            row = weights[word_idx, :]

            axpy(
                a = word_cnt,
                x = row,
                y = out
            )

        scale(
            vec   = out,
            alpha = 1. / _sum
        )

    cdef inline int32 _sample_negative(Worker self) nogil:
        """ Sample a single negative index. Will not return an index for label that is marked as 'positive'

        Returns
        -------
        idx : int32
            sampled index
        """

        cdef:
            int32 ret
            uint8 condition = 1

        while condition:
            ret       = self.negatives[self.neg_ptr]
            condition = self.labels_set[ret]

            self.neg_ptr += 1
            if self.neg_ptr >= self.negatives.shape[0]:
                self.neg_ptr = 0

        return ret

    cdef inline int32 _next_idx(Worker self) nogil:
        cdef:
            int32 idx

        idx = self.indices[self.idx_ptr]

        self.idx_ptr += 1
        if self.idx_ptr >= self.indices.shape[0]:
            self.idx_ptr = 0

        return idx

    cdef void _zero_grad(Worker self) nogil:
        zero(self.grad)

    cdef void _set_labels(Worker self, IntVector targets) nogil:
        """ Updates a set of positive labels with 'targets'. Previous set is zeroed-out
        Stores currently positive labels in self.prev_targets

        Parameters
        ----------
        targets : IntVector
            positive labels
        """
        cdef int32 idx, y

        # unset previous targets
        for idx in range(self.prev_targets.shape[0]):
            y = self.prev_targets[idx]
            self.labels_set[y] = 0

        # set current targets
        self.prev_targets = targets
        for idx in range(targets.shape[0]):
            y = targets[idx]
            self.labels_set[y] = 1

    cdef void _step(Worker self, FloatMatrix weights, SparseRow features, FloatVector gradient, real alpha=1.) nogil:
        """ Gradient descent step. Updates matrix 'W' at 'indices' using learning step 'alpha' and gradient 'grad'

        Parameters
        ----------
        weights : FloatMatrix
            weight matrix to update
        features : SparseRow
            features to update
        gradient : FloatVector
            gradient wrt to W
        alpha : float
            learning step
        """
        cdef:
            IntVector   indices
            FloatVector counts
            real        _sum

            int32 word_idx
            real  word_cnt

            int32 idx
            real  eta

        if self.lr == 0.:
            return

        indices, counts = features.first, features.second

        for idx in range(indices.shape[0]):

            word_idx = indices[idx]
            word_cnt = counts[idx]

            eta = (-1) * alpha * word_cnt * self.lr

            axpy(
                a = eta,
                x = gradient,
                y = weights[word_idx, :]
            )

            if self.d2h_norm > 0:
                normalize(
                    vec      = weights[word_idx, :],
                    max_norm = self.d2h_norm
                )

    cdef inline FloatVector fprop_target(Worker self, int32 y) nogil:

        cdef:
            SparseRow desc

        if self.model == MODEL.zsl:
            self._fprop(
                features = self.cached_D.row(y),
                weights  = self.l2h,
                out      = self.lab_hidden
            )

        else:
            copy(
                src  = self.l2h[y, :],
                dest = self.lab_hidden
            )

        return self.lab_hidden

    cdef inline void bprop_target(Worker self, int32 y, FloatVector grad) nogil:
        cdef:
            SparseRow desc
            real desc_sum

        if self.model == MODEL.zsl:
            desc     = self.cached_D.row(y)
            desc_sum = sum(desc.second)

            self._step(
                weights  = self.l2h,
                features = desc,
                gradient = grad,
                alpha    = 1./desc_sum
            )

        else:
            axpy(
                a = (-self.lr),
                x = grad,
                y = self.l2h[y, :]
            )

            if self.l2h_norm > 0:
                normalize(
                    vec      = self.l2h[y, :],
                    max_norm = self.l2h_norm
                )
