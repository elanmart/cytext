# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: embedsignature=True
# cython: language=3
# distutils: language=c++


from __future__ import print_function
import json
import multiprocessing
import os
import threading
import datetime as dt

from typedefs  cimport (int32, float32, uint8, sparse_row)
from cblas     cimport (dot, scale, axpy, copy, zero, unsafe_axpy, mul, sub,
                        softmax, normalize, constraint, unif, set_num_threads)
from random    cimport mt19937, uniform_int_distribution
from sparse    cimport FastCSR
from constans  cimport *  # TODO(elan)

# numpy-world
import scipy.sparse as sp
import  numpy as np
cimport numpy as np

# C / C++
from libc.math   cimport fabs, log as c_log, exp as c_exp
from libc.stdlib cimport rand as c_rand
from libc.time cimport time, time_t



cdef class OurModel:
    """ Implementation of our ZSL model.
    It's the interface visible to the user that manages initialization and starting Worker threads.

    Attributes
    ----------
    doc2hid: float32[:, ::1]
        embedding matrix for document words.
    lab2hid: float32[:, ::1]
        embedding matrix for label description words
    sampling_indices: int32[::1]
        sampling order of the negative examples. Initialized before training for efficiency
    example_indices: int32[::1]
        sampling order of the training examples. Initialized before training for efficiency
    """

    cdef:
        APPROACH approach
        LOSS     loss
        MODEL    model

        int32   max_trials
        uint8   always_update
        float32 margin
        int32   dim
        int32   neg
        int32   nb_epoch
        float32 lr_high
        float32 lr_low
        float32 proba_pow

        float32[:, ::1] doc2hid
        float32[:, ::1] lab2hid

        float32[:, ::1] label_representations

        int32[::1] sampling_indices
        int32[::1] example_indices

        np.uint64_t indices_table_size
        np.uint64_t neg_table_size

        int32   validation_freq
        float32 validation_loss_pos
        float32 validation_loss_neg

        int32   n_jobs
        int32   seed
        uint8 _loaded

        int32 cbk_freq
        time_t start
        time_t last
        object barrier
        object can_run
        object can_write

        object config
        object log_handler
        object story
        object str_end

        float32 lab2hid_norm
        float32 doc2hid_norm

        uint8 doc_normalize

    def __init__(self, model='our', loss='ns', approach='sample',
                 d2h_norm=0., l2h_norm=0.,
                 max_trials=1024, always_update=1, margin=0.1,
                 dim=100, neg=100,
                 nb_epoch=1, lr_high=0.1, lr_low=0.0001, proba_pow=0.5,
                 n_jobs=-1, seed=None, validation_freq=0, log_handler=None, cbk_freq=1000, story=None, str_end=''):
        """
        Parameters
        ----------
        approach: multilabel_approach
            if multilabel_approach.sample => for each example, sample only one positive label for which update is made
            if multilabel_approach.full   => update all positive labels
        dim: int32
            dimensionality of the hidden layer
        neg: int32
            number of negative samples in negative sampling
        nb_epoch: int32
            number of epochs to run
        n_jobs: int32
            number of threads to use
        seed: int32
            seed for numpy rng
        lr_high: float32
            starting learning rate
        lr_low: float32
            final learnign rate
        """

        assert model         in {'our', 'ft'}
        assert loss          in {'ns', 'warp', 'sft'}
        assert approach      in {'sample', 'full'}
        assert always_update in {0, 1}

        if n_jobs < 0:
            n_jobs += multiprocessing.cpu_count() + 1
        seed = seed or c_rand()
        loss_map     = {'ns':     LOSS.ns,         'warp': LOSS.warp, 'sft': LOSS.sft}
        model_map    = {'our':    MODEL.our,       'ft':   MODEL.ft}
        approach_map = {'sample': APPROACH.sample, 'full': APPROACH.full}

        # params
        loss     = loss_map[loss]
        model    = model_map[model]
        approach = approach_map[approach]

        self.loss     = loss
        self.model    = model
        self.approach = approach

        self.max_trials    = max_trials
        self.always_update = always_update
        self.margin        = margin

        self.dim       = dim
        self.neg       = neg
        self.nb_epoch  = nb_epoch
        self.lr_high   = lr_high
        self.lr_low    = lr_low
        self.proba_pow = proba_pow

        self.n_jobs          = n_jobs
        self.seed            = seed
        self.validation_freq = validation_freq
        self.n_jobs          = n_jobs

        # config
        self.config = {}
        for k, v in locals().items():
            if k != 'self':
                self.config[k] = v

        # sync threading
        self.barrier   = threading.Barrier(self.n_jobs)
        self.can_run   = threading.Event()
        self.can_write = threading.Event()
        self.can_run.set()
        self.can_write.clear()

        # misc
        self.log_handler           = log_handler
        self.label_representations = np.empty((0, dim), dtype=np.float32)
        self.sampling_indices      = np.empty((0, ),    dtype=np.int32)
        self.example_indices       = np.empty((0, ),    dtype=np.int32)
        self._loaded               = 0

        self.start    = 0
        self.last     = 0
        self.cbk_freq = cbk_freq
        self.story = story

        self.doc_normalize = 0
        self.doc2hid_norm = d2h_norm
        self.lab2hid_norm = l2h_norm
        self.str_end = str_end

    def set_log_handler(self, handler):
        self.log_handler = handler

    def callback(self, key, value):
        """ Report progress to the log handler or stdout """
        cdef:
            float rate, t_left, progress, elapsed, loss
            int i, n_iters, left
            int sec, th, tmin
            int32 since_last, since_start
            time_t tm
            float32 estimated_total, estimated_left

        if self.log_handler is not None:
            self.log_handler(key, value)

        else:
            i, n_iters, _, loss, _ = value
            tm = time(NULL)

            if self.start == 0:
                self.start = tm
                self.last  = tm
                return

            since_last  = tm - self.last
            since_start = tm - self.start
            self.last   = tm

            rate = <float32> self.cbk_freq / since_last
            left = n_iters - i
            t_left = (<float32> left / rate)

            progress = <float32> i / n_iters
            td = dt.datetime.now().strftime('%A, %H:%M:%S')

            sec     = <int32> t_left
            th      = <int32> t_left // 3600
            tmin    = <int32> (t_left // 60) % 60
            elapsed = since_start / 60

            estimated_total = <float32> elapsed / progress
            estimated_left  = estimated_total * (1 - progress)

            msg = "\r {} ::: loss: {:.3f} ::: {:.3f}%, elapsed: {:.1f} [min], total: {:.1f} [min], ETA: {:.1f} [min]    "
            msg = msg.format(td, loss, 100*progress, elapsed, estimated_total, estimated_left)

            if self.story is not None:
                self.story.append(loss)

            print(msg, end=self.str_end, flush=True)

    def _unsafe_save(OurModel self, directory):
        """ This will result in a deadlock if called on an instance that is not training """
        self.can_run.clear()
        self.can_write.wait()

        self.save(directory)

        self.can_run.set()

    def save(OurModel self, directory):
        os.makedirs(directory, exist_ok=True)

        d2h_p = os.path.join(directory, "doc2hid.npy")
        l2h_p = os.path.join(directory, "lab2hid.npy")
        rep_p = os.path.join(directory, "lab_rep.npy")
        cfg_p = os.path.join(directory, "config.json")

        with open(d2h_p, 'wb') as f: np.save(f, np.asarray(self.doc2hid))
        with open(l2h_p, 'wb') as f: np.save(f, np.asarray(self.lab2hid))
        with open(rep_p, 'wb') as f: np.save(f, np.asarray(self.label_representations))
        with open(cfg_p, 'w')  as f: json.dump(self.config, f)

    def load(OurModel self, directory, weights_only=True):
        d2h_p = os.path.join(directory, "doc2hid.npy")
        l2h_p = os.path.join(directory, "lab2hid.npy")
        rep_p = os.path.join(directory, "lab_rep.npy")
        cfg_p = os.path.join(directory, "config.json")

        with open(d2h_p, 'rb') as f: self.doc2hid = np.load(f)
        with open(l2h_p, 'rb') as f: self.lab2hid = np.load(f)

        if not weights_only:
            with open(cfg_p, 'r')  as f: self._unpack_confg(json.load(f))
            with open(rep_p, 'rb') as f: self.label_representations = np.load(f)

        self._loaded = 1

    def _unpack_confg(OurModel self, cfg):
        self.config          = cfg

        self.loss            = cfg['loss']
        self.approach        = cfg['approach']
        self.model           = cfg['model']
        self.max_trials      = cfg['max_trials']
        self.always_update   = cfg['always_update']
        self.margin          = cfg['margin']
        self.dim             = cfg['dim']
        self.neg             = cfg['neg']
        self.nb_epoch        = cfg['nb_epoch']
        self.lr_high         = cfg['lr_high']
        self.lr_low          = cfg['lr_low']
        self.proba_pow       = cfg['proba_pow']
        self.n_jobs          = cfg['n_jobs']
        self.seed            = cfg['seed']
        self.validation_freq = cfg['validation_freq']
        self.n_jobs          = cfg['n_jobs']

    def _get_probas(self, Y):
        """ Default probabilities initializer. Computes unigram distribution raised to 1/2 power

        Parameters
        ----------
        Y : sp.csr_matrix
            training label matrix

        Returns
        -------
        probabilities: float32[::1] of shape (Y.shape[1], )
            probabilities for sampling each of the labels.
        """
        probabilities = Y.sum(axis=0).A1.ravel()
        probabilities = np.power(probabilities, self.proba_pow)
        probabilities /= probabilities.sum()

        return probabilities

    def _initialize_tables(self, doc_dim, lab_dim, n_examples, n_examples_valid, n_labels, probabilities):
        """ Initialize approximation and sampling tables used by the model.

        Parameters
        ----------
        Y : sp.csr_matrix
        probabilities : float32[::1]
            sampling probabilities for negative labels

        Parameters
        ----------
        doc_dim : int
            number of embedding rows in document embedding matrix
        lab_dim : int
            number of embedding rows in label embedding matrix
        n_labels : int
            number of labels seen during training
        probabilities : float[::1] of shape (n_labels, )
            Probabilities according to which we will sample negative indices
        """

        # approx
        init_sigmoid()
        init_log()
        init_warp()
        np.random.seed(self.seed)

        # sampling tables
        self.indices_table_size = self.nb_epoch * (n_examples + n_examples_valid)
        self.neg_table_size     = MAX_NEG_TABLE_SZ

        self.indices_table_size = min(self.indices_table_size, MAX_NEG_TABLE_SZ)
        self.neg_table_size     = min(self.neg_table_size,     MAX_NEG_TABLE_SZ)

        self.sampling_indices = np.asarray(
            np.random.choice(
                np.arange(n_labels),
                size=self.neg_table_size,
                p=probabilities),
            dtype=np.int32, order='C')

        self.example_indices = np.asarray(
            np.random.choice(
                np.arange(n_examples),
                size=self.indices_table_size),
            dtype=np.int32, order='C')

        # weights
        if not self._loaded:
            self.doc2hid = np.random.randn(doc_dim, self.dim).astype(np.float32) / 100.
            self.lab2hid = np.random.randn(lab_dim, self.dim).astype(np.float32) / 100.

    def fit(self, X_doc, X_lab, Y, eval_set=None):
        """
        Parameters
        ----------
        X_doc : sp.csr_matrix
            Documents to train on
        X_lab : sp.csr_matrix
            Label representations
        Y : sp.csr_matrix
            Labels
        """
        # init
        probabilities    = self._get_probas(Y)
        n_examples       = Y.shape[0]
        n_examples_valid = 0 if eval_set is None else eval_set[1].shape[0]

        doc_dim = X_doc.shape[1]
        lab_dim = X_lab.shape[1] if (self.model == MODEL.our) else Y.shape[1]

        self._initialize_tables(doc_dim=doc_dim, lab_dim=lab_dim,
                                n_examples=n_examples, n_examples_valid=n_examples_valid, n_labels=Y.shape[1],
                                probabilities=probabilities)

        # move data to cython
        X_doc  = FastCSR(X_doc)
        X_lab  = FastCSR(X_lab)
        Y      = FastCSR(Y)

        # optional validation data
        if eval_set is not None:
            X_eval, Y_eval = eval_set
            X_eval = FastCSR(X_eval)
            Y_eval = FastCSR(Y_eval)
            eval_set = (X_eval, Y_eval)

        # start workers
        threads = []
        for i in range(self.n_jobs):
            print("Starting thread no: {}".format(i))
            t = threading.Thread(target=self._dispatch,
                                 args=(i, X_doc, X_lab, Y, eval_set),
                                 daemon=True)
            threads.append(t)
            t.start()

        # run until completiion
        for t in threads:
            t.join()

        print()

    def _dispatch(OurModel self, int32 thread_id, FastCSR X_doc, FastCSR X_lab, FastCSR Y, eval_set=None):
        """ Start a Worker thread

        Parameters
        ----------
        thread_id : int
            unique thread identifier
        X_doc : FastCSR
            documents
        X_lab : FastCSR
            label descriptions
        Y : FastCSR
            labels
        """
        cdef Worker w
        w = Worker(parent=self, thread_id=thread_id, base_seed=self.seed, n_labels=X_lab.shape[0])
        w.fit(X_doc, X_lab, Y, eval_set=eval_set)

    def compute_base_scores(OurModel self, X_train, relative=False):
        X_train = FastCSR(X_train)

        w = Worker(self)
        base_scores = w.compute_base_scores(X_train)

        return base_scores

    def compute_ranks(OurModel self, X_doc, Y, X_lab, filename, base_scores=None, do_normalize=False):
        cdef:
            int32 num_docs, num_labels

        num_docs, num_labels = X_doc.shape[0], X_lab.shape[0]

        X_doc   = FastCSR(X_doc)
        Y       = FastCSR(Y)

        with open(filename, 'w') as f:
            to_evaluate = set(range(num_labels))
            write_rankfile_header(to_evaluete=to_evaluate, fout=f)

            w = Worker(self, n_labels=X_lab.shape[0])
            w.compute_ranks(X_doc=X_doc, Y=Y, base_scores=base_scores, file=f)

    def np_softmax(OurModel, arr):
        maxes  = np.max(arr, axis=1)
        arr = np.exp(arr - maxes.reshape(-1, 1))
        arr = arr / np.sum(arr, axis=1).reshape(-1, 1)

        return arr

    def predict_raw(OurModel self, X_doc, Y):
        doc_repr = Worker(self).get_doc_repr(X_doc)

        doc_repr = np.asarray(doc_repr)
        lab2hid  = np.asarray(self.lab2hid)

        scores = doc_repr @ lab2hid.T
        scores = self.np_softmax(scores)

        return doc_repr, scores

    def set_doc_normalize(OurModel self, x):
        self.doc_normalize = <uint8> x

    def compute_label_representations(OurModel self, X_lab, do_normalize=False):
        X_lab = FastCSR(X_lab)

        w = Worker(self, n_labels=X_lab.py_shape[0])
        w.compute_label_representations(X_lab, do_normalize=do_normalize)


cdef class Worker:
    """ Implementation of our algorithm. Runs training loop, provides evaluation methods

    Attributes
    ----------
    thread_id : int32
        unique identifier
    parent : OurModel
        parent caller, see __init__ docstring
    X_lab : FastCSR
        label descriptions matrix
    doc2hid : float32[:, ::1]
        see OurModel.doc2hid. This is shared across multiple workers
    lab2hid : float32[:, ::1]
        see OurModel.lab2hid. This is shared across multiple workers
    sampling_indices : int32[::1]
        see OurModel.sampling_indices. This is shared across multiple workers
    example_indices : int32[::1]
        see OurModel.example_indices. This is shared across multiple workers
    locked : int32[::1]
        labels that are considered relevant for a currently processed example.
    doc_hidden : float32[::1] of shape (dim, )
        hidden representation of a document.
    lab_hidden : float32[::1] of shape (dim, )
        hidden representation of a label.
    grad : float32[::1] of shape (dim, )
        gradient vector
    hidden_cache : float32[::1] of shape (dim, )
        cached hidden representation. Preallocated for efficiency
    labels_set : uint8[::1]
        set of positive labels for a given example. if labels_set[i] == 1, then label i is relevant for given example
    idx_data : int32
        index into precomputed array on training data indices
    idx_neg : int32
        index into precomputed array on negative labels indices
    lr : float32
        current learning rate
    engine : mt19937
        random engine
    """

    cdef:
        int32 thread_id
        OurModel parent

        FastCSR X_lab

        FastCSR X_valid
        FastCSR Y_valid
        uint8 do_eval

        float32[:, ::1] doc2hid
        float32[:, ::1] lab2hid

        int32[::1] sampling_indices
        int32[::1] example_indices
        np.uint64_t indices_table_size
        np.uint64_t neg_table_size

        int32[::1]   locked
        float32[::1] doc_hidden
        float32[::1] lab_hidden
        float32[::1] grad
        float32[::1] hidden_cache
        float32[::1] output

        uint8[::1] labels_set

        np.uint64_t idx_data
        np.uint64_t idx_neg
        int32 n_labels

        float32 lr

        mt19937 engine

        # debug
        int32 cbk_freq
        int32 grad_step_idx
        float32 loss_pos
        float32 loss_neg

    def __init__(Worker self, OurModel parent, int32 thread_id=0, int32 base_seed=0,
                 int32 n_labels=int(2**24)):
        """
        Parameters
        ----------
        parent : OurModel
            Worker should only be called by a shared parent which takes care of initialization of shared arrays
        thread_id : int32
            unique worker identifier
        base_seed : int32
            seed for random number generator
        n_labels : int32
            number of labels to train on
        """
        set_num_threads(1)

        # set attrs
        self.parent    = parent
        self.thread_id = thread_id
        self.engine    = mt19937(base_seed + thread_id)

        # shared arrays
        self.doc2hid          = parent.doc2hid
        self.lab2hid          = parent.lab2hid
        self.sampling_indices = parent.sampling_indices
        self.example_indices  = parent.example_indices
        self.lr               = parent.lr_high

        self.locked       = np.zeros((1, ),               dtype=np.int32)
        self.labels_set   = np.zeros((n_labels, ),        dtype=np.uint8)
        self.doc_hidden   = np.empty((self.parent.dim, ), dtype=np.float32)
        self.lab_hidden   = np.empty((self.parent.dim, ), dtype=np.float32)
        self.hidden_cache = np.empty((self.parent.dim, ), dtype=np.float32)
        self.grad         = np.empty((self.parent.dim, ), dtype=np.float32)
        self.output       = np.empty((n_labels, ),        dtype=np.float32)

        # initialize indices
        self.idx_data = <np.uint64_t> ((self.thread_id / self.parent.n_jobs) * self.parent.indices_table_size)
        self.idx_neg  = <np.uint64_t> ((self.thread_id / self.parent.n_jobs) * self.parent.neg_table_size)
        self.indices_table_size = self.parent.indices_table_size
        self.neg_table_size     = self.parent.neg_table_size

        self.cbk_freq = self.parent.cbk_freq

    def fit(Worker self, FastCSR X_doc, FastCSR X_lab, FastCSR Y, eval_set=None):
        """ Fit the worker on (X_doc, Y) data, with X_lab descriptions.

        Parameters
        ----------
        X_doc : FastCSR
            documents
        X_lab : FastCSR
            descriptions
        Y : FastCSR
            labels
        """
        self.n_labels = X_lab.shape[0]
        self.X_lab    = X_lab

        # validation data
        self.X_valid, self.Y_valid = eval_set or (None, None)
        self.do_eval = self.X_valid is not None

        # debug
        self.grad_step_idx = 0
        self.loss_pos = 0
        self.loss_neg = 0

        with nogil:
            self._fit(X_doc, Y)

    cdef inline int32 _next_idx(Worker self) nogil:
        cdef:
            int32 idx

        idx = self.example_indices[self.idx_data]

        self.idx_data += 1
        if self.idx_data >= self.indices_table_size:
            self.idx_data = 0

        return idx

    cdef void _fit(Worker self, FastCSR X_doc, FastCSR Y) nogil:
        """ gil-free `fit()` method.

        Parameters
        ----------
        X_doc : FastCSR
            documents
        Y : FastCSR
            labels
        n_iters : int32
            number of iterations the worker should run
        """
        cdef:
            np.int64_t i, idx    # indexing vars
            sparse_row  document # currently processed document
            int32[::1]  targets  # positive labels for currently processed document
            np.int64_t n_iters, validation_freq

        # setup
        self.loss_pos   = self.loss_neg = 0.
        validation_freq = self.parent.validation_freq / self.parent.n_jobs
        n_iters         = (X_doc.shape[0] * self.parent.nb_epoch) / self.parent.n_jobs

        for i in range(n_iters):
            # read an example
            idx      = self._next_idx()
            document = X_doc.take_row(idx)
            targets  = Y.take_nnz(idx)

            if document[0].shape[0] == 0 or targets.shape[0] == 0:
                continue

            # update step
            self._update(document, targets)

            # bookkeeping
            self.lr = self.parent.lr_high - ((self.parent.lr_high - self.parent.lr_low) * (i / (<float32> n_iters)))

            # callback & printing
            if i % self.cbk_freq == 0:
                # self.callback()

                if self.thread_id == 0:
                    # report back
                    with gil:
                        key   = "training"
                        value = (i, n_iters, self.lr, self.loss_pos / self.cbk_freq, self.loss_neg / self.cbk_freq)
                        self.parent.callback(key, value)

                    # reset loss
                    self.loss_pos = self.loss_neg = 0.

            # evaluate
            if self.do_eval and ((i+1) % validation_freq) == 0:
                self._do_validation()

    cdef void callback(Worker self) nogil:
        with gil:
            if not self.parent.can_run.is_set():
                self.parent.barrier.wait()
                self.parent.can_write.set()
                self.parent.can_run.wait()
                self.parent.can_write.clear()

    cdef void _update(Worker self, sparse_row document, int32[::1] targets) nogil:
        """ Given an annotated document, perform a single learning step

        Parameters
        ----------
        document : sparse_row
            word indices & weights
        targets : int32[::1]
            relevant labels
        """
        cdef:
            int32 y_idx, y_sz
            int32 num_documents
            uniform_int_distribution[int32] uniform

        self._set_labels(targets)
        zero(self.grad)

        # sample only one target if neccessary
        if self.parent.approach == APPROACH.sample:
            y_sz    = targets.shape[0]
            uniform = uniform_int_distribution[int32](0, y_sz-1)
            y_idx   = uniform(self.engine)
            targets = targets[y_idx:y_idx+1]

        # compute doc representation
        self._fprop(document, self.doc2hid, self.doc_hidden)

        if   self.parent.loss == LOSS.ns:    self.update_ns(document, targets)
        elif self.parent.loss == LOSS.warp:  self.update_warp(document, targets)
        elif self.parent.loss == LOSS.sft:   self.update_softmax(document, targets)

    cdef void update_ns(Worker self, sparse_row document, int32[::1] targets) nogil:
        cdef:
            float32 loss_pos, loss_neg
            float32 _pos, _neg,
            float32 min_pos, max_neg
            int32 y_idx
            float32 doc_sum

        loss_pos = 0.
        loss_neg = 0.
        min_pos  = 99999.
        max_neg  = -99999.

        # compute loss & update label-desc embedding matrix
        for y_idx in range(targets.shape[0]):
            _pos     = self._binary_logistic(targets[y_idx], 1)
            min_pos  = min(_pos, min_pos)
            loss_pos += _pos

        for y_idx in range(self.parent.neg):
            _neg     = self._binary_logistic(self._sample_negative(), 0)
            max_neg  = max(_neg, max_neg)
            loss_neg += _neg

        # keep track of the loss
        loss_pos /= targets.shape[0]
        loss_neg /= self.parent.neg

        self.loss_pos += (min_pos - max_neg)
        self.loss_neg += (loss_pos - loss_neg) / (1e-6 + (fabs(loss_pos) + fabs(loss_neg)) / 2)

        # finalize gradient computation
        doc_sum = document[2]
        scale(self.grad, 1/doc_sum)

        # update document matrix
        self._step(W=self.doc2hid, where=document, grad=self.grad)

    cdef void update_warp(Worker self, sparse_row document, int32[::1] targets) nogil:
        cdef:
            int32      worst_idx,   best_idx
            float32    worst_score, best_score
            int32 idx, y
            int32 n_trials
            float32 doc_sum
            float32 score, L

        worst_idx   = best_idx = 0
        worst_score = 99999.
        best_score  = -99999.
        n_trials    = 0

        for idx in range(targets.shape[0]):
            y = targets[idx]

            self.fprop_target(y)
            score = dot(self.lab_hidden, self.doc_hidden)

            if score < worst_score:
                worst_score = score
                worst_idx   = y
                copy(self.lab_hidden, self.hidden_cache)

        while (n_trials < self.parent.max_trials) and (best_score <= (worst_score - self.parent.margin)):
            y = self._sample_negative()

            self.fprop_target(y)
            score = dot(self.lab_hidden, self.doc_hidden)

            if score > best_score:
                best_score  = score
                best_idx    = y

            n_trials += 1

        if best_score > (worst_score - self.parent.margin) or self.parent.always_update:
            L  = warploss(<int32> ((self.n_labels - targets.shape[0]) / max(n_trials, 1)))
            doc_sum = document[2]

            scale(self.doc_hidden, -L)
            self.bprop_target(y=worst_idx, grad=self.doc_hidden)

            scale(self.doc_hidden, -1)
            self.bprop_target(y=best_idx,  grad=self.doc_hidden)

            sub(v1=self.lab_hidden, v2=self.hidden_cache, dest=self.grad)
            scale(self.grad, L / doc_sum)
            self._step(W=self.doc2hid, where=document, grad=self.grad)

        self.loss_pos += n_trials
        self.loss_neg += (best_score - worst_score) / (1e-6 + (fabs(best_score) + fabs(worst_score)) / 2)

    cdef void update_softmax(Worker self, sparse_row document, int32[::1] targets) nogil:
        cdef:
            float32 alpha, doc_sum
            float32 label
            float32 score, score_true
            int32 y_true, y_temp

        y_true = targets[0]
        mul(M=self.lab2hid, x=self.doc_hidden, dest=self.output)
        softmax(self.output)

        for y_temp in range(self.output.shape[0]):
            score = self.output[y_temp]
            label = <float32> (y_true == y_temp)

            alpha = score - label
            axpy(x=self.lab2hid[y_temp, :], a=alpha, y=self.grad)

            copy(src=self.doc_hidden, dest=self.hidden_cache)
            scale(self.hidden_cache, alpha=alpha)
            self.bprop_target(y=y_temp, grad=self.hidden_cache)

        score_true = self.output[y_true]
        self.loss_pos += -log(score_true)

        doc_sum = document[2]
        scale(self.grad, 1/doc_sum)
        self._step(W=self.doc2hid, where=document, grad=self.grad)

    cdef float32 _binary_logistic(Worker self, int32 y, uint8 label) nogil:
        """ Given a label y and a binary indicator if its positive, make a step using binary_logistic loss

        This function ovewrwrites self.hidden_cache, self.lab_hidden
        This function updates self.lab2hid, self.grad

        Parameters
        ----------
        y : int32
            label
        label : uint8
            if 1, label is relevant for currently examined document. Else negative.

        Returns
        -------
        loss: float
            loss suffered for this label.
        """
        cdef:
            float32 z, score, dLoss_dZ
            float32 loss=0.

        # read this labels' description and fprop it into lab_hidden
        self.fprop_target(y)

        # compute score and derivative of the loss
        z = dot(self.doc_hidden, self.lab_hidden)
        score = sigmoid(z)
        dLoss_dZ = score - (<float32> label)

        # update grad (g := g + x*a)
        axpy(x=self.lab_hidden, a=dLoss_dZ, y=self.grad)

        # compute gradient wrt to labels' description
        copy(src=self.doc_hidden, dest=self.hidden_cache)
        scale(self.hidden_cache, alpha=dLoss_dZ)
        self.bprop_target(y=y, grad=self.hidden_cache)

        # loss  = -log(score) if label else -log(1. - score)
        return score

    cdef void _fprop(Worker self, sparse_row features, float32[:, ::1] Theta, float32[::1] destination) nogil:
        """ Forward propagation: average embeddings for each feature using matrix weights, store in destination

        Parameters
        ----------
        features : sparse_row
            features to use for computing emedding
        Theta : np.ndarray[float, ndim=2]
            weight matrix to use
        destination : np.ndarray[float, ndim=1]
            vector to store resulting embedding
        """
        cdef:
            int32 idx, size
            int32   word_idx
            float32 word_cnt

            int32[::1]   indices
            float32[::1] counts

            float32[::1] row
            float32      _sum

        # unpack features
        indices, counts, _sum = features

        # prepare
        total_count = 0.
        size = indices.shape[0]
        zero(destination)

        # embedd
        for i in range(size):
            word_idx = indices[i]
            word_cnt = counts[i]

            row = Theta[word_idx, :]
            axpy(row, word_cnt, destination)

        scale(destination, 1/_sum)

    cdef inline int32 _sample_negative(Worker self) nogil:
        """ Sample a single negative index. Will not return an index for label that is marked as 'positive'

        Returns
        -------
        idx : int32
            sampled index
        """
        cdef:
            int32 ret
            uint8 cond=1

        while cond:
            ret  = self.sampling_indices[self.idx_neg]
            cond = self.labels_set[ret]

            self.idx_neg += 1
            if self.idx_neg >= self.neg_table_size:
                self.idx_neg = 0

        return ret

    cdef void _set_labels(Worker self, int32[::1] targets) nogil:
        """ Updates a set of positive labels with 'targets'. Previous set is zeroed-out
        Stores currently positive labels in self.locked

        Parameters
        ----------
        targets : int32[::1]
            positive labels
        """
        cdef int32 idx, y

        # unset previous targets
        for idx in range(self.locked.shape[0]):
            y = self.locked[idx]
            self.labels_set[y] = 0

        # set current targets
        self.locked = targets
        for idx in range(targets.shape[0]):
            y = targets[idx]
            self.labels_set[y] = 1

    cdef void _step(Worker self, float32[:, ::1] W, sparse_row where, float32[::1] grad, float32 alpha=1.) nogil:
        """ Gradient descent step. Updates matrix 'W' at 'indices' using learning step 'alpha' and gradient 'grad'

        Parameters
        ----------
        W : float32[:, ::1]
            weight matrix to update
        where : sparse_row
            features to update
        grad : float32[::1]
            gradient wrt to W
        alpha : float
            learning step
        """
        cdef:
            int32[::1]   indices
            float32[::1] counts
            float32      _sum

            int32   word_idx
            float32 word_cnt

            int32   idx
            float32 eta

        if self.lr == 0.:
            return

        indices, counts, _sum = where

        for idx in range(indices.shape[0]):
            word_idx = indices[idx]
            word_cnt = counts[idx]

            eta = (-1) * alpha * word_cnt * self.lr
            axpy(x=grad, a=eta, y=W[word_idx, :])

            if self.parent.doc2hid_norm > 0:
                constraint(W[word_idx, :], self.parent.doc2hid_norm)

    cdef void _do_validation(Worker self) nogil:
        cdef:
            int32 n
            int32 i, start, stop
            sparse_row document
            int32[::1] targets
            float32 cached_lr, cached_loss_pos, cached_loss_neg

        # cache
        cached_loss_pos, self.loss_pos = self.loss_pos, 0.
        cached_loss_neg, self.loss_neg = self.loss_neg, 0.
        cached_lr, self.lr = self.lr, 0.

        # wait for all threads & zero-out the accumulated validation losses
        with gil:
            if self.thread_id == 0:
                self.parent.validation_loss_pos = self.parent.validation_loss_neg =  0.
            self.parent.barrier.wait()

        # setup counters
        n     = self.X_valid.shape[0] // self.parent.n_jobs
        start = self.thread_id * n
        stop  = min(start + n, self.X_valid.shape[0] - 1)

        # run validation loop
        for i in range(start, stop):
            document = self.X_valid.take_row(i)
            targets  = self.Y_valid.take_nnz(i)

            self._update(document, targets)

        # wait for all threads and merge results
        with gil:
            # merge results
            self.loss_pos /= (stop - start)
            self.loss_neg /= (stop - start)

            self.parent.validation_loss_pos += (self.loss_pos / self.parent.n_jobs)
            self.parent.validation_loss_neg += (self.loss_neg / self.parent.n_jobs)

            # wait for all threads and write results back to parent
            self.parent.barrier.wait()
            if self.thread_id == 0:
                loss_pos, loss_neg = self.parent.validation_loss_pos, self.parent.validation_loss_neg
                self.parent.callback("validation", (loss_pos, loss_neg))

        # load cache
        self.lr = cached_lr
        self.loss_pos = cached_loss_pos
        self.loss_neg = cached_loss_neg

    def compute_base_scores(Worker self, FastCSR X_train):
        cdef:
            int32 i, j, cnt
            float32[::1] scores
            float32[::1] base_scores
            sparse_row features

        n_docs      = X_train.shape[0]
        n_labels    = self.parent.label_representations.shape[0]

        base_scores = np.zeros((n_labels, ), dtype=np.float32, order='C')
        scores      = np.zeros((n_labels, ), dtype=np.float32, order='C')

        cnt = 0
        for i in range(n_docs):
            features = X_train.take_row(i)
            if features[0].shape[0] == 0:
                continue

            self._fprop(features=features, Theta=self.doc2hid, destination=self.doc_hidden)
            if self.parent.doc_normalize:
                normalize(self.doc_hidden)

            mul(M=self.parent.label_representations, x=self.doc_hidden, dest=scores)
            for j in range(scores.shape[0]):
                base_scores[j] += scores[j]

            cnt += 1

        for j in range(base_scores.shape[0]):
                base_scores[j] /= cnt

        return np.asarray(base_scores)

    cdef float32[::1] evaluate_example(Worker self, sparse_row features, int32[::1] targets, float32[::1] scores):
        self._fprop(features=features, Theta=self.doc2hid, destination=self.doc_hidden)
        if self.parent.doc_normalize:
            normalize(self.doc_hidden)

        mul(M=self.parent.label_representations, x=self.doc_hidden, dest=scores)

        return scores

    cdef int32[::1] compute_example_ranks(Worker self, int32[::1] targets, float32[::1] scores):
        cdef:
            int32 i, j, r
            int32 y
            uint8 dec
            float32 s
            int32[::1] ranks

        self._set_labels(targets)
        ranks = np.empty((targets.shape[0], ), dtype=np.int32)

        for i in range(targets.shape[0]):
            r = 0
            y = targets[i]
            s = scores[y]
            for j in range(scores.shape[0]):
                if scores[j] > s and self.labels_set[j] == 0:
                    r += 1
                if scores[j] == s and self.labels_set[j] == 0:
                    dec = <uint8>  (c_rand() > 0.5)
                    r += dec
            ranks[i] = r

        return ranks

    def compute_ranks(Worker self, FastCSR X_doc, FastCSR Y, float32[::1] base_scores, object file):
        cdef:
            int32 i, j
            int32 n_docs, n_labels

            sparse_row features
            int32[::1]   word_indices
            float32[::1] word_counts
            float32      _sum

            int32[::1] targets
            float32[::1] scores
            int32[::1] ranks

        n_docs, n_labels = Y.shape
        scores = np.empty((n_labels, ), dtype=np.float32, order='C')

        for i in range(n_docs):
            features = X_doc.take_row(i)
            targets  = Y.take_nnz(i)
            word_indices, word_counts, _sum = features

            if word_indices.shape[0] == 0 or targets.shape[0] == 0:
                continue

            scores = self.evaluate_example(features=features, targets=targets, scores=scores)
            if base_scores is not None:
                for j in range(scores.shape[0]): scores[j] -= base_scores[j]

            ranks = self.compute_example_ranks(targets=targets, scores=scores)
            write_rankfile_row(targets=targets, ranks=ranks, fout=file)

            if i % 100 == 0:
                print("\r {:.3f}%                 ".format((100. * i) / n_docs), end='')
        print()

    cdef _compute_label_repr_our(Worker self, FastCSR X_desc, uint8 do_normalize):
        cdef:
            float32[::1] repr_row
            sparse_row features
            int32 i

        for i in range(X_desc.shape[0]):
            features = X_desc.take_row(i)
            repr_row = self.parent.label_representations[i, :]

            self._fprop(features=features, Theta=self.lab2hid, destination=repr_row)

            if do_normalize:
                normalize(repr_row)

    def compute_label_representations(Worker self, FastCSR X_desc, uint8 do_normalize):
        if self.parent.model == MODEL.our:
            self.parent.label_representations = np.empty((X_desc.shape[0], self.parent.dim), dtype=np.float32)
            self._compute_label_repr_our(X_desc, do_normalize)
        else:
            self.parent.label_representations = self.parent.lab2hid

    cdef inline void fprop_target(Worker self, int32 y) nogil:
        cdef:
            sparse_row desc

        if self.parent.model == MODEL.our:
            desc = self.X_lab.take_row(y)
            self._fprop(features=desc, Theta=self.lab2hid, destination=self.lab_hidden)

        else:
            copy(self.lab2hid[y, :], self.lab_hidden)

    cdef inline void bprop_target(Worker self, int32 y, float32[::1] grad) nogil:
        cdef:
            sparse_row desc
            float32 desc_sum

        if self.parent.model == MODEL.our:
            desc    = self.X_lab.take_row(y)
            desc_sum = desc[2]
            self._step(W=self.lab2hid, where=desc, grad=grad, alpha=1./desc_sum)

        else:
            axpy(x=grad, a=(-self.lr), y=self.lab2hid[y, :])
            if self.parent.lab2hid_norm > 0:
                constraint(self.lab2hid[y, :], self.parent.lab2hid_norm)

    cdef float32[:, :] get_doc_repr(Worker self, X):
        cdef:
            FastCSR X_doc
            sparse_row document
            int32 idx, n

        n = X.shape[0]
        X_doc = FastCSR(X)
        ret = np.empty((n, self.parent.dim), dtype=np.float32)

        for idx in range(n):
            document = X_doc.take_row(idx)
            self._fprop(features=document, Theta=self.doc2hid, destination=ret[idx, :])

        return ret
