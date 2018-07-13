import multiprocessing
import warnings
from contextlib import contextmanager

import numpy as np

from .model import Worker


class CytextModel:

    def __init__(self,
                 model      = 'zsl',
                 loss       = 'ns',
                 approach   = 'sample',
                 d2h_norm   = 0.,
                 l2h_norm   = 0.,
                 max_trials = 1024,
                 margin     = 0.1,
                 dim        = 100,
                 neg        = 100,
                 nb_epoch   = 1,
                 lr_high    = 0.1,
                 lr_low     = 0.0001,
                 proba_pow  = 0.5,
                 n_jobs     = -1,
                 table_size = 10_000_000,
                 seed       = None):

        assert model    in {'zsl', 'ft'}
        assert loss     in {'ns', 'warp', 'softmax'}
        assert approach in {'sample', 'full'}

        if n_jobs < 0:
            n_jobs += multiprocessing.cpu_count() + 1

        self.model      = model
        self.loss       = loss
        self.approach   = approach
        self.d2h_norm   = d2h_norm
        self.l2h_norm   = l2h_norm
        self.max_trials = max_trials
        self.margin     = margin
        self.dim        = dim
        self.neg        = neg
        self.nb_epoch   = nb_epoch
        self.lr_high    = lr_high
        self.lr_low     = lr_low
        self.proba_pow  = proba_pow
        self.n_jobs     = n_jobs
        self.table_size = table_size
        self.seed       = seed

        self._callback = None
        self._workers  = []

        self._indices   = None
        self._negatives = None

        self.doc2hid_ = None
        self.lab2hid_ = None

        np.random.seed(self.seed)

    def fit(self, X, D, Y,
            validation_set=None, validation_mode='ranks',
            train_callback=None, validation_callback=None):

        try:
            for _ in range(self.nb_epoch):
                self.fit_one_epoch(X, D, Y, train_callback)

        except KeyboardInterrupt:
            warnings.warn('Keyboard Interrupt was caught, the model will stop training now.')

        # save some memory
        self._indices   = None
        self._negatives = None

    def fit_one_epoch(self, X, D, Y, callback=None):

        assert X.shape[0] == D.shape[0] == Y.shape[0]
        assert D.shape[0] == Y.shape[1]

        # initialize the weights
        self.doc2hid_ = self.doc2hid_ or self._init_weights(X.shape[1])
        self.lab2hid_ = self.lab2hid_ or self._init_weights(D.shape[1])

        # initialize tables with sampling indices
        self._reset_sampling_tables(n_examples=X.shape[0],
                                    n_labels=D.shape[0],
                                    Y=Y)

        # run workers
        with self._worker_pool() as pool:
            for worker in pool:
                worker.train(X, D, Y, callback)

    def predict_topk(self):
        pass

    def predict_scores(self):
        pass

    def predict_ranks(self):
        pass

    def _init_weights(self, n_words):
        return np.random.randn(n_words, self.dim).astype(np.float32) / 10

    def _reset_sampling_tables(self, n_examples, n_labels, Y):

        # compute probabilities for unigram sampling
        probabilities  = Y.tocsc().sum(axis=0).A1.ravel()
        probabilities  = np.power(probabilities, self.proba_pow)
        probabilities /= probabilities.sum()

        # indices for negative sampling
        self._negatives = np.asarray(
            np.random.choice(
                a    = np.arange(Y.shape[1]),
                size = min(self.max_trials * n_examples, self.table_size),
                p    = probabilities),
            dtype=np.int32,
            order='C')

        # example indices
        self._indices = np.arange(n_examples, dtype=np.int32)
        np.random.shuffle(self._indices)
        self._indices = np.ascontiguousarray(self._indices)

    @contextmanager
    def _worker_pool(self, n=None, clear=False):

        n = n or self.n_jobs

        self._workers = [
            Worker(thread_no, )
            for thread_no in range(n)
        ]

        yield self._workers

        self._workers = [
            _w.join()
            for _w in self._workers
            if _w.active
        ]

        if clear:
            self._workers.clear()
