# -*- coding: utf-8 -*-
"""
Created on 2020/11/30 22:18

@author: John_Fengz
"""

import numpy as np
from numba import jit
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.utils import check_random_state, _safe_indexing
from sklearn.base import BaseEstimator, ClassifierMixin


def _kernel(x1, x2, kernel='rbf'):
    """
    Calculate the kernel matrix of two datasets.

    Parameters
    ----------
    x1 : array-like (numpy.array).
    x2 : array-like (numpy.array).
    kernel : string (linear, rbf, poly)

    Returns
    -------
        Kernel matrix of two datasets
    """

    if kernel == 'linear':
        return linear_kernel(x1, x2)

    elif kernel == 'rbf':
        return rbf_kernel(x1, x2)

    elif kernel == 'poly':
        return polynomial_kernel(x1, x2)

    else:
        raise Exception('kernel must be one of linear, rbf, poly.')


def _get_min(y):
    y_set = set(y)
    count = {}
    for i in y_set:
        num = len(np.flatnonzero(y == i))
        count[i] = num
    idx_all = [x[0] for x in sorted(count.items(), key=lambda x: x[1], reverse=True)]
    idx = idx_all[1:]
    num = count[idx_all[0]]
    return num, idx


class KSIDO(BaseEstimator, ClassifierMixin):
    def __init__(self, p=0.1, k_neighbors=5, max_iter=None, random_state=42):
        self.p = p
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.max_iter = int(1 / p) if max_iter is None else max_iter
        self.svm = None
        self.X_train = None
        self.X_inf = None
        self.X_sel = None
        self.y_sel = None
        self.K_train = None
        self.D = None

    def _svm(self):
        return SVC(kernel='precomputed', probability=True, random_state=self.random_state)

    def _inf_sel(self, clf, X, y, bias=0):
        num_maj, min_idx = _get_min(y)
        random_state = check_random_state(self.random_state + bias)
        _X_inf = []
        _X_sel = []
        _y_sel = []
        for i in min_idx:
            class_indices = np.flatnonzero(y == i)
            X_class = _safe_indexing(X, class_indices)
            y_class = _safe_indexing(y, class_indices)
            n_samples = len(X_class)
            num_S_inf = int(self.p * (num_maj - n_samples))
            K_class_X = _kernel(X_class, self.X_train)
            K_class = None
            if self.X_inf is None:
                K_class = rbf_kernel(X_class, X)
            else:
                E = np.eye(len(self.X_sel), len(self.X_sel))
                D = np.diag(random_state.rand(len(self.X_sel)))
                K_class_sel = _kernel(X_class, self.X_inf) @ (E - D) + \
                              _kernel(X_class, self.X_sel) @ D
                K_class = np.hstack((K_class_X, K_class_sel))
            class_prob = clf.predict_proba(K_class)[:, i]
            X_inf_idx = np.argsort(class_prob)[:num_S_inf]

            # informative instances
            X_inf_ = _safe_indexing(X_class, X_inf_idx)
            K_inf_class = rbf_kernel(X_inf_, X_class)
            dis_mat = np.sqrt(2.0 * (1.0 - K_inf_class))
            random_state = check_random_state(random_state)
            sel_idx = [np.argsort(x)[1:(1 + self.k_neighbors)] for x in dis_mat]
            sel_idx = [random_state.choice(x, 1)[0] for x in sel_idx]
            X_sel_ = _safe_indexing(X_class, sel_idx)
            y_sel_ = _safe_indexing(y_class, sel_idx)

            _X_inf.append(X_inf_)
            _X_sel.append(X_sel_)
            _y_sel.append(y_sel_)

        _X_inf = np.vstack(_X_inf)
        _X_sel = np.vstack(_X_sel)
        _y_sel = np.hstack(_y_sel)
        return _X_inf, _X_sel, _y_sel

    def fit(self, X_train, y_train):
        random_state = check_random_state(self.random_state)
        self.X_train = X_train
        self.K_train = _kernel(X_train, X_train)
        X_infs = list()
        X_sels = list()
        y_sels = list()

        # train initial SVM
        self.svm = self._svm()
        self.svm.fit(self.K_train, y_train)

        for n in range(self.max_iter):
            # print('Iteration {}'.format(n))
            X_inf, X_sel, y_sel = self._inf_sel(self.svm, X_train, y_train, bias=0)
            X_infs.append(X_inf)
            X_sels.append(X_sel)
            y_sels.append(y_sel)
            self.X_inf = np.vstack(X_infs)
            self.X_sel = np.vstack(X_sels)
            self.y_sel = np.hstack(y_sels)
            # print(len(self.X_inf))
            E = np.eye(len(self.X_sel), len(self.X_sel))
            D = np.diag(random_state.rand(len(self.X_sel)))
            self.D = D
            K_train_syn = _kernel(X_train, self.X_inf) @ (E - D) + \
                          _kernel(X_train, self.X_sel) @ D
            """
            K_syn_sel = (E - D) @ _kernel(self.X_inf, self.X_inf) @ (E - D) + \
                        (E - D) @ _kernel(self.X_inf, self.X_sel) @ D + \
                        D @ _kernel(self.X_sel, self.X_inf) @ (E - D) + \
                        D @ _kernel(self.X_sel, self.X_sel) @ D
            """

            K_inf = _kernel(self.X_inf, self.X_inf)
            K_inf_sel = _kernel(self.X_inf, self.X_sel)
            K_sel = _kernel(self.X_sel, self.X_sel)
            K_syn = (E - D) @ K_inf @ (E - D) + \
                    (E - D) @ K_inf_sel @ D + \
                    D @ K_inf_sel.T @ (E - D) + \
                    D @ K_sel @ D
            K_train_new = np.vstack((np.hstack((self.K_train, K_train_syn)),
                                     np.hstack((K_train_syn.T, K_syn))))
            y_train_new = np.hstack((y_train, self.y_sel))
            self.svm.fit(K_train_new, y_train_new)
        return self

    def predict_proba(self, X_test):
        E = np.eye(len(self.X_sel), len(self.X_sel))
        D = self.D
        K_test = _kernel(X_test, self.X_train)
        K_test_sel = _kernel(X_test, self.X_inf) @ (E - D) + \
                     _kernel(X_test, self.X_sel) @ D
        K_test = np.hstack((K_test, K_test_sel))

        return self.svm.predict_proba(K_test)

    def predict(self, X_test):
        E = np.eye(len(self.X_sel), len(self.X_sel))
        D = self.D
        K_test = _kernel(X_test, self.X_train)
        K_test_sel = _kernel(X_test, self.X_inf) @ (E - D) + \
                      _kernel(X_test, self.X_sel) @ D
        K_test = np.hstack((K_test, K_test_sel))

        return self.svm.predict(K_test)
