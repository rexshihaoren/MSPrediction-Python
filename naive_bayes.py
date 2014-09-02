# -*- coding: utf-8 -*-

"""
The :mod:`sklearn.naive_bayes` module implements Naive Bayes algorithms. These
are supervised learning methods based on applying Bayes' theorem with strong
(naive) feature independence assumptions.
"""

# Author: Vincent Michel <vincent.michel@inria.fr>
#         Minor fixes by Fabian Pedregosa
#         Amit Aides <amitibo@tx.technion.ac.il>
#         Yehuda Finkelstein <yehudaf@tx.technion.ac.il>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
#         (parts based on earlier work by Mathieu Blondel)
#
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import issparse
import warnings
from scipy.misc import factorial
from scipy.stats import chisquare, itemfreq
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.utils import array2d, atleast2d_or_csr, column_or_1d, check_arrays
from sklearn.utils.extmath import safe_sparse_dot, logsumexp
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.externals import six

__all__ = ['BernoulliNB', 'GaussianNB','GaussianNB2','MultinomialNB', 'PoissonNB', 'MixNB','MixNB2']


class BaseNB(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    """Abstract base class for naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X):
        """Compute the unnormalized posterior log probability of X

        I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of
        shape [n_classes, n_samples].

        Input is passed to _joint_log_likelihood as-is by predict,
        predict_proba and predict_log_proba.
        """

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        probas = np.exp(self.predict_log_proba(X))
        rowsum = np.sum(probas, axis = 1)
        # if np.array_equal(rowsum, np.ones(rowsum.shape[0])):
        #     print "rowsum are 1"
        # else:
        #     print "rowsums are't 1"
        return probas / rowsum.reshape(rowsum.shape[0],1)


class GaussianNB(BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB)

    Attributes
    ----------
    `class_prior_` : array, shape = [n_classes]
        probability of each class.

    `theta_` : array, shape = [n_classes, n_features]
        mean of each feature per class

    `sigma_` : array, shape = [n_classes, n_features]
        variance of each feature per class

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    def fit(self, X, y):
        """Fit Gaussian Naive Bayes according to X, y

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = check_arrays(X, y, sparse_format='dense')
        y = column_or_1d(y, warn=True)

        n_samples, n_features = X.shape

        self.classes_ = unique_y = np.unique(y)
        n_classes = unique_y.shape[0]

        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        epsilon = 1e-9
        for i, y_i in enumerate(unique_y):
            Xi = X[y == y_i, :]
            self.theta_[i, :] = np.mean(Xi, axis=0)
            self.sigma_[i, :] = np.var(Xi, axis=0) + epsilon
            self.class_prior_[i] = np.float(Xi.shape[0]) / n_samples
        return self

    def _joint_log_likelihood(self, X):
        X = array2d(X)
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood



class GaussianNB2(BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB)

    Attributes
    ----------
    `class_prior_` : array, shape = [n_classes]
        probability of each class.

    `theta_` : array, shape = [n_classes, n_features]
        mean of each feature per class

    `sigma_` : array, shape = [n_classes, n_features]
        variance of each feature per class

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    def fit(self, X, y):
        """Fit Gaussian Naive Bayes according to X, y

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """

        #X, y = check_arrays(X, y, sparse_format='dense')
        y = column_or_1d(y, warn=True)

        n_samples, n_features = X.shape

        self.classes_ = unique_y = np.unique(y)
        n_classes = unique_y.shape[0]

        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        epsilon = 1e-9
        for i, y_i in enumerate(unique_y):
            Xi = X[y == y_i, :]
            # nanmean, nanvar both ignore na
            self.theta_[i, :] = np.nanmean(Xi, axis=0)
            self.sigma_[i, :] = np.nanvar(Xi, axis=0) + epsilon
            self.class_prior_[i] = np.float(Xi.shape[0]) / n_samples
        return self

    def _joint_log_likelihood(self, X):
        # Here sigma is Var not Std
        norm_func = lambda x, sigma, theta: 1 if np.isnan(x) else -0.5 * np.log(2 * np.pi*sigma) - 0.5 * ((x - theta)**2/sigma) 
        # norm_func = lambda x, sigma, theta: -np.log(sigma)- 0.5 * np.log(2 * np.pi) - 0.5 * ((x - theta)/sigma) ** 2
        norm_func = np.vectorize(norm_func)
        #X = array2d(X)
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = np.sum(norm_func(X, self.sigma_[i,:], self.theta_[i,:]), axis = 1)
            # n_ij = - 0.5 * np.sum(np.log(2 * np.pi)) - np.sum(np.log(self.sigma_[i, :]))
            # n_ij -= 0.5 * np.sum(((X - self.theta_[i, :])/
            #                      self.sigma_[i, :]) ** 2)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

class BaseDiscreteNB(BaseNB):
    """Abstract base class for naive Bayes on discrete/categorical data

    Any estimator based on this class should provide:

    __init__
    _joint_log_likelihood(X) as per BaseNB
    """

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of"
                                 " classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = (np.log(self.class_count_)
                                     - np.log(self.class_count_.sum()))
        else:
            self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible
        (as long as fitting in the memory budget) to hide the overhead.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        classes : array-like, shape = [n_classes]
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns self.
        """
        X = atleast2d_or_csr(X, dtype=np.float64)
        _, n_features = X.shape

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_effective_classes = len(classes) if len(classes) > 1 else 2
            self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
            self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                           dtype=np.float64)

        Y = label_binarize(y, classes=self.classes_)
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        n_samples, n_classes = Y.shape

        if X.shape[0] != Y.shape[0]:
            msg = "X.shape[0]=%d and y.shape[0]=%d are incompatible."
            raise ValueError(msg % (X.shape[0], y.shape[0]))

        # convert to float to support sample weight consistently
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            Y *= array2d(sample_weight).T

        # Count raw events from data before updating the class log prior
        # and feature log probas
        self._count(X, Y)

        # XXX: OPTIM: we could introduce a public finalization method to
        # be called by the user explicitly just once after several consecutive
        # calls to partial_fit and prior any call to predict[_[log_]proba]
        # to avoid computing the smooth log probas at each call to partial fit
        self._update_feature_log_prob()
        self._update_class_log_prior()
        return self

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_arrays(X, y, sparse_format='csr')
        X = X.astype(np.float)
        y = column_or_1d(y, warn=True)
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # convert to float to support sample weight consistently
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            Y *= array2d(sample_weight).T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                       dtype=np.float64)
        self._count(X, Y)
        self._update_feature_log_prob()
        self._update_class_log_prior(class_prior=class_prior)
        return self

    # XXX The following is a stopgap measure; we need to set the dimensions
    # of class_log_prior_ and feature_log_prob_ correctly.
    def _get_coef(self):
        return (self.feature_log_prob_[1:]
                if len(self.classes_) == 2 else self.feature_log_prob_)

    def _get_intercept(self):
        return (self.class_log_prior_[1:]
                if len(self.classes_) == 2 else self.class_log_prior_)

    coef_ = property(_get_coef)
    intercept_ = property(_get_intercept)







class MultinomialNB(BaseDiscreteNB):
    """
    Naive Bayes classifier for multinomial models

    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    fit_prior : boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like, size (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    Attributes
    ----------
    `class_log_prior_` : array, shape (n_classes, )
        Smoothed empirical log probability for each class.

    `intercept_` : property
        Mirrors ``class_log_prior_`` for interpreting MultinomialNB
        as a linear model.

    `feature_log_prob_`: array, shape (n_classes, n_features)
        Empirical log probability of features
        given a class, ``P(x_i|y)``.

    `coef_` : property
        Mirrors ``feature_log_prob_`` for interpreting MultinomialNB
        as a linear model.

    `class_count_` : array, shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    `feature_count_` : array, shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB()
    >>> clf.fit(X, y)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    >>> print(clf.predict(X[2]))
    [3]

    Notes
    -----
    For the rationale behind the names `coef_` and `intercept_`, i.e.
    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
    Tackling the poor assumptions of naive Bayes text classifiers, ICML.

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    http://nlp.stanford.edu/IR-book/html/htmledition/
        naive-bayes-text-classification-1.html
    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = (np.log(smoothed_fc)
                                  - np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        X = atleast2d_or_csr(X)
        return (safe_sparse_dot(X, self.feature_log_prob_.T)
                + self.class_log_prior_)


class BernoulliNB(BaseDiscreteNB):
    """Naive Bayes classifier for multivariate Bernoulli models.

    Like MultinomialNB, this classifier is suitable for discrete data. The
    difference is that while MultinomialNB works with occurrence counts,
    BernoulliNB is designed for binary/boolean features.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    binarize : float or None, optional
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.

    fit_prior : boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like, size=[n_classes,]
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    Attributes
    ----------
    `class_log_prior_` : array, shape = [n_classes]
        Log probability of each class (smoothed).

    `feature_log_prob_` : array, shape = [n_classes, n_features]
        Empirical log probability of features given a class, P(x_i|y).

    `class_count_` : array, shape = [n_classes]
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    `feature_count_` : array, shape = [n_classes, n_features]
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(2, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 4, 5])
    >>> from sklearn.naive_bayes import BernoulliNB
    >>> clf = BernoulliNB()
    >>> clf.fit(X, Y)
    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    >>> print(clf.predict(X[2]))
    [3]

    References
    ----------

    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

    A. McCallum and K. Nigam (1998). A comparison of event models for naive
    Bayes text classification. Proc. AAAI/ICML-98 Workshop on Learning for
    Text Categorization, pp. 41-48.

    V. Metsis, I. Androutsopoulos and G. Paliouras (2006). Spam filtering with
    naive Bayes -- Which naive Bayes? 3rd Conf. on Email and Anti-Spam (CEAS).
    """

    def __init__(self, alpha=1.0, binarize=.0, fit_prior=True,
                 class_prior=None):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self):
        """Apply smoothing to raw counts and recompute log probabilities"""
        n_classes = len(self.classes_)
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = self.class_count_ + self.alpha * n_classes

        self.feature_log_prob_ = (np.log(smoothed_fc)
                                  - np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""

        X = atleast2d_or_csr(X)

        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)

        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError("Expected input with %d features, got %d instead"
                             % (n_features, n_features_X))

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        return jll

class PoissonNB(BaseNB):
    """
    Poisson Naive Bayes (PoissonNB)

    Attributes
    ----------
    `class_prior_` : array, shape = [n_classes]
        probability of each class.

    `lamb_` : array, shape = [n_classes, n_features]
        mean of each feature per class
    `model_` : array, shape = [n_classes, ]
        model class

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import PossionNB
    >>> clf = PoissonNB()
    >>> clf.fit(X, Y)
    PossionNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    def fit(self, X, y):
        """Fit Poisson Naive Bayes according to X, y

        Parameters
        ----------
        X : structured array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features. Each coloumn can
            be accessed by column name.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = check_arrays(X, y, sparse_format='dense')
        y = column_or_1d(y, warn=True)

        n_samples, n_features = X.shape

        self.classes_ = unique_y = np.unique(y)
        n_classes = unique_y.shape[0]

        self.lamb_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        epsilon = 1e-9
        for i, y_i in enumerate(unique_y):
            Xi = X[y == y_i, :]
            L, R = Xi.shape
            for j in range(0, R):
                # Estimate lambda
                lamb = np.mean(Xi[:,j])
                self.lamb_[i, j] = lamb
            self.class_prior_[i] = np.float(Xi.shape[0]) / n_samples
        return self

    def _joint_log_likelihood(self, X):
        #X = array2d(X)
        joint_log_likelihood = []
        func = lambda x, lamb: x* np.log(lamb)- lamb - np.log(factorial(x))
        vecfunc = np.vectorize(func)
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = np.sum(vecfunc(X, self.lamb_[i,:]), axis = 1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

class MixNB(BaseNB):
    """
    Mix Naive Bayes (MixNB)
    Different features having different distributions

    Attributes
    ----------
    `class_prior_` : array, shape = [n_classes]
        probability of each class.

    `lamb_` : array, shape = [n_classes, n_features]
        mean of each feature per class
    `model_` : array, shape = [n_classes, ]
        model class

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import PossionNB
    >>> clf = PoissonNB()
    >>> clf.fit(X, Y)
    PossionNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """
    # norm_func = lambda x, sigma, theta: 1 if np.isnan(x) else -np.log(sigma)- 0.5 * np.log(2 * np.pi) - 0.5 * ((x - theta)/sigma) ** 2
    norm_func = lambda x, sigma, theta: 1 if np.isnan(x) else -0.5 * np.log(2 * np.pi*sigma) - 0.5 * ((x - theta)**2/sigma) 
    # if x is na
    # return 1
    # else 
    # return -np.log(sigma)- 0.5 * np.log(2 * np.pi) - 0.5 * ((x - theta)/sigma) ** 2
    norm_func = np.vectorize(norm_func)
    pois_func = lambda x, lamb: 1 if np.isnan(x) else x* np.log(lamb)- lamb - np.log(factorial(x))
    pois_func = np.vectorize(pois_func)

    def __init__(self, models = ['norm', 'poisson'], funcs = {'norm': norm_func,'poisson': pois_func}):
        self.models = models
        self.funcs = funcs

    def fit(self, X, y):
        """Fit Mix Naive Bayes according to X, y

        Parameters
        ----------
        X : structured array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features. Each coloumn can
            be accessed by column name.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """

        #X, y = check_arrays(X, y, sparse_format='dense')
        y = column_or_1d(y, warn=True)

        n_samples, n_features = X.shape
        self.n_samples = n_samples
        self.features_ = n_features
        self.classes_ = unique_y = np.unique(y)
        n_classes = unique_y.shape[0]
        self.class_prior_ = np.zeros(n_classes)
        epsilon = 1e-9
        self.distnames = np.array(range(n_features), dtype = object)
        self.optmodels = np.zeros((n_classes, n_features), dtype = object)
        num_samples = X.shape[0]
        for j in range(0, n_features):
            # Feature column
            fcol = X[:,j]
            if sum(fcol>=0) != num_samples:
                distname = 'norm'
            else:
                distname, _ = self._max_fit(fcol, y)
            self.distnames[j] = distname
            for i, y_i in enumerate(unique_y):
                fcoli = fcol[y == y_i]
                self.optmodels[i, j] = self._fit_model(fcoli, distname)
                # This step seems redundant but I don't see any better place to put this
                self.class_prior_[i] = np.float(fcoli.shape[0]) / n_samples
        return self

    def _max_fit(self, fcol, y):
        """Determine the best fit for one feature column

        Parameters
        ----------
        fcol: feature column


        Returns
        ----------

        distname: optimal distribution name
        optmodel: optimal distribution function with feature as argument

        NOTE:

        Works for Discrete Fit

        """
        goodness = {}
        funcs = {}
        for dis in self.models:
            func = self._fit_model(fcol,dis)
            funcs[dis] = func
            # Use Chi-square to measure goodness
            goodness[dis] = self._get_goodness(func, fcol)
        distname = min(goodness, key=goodness.get)
        optmodel = funcs[distname]
        return distname, optmodel


    def _get_goodness(self, func, fcol):
        """Calculate the goodness with Pearson's chi-squared test

        Parameters
        ----------
        func: fit function
        fcol: feature column


        Returns
        ----------

        goodness

        """
        itfreq = itemfreq(fcol)
        uniqueVars = itfreq[:,0]
        freq = itfreq[:,1]
        #freq = freq/sum(freq)
        freq = freq/self.n_samples
        predFreq = np.exp(func(uniqueVars))
        # predFreq = predFreq/sum(predFreq)
        goodness = chisquare(predFreq,freq)[0]
        return goodness


    def _fit_model(self, fcol, dis):

        """Determine the best fit for one feature column given distribution name

        Parameters
        ----------
        fcol: feature column, array
        dis: distribution name, String


        Returns
        ----------
        function: fit model with feature as argument

        """
        if dis == 'poisson':
            lamb = np.nanmean(fcol, axis = 0)
            func = lambda x: self.funcs[dis](x, lamb)
        if dis ==   'norm':
            sigma = np.nanvar(fcol, axis=0)
            theta = np.nanmean(fcol, axis = 0)
            func = lambda x: self.funcs[dis](x, sigma, theta)
        return np.vectorize(func)

    def _joint_log_likelihood(self, X):
        #X = array2d(X)
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = np.sum([self.optmodels[i, :][j](X[:,j]) for j in range(self.features_)],axis = 0)
            # n_ij = np.sum(self.optmodels[i, :](X), axis = 1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood


class MixNB2(BaseNB):
    """
    Mix Naive Bayes2 (MixNB)
    Different features having different distributions: if a feature has less than 5 discrete values, use ratio of + - samples to characterize this feature; Other wise follow MixNB

    Attributes
    ----------
    `class_prior_` : array, shape = [n_classes]
        probability of each class.

    `lamb_` : array, shape = [n_classes, n_features]
        mean of each feature per class
    `model_` : array, shape = [n_classes, ]
        model class

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import PossionNB
    >>> clf = PoissonNB()
    >>> clf.fit(X, Y)
    PossionNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """
    # norm_func = lambda x, sigma, theta: 1 if np.isnan(x) else -np.log(sigma)- 0.5 * np.log(2 * np.pi) - 0.5 * ((x - theta)/sigma) ** 2
    norm_func = lambda x, sigma, theta: 1 if np.isnan(x) else -0.5 * np.log(2 * np.pi*sigma) - 0.5 * ((x - theta)**2/sigma) 
    # if x is na
    # return 1
    # else 
    # return -np.log(sigma)- 0.5 * np.log(2 * np.pi) - 0.5 * ((x - theta)/sigma) ** 2
    norm_func = np.vectorize(norm_func)
    pois_func = lambda x, lamb: 1 if np.isnan(x) else x* np.log(lamb)- lamb - np.log(factorial(x))
    pois_func = np.vectorize(pois_func)
    # rat denotes the ratio between freq of + sample (or - sample), it's a dictionary, with different discrete values for a feature as keys
    ratio_func = lambda x, rat: 1 if np.isnan(x) else np.log(rat[x])
    ratio_func = np.vectorize(ratio_func)


    def __init__(self, models = ['norm', 'poisson', 'ratio'], funcs = {'norm': norm_func,'poisson': pois_func, 'ratio': ratio_func}):
        self.models = models
        self.funcs = funcs

    def fit(self, X, y):
        """Fit Mix Naive Bayes according to X, y

        Parameters
        ----------
        X : structured array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features. Each coloumn can
            be accessed by column name.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """

        #X, y = check_arrays(X, y, sparse_format='dense')
        y = column_or_1d(y, warn=True)
        n_samples, n_features = X.shape
        self.n_samples = n_samples
        self.features_ = n_features
        self.classes_ = unique_y = np.unique(y)
        n_classes = unique_y.shape[0]
        self.class_prior_ = np.zeros(n_classes)
        epsilon = 1e-9
        self.distnames = np.array(range(n_features), dtype = object)
        self.optmodels = np.zeros((n_classes, n_features), dtype = object)
        num_samples = X.shape[0]
        for j in range(0, n_features):
            # Feature column
            fcol = X[:,j]
            if itemfreq(fcol).shape[0] <= 5:
                distname = 'ratio'
            elif sum(fcol>=0) != num_samples:
                distname = 'norm'
            else:
                distname, _ = self._max_fit(fcol, y)
            self.distnames[j] = distname
            for i, y_i in enumerate(unique_y):
                fcoli = fcol[y == y_i]
                self.optmodels[i, j] = self._fit_model(fcoli, distname)
                # This step seems redundant but I don't see any better place to put this
                self.class_prior_[i] = np.float(fcoli.shape[0]) / n_samples
        return self

    def _max_fit(self, fcol, y):
        """Determine the best fit for one feature column

        Parameters
        ----------
        fcol: feature column


        Returns
        ----------

        distname: optimal distribution name
        optmodel: optimal distribution function with feature as argument

        NOTE:

        Works for Discrete Fit

        """
        goodness = {}
        funcs = {}
        for dis in ['norm','poisson']:
            func = self._fit_model(fcol,dis)
            funcs[dis] = func
            # Use Chi-square to measure goodness
            goodness[dis] = self._get_goodness(func, fcol)
        distname = min(goodness, key=goodness.get)
        optmodel = funcs[distname]
        return distname, optmodel


    def _get_goodness(self, func, fcol):
        """Calculate the goodness with Pearson's chi-squared test

        Parameters
        ----------
        func: fit function
        fcol: feature column


        Returns
        ----------

        goodness

        """
        itfreq = itemfreq(fcol)
        uniqueVars = itfreq[:,0]
        freq = itfreq[:,1]
        #freq = freq/sum(freq)
        freq = freq/self.n_samples
        predFreq = np.exp(func(uniqueVars))
        # predFreq = predFreq/sum(predFreq)
        goodness = chisquare(predFreq,freq)[0]
        return goodness


    def _fit_model(self, fcol, dis):

        """Determine the best fit for one feature column given distribution name

        Parameters
        ----------
        fcol: feature column, array
        dis: distribution name, String


        Returns
        ----------
        function: fit model with feature as argument

        """
        if dis == 'ratio':
            itfreq = itemfreq(fcol)
            uniqueVars = itfreq[:,0]
            freq = itfreq[:,1]
            rat = freq/self.n_samples
            rat = dict(zip(uniqueVars, rat.T))
            func = lambda x: self. funcs[dis](x, rat)
        if dis == 'poisson':
            lamb = np.nanmean(fcol, axis = 0)
            func = lambda x: self.funcs[dis](x, lamb)
        if dis ==   'norm':
            sigma = np.nanvar(fcol, axis=0)
            theta = np.nanmean(fcol, axis = 0)
            func = lambda x: self.funcs[dis](x, sigma, theta)
        return np.vectorize(func)

    def _joint_log_likelihood(self, X):
        #X = array2d(X)
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = np.sum([self.optmodels[i, :][j](X[:,j]) for j in range(self.features_)],axis = 0)
            # n_ij = np.sum(self.optmodels[i, :](X), axis = 1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood