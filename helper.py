from __future__ import division, print_function, absolute_import
import numpy as np
from scipy import interpolate
from scipy.interpolate.interpnd import LinearNDInterpolator, NDInterpolatorBase, \
     CloughTocher2DInterpolator, _ndim_coords_from_arrays
from scipy.spatial import cKDTree
from scipy.misc import factorial as facto
from scipy.optimize import fmin, fmin_bfgs, fminbound
import EnjoyLifePred as ELP
# from sklearn.naive_bayes import PoissonNB
from statsmodels.discrete.discrete_model import Poisson as pois
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import os
os.environ['R_HOME'] = "/opt/local/Library/Frameworks/R.framework/Resources"
from rpy2.robjects.packages import importr
MASS = importr('MASS')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

__all__ = ['griddata', 'NearestNDInterpolator', 'LinearNDInterpolator',
           'CloughTocher2DInterpolator']
def griddata(points, values, xi, method='linear', fill_value=np.nan, tol = 1e-6):
    points = _ndim_coords_from_arrays(points)

    if points.ndim < 2:
        ndim = points.ndim
    else:
        ndim = points.shape[-1]

    if ndim == 1 and method in ('nearest', 'linear', 'cubic'):
        from interpolate import interp1d
        points = points.ravel()
        if isinstance(xi, tuple):
            if len(xi) != 1:
                raise ValueError("invalid number of dimensions in xi")
            xi, = xi
        # Sort points/values together, necessary as input for interp1d
        idx = np.argsort(points)
        points = points[idx]
        values = values[idx]
        ip = interp1d(points, values, kind=method, axis=0, bounds_error=False,
                      fill_value=fill_value)
        return ip(xi)
    elif method == 'nearest':
        ip = NearestNDInterpolator(points, values)
        return ip(xi)
    elif method == 'linear':
        ip = LinearNDInterpolator(points, values, fill_value=fill_value)
        return ip(xi)
    elif method == 'cubic' and ndim == 2:
        ip = CloughTocher2DInterpolator(points, values, fill_value=fill_value, tol = tol)
        return ip(xi)
    else:
        raise ValueError("Unknown interpolation method %r for "
                         "%d dimensional data" % (method, ndim))



def argmax(fucntion):
    """a function with only one variable, we out put the argmax"""
    x = fmin(- function)
    return x

def funcadd(a, b):
    return lambda lamb: a(lamb)+ b(lamb)

def testargmax():
    data_path = '../MSPrediction-R/Data Scripts/data/predData.h5'
    obj = 'fam2_bin'
    target = 'EnjoyLife'
    # X, y, featureNames = ELP.pred_prep(data_path, obj, target)
    X = np.random.randint(5, size= (10,6))
    # loglike for Poisson
    func = lambda lamb, x: x* np.log(lamb)- lamb - np.log(facto(x))
    # func = lambda lamb, x: facto(x) - lamb + x
    vecfunc = np.vectorize(func)
    X = X.astype(int)
    # X = np.random.poisson(5,100)
    [nrow, ncol] = X.shape
    negloglikemat = np.zeros(X.shape, dtype = object)
    negloglikecolsum = np.zeros(ncol, dtype = object)
    la = np.zeros(X.shape, dtype = object)
    # lambinit = np.zeros(ncol)
    for j in range(0, ncol):
        negloglikemat[:,j] = map(lambda lamb: lambda x: -vecfunc(lamb,x), X[:,j])
        temp = lambda lamb: 0
        for i, item in enumerate(negloglikemat[:,j]):
            temp = funcadd(item, temp)
        negloglikecolsum[j] = temp
        lambopt = fmin(func = negloglikecolsum[j], xtol=0.0001, ftol=0.0001, x0 = 0.0)
        # lambopt = fminbound(func = negloglikecolsum[j], x1 = 0.00, x2 = 5.00, xtol=0.001, maxfun = 100, disp = 2)
        # la[:, j] = fmin(func = negloglikecolsum[j], xtol=0.0001, ftol=0.0001, x0 = 0.00, disp = True)
        la[:, j] = lambopt
        # fmin_bfgs
        # loglikecolsum[i] = lambda lamb: np.sum(loglikemat[:,i](lamb))
    # initial guess for lamb

    return X, negloglikemat, negloglikecolsum, la
    # optlamb = fmin(loglikecolsum, lambinit)


def newTest():
    